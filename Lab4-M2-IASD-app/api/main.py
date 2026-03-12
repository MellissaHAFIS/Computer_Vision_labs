"""
api/main.py
──────────────────────────────────────────────────────────────────────────────
FastAPI backend — exposes two endpoints consumed by the Streamlit frontend.

Endpoints
─────────
POST /train
    Body (JSON):
        {
            "model_name"   : "U-Net" | "ResNet" | "Inception",
            "learning_rate": 1e-3,
            "epochs"       : 10,
            "batch_size"   : 32,
            "dropout_rate" : 0.5,
            "image_size"   : 224
        }
    Response (JSON):
        {
            "train_losses" : [0.72, 0.55, ...],
            "val_losses"   : [0.80, 0.61, ...],
            "train_accs"   : [0.61, 0.74, ...],
            "val_accs"     : [0.55, 0.70, ...],
            "train_aucs"   : [...],
            "val_aucs"     : [...],
            "best_val_auc" : 0.94
        }

POST /predict
    Body (JSON):
        {
            "model_name": "U-Net" | "ResNet" | "Inception",
            "image_size": 224
        }
    Response (JSON):
        {
            "predictions": [
                {"id": "img_0001", "prediction": 0.91},
                {"id": "img_0002", "prediction": 0.07},
                ...
            ]
        }

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

# Allow importing from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image

from models.unet import UNet
from models.resnet import ResNet
from models.inception import Inception


app = FastAPI(title="Pneumonia Detection API")

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
TRAIN_DIR   = ROOT / "data" / "train"
VAL_DIR     = ROOT / "data" / "val"
TEST_DIR    = ROOT / "submission" / "test_for_students"
WEIGHTS_DIR = ROOT / "api" / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(torch.backends.mps.is_available())  # True si le GPU Apple peut être utilisé

# ── Model registry ─────────────────────────────────────────────────────────────
# FIX 1: MODEL_REGISTRY was imported from models/__init__.py but never defined
# there. We define it here directly so the file is self-contained.
# Each factory accepts (in_channels, dropout_rate) to match the /train call.

def _make_unet(in_channels: int = 3, dropout_rate: float = 0.4) -> nn.Module:
    return UNet(in_channels=in_channels, dropout_rate=dropout_rate)

def _make_resnet(in_channels: int = 3, dropout_rate: float = 0.4) -> nn.Module:
    return ResNet(in_channels=in_channels, dropout_rate=dropout_rate)

def _make_inception(in_channels: int = 3, dropout_rate: float = 0.4) -> nn.Module:
    return Inception(in_channels=in_channels, dropout_rate=dropout_rate)

MODEL_REGISTRY = {
    "U-Net":     _make_unet,
    "ResNet":    _make_resnet,
    "Inception": _make_inception,
}


# ── Request / Response schemas ─────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_name:    str   = Field(..., description="U-Net | ResNet | Inception")
    learning_rate: float = Field(1e-3,  ge=1e-6, le=1.0)
    epochs:        int   = Field(10,    ge=1,    le=100)
    batch_size:    int   = Field(32,    ge=4,    le=128)
    dropout_rate:  float = Field(0.5,   ge=0.0,  le=0.9)
    image_size:    int   = Field(224,   ge=64,   le=512)


class TrainResponse(BaseModel):
    train_losses: list[float]
    val_losses:   list[float]
    train_accs:   list[float]
    val_accs:     list[float]
    train_aucs:   list[float]
    val_aucs:     list[float]
    best_val_auc: float


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="U-Net | ResNet | Inception")
    image_size: int = Field(224, ge=64, le=512)


class PredictionItem(BaseModel):
    id:         str
    prediction: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_transforms(image_size: int, augment: bool = False):
    """
    Return torchvision transforms for train (with augmentation) or val/test.
    FIX 2: the original Compose([]) lists were completely empty (only comments).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(image_size: int, batch_size: int):
    """Build train and val DataLoaders from ImageFolder structure."""
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=get_transforms(image_size, augment=True))
    val_dataset   = datasets.ImageFolder(str(VAL_DIR),   transform=get_transforms(image_size, augment=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader


# ── Training helpers ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, model_name: str) -> tuple[float, float, float]:
    """
    Run one full pass over the training set.

    FIX 3: Inception returns (logits, aux_logits) during training.
    We unpack the tuple when model_name == "Inception" and add the
    auxiliary loss (weighted 0.3) to the main loss, exactly as in the
    original GoogLeNet paper.
    """
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    is_inception = model_name == "Inception"

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        output = model(images)

        # Inception returns (main_logits, aux_logits) during training
        if is_inception and isinstance(output, tuple):
            logits, aux_logits = output
            loss = criterion(logits, labels) + 0.3 * criterion(aux_logits, labels)
        else:
            logits = output
            loss   = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = float(np.mean((np.array(all_probs) >= 0.5) == np.array(all_labels)))
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return avg_loss, accuracy, auc


def evaluate(model, loader, criterion) -> tuple[float, float, float]:
    """
    Evaluate the model on a validation loader.
    Note: Inception.use_aux only fires during model.train(), so eval() is safe.
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            logits = model(images)   # always a plain tensor in eval mode
            loss   = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = float(np.mean((np.array(all_probs) >= 0.5) == np.array(all_labels)))
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return avg_loss, accuracy, auc


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    if req.model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model_name}'. Choose from {list(MODEL_REGISTRY)}"
        )

    train_loader, val_loader = get_dataloaders(req.image_size, req.batch_size)

    # FIX 4: original call was MODEL_REGISTRY[name](in_channels=3, dropout_rate=...)
    # but UNet/ResNet/Inception don't accept dropout_rate as a constructor arg.
    # The factory wrappers (_make_*) above absorb that parameter safely.
    model     = MODEL_REGISTRY[req.model_name](in_channels=3, dropout_rate=req.dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=req.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.35]).to(DEVICE)) # nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    train_aucs,   val_aucs   = [], []
    best_val_auc = 0.0

    for epoch in range(req.epochs):
        tr_loss, tr_acc, tr_auc = train_one_epoch(model, train_loader, optimizer, criterion, req.model_name)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, criterion)

        scheduler.step(vl_auc)

        train_losses.append(round(tr_loss, 6))
        val_losses.append(round(vl_loss, 6))
        train_accs.append(round(tr_acc, 6))
        val_accs.append(round(vl_acc, 6))
        train_aucs.append(round(tr_auc, 6))
        val_aucs.append(round(vl_auc, 6))

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            weight_path  = WEIGHTS_DIR / f"{req.model_name.replace(' ', '_')}_best.pt"
            torch.save(model.state_dict(), weight_path)

        print(f"Epoch {epoch+1}/{req.epochs} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} auc {tr_auc:.4f} | "
              f"Val   loss {vl_loss:.4f} acc {vl_acc:.4f} auc {vl_auc:.4f}")

    return TrainResponse(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        train_aucs=train_aucs,
        val_aucs=val_aucs,
        best_val_auc=best_val_auc,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model_name}'")

    weight_path = WEIGHTS_DIR / f"{req.model_name.replace(' ', '_')}_best.pt"
    if not weight_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No saved weights found for '{req.model_name}'. Train it first."
        )

    model = MODEL_REGISTRY[req.model_name](in_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    transform = get_transforms(req.image_size, augment=False)

    image_paths = (
        sorted(TEST_DIR.glob("*.jpeg")) +
        sorted(TEST_DIR.glob("*.jpg"))  +
        sorted(TEST_DIR.glob("*.png"))
    )

    if not image_paths:
        raise HTTPException(status_code=404, detail=f"No images found in {TEST_DIR}")

    predictions = []
    with torch.no_grad():
        for img_path in image_paths:
            image  = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            logit  = model(tensor)
            prob   = torch.sigmoid(logit).item()
            predictions.append(PredictionItem(id=img_path.stem, prediction=round(prob, 6)))

    return PredictResponse(predictions=predictions)


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "device": str(DEVICE), "models": list(MODEL_REGISTRY.keys())}