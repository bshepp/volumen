"""
Local validation training run.
Reduced settings for GTX 1650 SUPER (4GB VRAM):
  - 64^3 patches (instead of 128^3)
  - 16 base filters (instead of 32)
  - depth 3 (instead of 4)
  - batch_size 1
  - 5 epochs
  - Subset of training data
This is NOT for competition — just to validate the full loop works.
"""
import os
import sys
import time
import json

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import VesuviusDataset, get_scroll_splits, get_deprecated_ids
from src.losses import CompositeLoss
from src.model import get_model, count_parameters

# ---- Config ----
DATA_DIR = r"f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection"
OUTPUT_DIR = r"f:\kaggle\vesuvius_challenge\outputs_local"
EPOCHS = 5
BATCH_SIZE = 1
PATCH_SIZE = 64
BASE_FILTERS = 16
DEPTH = 3
LR = 1e-3
CLDICE_ITERS = 5
NUM_WORKERS = 0  # safer on Windows
TRAIN_SAMPLES = 20  # use only a subset for speed
VAL_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Device: {device}, AMP: {use_amp}")

# Data
deprecated = get_deprecated_ids(DATA_DIR)
train_ids, val_ids = get_scroll_splits(
    os.path.join(DATA_DIR, "train.csv"),
    val_scroll_ids=["26002"],
    deprecated_ids=set(deprecated),
)
# Use subset
train_ids = train_ids[:TRAIN_SAMPLES]
val_ids = val_ids[:VAL_SAMPLES]
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

train_ds = VesuviusDataset(
    data_dir=DATA_DIR,
    sample_ids=train_ids,
    patch_size=PATCH_SIZE,
    patches_per_volume=2,
    surface_bias=0.7,
    augment=True,
)
val_ds = VesuviusDataset(
    data_dir=DATA_DIR,
    sample_ids=val_ids,
    patch_size=PATCH_SIZE,
    patches_per_volume=1,
    surface_bias=0.8,
    augment=False,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

# Model
model = get_model(in_channels=6, num_classes=3, base_filters=BASE_FILTERS, depth=DEPTH).to(device)
print(f"Model: {count_parameters(model):,} params")

# Loss, optimizer, scheduler
loss_fn = CompositeLoss(cldice_iters=CLDICE_ITERS)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
scaler = GradScaler(enabled=use_amp)

# ---- Training ----
best_val_loss = float("inf")
history = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    
    # Train
    model.train()
    train_loss = 0.0
    n_train = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast(dtype=torch.float16):
                logits = model(images)
                losses = loss_fn(logits, labels)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            losses = loss_fn(logits, labels)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss += losses["total"].item()
        n_train += 1
        
        if batch_idx % 10 == 0:
            print(f"  E{epoch} [{batch_idx}/{len(train_loader)}] "
                  f"loss={losses['total'].item():.4f} "
                  f"(ce={losses['ce_dice'].item():.3f} cl={losses['cldice'].item():.3f} bd={losses['boundary'].item():.3f})")

    avg_train = train_loss / max(n_train, 1)

    # Validate
    model.eval()
    val_loss = 0.0
    n_val = 0
    surface_tp = surface_fp = surface_fn = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with autocast(dtype=torch.float16):
                    logits = model(images)
                    losses = loss_fn(logits, labels)
            else:
                logits = model(images)
                losses = loss_fn(logits, labels)
            val_loss += losses["total"].item()
            n_val += 1
            
            preds = logits.argmax(dim=1)
            pred_s = (preds == 1)
            gt_s = (labels == 1)
            surface_tp += (pred_s & gt_s).sum().item()
            surface_fp += (pred_s & ~gt_s).sum().item()
            surface_fn += (~pred_s & gt_s).sum().item()

    avg_val = val_loss / max(n_val, 1)
    dice_denom = 2 * surface_tp + surface_fp + surface_fn
    surf_dice = 2 * surface_tp / max(dice_denom, 1)

    scheduler.step()
    elapsed = time.time() - t0

    print(f"Epoch {epoch}/{EPOCHS}: train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
          f"surf_dice={surf_dice:.4f} time={elapsed:.1f}s")

    record = {"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "surf_dice": surf_dice}
    history.append(record)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
        }, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  ** Best model saved (val_loss={best_val_loss:.4f}) **")

# Save last model too
torch.save({
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
}, os.path.join(OUTPUT_DIR, "last_model.pth"))

with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\n=== Local validation training complete ===")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Outputs: {OUTPUT_DIR}")
