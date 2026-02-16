"""Quick validation that the full training pipeline works with real data."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import VesuviusDataset, get_scroll_splits, get_deprecated_ids
from src.model import get_model, count_parameters
from src.losses import CompositeLoss
import torch

data_dir = r"f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection"
deprecated = get_deprecated_ids(data_dir)
train_ids, val_ids = get_scroll_splits(
    os.path.join(data_dir, "train.csv"),
    val_scroll_ids=["26002"],
    deprecated_ids=set(deprecated),
)

print(f"Train: {len(train_ids)} samples, Val: {len(val_ids)} samples")

# Model
model = get_model(in_channels=6, num_classes=3, base_filters=32, depth=4)
print(f"Full model: {count_parameters(model):,} params")

# Loss
loss_fn = CompositeLoss()

# Test one forward + backward pass with real data
ds = VesuviusDataset(
    data_dir=data_dir,
    sample_ids=train_ids[:1],
    patch_size=64,
    patches_per_volume=1,
    augment=True,
)
feat, lbl = ds[0]
feat = feat.unsqueeze(0)  # (1, 6, 64, 64, 64)
lbl = lbl.unsqueeze(0)  # (1, 64, 64, 64)

print(f"Input: {feat.shape}, Label: {lbl.shape}")

# Forward
logits = model(feat)
losses = loss_fn(logits, lbl)
print(f"Loss total: {losses['total'].item():.4f}")
print(f"  CE+Dice: {losses['ce_dice'].item():.4f}")
print(f"  clDice:  {losses['cldice'].item():.4f}")
print(f"  Boundary: {losses['boundary'].item():.4f}")

# Backward
losses["total"].backward()
grad_norm = sum(
    p.grad.norm().item() for p in model.parameters() if p.grad is not None
)
print(f"Gradient norm: {grad_norm:.4f}")
print("Full training pipeline validated!")
