# Pipeline: V1 (src/) — See PIPELINES.md
"""
End-to-end smoke test for the full pipeline.
Uses a tiny synthetic volume to verify all components work together.
Tests both CPU features (legacy) and GPU features (new).
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.features import (
    compute_features, compute_features_for_patch, FEATURE_NAMES,
    GPUFeatureExtractor,
)
from src.model import get_model, count_parameters
from src.losses import CompositeLoss
from src.postprocess import postprocess_prediction
from src.evaluate import compute_competition_score
from src.inference import sliding_window_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("SMOKE TEST: Vesuvius Surface Detection Pipeline")
print(f"Device: {device}")
print("=" * 60)

# ---- 1. CPU Feature computation (legacy) ----
print("\n[1/7] CPU Feature computation (legacy)...")
vol = np.random.randint(0, 256, (64, 64, 64), dtype=np.uint8)
feat = compute_features(vol, normalize=True)
assert feat.shape == (6, 64, 64, 64), f"Expected (6,64,64,64), got {feat.shape}"
assert feat.dtype == np.float32
print(f"  OK: {feat.shape}, dtype={feat.dtype}")

patch = vol[10:42, 10:42, 10:42]
feat_patch = compute_features_for_patch(patch, normalize=True)
assert feat_patch.shape == (6, 32, 32, 32), f"Expected (6,32,32,32), got {feat_patch.shape}"
print(f"  Patch OK: {feat_patch.shape}")

# ---- 2. GPU Feature extractor ----
print("\n[2/7] GPU Feature extractor...")
feat_ext = GPUFeatureExtractor().to(device)
raw_tensor = torch.from_numpy(vol.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    gpu_feat = feat_ext(raw_tensor)
assert gpu_feat.shape == (1, 6, 64, 64, 64), f"Expected (1,6,64,64,64), got {gpu_feat.shape}"
print(f"  OK: input {raw_tensor.shape} -> output {gpu_feat.shape}")

raw_batch = torch.randn(4, 1, 32, 32, 32, device=device)
with torch.no_grad():
    gpu_feat_batch = feat_ext(raw_batch)
assert gpu_feat_batch.shape == (4, 6, 32, 32, 32)
print(f"  Batch OK: {raw_batch.shape} -> {gpu_feat_batch.shape}")

# ---- 3. Model ----
print("\n[3/7] Model...")
model = get_model(in_channels=6, num_classes=3, base_filters=16, depth=3)
n_params = count_parameters(model)
x = torch.randn(1, 6, 32, 32, 32)
with torch.no_grad():
    out = model(x)
assert out.shape == (1, 3, 32, 32, 32), f"Expected (1,3,32,32,32), got {out.shape}"
print(f"  OK: {n_params:,} params, input {x.shape} -> output {out.shape}")

# ---- 4. End-to-end GPU pipeline (raw -> features -> model) ----
print("\n[4/7] End-to-end GPU pipeline...")
model_gpu = model.to(device)
raw_in = torch.randn(4, 1, 32, 32, 32, device=device)
with torch.no_grad():
    features = feat_ext(raw_in)
    logits = model_gpu(features)
assert logits.shape == (4, 3, 32, 32, 32)
print(f"  OK: raw {raw_in.shape} -> feat {features.shape} -> logits {logits.shape}")

# ---- 5. Loss ----
print("\n[5/7] Loss function...")
loss_fn = CompositeLoss(cldice_iters=3)
logits_cpu = torch.randn(2, 3, 32, 32, 32)
targets = torch.randint(0, 3, (2, 32, 32, 32))
losses = loss_fn(logits_cpu, targets)
assert "total" in losses
assert all(torch.isfinite(v) for v in losses.values())
print(f"  OK: total={losses['total'].item():.4f}")

# ---- 6. Post-processing ----
print("\n[6/7] Post-processing...")
pred = np.zeros((64, 64, 64), dtype=np.uint8)
pred[15:25, 10:50, 10:50] = 1
pred[40:50, 10:50, 10:50] = 1
pred[25:40, 28:32, 28:32] = 1
pred[5, 5, 5] = 1
pred[60, 60, 60] = 1
print(f"  Before: {pred.sum()} voxels")
pred_clean = postprocess_prediction(pred, min_component_size=50, bridge_threshold=2)
print(f"  After: {pred_clean.sum()} voxels")
assert pred_clean.sum() < pred.sum()
print(f"  OK: reduced from {pred.sum()} to {pred_clean.sum()}")

# ---- 7. Inference pipeline (GPU features) ----
print("\n[7/7] Inference pipeline with GPU features...")
model_small = get_model(in_channels=6, num_classes=3, base_filters=16, depth=3).to(device)
feat_ext_inf = GPUFeatureExtractor().to(device)
vol_test = np.random.randint(0, 256, (64, 64, 64), dtype=np.uint8)
probs = sliding_window_inference(
    model_small, feat_ext_inf, vol_test,
    patch_size=32, stride=16, device=device, use_amp=False, batch_size=2,
)
assert probs.shape == (3, 64, 64, 64), f"Expected (3,64,64,64), got {probs.shape}"
pred_final = (probs.argmax(axis=0) == 1).astype(np.uint8)
print(f"  OK: probs={probs.shape}, pred surface voxels={pred_final.sum()}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED!")
print("=" * 60)
