# Pipeline: V2 (src_v2/) — See PIPELINES.md
"""
End-to-end smoke test for Pipeline V2.
Tests all V2 components with synthetic data: model, losses, dataset skeleton
precomputation, deep supervision, and inference.

This tests Pipeline V2 ONLY. For Pipeline V1 tests, see smoke_test.py.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src_v2.features import GPUFeatureExtractor
from src_v2.model import get_model, count_parameters, UNet3DDeepSup
from src_v2.losses import (
    FocalDiceLoss,
    SkeletonRecallLoss,
    CompositeLossV2,
    DeepSupCompositeLoss,
)
from src_v2.dataset import _compute_skeleton_mask
from src_v2.postprocess import postprocess_prediction
from src_v2.inference import sliding_window_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("SMOKE TEST: Pipeline V2 (Focal + DeepSup + SkeletonRecall)")
print(f"Device: {device}")
print("=" * 60)

# ---- 1. GPU Feature extractor (shared copy, sanity check) ----
print("\n[1/8] GPU Feature extractor...")
feat_ext = GPUFeatureExtractor().to(device)
raw_tensor = torch.randn(2, 1, 32, 32, 32, device=device)
with torch.no_grad():
    gpu_feat = feat_ext(raw_tensor)
assert gpu_feat.shape == (2, 6, 32, 32, 32), f"Expected (2,6,32,32,32), got {gpu_feat.shape}"
print(f"  OK: {raw_tensor.shape} -> {gpu_feat.shape}")

# ---- 2. UNet3DDeepSup — training mode ----
print("\n[2/8] UNet3DDeepSup (training mode)...")
model = get_model(in_channels=6, num_classes=3, base_filters=16, depth=4)
assert isinstance(model, UNet3DDeepSup)
n_params = count_parameters(model)
model.train()
x = torch.randn(1, 6, 32, 32, 32)
out = model(x)
assert isinstance(out, dict), f"Expected dict in training mode, got {type(out)}"
assert "logits" in out, "Missing 'logits' key"
assert out["logits"].shape == (1, 3, 32, 32, 32), f"Main logits shape: {out['logits'].shape}"
for key in ["aux_1", "aux_2", "aux_3"]:
    assert key in out, f"Missing '{key}' key"
    print(f"  {key}: {out[key].shape}")
print(f"  OK: {n_params:,} params, logits={out['logits'].shape}")

# ---- 3. UNet3DDeepSup — eval mode ----
print("\n[3/8] UNet3DDeepSup (eval mode)...")
model.eval()
with torch.no_grad():
    out_eval = model(x)
assert isinstance(out_eval, torch.Tensor), f"Expected tensor in eval mode, got {type(out_eval)}"
assert out_eval.shape == (1, 3, 32, 32, 32)
print(f"  OK: eval returns tensor {out_eval.shape}")

# ---- 4. FocalDiceLoss ----
print("\n[4/8] FocalDiceLoss...")
focal_dice = FocalDiceLoss(class_weights=(0.3, 3.0, 0.3), gamma=2.0)
logits = torch.randn(2, 3, 16, 16, 16)
targets = torch.randint(0, 3, (2, 16, 16, 16))
loss_fd = focal_dice(logits, targets)
assert loss_fd.ndim == 0 and torch.isfinite(loss_fd)
print(f"  OK: loss={loss_fd.item():.4f}")

# ---- 5. SkeletonRecallLoss + skeleton precomputation ----
print("\n[5/8] Skeleton precomputation + SkeletonRecallLoss...")
# Create a synthetic label with a thin surface
label_patch = np.zeros((32, 32, 32), dtype=np.uint8)
label_patch[14:18, 5:28, 5:28] = 1  # thin slab = surface
skel = _compute_skeleton_mask(label_patch, dilation_radius=2)
assert skel.shape == label_patch.shape
assert skel.dtype == np.uint8
assert skel.sum() > 0, "Skeleton should have some voxels"
assert skel.sum() <= (label_patch == 1).sum(), "Skeleton should be subset of surface"
print(f"  Skeleton: {skel.sum()} voxels out of {(label_patch == 1).sum()} surface voxels")

skel_loss_fn = SkeletonRecallLoss()
# Use logits/targets that match the skeleton spatial size (16x16x16)
logits_16 = torch.randn(1, 3, 16, 16, 16)
targets_16 = torch.randint(0, 3, (1, 16, 16, 16))
skel_tensor_16 = torch.zeros(1, 16, 16, 16)
skel_tensor_16[0, 7:9, 3:13, 3:13] = 1.0
loss_skel = skel_loss_fn(logits_16, targets_16, skel_tensor_16)
assert loss_skel.ndim == 0 and torch.isfinite(loss_skel)
print(f"  OK: skel_recall_loss={loss_skel.item():.4f}")

# Empty skeleton edge case
skel_empty_t = torch.zeros(1, 16, 16, 16)
loss_skel_empty = skel_loss_fn(logits_16, targets_16, skel_empty_t)
assert torch.isfinite(loss_skel_empty)
print(f"  OK: empty skeleton loss={loss_skel_empty.item():.4f} (should be ~0)")

# ---- 6. CompositeLossV2 (single scale) ----
print("\n[6/8] CompositeLossV2 (single scale)...")
comp_loss = CompositeLossV2()
skeleton_batch = torch.zeros(2, 16, 16, 16)
skeleton_batch[0, 7:9, 3:13, 3:13] = 1.0
losses = comp_loss(logits, targets, skeleton_batch)
assert "total" in losses
assert all(torch.isfinite(v) for v in losses.values())
print(f"  OK: total={losses['total'].item():.4f}, "
      f"focal_dice={losses['focal_dice'].item():.4f}, "
      f"skel_recall={losses['skel_recall'].item():.4f}, "
      f"boundary={losses['boundary'].item():.4f}")

# ---- 7. DeepSupCompositeLoss ----
print("\n[7/8] DeepSupCompositeLoss (multi-scale)...")
ds_loss = DeepSupCompositeLoss()
model.train()
# Build fake deep supervision outputs
outputs_dict = {
    "logits": torch.randn(2, 3, 32, 32, 32),
    "aux_1": torch.randn(2, 3, 16, 16, 16),
    "aux_2": torch.randn(2, 3, 8, 8, 8),
    "aux_3": torch.randn(2, 3, 4, 4, 4),
}
targets_32 = torch.randint(0, 3, (2, 32, 32, 32))
skeleton_32 = torch.zeros(2, 32, 32, 32)
skeleton_32[0, 14:18, 5:28, 5:28] = 1.0
ds_losses = ds_loss(outputs_dict, targets_32, skeleton_32)
assert "total" in ds_losses
assert "main_loss" in ds_losses
assert "aux_1_loss" in ds_losses
assert all(torch.isfinite(v) for v in ds_losses.values())
print(f"  OK: total={ds_losses['total'].item():.4f}, "
      f"main={ds_losses['main_loss'].item():.4f}, "
      f"aux_1={ds_losses['aux_1_loss'].item():.4f}, "
      f"aux_2={ds_losses['aux_2_loss'].item():.4f}, "
      f"aux_3={ds_losses['aux_3_loss'].item():.4f}")

# ---- 8. End-to-end: raw -> GPU features -> model -> loss (training) ----
print("\n[8/8] End-to-end GPU pipeline + inference...")
model_gpu = get_model(in_channels=6, num_classes=3, base_filters=16, depth=3).to(device)
feat_ext_gpu = GPUFeatureExtractor().to(device)

# Training forward pass
model_gpu.train()
raw_in = torch.randn(2, 1, 32, 32, 32, device=device)
with torch.no_grad():
    features = feat_ext_gpu(raw_in)
outputs_train = model_gpu(features)
assert isinstance(outputs_train, dict)
print(f"  Training: raw {raw_in.shape} -> feat {features.shape} -> logits {outputs_train['logits'].shape}")

# Inference pass
model_gpu.eval()
vol_test = np.random.randint(0, 256, (48, 48, 48), dtype=np.uint8)
probs = sliding_window_inference(
    model_gpu, feat_ext_gpu, vol_test,
    patch_size=32, stride=16, device=device, use_amp=False, batch_size=2,
)
assert probs.shape == (3, 48, 48, 48), f"Expected (3,48,48,48), got {probs.shape}"
pred_surface = (probs.argmax(axis=0) == 1).astype(np.uint8)
print(f"  Inference: probs={probs.shape}, surface voxels={pred_surface.sum()}")

# Post-processing
pred_test = np.zeros((48, 48, 48), dtype=np.uint8)
pred_test[10:20, 10:40, 10:40] = 1
pred_clean = postprocess_prediction(pred_test, min_component_size=50)
print(f"  Postprocess: {pred_test.sum()} -> {pred_clean.sum()} voxels")

print("\n" + "=" * 60)
print("ALL PIPELINE V2 SMOKE TESTS PASSED!")
print("=" * 60)
