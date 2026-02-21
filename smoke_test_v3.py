# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
End-to-end smoke test for Pipeline V3.
Tests MultiScaleFusionUNet, losses, and inference.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src_v3.features import GPUFeatureExtractor
from src_v3.model import get_model, count_parameters, MultiScaleFusionUNet
from src_v3.losses import CompositeLossV3
from src_v3.dataset import _compute_skeleton_mask
from src_v3.postprocess import postprocess_prediction
from src_v3.inference import sliding_window_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("SMOKE TEST: Pipeline V3 (Multi-Scale Learned Fusion)")
print("Device: {}".format(device))
print("=" * 60)

# ---- 1. GPU Feature extractor ----
print("\n[1/6] GPU Feature extractor...")
feat_ext = GPUFeatureExtractor().to(device)
raw_tensor = torch.randn(2, 1, 128, 128, 128, device=device)
with torch.no_grad():
    gpu_feat = feat_ext(raw_tensor)
assert gpu_feat.shape == (2, 6, 128, 128, 128), "Expected (2,6,128,128,128), got {}".format(gpu_feat.shape)
print("  OK: {} -> {}".format(raw_tensor.shape, gpu_feat.shape))

# ---- 2. MultiScaleFusionUNet ----
print("\n[2/6] MultiScaleFusionUNet...")
model = get_model(in_channels=6, num_classes=3, base_filters=8)
assert isinstance(model, MultiScaleFusionUNet)
n_params = count_parameters(model)
x = torch.randn(1, 6, 128, 128, 128)
out = model(x)
assert out.shape == (1, 3, 128, 128, 128), "Expected (1,3,128,128,128), got {}".format(out.shape)
print("  OK: {} params, logits={}".format(n_params, out.shape))

# ---- 3. Skeleton precomputation ----
print("\n[3/6] Skeleton precomputation...")
label_patch = np.zeros((32, 32, 32), dtype=np.uint8)
label_patch[14:18, 5:28, 5:28] = 1
skel = _compute_skeleton_mask(label_patch, dilation_radius=2)
assert skel.shape == label_patch.shape
assert skel.sum() > 0
print("  OK: skeleton {} voxels".format(skel.sum()))

# ---- 4. CompositeLossV3 ----
print("\n[4/6] CompositeLossV3...")
comp_loss = CompositeLossV3()
logits = torch.randn(2, 3, 128, 128, 128)
targets = torch.randint(0, 3, (2, 128, 128, 128))
skeleton_batch = torch.zeros(2, 128, 128, 128)
skeleton_batch[0, 60:68, 50:78, 50:78] = 1.0
losses = comp_loss(logits, targets, skeleton_batch)
assert "total" in losses
assert torch.isfinite(losses["total"])
print("  OK: total={:.4f}".format(losses["total"].item()))

# ---- 5. End-to-end GPU pipeline + inference ----
print("\n[5/6] End-to-end GPU pipeline + inference...")
model_gpu = get_model(in_channels=6, num_classes=3, base_filters=8).to(device)
feat_ext_gpu = GPUFeatureExtractor().to(device)

model_gpu.train()
raw_in = torch.randn(2, 1, 128, 128, 128, device=device)
with torch.no_grad():
    features = feat_ext_gpu(raw_in)
logits_train = model_gpu(features)
assert logits_train.shape == (2, 3, 128, 128, 128)
print("  Training: raw {} -> feat {} -> logits {}".format(
    raw_in.shape, features.shape, logits_train.shape))

model_gpu.eval()
vol_test = np.random.randint(0, 256, (160, 160, 160), dtype=np.uint8)
probs = sliding_window_inference(
    model_gpu, feat_ext_gpu, vol_test,
    patch_size=128, stride=64, device=device, use_amp=False, batch_size=2,
)
assert probs.shape == (3, 160, 160, 160), "Expected (3,160,160,160), got {}".format(probs.shape)
pred_surface = (probs.argmax(axis=0) == 1).astype(np.uint8)
print("  Inference: probs={}, surface voxels={}".format(probs.shape, pred_surface.sum()))

# ---- 6. Post-processing ----
print("\n[6/6] Post-processing...")
pred_test = np.zeros((64, 64, 64), dtype=np.uint8)
pred_test[10:20, 10:54, 10:54] = 1
pred_clean = postprocess_prediction(pred_test, min_component_size=50)
print("  Postprocess: {} -> {} voxels".format(pred_test.sum(), pred_clean.sum()))

print("\n" + "=" * 60)
print("ALL PIPELINE V3 SMOKE TESTS PASSED!")
print("=" * 60)
