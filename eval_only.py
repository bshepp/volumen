"""Quick post-process + evaluate on the raw prediction from inference."""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import tifffile

sys.path.insert(0, os.path.dirname(__file__))

from src.features import compute_features
from src.model import get_model
from src.postprocess import postprocess_prediction
from src.evaluate import compute_competition_score
from src.inference import sliding_window_inference

DATA_DIR = r"f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection"
MODEL_PATH = r"f:\kaggle\vesuvius_challenge\outputs_local\best_model.pth"
OUTPUT_DIR = r"f:\kaggle\vesuvius_challenge\outputs_local"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

# ---- Load model + compute features + run inference ----
print("Loading model...")
model = get_model(in_channels=6, num_classes=3, base_filters=16, depth=3)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

print("Loading test volume...")
test_vol = tifffile.imread(os.path.join(DATA_DIR, "test_images", "1407735.tif"))

print("Computing features...")
t0 = time.time()
features = compute_features(test_vol, normalize=True)
print(f"Features: {features.shape}, {time.time()-t0:.1f}s")

print("Sliding window inference (64^3 patches, stride 32)...")
t0 = time.time()
probs = sliding_window_inference(
    model, features, patch_size=64, stride=32,
    device=device, use_amp=use_amp, num_classes=3,
)
print(f"Inference: {time.time()-t0:.1f}s")

# ---- Extract raw prediction ----
pred_raw = (probs.argmax(axis=0) == 1).astype(np.uint8)
print(f"\nRaw prediction: {pred_raw.sum()} surface voxels ({pred_raw.sum()/pred_raw.size*100:.1f}%)")

# ---- Post-processing ----
print("Post-processing...")
t0 = time.time()
pred_clean = postprocess_prediction(
    pred_raw,
    min_component_size=500,
    bridge_threshold=3,
    fill_holes=True,
    min_sheet_spacing=10,
)
print(f"Post-processing: {time.time()-t0:.1f}s")
print(f"After post-processing: {pred_clean.sum()} voxels ({pred_clean.sum()/pred_clean.size*100:.1f}%)")

# Save
tifffile.imwrite(os.path.join(OUTPUT_DIR, "pred_1407735.tif"), pred_clean)

# ---- Evaluate against deprecated GT ----
print("\n=== Evaluation against deprecated label ===")
gt = tifffile.imread(os.path.join(DATA_DIR, "deprecated_train_labels", "1407735.tif"))
print(f"GT surface: {gt.sum()} voxels ({gt.sum()/gt.size*100:.1f}%)")

scores_raw = compute_competition_score(pred_raw, gt)
print(f"\nRaw prediction score: {scores_raw['score']:.4f}")
print(f"  SurfaceDice: {scores_raw['surface_dice']:.4f}")
print(f"  VOI:         {scores_raw['voi_score']:.4f}")
print(f"  Topo:        {scores_raw['topo_score']:.4f}")

scores_pp = compute_competition_score(pred_clean, gt)
print(f"\nPost-processed score: {scores_pp['score']:.4f}")
print(f"  SurfaceDice: {scores_pp['surface_dice']:.4f}")
print(f"  VOI:         {scores_pp['voi_score']:.4f}")
print(f"  Topo:        {scores_pp['topo_score']:.4f}")

improvement = scores_pp['score'] - scores_raw['score']
print(f"\nPost-processing improvement: {'+' if improvement >= 0 else ''}{improvement:.4f}")
print("\nDone!")
