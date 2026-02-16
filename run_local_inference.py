"""
Run inference on the test volume using the locally-trained validation model.
Also evaluates against the deprecated label for sample 1407735.
"""
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

# ---- Config (must match local training) ----
DATA_DIR = r"f:\kaggle\vesuvius_challenge\vesuvius-challenge-surface-detection"
MODEL_PATH = r"f:\kaggle\vesuvius_challenge\outputs_local\best_model.pth"
OUTPUT_DIR = r"f:\kaggle\vesuvius_challenge\outputs_local"

# Reduced model (matches local training)
BASE_FILTERS = 16
DEPTH = 3

# Inference settings
PATCH_SIZE = 64  # smaller patches for 4GB VRAM
STRIDE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Device: {device}")

# ---- Load model ----
print("Loading model...")
model = get_model(in_channels=6, num_classes=3, base_filters=BASE_FILTERS, depth=DEPTH)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()
print(f"Model loaded from epoch {ckpt['epoch']}")

# ---- Load test volume ----
print("Loading test volume (1407735)...")
test_vol = tifffile.imread(os.path.join(DATA_DIR, "test_images", "1407735.tif"))
print(f"Volume: {test_vol.shape}, dtype={test_vol.dtype}")

# ---- Compute features ----
print("Computing features...")
t0 = time.time()
features = compute_features(test_vol, normalize=True)
print(f"Features: {features.shape}, computed in {time.time()-t0:.1f}s")

# ---- Sliding window inference ----
print("Running sliding window inference...")
t0 = time.time()
probs = sliding_window_inference(
    model, features,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    device=device,
    use_amp=use_amp,
    num_classes=3,
)
print(f"Inference done in {time.time()-t0:.1f}s")

# ---- Extract surface prediction ----
pred_classes = probs.argmax(axis=0)
pred_surface_raw = (pred_classes == 1).astype(np.uint8)
print(f"Raw prediction: {pred_surface_raw.sum()} surface voxels "
      f"({pred_surface_raw.sum()/pred_surface_raw.size*100:.1f}%)")

# ---- Post-processing ----
print("Post-processing...")
pred_surface = postprocess_prediction(
    pred_surface_raw,
    min_component_size=500,
    bridge_threshold=3,
    fill_holes=True,
    min_sheet_spacing=10,
)
print(f"After post-processing: {pred_surface.sum()} voxels "
      f"({pred_surface.sum()/pred_surface.size*100:.1f}%)")

# ---- Save prediction ----
pred_path = os.path.join(OUTPUT_DIR, "pred_1407735.tif")
tifffile.imwrite(pred_path, pred_surface)
print(f"Prediction saved to: {pred_path}")

# ---- Evaluate against deprecated label ----
print("\n=== Evaluation against deprecated label ===")
gt = tifffile.imread(os.path.join(DATA_DIR, "deprecated_train_labels", "1407735.tif"))
print(f"Ground truth: {gt.shape}, unique={np.unique(gt)}")

scores = compute_competition_score(pred_surface, gt)
print(f"\nCompetition Score: {scores['score']:.4f}")
print(f"  SurfaceDice@2:   {scores['surface_dice']:.4f}")
print(f"  VOI_score:       {scores['voi_score']:.4f}")
print(f"  TopoScore:       {scores['topo_score']:.4f}")

# Also score the raw (pre-postprocess) prediction
scores_raw = compute_competition_score(pred_surface_raw, gt)
print(f"\nRaw (no post-process) Score: {scores_raw['score']:.4f}")
print(f"  SurfaceDice@2:   {scores_raw['surface_dice']:.4f}")
print(f"  VOI_score:       {scores_raw['voi_score']:.4f}")
print(f"  TopoScore:       {scores_raw['topo_score']:.4f}")

# Class distribution of prediction vs GT
print(f"\nPrediction surface coverage: {pred_surface.sum()/pred_surface.size*100:.1f}%")
print(f"GT surface coverage:         {gt.sum()/gt.size*100:.1f}%")
