# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
Full inference pipeline for Vesuvius Challenge Surface Detection — Pipeline V3.

MultiScaleFusionUNet returns fused logits directly. Sliding window 128³;
model handles multi-scale internally.

WARNING: This is Pipeline V3. Do NOT import from src/ or src_v2/.
See PIPELINES.md in the project root.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import tifffile

from .features import GPUFeatureExtractor
from .model import get_model
from .postprocess import postprocess_prediction


def sliding_window_inference(
    model: torch.nn.Module,
    feat_extractor: GPUFeatureExtractor,
    volume: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
    num_classes: int = 3,
    batch_size: int = 4,
) -> np.ndarray:
    """Run sliding window inference with GPU feature extraction."""
    model.eval()
    feat_extractor.eval()
    Z, Y, X = volume.shape

    pad_z = max(0, patch_size - Z)
    pad_y = max(0, patch_size - Y)
    pad_x = max(0, patch_size - X)

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        volume = np.pad(volume, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="reflect")

    Zp, Yp, Xp = volume.shape

    prob_sum = np.zeros((num_classes, Zp, Yp, Xp), dtype=np.float32)
    count = np.zeros((Zp, Yp, Xp), dtype=np.float32)

    z_starts = list(range(0, Zp - patch_size + 1, stride))
    y_starts = list(range(0, Yp - patch_size + 1, stride))
    x_starts = list(range(0, Xp - patch_size + 1, stride))

    if z_starts[-1] + patch_size < Zp:
        z_starts.append(Zp - patch_size)
    if y_starts[-1] + patch_size < Yp:
        y_starts.append(Yp - patch_size)
    if x_starts[-1] + patch_size < Xp:
        x_starts.append(Xp - patch_size)

    origins = [(z0, y0, x0) for z0 in z_starts for y0 in y_starts for x0 in x_starts]
    total_patches = len(origins)
    print(f"  [V3] Sliding window: {total_patches} patches "
          f"({len(z_starts)}x{len(y_starts)}x{len(x_starts)}), "
          f"batch_size={batch_size}")

    vol_f32 = volume.astype(np.float32) / 255.0

    with torch.no_grad():
        for batch_start in range(0, total_patches, batch_size):
            batch_origins = origins[batch_start : batch_start + batch_size]
            B = len(batch_origins)

            patches = np.zeros((B, 1, patch_size, patch_size, patch_size), dtype=np.float32)
            for i, (z0, y0, x0) in enumerate(batch_origins):
                patches[i, 0] = vol_f32[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size]

            patch_tensor = torch.from_numpy(patches).to(device)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    features = feat_extractor(patch_tensor)
                    logits = model(features)
            else:
                features = feat_extractor(patch_tensor)
                logits = model(features)

            probs = F.softmax(logits, dim=1).cpu().float().numpy()

            for i, (z0, y0, x0) in enumerate(batch_origins):
                prob_sum[:, z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] += probs[i]
                count[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] += 1.0

            done = min(batch_start + batch_size, total_patches)
            if done % 20 == 0 or done == total_patches:
                print(f"    [V3] Patch {done}/{total_patches}", end="\r")

    print(f"    [V3] Done: {total_patches} patches")
    count = np.maximum(count, 1.0)
    prob_avg = prob_sum / count[np.newaxis]
    return prob_avg[:, :Z, :Y, :X]


def test_time_augmentation(
    model: torch.nn.Module,
    feat_extractor: GPUFeatureExtractor,
    volume: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
    num_classes: int = 3,
    batch_size: int = 4,
) -> np.ndarray:
    """Run inference with test-time augmentation (8 flip combinations)."""
    Z, Y, X = volume.shape
    prob_sum = np.zeros((num_classes, Z, Y, X), dtype=np.float32)

    flip_configs = [(fz, fy, fx) for fz in [False, True] for fy in [False, True] for fx in [False, True]]

    for i, (fz, fy, fx) in enumerate(flip_configs):
        print(f"  [V3] TTA {i+1}/8: flip_z={fz}, flip_y={fy}, flip_x={fx}")

        vol_aug = volume.copy()
        if fz:
            vol_aug = np.flip(vol_aug, axis=0)
        if fy:
            vol_aug = np.flip(vol_aug, axis=1)
        if fx:
            vol_aug = np.flip(vol_aug, axis=2)
        vol_aug = np.ascontiguousarray(vol_aug)

        probs = sliding_window_inference(
            model, feat_extractor, vol_aug,
            patch_size, stride, device, use_amp, num_classes, batch_size,
        )

        if fx:
            probs = np.flip(probs, axis=3)
        if fy:
            probs = np.flip(probs, axis=2)
        if fz:
            probs = np.flip(probs, axis=1)

        prob_sum += np.ascontiguousarray(probs)

    return prob_sum / len(flip_configs)


def run_inference(
    model_path: str,
    test_volume_path: str,
    output_path: str,
    patch_size: int = 128,
    stride: int = 64,
    use_tta: bool = True,
    use_postprocess: bool = True,
    device: Optional[torch.device] = None,
    base_filters: int = 16,
    batch_size: int = 4,
) -> np.ndarray:
    """Full inference pipeline for Pipeline V3."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("[V3] Device: {}, AMP: {}".format(device, use_amp))

    print("[V3] Loading model...")
    model = get_model(in_channels=6, num_classes=3, base_filters=base_filters)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    feat_extractor = GPUFeatureExtractor().to(device)
    feat_extractor.eval()

    print("[V3] Loading test volume...")
    volume = tifffile.imread(test_volume_path)
    print("  Volume shape: {}, dtype: {}".format(volume.shape, volume.dtype))

    if use_tta:
        print("[V3] Running inference with TTA...")
        probs = test_time_augmentation(
            model, feat_extractor, volume,
            patch_size, stride, device, use_amp, batch_size=batch_size,
        )
    else:
        print("[V3] Running inference...")
        probs = sliding_window_inference(
            model, feat_extractor, volume,
            patch_size, stride, device, use_amp, batch_size=batch_size,
        )

    pred_classes = probs.argmax(axis=0)
    pred_surface = (pred_classes == 1).astype(np.uint8)

    print("  [V3] Predicted surface voxels: {} ({:.1f}%)".format(
        pred_surface.sum(), pred_surface.sum() / pred_surface.size * 100))

    if use_postprocess:
        print("[V3] Applying post-processing...")
        pred_surface = postprocess_prediction(pred_surface)
        print("  [V3] After post-processing: {} voxels ({:.1f}%)".format(
            pred_surface.sum(), pred_surface.sum() / pred_surface.size * 100))

    print("[V3] Saving prediction to {}".format(output_path))
    tifffile.imwrite(output_path, pred_surface)

    return pred_surface
