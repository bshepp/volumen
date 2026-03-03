# Pipeline: nnU-Net (src_nnunet/) — See PIPELINES.md
"""
nnU-Net inference with 1st place post-processing.

Runs nnU-Net v2 prediction on test volumes, then applies the 1st place
post-processing pipeline (binary closing, height-map patching, LUT-based
hole plugging, global fill holes).

Usage:
    python -m src_nnunet.predict \\
        --input-dir /path/to/test_images \\
        --output-dir /path/to/predictions \\
        --model-dir /path/to/nnUNet_results/Dataset011_Vesuvius/nnUNetTrainer__nnUNetPlans__3d_fullres \\
        --folds all

Post-processing source:
    "1st Place Solution for the Vesuvius Challenge Surface Detection"
    https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/
    writeups/1st-place-solution-for-the-vesuvius-challenge-su
"""

import argparse
import os
import time

import numpy as np
import tifffile
import torch
from scipy import ndimage
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label as ndimage_label,
)


# ---------------------------------------------------------------------------
# nnU-Net inference
# ---------------------------------------------------------------------------

def run_nnunet_inference(
    input_dir: str,
    output_dir: str,
    model_dir: str,
    folds: str = "all",
    use_mirroring: bool = True,
    device: str = "cuda",
    step_size: float = 0.5,
):
    """Run nnU-Net prediction on all TIF files in input_dir."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    os.makedirs(output_dir, exist_ok=True)

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    fold_list = [folds] if folds == "all" else [int(f) for f in folds.split(",")]
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=fold_list,
        checkpoint_name="checkpoint_final.pth",
    )

    predictor.predict_from_files(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )

    return output_dir


# ---------------------------------------------------------------------------
# Post-processing (inlined from src_v2/postprocess.py for pipeline isolation)
# ---------------------------------------------------------------------------

def postprocess_prediction(
    pred: np.ndarray,
    min_component_size: int = 500,
    closing_radius: int = 3,
    enable_patching: bool = True,
    enable_hole_plugging: bool = True,
    connectivity: int = 26,
) -> np.ndarray:
    """Post-processing pipeline adapted from the 1st place solution."""
    t_total = time.time()
    pred = pred.astype(np.uint8)
    struct = _get_structure(connectivity)

    t0 = time.time()
    pred = _remove_small_components(pred, min_component_size, connectivity)
    print(f"  Step 1 (CC filter): {time.time() - t0:.2f}s")

    labeled, n = ndimage_label(pred, structure=struct)
    if n == 0:
        return pred

    slices = find_objects(labeled)
    footprint = _make_ball_footprint(closing_radius) if closing_radius > 0 else None
    pad = closing_radius if closing_radius > 0 else 0
    result = np.zeros_like(pred, dtype=np.uint8)

    t_close = time.time()
    t_patch = 0.0
    t_plug = 0.0

    for comp_id, sl in enumerate(slices, 1):
        if sl is None:
            continue
        padded_sl = _pad_slices(sl, pred.shape, pad)
        crop = (labeled[padded_sl] == comp_id).astype(np.uint8)

        if closing_radius > 0:
            crop = binary_closing(crop, structure=footprint).astype(np.uint8)

        if enable_patching:
            tp = time.time()
            crop = _height_map_patch_crop(crop)
            t_patch += time.time() - tp

        if enable_hole_plugging:
            tg = time.time()
            crop = _plug_holes_lut(crop)
            t_plug += time.time() - tg

        result[padded_sl] |= crop

    print(f"  Step 2 (closing): {time.time() - t_close - t_patch - t_plug:.2f}s")
    print(f"  Step 3 (patching): {t_patch:.2f}s")
    print(f"  Step 4 (hole plug): {t_plug:.2f}s")

    t0 = time.time()
    result = binary_fill_holes(result).astype(np.uint8)
    print(f"  Step 5 (fill holes): {time.time() - t0:.2f}s")
    print(f"  Post-processing total: {time.time() - t_total:.2f}s")

    return result


def _pad_slices(sl, shape, pad):
    return tuple(
        slice(max(0, s.start - pad), min(dim, s.stop + pad))
        for s, dim in zip(sl, shape)
    )


def _remove_small_components(mask, min_size, connectivity):
    struct = _get_structure(connectivity)
    labeled, n = ndimage_label(mask, structure=struct)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros_like(mask)
    for i, size in enumerate(sizes, 1):
        if size >= min_size:
            keep[labeled == i] = 1
    return keep.astype(np.uint8)


def _make_ball_footprint(radius):
    zz, yy, xx = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
        -radius:radius + 1,
    ]
    return (zz ** 2 + yy ** 2 + xx ** 2) <= radius ** 2


def _height_map_patch_crop(crop):
    if crop.sum() == 0:
        return crop

    best_axis, best_area = 0, 0
    for axis in range(3):
        area = crop.max(axis=axis).sum()
        if area > best_area:
            best_area = area
            best_axis = axis

    crop_t = np.moveaxis(crop, best_axis, 0)
    D, H, W = crop_t.shape

    depth_coords = np.arange(D, dtype=np.float32).reshape(D, 1, 1)
    valid_3d = crop_t.astype(bool)
    count_map = valid_3d.sum(axis=0)
    has_voxels = count_map > 0

    height_map = np.full((H, W), np.nan, dtype=np.float32)
    thick_map = np.zeros((H, W), dtype=np.float32)
    depth_sum = (depth_coords * valid_3d).sum(axis=0)
    height_map[has_voxels] = depth_sum[has_voxels] / count_map[has_voxels]
    thick_map[has_voxels] = count_map[has_voxels]

    filled_proj = binary_fill_holes(has_voxels)
    gap_mask = filled_proj & ~has_voxels
    if not gap_mask.any():
        return crop

    holes_before = _count_internal_holes(crop)

    fill_row_h = np.full((H, W), np.nan, dtype=np.float32)
    fill_row_t = np.full((H, W), np.nan, dtype=np.float32)
    for r in range(H):
        valid_cols = np.where(has_voxels[r])[0]
        gap_cols = np.where(gap_mask[r])[0]
        if len(valid_cols) >= 2 and len(gap_cols) > 0:
            fill_row_h[r, gap_cols] = np.interp(gap_cols, valid_cols, height_map[r, valid_cols])
            fill_row_t[r, gap_cols] = np.interp(gap_cols, valid_cols, thick_map[r, valid_cols])

    fill_col_h = np.full((H, W), np.nan, dtype=np.float32)
    fill_col_t = np.full((H, W), np.nan, dtype=np.float32)
    for c in range(W):
        valid_rows = np.where(has_voxels[:, c])[0]
        gap_rows = np.where(gap_mask[:, c])[0]
        if len(valid_rows) >= 2 and len(gap_rows) > 0:
            fill_col_h[gap_rows, c] = np.interp(gap_rows, valid_rows, height_map[valid_rows, c])
            fill_col_t[gap_rows, c] = np.interp(gap_rows, valid_rows, thick_map[valid_rows, c])

    not_valid = ~has_voxels
    row_dist = distance_transform_edt(not_valid, sampling=[1e6, 1])
    col_dist = distance_transform_edt(not_valid, sampling=[1, 1e6])

    gap_r, gap_c = np.where(gap_mask)
    hr = fill_row_h[gap_r, gap_c]
    hc = fill_col_h[gap_r, gap_c]
    tr = fill_row_t[gap_r, gap_c]
    tc = fill_col_t[gap_r, gap_c]
    dr = np.maximum(row_dist[gap_r, gap_c], 1e-6)
    dc = np.maximum(col_dist[gap_r, gap_c], 1e-6)

    valid_r = ~np.isnan(hr)
    valid_c = ~np.isnan(hc)
    both = valid_r & valid_c
    only_r = valid_r & ~valid_c
    only_c = valid_c & ~valid_r

    wr = np.where(both, 1.0 / dr, 0.0)
    wc = np.where(both, 1.0 / dc, 0.0)
    w_total = np.maximum(wr + wc, 1e-12)

    h_avg = np.where(
        both,
        (np.nan_to_num(hr) * wr + np.nan_to_num(hc) * wc) / w_total,
        np.where(only_r, hr, np.where(only_c, hc, np.nan)),
    )
    t_avg = np.where(
        both,
        (np.nan_to_num(tr) * wr + np.nan_to_num(tc) * wc) / w_total,
        np.where(only_r, tr, np.where(only_c, tc, 0.0)),
    )

    patched_t = crop_t.copy()
    for idx in range(len(gap_r)):
        h_val = h_avg[idx]
        if np.isnan(h_val):
            continue
        r, c = gap_r[idx], gap_c[idx]
        center = int(round(h_val))
        half = max(0, int(round(t_avg[idx] / 2)))
        z0 = max(0, center - half)
        z1 = min(D - 1, center + half)
        patched_t[z0:z1 + 1, r, c] = 1

    patched_3d = np.moveaxis(patched_t, 0, best_axis)

    holes_after = _count_internal_holes(patched_3d)
    if holes_after > holes_before:
        return crop

    return patched_3d.astype(np.uint8)


def _count_internal_holes(mask):
    inv = 1 - mask.astype(np.uint8)
    struct6 = generate_binary_structure(3, 1)
    labeled, n = ndimage_label(inv, structure=struct6)
    if n == 0:
        return 0
    border = np.zeros(n + 1, dtype=bool)
    border[labeled[0]] = True
    border[labeled[-1]] = True
    border[labeled[:, 0]] = True
    border[labeled[:, -1]] = True
    border[labeled[:, :, 0]] = True
    border[labeled[:, :, -1]] = True
    border[0] = True
    return int(n - border[1:].sum())


# --- LUT-based 1-voxel hole plugging ---

_HOLE_PLUG_LUT = None


def _build_hole_plug_lut():
    face_diags = [
        (0, 3, 1, 2), (1, 2, 0, 3),
        (4, 7, 5, 6), (5, 6, 4, 7),
        (0, 5, 1, 4), (1, 4, 0, 5),
        (2, 7, 3, 6), (3, 6, 2, 7),
        (0, 6, 2, 4), (2, 4, 0, 6),
        (1, 7, 3, 5), (3, 5, 1, 7),
    ]
    lut = np.zeros(256, dtype=np.uint8)
    for pattern in range(256):
        add = 0
        for fa, fb, g1, g2 in face_diags:
            if ((pattern >> fa) & 1) and ((pattern >> fb) & 1) \
                    and not ((pattern >> g1) & 1) \
                    and not ((pattern >> g2) & 1):
                add |= (1 << g1)
        lut[pattern] = add
    return lut


def _get_hole_plug_lut():
    global _HOLE_PLUG_LUT
    if _HOLE_PLUG_LUT is None:
        _HOLE_PLUG_LUT = _build_hole_plug_lut()
    return _HOLE_PLUG_LUT


def _plug_holes_lut(mask, max_iterations=5):
    lut = _get_hole_plug_lut()
    result = mask.astype(np.uint8).copy()
    Z, Y, X = result.shape
    if Z < 2 or Y < 2 or X < 2:
        return result

    offsets = [(dz, dy, dx) for dz in range(2) for dy in range(2) for dx in range(2)]

    for _ in range(max_iterations):
        pattern = np.zeros((Z - 1, Y - 1, X - 1), dtype=np.uint8)
        for bit, (dz, dy, dx) in enumerate(offsets):
            pattern |= (result[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx] << bit)

        additions = lut[pattern]
        if not additions.any():
            break

        for bit, (dz, dy, dx) in enumerate(offsets):
            add_bit = ((additions >> bit) & 1).astype(np.uint8)
            result[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx] |= add_bit

    return result


def _get_structure(connectivity):
    if connectivity == 6:
        return generate_binary_structure(3, 1)
    elif connectivity == 18:
        return generate_binary_structure(3, 2)
    else:
        return generate_binary_structure(3, 3)


# ---------------------------------------------------------------------------
# Main: nnU-Net predict + postprocess
# ---------------------------------------------------------------------------

def predict_and_postprocess(
    input_dir: str,
    output_dir: str,
    model_dir: str,
    folds: str = "all",
    use_mirroring: bool = True,
    device: str = "cuda",
    step_size: float = 0.5,
    min_component_size: int = 500,
    closing_radius: int = 3,
    enable_patching: bool = True,
    enable_hole_plugging: bool = True,
):
    """Run nnU-Net inference then post-process each prediction."""
    raw_dir = os.path.join(output_dir, "raw_nnunet")

    print("=" * 60)
    print("Step 1: nnU-Net inference")
    print("=" * 60)
    run_nnunet_inference(
        input_dir=input_dir,
        output_dir=raw_dir,
        model_dir=model_dir,
        folds=folds,
        use_mirroring=use_mirroring,
        device=device,
        step_size=step_size,
    )

    print("\n" + "=" * 60)
    print("Step 2: Post-processing")
    print("=" * 60)
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".tif"):
            continue

        print(f"\nPost-processing {fname}...")
        pred = tifffile.imread(os.path.join(raw_dir, fname))
        surface_mask = (pred == 1).astype(np.uint8)

        processed = postprocess_prediction(
            surface_mask,
            min_component_size=min_component_size,
            closing_radius=closing_radius,
            enable_patching=enable_patching,
            enable_hole_plugging=enable_hole_plugging,
        )

        out_path = os.path.join(final_dir, fname)
        tifffile.imwrite(out_path, processed.astype(np.uint8))
        print(f"  Saved to {out_path}")

    print(f"\nAll predictions saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net inference + 1st place post-processing"
    )
    parser.add_argument("--input-dir", required=True, help="Directory with test TIF volumes")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions")
    parser.add_argument("--model-dir", required=True, help="nnU-Net trained model directory")
    parser.add_argument("--folds", default="all", help="Fold(s) to use (default: all)")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation (mirroring)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--step-size", type=float, default=0.5, help="Sliding window step size (0-1)")
    parser.add_argument("--min-component-size", type=int, default=500)
    parser.add_argument("--closing-radius", type=int, default=3)
    parser.add_argument("--no-patching", action="store_true")
    parser.add_argument("--no-hole-plugging", action="store_true")
    args = parser.parse_args()

    predict_and_postprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        folds=args.folds,
        use_mirroring=not args.no_tta,
        device=args.device,
        step_size=args.step_size,
        min_component_size=args.min_component_size,
        closing_radius=args.closing_radius,
        enable_patching=not args.no_patching,
        enable_hole_plugging=not args.no_hole_plugging,
    )


if __name__ == "__main__":
    main()
