# Pipeline: V2 (src_v2/) — See PIPELINES.md
"""
Topology-aware post-processing for surface segmentation predictions.

Primary pipeline adapted from the 1st place solution for the Vesuvius
Challenge Surface Detection competition:
    "1st Place Solution for the Vesuvius Challenge Surface Detection"
    https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/
    writeups/1st-place-solution-for-the-vesuvius-challenge-su

Pipeline steps:
  1. Connected component filtering — remove small fragments
  2. Per-sheet binary closing — close small holes/cavities (spherical footprint)
  3. Height-map patching — repair larger gaps via projection and interpolation
  4. 1-voxel hole plugging — LUT-based 2x2x2 micro-repair for watertightness
  5. Global binary fill holes — fill remaining enclosed cavities

All per-sheet operations (steps 2-4) use bounding-box cropping to avoid
full-volume array operations per component.

Legacy steps (remove_bridges, fill_small_holes, spacing_validation) are
retained as importable functions for backward compatibility.
"""

import time

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label as ndimage_label,
)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def postprocess_prediction(
    pred: np.ndarray,
    min_component_size: int = 500,
    closing_radius: int = 3,
    enable_patching: bool = True,
    enable_hole_plugging: bool = True,
    connectivity: int = 26,
) -> np.ndarray:
    """Post-processing pipeline adapted from the 1st place solution.

    All per-sheet operations use bounding-box crops for performance.

    Args:
        pred: Binary 3D mask (Z, Y, X), uint8 or bool.
        min_component_size: Remove components smaller than this (voxels).
        closing_radius: Radius for spherical binary closing footprint.
        enable_patching: Run height-map gap repair.
        enable_hole_plugging: Run LUT-based 1-voxel hole plugging.
        connectivity: 6, 18, or 26 for connected components.

    Returns:
        Cleaned binary mask, same shape, uint8.
    """
    t_total = time.time()
    pred = pred.astype(np.uint8)
    struct = _get_structure(connectivity)

    t0 = time.time()
    pred = remove_small_components(pred, min_component_size, connectivity)
    print(f'  Step 1 (CC filter): {time.time() - t0:.2f}s')

    labeled, n = ndimage_label(pred, structure=struct)
    if n == 0:
        return pred

    slices = find_objects(labeled)
    footprint = make_ball_footprint(closing_radius) if closing_radius > 0 else None
    pad = closing_radius if closing_radius > 0 else 0
    result = np.zeros_like(pred, dtype=np.uint8)

    t_close = time.time()
    t_patch = 0.0
    t_plug = 0.0
    n_sheets = 0

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
            crop = plug_holes_lut(crop)
            t_plug += time.time() - tg

        result[padded_sl] |= crop
        n_sheets += 1

    print(f'  Step 2 (closing, {n_sheets} sheets): {time.time() - t_close - t_patch - t_plug:.2f}s')
    print(f'  Step 3 (patching): {t_patch:.2f}s')
    print(f'  Step 4 (hole plug): {t_plug:.2f}s')

    t0 = time.time()
    result = binary_fill_holes(result).astype(np.uint8)
    print(f'  Step 5 (fill holes): {time.time() - t0:.2f}s')
    print(f'  Post-processing total: {time.time() - t_total:.2f}s')

    return result


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------

def _pad_slices(sl, shape, pad):
    """Expand a tuple of slices by ``pad`` voxels, clipped to volume bounds."""
    return tuple(
        slice(max(0, s.start - pad), min(dim, s.stop + pad))
        for s, dim in zip(sl, shape)
    )


# ---------------------------------------------------------------------------
# Step 1: Connected component filtering
# ---------------------------------------------------------------------------

def remove_small_components(
    mask: np.ndarray,
    min_size: int = 500,
    connectivity: int = 26,
) -> np.ndarray:
    """Remove connected components smaller than min_size."""
    struct = _get_structure(connectivity)
    labeled, n_components = ndimage_label(mask, structure=struct)

    if n_components == 0:
        return mask

    component_sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))

    keep_mask = np.zeros_like(mask)
    for i, size in enumerate(component_sizes, 1):
        if size >= min_size:
            keep_mask[labeled == i] = 1

    return keep_mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 2: Per-sheet binary closing (via bbox crop in main loop)
# ---------------------------------------------------------------------------

def make_ball_footprint(radius: int) -> np.ndarray:
    """Spherical binary structuring element of given radius.

    Returns a (2*radius+1)^3 boolean array where voxels within Euclidean
    distance ``radius`` of the center are True.
    """
    zz, yy, xx = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
        -radius:radius + 1,
    ]
    return (zz ** 2 + yy ** 2 + xx ** 2) <= radius ** 2


def per_sheet_binary_closing(
    mask: np.ndarray,
    radius: int = 3,
    connectivity: int = 26,
) -> np.ndarray:
    """Apply binary_closing with a spherical footprint to each sheet separately.

    Uses bounding-box cropping per component for performance.
    """
    footprint = make_ball_footprint(radius)
    struct = _get_structure(connectivity)
    labeled, n = ndimage_label(mask, structure=struct)
    slices = find_objects(labeled)
    result = np.zeros_like(mask, dtype=np.uint8)

    for i, sl in enumerate(slices, 1):
        if sl is None:
            continue
        padded_sl = _pad_slices(sl, mask.shape, radius)
        sheet_crop = (labeled[padded_sl] == i)
        closed_crop = binary_closing(sheet_crop, structure=footprint)
        result[padded_sl] |= closed_crop

    return result


# ---------------------------------------------------------------------------
# Step 3: Height-map patching (operates on a single-component bbox crop)
# ---------------------------------------------------------------------------

def _height_map_patch_crop(crop: np.ndarray) -> np.ndarray:
    """Height-map patching on a single-component bounding-box crop.

    The crop must contain exactly one connected component (the sheet).
    Repairs gaps by projecting to 2D, interpolating height/thickness,
    and reconstructing 3D voxels. Discards the patch if it introduces
    new internal holes.
    """
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
            fill_row_h[r, gap_cols] = np.interp(
                gap_cols, valid_cols, height_map[r, valid_cols])
            fill_row_t[r, gap_cols] = np.interp(
                gap_cols, valid_cols, thick_map[r, valid_cols])

    fill_col_h = np.full((H, W), np.nan, dtype=np.float32)
    fill_col_t = np.full((H, W), np.nan, dtype=np.float32)
    for c in range(W):
        valid_rows = np.where(has_voxels[:, c])[0]
        gap_rows = np.where(gap_mask[:, c])[0]
        if len(valid_rows) >= 2 and len(gap_rows) > 0:
            fill_col_h[gap_rows, c] = np.interp(
                gap_rows, valid_rows, height_map[valid_rows, c])
            fill_col_t[gap_rows, c] = np.interp(
                gap_rows, valid_rows, thick_map[valid_rows, c])

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


def height_map_patching(
    mask: np.ndarray,
    connectivity: int = 26,
    min_sheet_size: int = 100,
) -> np.ndarray:
    """Repair gaps in sheets via height-map projection and interpolation.

    Uses bounding-box cropping per component for performance.
    """
    struct = _get_structure(connectivity)
    labeled, n = ndimage_label(mask, structure=struct)
    slices = find_objects(labeled)
    result = mask.copy().astype(np.uint8)

    for comp_id, sl in enumerate(slices, 1):
        if sl is None:
            continue
        crop = (labeled[sl] == comp_id).astype(np.uint8)
        if crop.sum() < min_sheet_size:
            continue
        patched = _height_map_patch_crop(crop)
        result[sl] |= patched

    return result


def _count_internal_holes(mask: np.ndarray) -> int:
    """Count enclosed cavities that do not touch the volume boundary."""
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


# ---------------------------------------------------------------------------
# Step 4: 1-voxel hole plugging via 2x2x2 LUT
# ---------------------------------------------------------------------------

_HOLE_PLUG_LUT = None


def _build_hole_plug_lut() -> np.ndarray:
    """Build 256-entry LUT for 2x2x2 neighborhood hole plugging.

    For each of the 256 possible binary patterns in a 2x2x2 cube, detect
    face-diagonal gaps (two foreground voxels at opposite corners of a cube
    face with both gap voxels empty) and add one bridging voxel per gap.

    This targets micro-perforations in surfaces without over-expanding at
    sheet boundaries, unlike generic tunnel detection which can aggressively
    thicken sheets.

    Bit-index to (z, y, x) mapping:
      0:(0,0,0)  1:(0,0,1)  2:(0,1,0)  3:(0,1,1)
      4:(1,0,0)  5:(1,0,1)  6:(1,1,0)  7:(1,1,1)
    """
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


def _get_hole_plug_lut() -> np.ndarray:
    global _HOLE_PLUG_LUT
    if _HOLE_PLUG_LUT is None:
        _HOLE_PLUG_LUT = _build_hole_plug_lut()
    return _HOLE_PLUG_LUT


def plug_holes_lut(
    mask: np.ndarray,
    max_iterations: int = 5,
) -> np.ndarray:
    """Plug 1-voxel holes using a 2x2x2 neighborhood lookup table.

    Iterates until convergence or max_iterations.  Uses numpy vectorization
    for performance on large volumes.
    """
    lut = _get_hole_plug_lut()
    result = mask.astype(np.uint8).copy()
    Z, Y, X = result.shape

    if Z < 2 or Y < 2 or X < 2:
        return result

    offsets = [
        (dz, dy, dx)
        for dz in range(2) for dy in range(2) for dx in range(2)
    ]

    for _ in range(max_iterations):
        pattern = np.zeros((Z - 1, Y - 1, X - 1), dtype=np.uint8)
        for bit, (dz, dy, dx) in enumerate(offsets):
            pattern |= (
                result[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx] << bit
            )

        additions = lut[pattern]

        if not additions.any():
            break

        for bit, (dz, dy, dx) in enumerate(offsets):
            add_bit = ((additions >> bit) & 1).astype(np.uint8)
            result[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx] |= add_bit

    return result


# ---------------------------------------------------------------------------
# Legacy pipeline components (backward compatibility)
# ---------------------------------------------------------------------------

def remove_bridges(
    mask: np.ndarray,
    thickness_threshold: int = 3,
    connectivity: int = 26,
) -> np.ndarray:
    """Detect and remove thin bridges between separate sheet surfaces.

    Strategy:
      1. Erode the mask by thickness_threshold iterations
      2. Label connected components in the eroded mask
      3. Dilate each component back separately
      4. Where multiple dilated components overlap -> bridge -> remove
    """
    if mask.sum() == 0:
        return mask

    struct = _get_structure(connectivity)

    eroded = binary_erosion(mask, structure=struct, iterations=thickness_threshold)
    eroded = eroded.astype(np.uint8)

    if eroded.sum() == 0:
        return mask

    labeled_eroded, n_eroded = ndimage_label(eroded, structure=struct)

    if n_eroded <= 1:
        return mask

    territories = np.zeros(mask.shape, dtype=np.int32)
    conflict = np.zeros(mask.shape, dtype=bool)

    for comp_id in range(1, n_eroded + 1):
        comp_mask = (labeled_eroded == comp_id).astype(np.uint8)
        dilated = binary_dilation(
            comp_mask, structure=struct, iterations=thickness_threshold + 1
        )
        dilated = dilated & mask.astype(bool)

        overlap = (territories > 0) & dilated
        conflict[overlap] = True

        territories[dilated & (territories == 0)] = comp_id

    result = mask.copy()
    result[conflict] = 0

    result = remove_small_components(result, min_size=100, connectivity=connectivity)

    return result


def fill_small_holes(
    mask: np.ndarray,
    max_hole_size: int = 200,
) -> np.ndarray:
    """Fill small enclosed holes/cavities in the mask.

    Large holes are left alone as they may represent real gaps between sheets.
    """
    inverted = 1 - mask.astype(np.uint8)
    struct = generate_binary_structure(3, 1)
    labeled_holes, n_holes = ndimage_label(inverted, structure=struct)

    if n_holes == 0:
        return mask

    result = mask.copy()

    for hole_id in range(1, n_holes + 1):
        hole_mask = labeled_holes == hole_id
        hole_size = hole_mask.sum()

        touches_boundary = (
            hole_mask[0, :, :].any()
            or hole_mask[-1, :, :].any()
            or hole_mask[:, 0, :].any()
            or hole_mask[:, -1, :].any()
            or hole_mask[:, :, 0].any()
            or hole_mask[:, :, -1].any()
        )

        if not touches_boundary and hole_size <= max_hole_size:
            result[hole_mask] = 1

    return result.astype(np.uint8)


def spacing_validation(
    mask: np.ndarray,
    min_spacing: int = 10,
    connectivity: int = 26,
    max_components_to_check: int = 20,
) -> np.ndarray:
    """Remove surface pairs that are closer than min_spacing voxels apart."""
    struct = _get_structure(connectivity)
    labeled, n_components = ndimage_label(mask, structure=struct)

    if n_components <= 1:
        return mask

    result = mask.copy()
    component_sizes = {}
    for i in range(1, n_components + 1):
        component_sizes[i] = (labeled == i).sum()

    sorted_comps = sorted(component_sizes.items(), key=lambda x: -x[1])
    comps_to_check = [c[0] for c in sorted_comps[:max_components_to_check]]

    removed = set()
    for idx_a, comp_a in enumerate(comps_to_check):
        if comp_a in removed:
            continue
        mask_a = labeled == comp_a
        dist_from_a = distance_transform_edt(~mask_a)

        for comp_b in comps_to_check[idx_a + 1:]:
            if comp_b in removed:
                continue
            mask_b = labeled == comp_b
            min_dist = dist_from_a[mask_b].min() if mask_b.any() else float("inf")

            if min_dist < min_spacing:
                if component_sizes[comp_a] < component_sizes[comp_b]:
                    result[mask_a] = 0
                    removed.add(comp_a)
                    break
                else:
                    result[mask_b] = 0
                    removed.add(comp_b)

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_structure(connectivity: int) -> np.ndarray:
    """Binary structure element for 6, 18, or 26 connectivity."""
    if connectivity == 6:
        return generate_binary_structure(3, 1)
    elif connectivity == 18:
        return generate_binary_structure(3, 2)
    else:
        return generate_binary_structure(3, 3)
