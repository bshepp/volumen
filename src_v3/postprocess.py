# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
Topology-aware post-processing for surface segmentation predictions.

WARNING: This is Pipeline V3. Do NOT import from src/ or src_v2/.
See PIPELINES.md in the project root.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label as ndimage_label,
)


def postprocess_prediction(
    pred: np.ndarray,
    min_component_size: int = 500,
    bridge_threshold: int = 3,
    fill_holes: bool = True,
    min_sheet_spacing: int = 10,
    connectivity: int = 26,
) -> np.ndarray:
    pred = pred.astype(np.uint8)
    pred = remove_small_components(pred, min_component_size, connectivity)
    pred = remove_bridges(pred, bridge_threshold, connectivity)
    if fill_holes:
        pred = fill_small_holes(pred, max_hole_size=200)
    pred = spacing_validation(pred, min_sheet_spacing, connectivity)
    return pred.astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 500,
    connectivity: int = 26,
) -> np.ndarray:
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


def remove_bridges(
    mask: np.ndarray,
    thickness_threshold: int = 3,
    connectivity: int = 26,
) -> np.ndarray:
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


def fill_small_holes(mask: np.ndarray, max_hole_size: int = 200) -> np.ndarray:
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
            hole_mask[0, :, :].any() or hole_mask[-1, :, :].any()
            or hole_mask[:, 0, :].any() or hole_mask[:, -1, :].any()
            or hole_mask[:, :, 0].any() or hole_mask[:, :, -1].any()
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
    struct = _get_structure(connectivity)
    labeled, n_components = ndimage_label(mask, structure=struct)
    if n_components <= 1:
        return mask
    result = mask.copy()
    component_sizes = {i: (labeled == i).sum() for i in range(1, n_components + 1)}
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


def _get_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return generate_binary_structure(3, 1)
    elif connectivity == 18:
        return generate_binary_structure(3, 2)
    return generate_binary_structure(3, 3)
