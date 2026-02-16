# Pipeline: V2 (src_v2/) — See PIPELINES.md
"""
Topology-aware post-processing for surface segmentation predictions.

Steps:
  1. Connected component filtering — remove small fragments
  2. Bridge detection and removal — break artificial mergers between sheets
  3. Hole filling — fill small enclosed cavities
  4. Sheet spacing validation — flag surfaces closer than bilaminar thickness

From our analysis:
  - Median sheet spacing: ~56 voxels
  - Minimum observed spacing: ~4 voxels (in compressed regions)
  - Sheet thickness (surface label): ~5-12 voxels
  - Minimum bilaminar thickness: ~10 voxels
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
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
    """Full post-processing pipeline for a binary surface prediction.

    Args:
        pred: Binary 3D mask (Z, Y, X), uint8 or bool. 1 = surface, 0 = background.
        min_component_size: Remove connected components smaller than this (voxels).
        bridge_threshold: Thickness threshold for bridge detection (voxels).
        fill_holes: Whether to fill small internal holes.
        min_sheet_spacing: Minimum expected spacing between sheets (voxels).
        connectivity: 6, 18, or 26 connectivity for connected components.

    Returns:
        Cleaned binary mask, same shape, uint8.
    """
    pred = pred.astype(np.uint8)

    # Step 1: Remove small connected components
    pred = remove_small_components(pred, min_component_size, connectivity)

    # Step 2: Detect and remove bridges
    pred = remove_bridges(pred, bridge_threshold, connectivity)

    # Step 3: Fill small holes
    if fill_holes:
        pred = fill_small_holes(pred, max_hole_size=200)

    # Step 4: Sheet spacing validation (remove surfaces too close together)
    pred = spacing_validation(pred, min_sheet_spacing, connectivity)

    return pred.astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 500,
    connectivity: int = 26,
) -> np.ndarray:
    """Remove connected components smaller than min_size.

    This eliminates noise fragments that hurt VOI_split and TopoScore k=0.
    """
    struct = _get_structure(connectivity)
    labeled, n_components = ndimage_label(mask, structure=struct)

    if n_components == 0:
        return mask

    # Compute component sizes
    component_sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))

    # Keep only components above threshold
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
    """Detect and remove thin bridges between separate sheet surfaces.

    A bridge is a thin connection between two otherwise separate components.
    We detect bridges by erosion: if eroding the mask breaks a component
    into pieces, the connection was thin (a bridge).

    Strategy:
      1. Erode the mask by `thickness_threshold` iterations
      2. Find connected components in the eroded mask
      3. Dilate each component back separately
      4. Where multiple dilated components overlap with the original,
         the overlap region is a bridge — remove it

    This targets the VOI_merge penalty and TopoScore k=1 (tunnels).
    """
    if mask.sum() == 0:
        return mask

    struct = _get_structure(connectivity)

    # Erode
    eroded = binary_erosion(mask, structure=struct, iterations=thickness_threshold)
    eroded = eroded.astype(np.uint8)

    if eroded.sum() == 0:
        # Everything was thin — keep original
        return mask

    # Label eroded components
    labeled_eroded, n_eroded = ndimage_label(eroded, structure=struct)

    if n_eroded <= 1:
        # Single component or empty after erosion — no bridges to remove
        return mask

    # For each eroded component, dilate back and find its territory
    territories = np.zeros(mask.shape, dtype=np.int32)
    conflict = np.zeros(mask.shape, dtype=bool)

    for comp_id in range(1, n_eroded + 1):
        comp_mask = (labeled_eroded == comp_id).astype(np.uint8)
        # Dilate back to roughly original size
        dilated = binary_dilation(
            comp_mask, structure=struct, iterations=thickness_threshold + 1
        )
        # Only within the original mask
        dilated = dilated & mask.astype(bool)

        # Mark conflicts where territories overlap
        overlap = (territories > 0) & dilated
        conflict[overlap] = True

        territories[dilated & (territories == 0)] = comp_id

    # Remove conflict regions (bridges) from the mask
    result = mask.copy()
    result[conflict] = 0

    # Also remove any now-isolated small fragments created by bridge removal
    result = remove_small_components(result, min_size=100, connectivity=connectivity)

    return result


def fill_small_holes(
    mask: np.ndarray,
    max_hole_size: int = 200,
) -> np.ndarray:
    """Fill small enclosed holes/cavities in the mask.

    This targets TopoScore k=2 (spurious cavities).
    Large holes are left alone as they may represent real gaps between sheets.
    """
    # Invert to find holes
    inverted = 1 - mask.astype(np.uint8)
    struct = generate_binary_structure(3, 1)  # 6-connectivity for holes
    labeled_holes, n_holes = ndimage_label(inverted, structure=struct)

    if n_holes == 0:
        return mask

    result = mask.copy()

    for hole_id in range(1, n_holes + 1):
        hole_mask = labeled_holes == hole_id
        hole_size = hole_mask.sum()

        # Check if this hole touches the volume boundary — if so, it's not enclosed
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
    """Remove surface voxels that are too close to other surface voxels
    from a different connected component.

    If two separate sheet surfaces are closer than min_spacing,
    this suggests a labeling or prediction error. We remove the smaller
    component in the conflict zone.

    This serves as a physics-based prior from the observed sheet spacing
    distribution (median ~56 voxels, minimum ~4 in extreme compression).

    For efficiency, only the largest `max_components_to_check` components
    are compared pairwise (distance transforms on 320^3 are expensive).
    """
    struct = _get_structure(connectivity)
    labeled, n_components = ndimage_label(mask, structure=struct)

    if n_components <= 1:
        return mask

    # Compute component sizes and keep only the largest ones for pairwise check
    result = mask.copy()
    component_sizes = {}
    for i in range(1, n_components + 1):
        component_sizes[i] = (labeled == i).sum()

    # Sort by size descending, keep top N
    sorted_comps = sorted(component_sizes.items(), key=lambda x: -x[1])
    comps_to_check = [c[0] for c in sorted_comps[:max_components_to_check]]

    # For each pair of large components, check spacing
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
                # Remove the smaller one
                if component_sizes[comp_a] < component_sizes[comp_b]:
                    result[mask_a] = 0
                    removed.add(comp_a)
                    break  # comp_a is removed, skip remaining pairs
                else:
                    result[mask_b] = 0
                    removed.add(comp_b)

    return result.astype(np.uint8)


def _get_structure(connectivity: int) -> np.ndarray:
    """Get binary structure element for given connectivity.

    Args:
        connectivity: 6, 18, or 26.

    Returns:
        3x3x3 binary structure.
    """
    if connectivity == 6:
        return generate_binary_structure(3, 1)
    elif connectivity == 18:
        return generate_binary_structure(3, 2)
    else:  # 26
        return generate_binary_structure(3, 3)
