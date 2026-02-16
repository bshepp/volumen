# Pipeline: V1 (src/) — See PIPELINES.md
"""
Local evaluation module implementing the competition metric.

Competition score = 0.30 * TopoScore + 0.35 * SurfaceDice@tau + 0.35 * VOI_score

Components:
  1. SurfaceDice@tau (tau=2.0) — surface boundary proximity
  2. VOI_score (alpha=0.3) — instance split/merge via Variation of Information
  3. TopoScore — topological features via Betti number matching

This is a simplified local implementation for development iteration.
The full competition metric uses more sophisticated Betti matching
(persistent homology). This version uses connected component counts
as an approximation for Betti numbers.
"""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    generate_binary_structure,
    label as ndimage_label,
)


def compute_competition_score(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = 2.0,
    alpha_voi: float = 0.3,
    topo_weights: tuple = (0.34, 0.33, 0.33),
    spacing: tuple = (1.0, 1.0, 1.0),
    connectivity: int = 26,
) -> dict:
    """Compute the full competition metric.

    Args:
        pred: Binary prediction mask (Z, Y, X).
        gt: Binary ground truth mask (Z, Y, X).
        tau: Tolerance for SurfaceDice (in spacing units).
        alpha_voi: Scaling factor for VOI_score.
        topo_weights: Weights for Betti dimensions (k=0, k=1, k=2).
        spacing: Voxel spacing (sz, sy, sx).
        connectivity: Connectivity for connected components (default 26).

    Returns:
        Dict with 'score', 'surface_dice', 'voi_score', 'topo_score'
        and sub-components.
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    # Compute each component
    sd = surface_dice(pred_bin, gt_bin, tau=tau, spacing=spacing)
    voi = voi_score(pred_bin, gt_bin, alpha=alpha_voi, connectivity=connectivity)
    topo = topo_score(pred_bin, gt_bin, weights=topo_weights)

    # Final score
    score = 0.35 * sd + 0.35 * voi + 0.30 * topo

    return {
        "score": score,
        "surface_dice": sd,
        "voi_score": voi,
        "topo_score": topo,
    }


# ---- SurfaceDice@tau ----


def surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = 2.0,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """Compute SurfaceDice at tolerance tau.

    Fraction of surface points (both pred and GT) that lie within
    distance tau of each other.

    Edge cases:
      - Both empty -> 1.0
      - Exactly one empty -> 0.0
    """
    pred_empty = pred.sum() == 0
    gt_empty = gt.sum() == 0

    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0

    # Extract surfaces (border voxels)
    pred_surface = _extract_surface(pred)
    gt_surface = _extract_surface(gt)

    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 1.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0

    # Distance from GT surface to nearest pred surface point
    dt_pred = distance_transform_edt(~pred_surface.astype(bool), sampling=spacing)
    # Distance from pred surface to nearest GT surface point
    dt_gt = distance_transform_edt(~gt_surface.astype(bool), sampling=spacing)

    # Count matches
    gt_to_pred_matches = (dt_pred[gt_surface > 0] <= tau).sum()
    pred_to_gt_matches = (dt_gt[pred_surface > 0] <= tau).sum()

    gt_surface_count = gt_surface.sum()
    pred_surface_count = pred_surface.sum()

    # Average both directions
    sd = (gt_to_pred_matches + pred_to_gt_matches) / (
        gt_surface_count + pred_surface_count
    )
    return float(sd)


def _extract_surface(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels (foreground voxels adjacent to background)."""
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(mask.astype(bool), structure=generate_binary_structure(3, 1))
    surface = mask.astype(bool) & ~eroded
    return surface.astype(np.uint8)


# ---- VOI_score ----


def voi_score(
    pred: np.ndarray,
    gt: np.ndarray,
    alpha: float = 0.3,
    connectivity: int = 26,
) -> float:
    """Compute VOI-based score.

    VOI_score = 1 / (1 + alpha * VOI_total)

    Where VOI_total = VOI_split + VOI_merge, computed on connected
    component labelings.
    """
    struct = _get_structure(connectivity)

    # Connected components on the union foreground
    union_fg = ((pred > 0) | (gt > 0)).astype(np.uint8)

    if union_fg.sum() == 0:
        return 1.0

    # Label connected components in pred and gt
    pred_labeled, n_pred = ndimage_label(pred > 0, structure=struct)
    gt_labeled, n_gt = ndimage_label(gt > 0, structure=struct)

    if n_pred == 0 and n_gt == 0:
        return 1.0

    # Compute VOI using the labeled volumes on the union foreground
    # Focus on foreground voxels only
    fg_mask = union_fg > 0
    pred_labels_fg = pred_labeled[fg_mask]
    gt_labels_fg = gt_labeled[fg_mask]

    n_total = fg_mask.sum()

    # Compute contingency table
    # VOI_split = H(GT | Pred) = sum over pred clusters of
    #   (n_k/N) * sum over gt clusters of -(n_jk/n_k) * log(n_jk/n_k)
    # VOI_merge = H(Pred | GT) = similar with roles swapped

    voi_split = _conditional_entropy(gt_labels_fg, pred_labels_fg, n_total)
    voi_merge = _conditional_entropy(pred_labels_fg, gt_labels_fg, n_total)

    voi_total = voi_split + voi_merge
    score = 1.0 / (1.0 + alpha * voi_total)

    return float(score)


def _conditional_entropy(labels_a, labels_b, n_total):
    """Compute H(A|B) from label arrays."""
    # Get unique labels in B
    unique_b = np.unique(labels_b)
    h = 0.0
    for b_val in unique_b:
        mask_b = labels_b == b_val
        n_b = mask_b.sum()
        if n_b == 0:
            continue
        # Distribution of A within this B cluster
        a_in_b = labels_a[mask_b]
        unique_a_in_b, counts_a_in_b = np.unique(a_in_b, return_counts=True)
        for count in counts_a_in_b:
            p = count / n_b
            if p > 0:
                h -= (n_b / n_total) * p * np.log2(p + 1e-15)
    return h


# ---- TopoScore (simplified Betti number matching) ----


def topo_score(
    pred: np.ndarray,
    gt: np.ndarray,
    weights: tuple = (0.34, 0.33, 0.33),
) -> float:
    """Compute topological score using Betti number comparison.

    This is a simplified version that computes:
      - k=0: number of connected components (Betti_0)
      - k=1: approximated via Euler characteristic (tunnels/handles)
      - k=2: approximated via Euler characteristic (cavities)

    Uses a per-dimension F1-like score, then weighted average.

    NOTE: The full competition metric uses persistent homology for
    proper Betti matching. This is an approximation for local dev.
    """
    # k=0: connected components
    struct_26 = _get_structure(26)
    _, n_pred_cc = ndimage_label(pred > 0, structure=struct_26)
    _, n_gt_cc = ndimage_label(gt > 0, structure=struct_26)

    # k=2: cavities (connected components of the background fully enclosed)
    pred_cavities = _count_enclosed_cavities(pred)
    gt_cavities = _count_enclosed_cavities(gt)

    # k=1: tunnels/handles — approximate via Euler characteristic
    # chi = V - E + F - C for cubical complex, or chi = beta0 - beta1 + beta2
    # So beta1 = beta0 - chi + beta2
    pred_euler = _euler_number_3d(pred > 0)
    gt_euler = _euler_number_3d(gt > 0)
    pred_tunnels = max(0, n_pred_cc - pred_euler + pred_cavities)
    gt_tunnels = max(0, n_gt_cc - gt_euler + gt_cavities)

    betti_pred = [n_pred_cc, pred_tunnels, pred_cavities]
    betti_gt = [n_gt_cc, gt_tunnels, gt_cavities]

    # Per-dimension F1
    scores = []
    active_weights = []
    for k in range(3):
        bp = betti_pred[k]
        bg = betti_gt[k]
        if bp == 0 and bg == 0:
            # Inactive dimension — skip
            continue
        # Treat as: matched = min(bp, bg), unmatched_pred = bp - matched, etc.
        matched = min(bp, bg)
        fp = bp - matched
        fn = bg - matched
        if matched + fp + fn == 0:
            f1 = 1.0
        else:
            precision = matched / (matched + fp) if (matched + fp) > 0 else 0
            recall = matched / (matched + fn) if (matched + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        scores.append(f1)
        active_weights.append(weights[k])

    if not scores:
        return 1.0

    # Renormalize weights for active dimensions
    total_w = sum(active_weights)
    return sum(s * w / total_w for s, w in zip(scores, active_weights))


def _count_enclosed_cavities(mask: np.ndarray) -> int:
    """Count fully enclosed background cavities (Betti_2)."""
    inverted = 1 - (mask > 0).astype(np.uint8)
    struct = generate_binary_structure(3, 1)  # 6-connectivity
    labeled, n = ndimage_label(inverted, structure=struct)

    count = 0
    for i in range(1, n + 1):
        comp = labeled == i
        # Check if it touches any boundary
        touches = (
            comp[0].any()
            or comp[-1].any()
            or comp[:, 0].any()
            or comp[:, -1].any()
            or comp[:, :, 0].any()
            or comp[:, :, -1].any()
        )
        if not touches:
            count += 1
    return count


def _euler_number_3d(mask: np.ndarray) -> int:
    """Compute Euler number for a 3D binary image.

    Uses the formula: chi = V - E + F - C
    where V=vertices, E=edges, F=faces, C=cubes counted from
    the voxel grid of the foreground.

    This is a simplified computation.
    """
    m = mask.astype(bool)
    # Vertices (foreground voxels)
    v = m.sum()
    # Edges: pairs of adjacent foreground voxels along each axis
    e = (
        (m[:-1] & m[1:]).sum()
        + (m[:, :-1] & m[:, 1:]).sum()
        + (m[:, :, :-1] & m[:, :, 1:]).sum()
    )
    # Faces: 2x2 squares of foreground voxels (in each plane)
    f = (
        (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).sum()
        + (m[:-1, :, :-1] & m[1:, :, :-1] & m[:-1, :, 1:] & m[1:, :, 1:]).sum()
        + (m[:, :-1, :-1] & m[:, 1:, :-1] & m[:, :-1, 1:] & m[:, 1:, 1:]).sum()
    )
    # Cubes: 2x2x2 blocks of foreground voxels
    c = (
        m[:-1, :-1, :-1]
        & m[1:, :-1, :-1]
        & m[:-1, 1:, :-1]
        & m[1:, 1:, :-1]
        & m[:-1, :-1, 1:]
        & m[1:, :-1, 1:]
        & m[:-1, 1:, 1:]
        & m[1:, 1:, 1:]
    ).sum()

    chi = int(v - e + f - c)
    return chi


def _get_structure(connectivity: int) -> np.ndarray:
    """Get binary structure element for given connectivity."""
    if connectivity == 6:
        return generate_binary_structure(3, 1)
    elif connectivity == 18:
        return generate_binary_structure(3, 2)
    else:
        return generate_binary_structure(3, 3)


# ---- Convenience function for evaluation ----


def evaluate_on_deprecated_test(
    pred_path: str,
    data_dir: str,
    sample_id: str = "1407735",
) -> dict:
    """Evaluate a prediction against the deprecated label for sample 1407735.

    Args:
        pred_path: Path to predicted .tif mask.
        data_dir: Path to vesuvius-challenge-surface-detection/ directory.
        sample_id: Sample ID (default: 1407735, the test image).

    Returns:
        Metric dict.
    """
    import os
    import tifffile

    gt_path = os.path.join(data_dir, "deprecated_train_labels", f"{sample_id}.tif")
    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)

    # The deprecated label is binary {0, 1}
    # Our prediction should also be binary
    return compute_competition_score(pred, gt)
