# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
Local evaluation module implementing the competition metric.

WARNING: This is Pipeline V3. Do NOT import from src/ or src_v2/.
See PIPELINES.md in the project root.
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
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    sd = surface_dice(pred_bin, gt_bin, tau=tau, spacing=spacing)
    voi = voi_score(pred_bin, gt_bin, alpha=alpha_voi, connectivity=connectivity)
    topo = topo_score(pred_bin, gt_bin, weights=topo_weights)
    score = 0.35 * sd + 0.35 * voi + 0.30 * topo
    return {"score": score, "surface_dice": sd, "voi_score": voi, "topo_score": topo}


def surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = 2.0,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    pred_empty = pred.sum() == 0
    gt_empty = gt.sum() == 0
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0
    pred_surface = _extract_surface(pred)
    gt_surface = _extract_surface(gt)
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 1.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0
    dt_pred = distance_transform_edt(~pred_surface.astype(bool), sampling=spacing)
    dt_gt = distance_transform_edt(~gt_surface.astype(bool), sampling=spacing)
    gt_to_pred_matches = (dt_pred[gt_surface > 0] <= tau).sum()
    pred_to_gt_matches = (dt_gt[pred_surface > 0] <= tau).sum()
    gt_surface_count = gt_surface.sum()
    pred_surface_count = pred_surface.sum()
    sd = (gt_to_pred_matches + pred_to_gt_matches) / (gt_surface_count + pred_surface_count)
    return float(sd)


def _extract_surface(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask.astype(bool), structure=generate_binary_structure(3, 1))
    surface = mask.astype(bool) & ~eroded
    return surface.astype(np.uint8)


def voi_score(
    pred: np.ndarray,
    gt: np.ndarray,
    alpha: float = 0.3,
    connectivity: int = 26,
) -> float:
    struct = _get_structure(connectivity)
    union_fg = ((pred > 0) | (gt > 0)).astype(np.uint8)
    if union_fg.sum() == 0:
        return 1.0
    pred_labeled, n_pred = ndimage_label(pred > 0, structure=struct)
    gt_labeled, n_gt = ndimage_label(gt > 0, structure=struct)
    if n_pred == 0 and n_gt == 0:
        return 1.0
    fg_mask = union_fg > 0
    pred_labels_fg = pred_labeled[fg_mask]
    gt_labels_fg = gt_labeled[fg_mask]
    n_total = fg_mask.sum()
    voi_split = _conditional_entropy(gt_labels_fg, pred_labels_fg, n_total)
    voi_merge = _conditional_entropy(pred_labels_fg, gt_labels_fg, n_total)
    voi_total = voi_split + voi_merge
    return float(1.0 / (1.0 + alpha * voi_total))


def _conditional_entropy(labels_a, labels_b, n_total):
    unique_b = np.unique(labels_b)
    h = 0.0
    for b_val in unique_b:
        mask_b = labels_b == b_val
        n_b = mask_b.sum()
        if n_b == 0:
            continue
        a_in_b = labels_a[mask_b]
        unique_a_in_b, counts_a_in_b = np.unique(a_in_b, return_counts=True)
        for count in counts_a_in_b:
            p = count / n_b
            if p > 0:
                h -= (n_b / n_total) * p * np.log2(p + 1e-15)
    return h


def topo_score(
    pred: np.ndarray,
    gt: np.ndarray,
    weights: tuple = (0.34, 0.33, 0.33),
) -> float:
    struct_26 = _get_structure(26)
    _, n_pred_cc = ndimage_label(pred > 0, structure=struct_26)
    _, n_gt_cc = ndimage_label(gt > 0, structure=struct_26)
    pred_cavities = _count_enclosed_cavities(pred)
    gt_cavities = _count_enclosed_cavities(gt)
    pred_euler = _euler_number_3d(pred > 0)
    gt_euler = _euler_number_3d(gt > 0)
    pred_tunnels = max(0, n_pred_cc - pred_euler + pred_cavities)
    gt_tunnels = max(0, n_gt_cc - gt_euler + gt_cavities)
    betti_pred = [n_pred_cc, pred_tunnels, pred_cavities]
    betti_gt = [n_gt_cc, gt_tunnels, gt_cavities]
    scores = []
    active_weights = []
    for k in range(3):
        bp = betti_pred[k]
        bg = betti_gt[k]
        if bp == 0 and bg == 0:
            continue
        matched = min(bp, bg)
        fp = bp - matched
        fn = bg - matched
        if matched + fp + fn == 0:
            f1 = 1.0
        else:
            precision = matched / (matched + fp) if (matched + fp) > 0 else 0
            recall = matched / (matched + fn) if (matched + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
        active_weights.append(weights[k])
    if not scores:
        return 1.0
    total_w = sum(active_weights)
    return sum(s * w / total_w for s, w in zip(scores, active_weights))


def _count_enclosed_cavities(mask: np.ndarray) -> int:
    inverted = 1 - (mask > 0).astype(np.uint8)
    struct = generate_binary_structure(3, 1)
    labeled, n = ndimage_label(inverted, structure=struct)
    count = 0
    for i in range(1, n + 1):
        comp = labeled == i
        touches = (
            comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
            or comp[:, :, 0].any() or comp[:, :, -1].any()
        )
        if not touches:
            count += 1
    return count


def _euler_number_3d(mask: np.ndarray) -> int:
    m = mask.astype(bool)
    v = m.sum()
    e = (
        (m[:-1] & m[1:]).sum()
        + (m[:, :-1] & m[:, 1:]).sum()
        + (m[:, :, :-1] & m[:, :, 1:]).sum()
    )
    f = (
        (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).sum()
        + (m[:-1, :, :-1] & m[1:, :, :-1] & m[:-1, :, 1:] & m[1:, :, 1:]).sum()
        + (m[:, :-1, :-1] & m[:, 1:, :-1] & m[:, :-1, 1:] & m[:, 1:, 1:]).sum()
    )
    c = (
        m[:-1, :-1, :-1] & m[1:, :-1, :-1] & m[:-1, 1:, :-1] & m[1:, 1:, :-1]
        & m[:-1, :-1, 1:] & m[1:, :-1, 1:] & m[:-1, 1:, 1:] & m[1:, 1:, 1:]
    ).sum()
    return int(v - e + f - c)


def _get_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return generate_binary_structure(3, 1)
    elif connectivity == 18:
        return generate_binary_structure(3, 2)
    return generate_binary_structure(3, 3)


def evaluate_on_deprecated_test(
    pred_path: str,
    data_dir: str,
    sample_id: str = "1407735",
) -> dict:
    import os
    import tifffile
    gt_path = os.path.join(data_dir, "deprecated_train_labels", f"{sample_id}.tif")
    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)
    return compute_competition_score(pred, gt)
