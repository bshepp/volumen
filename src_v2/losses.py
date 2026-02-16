# Pipeline: V2 (src_v2/) — See PIPELINES.md
"""
Composite loss function for Pipeline V2 — topology-aware 3D surface segmentation.

Pipeline V2 differences from V1:
  1. Focal Loss replaces Cross-Entropy (handles class imbalance better)
  2. Skeleton Recall Loss replaces clDice (cheaper, uses precomputed skeletons)
  3. Deep Supervision wrapper applies loss at multiple decoder resolutions

Loss = 0.3 * (Focal + Dice) + 0.3 * SkeletonRecall + 0.2 * Boundary

WARNING: This is Pipeline V2. Do NOT import from src/ (Pipeline V1).
See PIPELINES.md in the project root for documentation.

References:
  - Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
  - Skeleton Recall: Kirchhof et al., "Skeleton Recall Loss for Connectivity
    Conserving and Resource Efficient Segmentation of Thin Tubular Structures"
  - Boundary Loss: Kervadec et al., "Boundary loss for highly unbalanced
    segmentation", MIDL 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


# ---- Component 1: Focal Loss + Dice ----


class FocalDiceLoss(nn.Module):
    """Combined Focal Loss and soft Dice loss.

    Focal loss replaces standard CE, down-weighting easy examples:
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This reduces the contribution of well-classified background voxels,
    letting the model focus on hard surface boundary cases.
    """

    def __init__(
        self,
        class_weights: tuple = (0.3, 3.0, 0.3),
        gamma: float = 2.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32)
        )
        self.gamma = gamma
        self.dice_smooth = dice_smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, Z, Y, X) raw model output.
            targets: (B, Z, Y, X) integer labels {0, 1, 2}.

        Returns:
            Scalar loss (focal + dice).
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, Z, Y, X)

        # --- Focal loss ---
        # Standard CE per-voxel
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, Z, Y, X)
        # Gather the log-prob and prob for the true class
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, Z, Y, X, C)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, Z, Y, X)

        # p_t = probability of the true class
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B, Z, Y, X)
        log_p_t = (log_probs * targets_one_hot).sum(dim=1)  # (B, Z, Y, X)

        # alpha_t = class weight for the true class
        alpha_t = self.class_weights.to(logits.device)[targets]  # (B, Z, Y, X)

        # Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = -(alpha_t * focal_weight * log_p_t).mean()

        # --- Soft Dice loss (per-class, weighted) ---
        dice_loss = 0.0
        total_weight = 0.0
        for c in range(num_classes):
            p = probs[:, c]  # (B, Z, Y, X)
            t = (targets == c).float()  # (B, Z, Y, X)
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_c = 1.0 - (2.0 * intersection + self.dice_smooth) / (
                union + self.dice_smooth
            )
            w = self.class_weights[c]
            dice_loss += w * dice_c
            total_weight += w

        dice_loss = dice_loss / total_weight

        return focal_loss + dice_loss


# ---- Component 2: Skeleton Recall Loss ----


class SkeletonRecallLoss(nn.Module):
    """Skeleton-based recall loss for topology preservation.

    Instead of differentiable skeletonization (expensive on GPU, as in clDice),
    this loss uses a precomputed skeleton mask from the DataLoader and simply
    measures how well the model's soft predictions recall that skeleton.

    Formula:
      recall = sum(pred_surface * skel_gt) / sum(skel_gt)
      loss   = 1 - recall

    The skeleton is computed on the CPU side in the Dataset using
    skimage.morphology.skeletonize + 2px dilation. This is ~90% cheaper
    than the iterative soft skeletonization used in clDice.

    Applied to the surface class (label=1) only.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skeleton: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, Z, Y, X) raw model output.
            targets: (B, Z, Y, X) integer labels (unused, kept for API compat).
            skeleton: (B, Z, Y, X) binary skeleton mask (1 = skeleton, 0 = not).

        Returns:
            Scalar skeleton recall loss.
        """
        probs = F.softmax(logits, dim=1)
        pred_surface = probs[:, 1]  # (B, Z, Y, X) — surface class prob

        skel_float = skeleton.float()  # (B, Z, Y, X)

        # Recall: how much of the skeleton does the prediction cover?
        recall_num = (pred_surface * skel_float).sum() + self.smooth
        recall_den = skel_float.sum() + self.smooth

        recall = recall_num / recall_den
        return 1.0 - recall


# ---- Component 3: Boundary (Distance) Loss ----
# Identical to V1 — copied here to maintain pipeline isolation.


class BoundaryLoss(nn.Module):
    """Distance-based boundary loss for surface proximity.

    Penalizes predictions based on their distance from the ground truth
    boundary. Uses precomputed distance transforms.

    Applied only to the surface class (label=1).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, Z, Y, X) raw model output.
            targets: (B, Z, Y, X) integer labels.

        Returns:
            Scalar boundary loss.
        """
        probs = F.softmax(logits, dim=1)
        pred_surface = probs[:, 1]  # (B, Z, Y, X)

        batch_size = targets.shape[0]
        total_loss = 0.0

        for b in range(batch_size):
            gt_binary = (targets[b] == 1).cpu().numpy().astype(np.uint8)

            if gt_binary.sum() == 0:
                total_loss = total_loss + pred_surface[b].mean()
                continue

            if gt_binary.sum() == gt_binary.size:
                total_loss = total_loss + (1.0 - pred_surface[b]).mean()
                continue

            dt_pos = distance_transform_edt(1 - gt_binary)
            dt_neg = distance_transform_edt(gt_binary)
            signed_dt = dt_pos - dt_neg

            signed_dt_tensor = torch.from_numpy(signed_dt.astype(np.float32)).to(
                logits.device
            )

            total_loss = total_loss + (pred_surface[b] * signed_dt_tensor).mean()

        return total_loss / batch_size


# ---- Composite Loss V2 (single-scale, used by the deep supervision wrapper) ----


class CompositeLossV2(nn.Module):
    """Base composite loss for Pipeline V2 (single output scale).

    L = w_focal_dice * (Focal + Dice) + w_skel * SkeletonRecall + w_boundary * Boundary

    Default weights: 0.3, 0.3, 0.2
    """

    def __init__(
        self,
        w_focal_dice: float = 0.3,
        w_skel: float = 0.3,
        w_boundary: float = 0.2,
        class_weights: tuple = (0.3, 3.0, 0.3),
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.w_focal_dice = w_focal_dice
        self.w_skel = w_skel
        self.w_boundary = w_boundary

        self.focal_dice = FocalDiceLoss(
            class_weights=class_weights, gamma=focal_gamma
        )
        self.skel_recall = SkeletonRecallLoss()
        self.boundary = BoundaryLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skeleton: torch.Tensor,
    ) -> dict:
        """
        Args:
            logits: (B, C, Z, Y, X) raw model output.
            targets: (B, Z, Y, X) integer labels.
            skeleton: (B, Z, Y, X) binary skeleton mask.

        Returns:
            Dict with 'total', 'focal_dice', 'skel_recall', 'boundary'.
        """
        loss_focal_dice = self.focal_dice(logits, targets)
        loss_skel = self.skel_recall(logits, targets, skeleton)
        loss_boundary = self.boundary(logits, targets)

        total = (
            self.w_focal_dice * loss_focal_dice
            + self.w_skel * loss_skel
            + self.w_boundary * loss_boundary
        )

        return {
            "total": total,
            "focal_dice": loss_focal_dice.detach(),
            "skel_recall": loss_skel.detach(),
            "boundary": loss_boundary.detach(),
        }


# ---- Deep Supervision Composite Loss ----


class DeepSupCompositeLoss(nn.Module):
    """Composite loss with deep supervision for Pipeline V2.

    Applies the full CompositeLossV2 to the main output, plus simplified
    losses at each auxiliary decoder resolution.

    Auxiliary losses are computed with downsampled targets/skeletons (nearest).

    Total = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss + 0.125 * aux3_loss

    The auxiliary weights decay by 2x at each coarser level, as those
    outputs are progressively less refined.
    """

    def __init__(
        self,
        w_focal_dice: float = 0.3,
        w_skel: float = 0.3,
        w_boundary: float = 0.2,
        class_weights: tuple = (0.3, 3.0, 0.3),
        focal_gamma: float = 2.0,
        aux_weights: tuple = (0.5, 0.25, 0.125),
    ):
        super().__init__()
        self.base_loss = CompositeLossV2(
            w_focal_dice=w_focal_dice,
            w_skel=w_skel,
            w_boundary=w_boundary,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )
        self.aux_weights = aux_weights

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        skeleton: torch.Tensor,
    ) -> dict:
        """
        Args:
            outputs: Dict from UNet3DDeepSup.forward():
                     {"logits": (B,C,Z,Y,X), "aux_1":..., "aux_2":..., "aux_3":...}
            targets: (B, Z, Y, X) integer labels at full resolution.
            skeleton: (B, Z, Y, X) binary skeleton at full resolution.

        Returns:
            Dict with 'total', 'focal_dice', 'skel_recall', 'boundary',
            'main_loss', 'aux_1_loss', 'aux_2_loss', 'aux_3_loss'.
        """
        main_logits = outputs["logits"]
        main_losses = self.base_loss(main_logits, targets, skeleton)

        total = main_losses["total"]
        result = {
            "total": total,
            "focal_dice": main_losses["focal_dice"],
            "skel_recall": main_losses["skel_recall"],
            "boundary": main_losses["boundary"],
            "main_loss": main_losses["total"].detach(),
        }

        # Auxiliary losses at coarser decoder stages
        for i, w in enumerate(self.aux_weights):
            aux_key = f"aux_{i + 1}"
            if aux_key not in outputs:
                continue

            aux_logits = outputs[aux_key]
            aux_shape = aux_logits.shape[2:]  # (Z', Y', X')

            # Downsample targets and skeleton to match auxiliary resolution
            # targets: (B, Z, Y, X) -> (B, 1, Z, Y, X) -> interpolate -> (B, Z', Y', X')
            targets_ds = F.interpolate(
                targets.unsqueeze(1).float(),
                size=aux_shape,
                mode="nearest",
            ).squeeze(1).long()

            skeleton_ds = F.interpolate(
                skeleton.unsqueeze(1).float(),
                size=aux_shape,
                mode="nearest",
            ).squeeze(1)

            aux_losses = self.base_loss(aux_logits, targets_ds, skeleton_ds)
            aux_loss = aux_losses["total"]
            total = total + w * aux_loss
            result[f"{aux_key}_loss"] = aux_loss.detach()

        result["total"] = total
        return result
