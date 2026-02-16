# Pipeline: V1 (src/) — See PIPELINES.md
"""
Composite loss function for topology-aware 3D surface segmentation.

Loss = 0.4 * (CE + Dice) + 0.3 * clDice + 0.3 * BoundaryLoss

Components:
  1. Weighted Cross-Entropy + Soft Dice Loss (standard segmentation)
  2. clDice Loss (centerline Dice - topology-preserving)
  3. Boundary Loss (distance-based surface proximity)

References:
  - clDice: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
    for Tubular Structure Segmentation", CVPR 2021
  - Boundary Loss: Kervadec et al., "Boundary loss for highly unbalanced
    segmentation", MIDL 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


# ---- Component 1: Weighted CE + Dice ----


class WeightedCEDiceLoss(nn.Module):
    """Combined weighted cross-entropy and soft Dice loss.

    Class weights upweight the rare surface class (label=1).
    """

    def __init__(
        self,
        class_weights: tuple = (0.3, 3.0, 0.3),
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32)
        )
        self.dice_smooth = dice_smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, Z, Y, X) raw model output
            targets: (B, Z, Y, X) integer labels {0, 1, 2}

        Returns:
            Scalar loss.
        """
        # Weighted cross-entropy
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.class_weights.to(logits.device)
        )

        # Soft Dice loss (per-class, then averaged with weighting)
        probs = F.softmax(logits, dim=1)  # (B, C, Z, Y, X)
        num_classes = logits.shape[1]

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

        return ce_loss + dice_loss


# ---- Component 2: clDice (Centerline Dice) ----


def soft_erode_3d(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Soft morphological erosion using min-pool."""
    pad = kernel_size // 2
    return -F.max_pool3d(
        -img, kernel_size=kernel_size, stride=1, padding=pad
    )


def soft_dilate_3d(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Soft morphological dilation using max-pool."""
    pad = kernel_size // 2
    return F.max_pool3d(img, kernel_size=kernel_size, stride=1, padding=pad)


def soft_open_3d(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Soft morphological opening (erode then dilate)."""
    return soft_dilate_3d(soft_erode_3d(img, kernel_size), kernel_size)


def soft_skeletonize_3d(
    img: torch.Tensor, num_iters: int = 10, kernel_size: int = 3
) -> torch.Tensor:
    """Differentiable soft skeletonization for 3D volumes.

    Iteratively peels away layers using the difference between
    the image and its morphological opening.

    Args:
        img: (B, 1, Z, Y, X) soft binary mask [0, 1].
        num_iters: Number of erosion iterations.
        kernel_size: Kernel size for morphological operations.

    Returns:
        Soft skeleton, same shape as input.
    """
    img_open = soft_open_3d(img, kernel_size)
    skel = F.relu(img - img_open)
    for _ in range(num_iters):
        img = soft_erode_3d(img, kernel_size)
        img_open = soft_open_3d(img, kernel_size)
        delta = F.relu(img - img_open)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CLDiceLoss(nn.Module):
    """Centerline Dice loss for topology-preserving segmentation.

    Computes Dice between the skeletons of prediction and ground truth,
    encouraging the model to preserve the connected centerline structure
    of thin surfaces.

    Applied only to the surface class (label=1).
    """

    def __init__(
        self,
        num_iters: int = 10,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, Z, Y, X) raw model output
            targets: (B, Z, Y, X) integer labels

        Returns:
            Scalar clDice loss.
        """
        # Extract surface class probabilities
        probs = F.softmax(logits, dim=1)
        pred_surface = probs[:, 1:2]  # (B, 1, Z, Y, X) — class 1
        gt_surface = (targets == 1).float().unsqueeze(1)  # (B, 1, Z, Y, X)

        # Soft skeletonize both
        skel_pred = soft_skeletonize_3d(pred_surface, self.num_iters)
        skel_gt = soft_skeletonize_3d(gt_surface, self.num_iters)

        # Topology precision: how much of pred skeleton is in GT mask?
        tprec_num = (skel_pred * gt_surface).sum()
        tprec_den = skel_pred.sum() + self.smooth

        # Topology sensitivity: how much of GT skeleton is in pred mask?
        tsens_num = (skel_gt * pred_surface).sum()
        tsens_den = skel_gt.sum() + self.smooth

        tprec = (tprec_num + self.smooth) / tprec_den
        tsens = (tsens_num + self.smooth) / tsens_den

        # clDice = 2 * tprec * tsens / (tprec + tsens)
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)

        return 1.0 - cl_dice


# ---- Component 3: Boundary (Distance) Loss ----


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

        # Compute signed distance transform for each sample in batch
        batch_size = targets.shape[0]
        total_loss = 0.0

        for b in range(batch_size):
            gt_binary = (targets[b] == 1).cpu().numpy().astype(np.uint8)

            if gt_binary.sum() == 0:
                # No surface in this patch — penalize any predicted surface
                total_loss = total_loss + pred_surface[b].mean()
                continue

            if gt_binary.sum() == gt_binary.size:
                # All surface — shouldn't happen, but handle gracefully
                total_loss = total_loss + (1.0 - pred_surface[b]).mean()
                continue

            # Signed distance transform: negative inside, positive outside
            dt_pos = distance_transform_edt(1 - gt_binary)  # distance from surface
            dt_neg = distance_transform_edt(gt_binary)  # distance inside surface
            signed_dt = dt_pos - dt_neg

            # Convert to tensor
            signed_dt_tensor = torch.from_numpy(signed_dt.astype(np.float32)).to(
                logits.device
            )

            # Boundary loss: expected distance under the predicted distribution
            # Lower is better when prediction aligns with GT surface
            total_loss = total_loss + (pred_surface[b] * signed_dt_tensor).mean()

        return total_loss / batch_size


# ---- Composite Loss ----


class CompositeLoss(nn.Module):
    """Combined topology-aware loss.

    L = w_ce_dice * (CE + Dice) + w_cldice * clDice + w_boundary * BoundaryLoss

    Default weights: 0.4, 0.3, 0.3
    """

    def __init__(
        self,
        w_ce_dice: float = 0.4,
        w_cldice: float = 0.3,
        w_boundary: float = 0.3,
        class_weights: tuple = (0.3, 3.0, 0.3),
        cldice_iters: int = 10,
    ):
        super().__init__()
        self.w_ce_dice = w_ce_dice
        self.w_cldice = w_cldice
        self.w_boundary = w_boundary

        self.ce_dice = WeightedCEDiceLoss(class_weights=class_weights)
        self.cldice = CLDiceLoss(num_iters=cldice_iters)
        self.boundary = BoundaryLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """
        Args:
            logits: (B, C, Z, Y, X) raw model output.
            targets: (B, Z, Y, X) integer labels.

        Returns:
            Dict with 'total', 'ce_dice', 'cldice', 'boundary' loss values.
        """
        loss_ce_dice = self.ce_dice(logits, targets)
        loss_cldice = self.cldice(logits, targets)
        loss_boundary = self.boundary(logits, targets)

        total = (
            self.w_ce_dice * loss_ce_dice
            + self.w_cldice * loss_cldice
            + self.w_boundary * loss_boundary
        )

        return {
            "total": total,
            "ce_dice": loss_ce_dice.detach(),
            "cldice": loss_cldice.detach(),
            "boundary": loss_boundary.detach(),
        }
