# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
Composite loss for Pipeline V3 — Focal+Dice + SkeletonRecall + Boundary.

WARNING: This is Pipeline V3. Do NOT import from src/ or src_v2/.
See PIPELINES.md in the project root.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class FocalDiceLoss(nn.Module):
    def __init__(self, class_weights=(0.3, 3.0, 0.3), gamma=2.0, dice_smooth=1.0):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.gamma = gamma
        self.dice_smooth = dice_smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        log_p_t = (log_probs * targets_one_hot).sum(dim=1)
        alpha_t = self.class_weights.to(logits.device)[targets]
        focal_loss = -(alpha_t * (1.0 - p_t) ** self.gamma * log_p_t).mean()
        dice_loss = 0.0
        total_weight = 0.0
        for c in range(num_classes):
            p, t = probs[:, c], (targets == c).float()
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_loss += self.class_weights[c] * (1.0 - (2 * inter + self.dice_smooth) / (union + self.dice_smooth))
            total_weight += self.class_weights[c]
        return focal_loss + dice_loss / total_weight


class SkeletonRecallLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, skeleton):
        pred_surface = F.softmax(logits, dim=1)[:, 1]
        skel = skeleton.float()
        recall_num = (pred_surface * skel).sum() + self.smooth
        recall_den = skel.sum() + self.smooth
        return 1.0 - (recall_num / recall_den)


class BoundaryLoss(nn.Module):
    def forward(self, logits, targets):
        pred_surface = F.softmax(logits, dim=1)[:, 1]
        batch_size = targets.shape[0]
        total = 0.0
        for b in range(batch_size):
            gt = (targets[b] == 1).cpu().numpy().astype(np.uint8)
            if gt.sum() == 0:
                total = total + pred_surface[b].mean()
                continue
            if gt.sum() == gt.size:
                total = total + (1.0 - pred_surface[b]).mean()
                continue
            dt_pos = distance_transform_edt(1 - gt)
            dt_neg = distance_transform_edt(gt)
            signed_dt = torch.from_numpy((dt_pos - dt_neg).astype(np.float32)).to(logits.device)
            total = total + (pred_surface[b] * signed_dt).mean()
        return total / batch_size


class CompositeLossV3(nn.Module):
    def __init__(self, w_focal_dice=0.3, w_skel=0.3, w_boundary=0.2, focal_gamma=2.0):
        super().__init__()
        self.focal_dice = FocalDiceLoss(gamma=focal_gamma)
        self.skel_recall = SkeletonRecallLoss()
        self.boundary = BoundaryLoss()
        self.w_focal_dice = w_focal_dice
        self.w_skel = w_skel
        self.w_boundary = w_boundary

    def forward(self, logits, targets, skeleton):
        lfd = self.focal_dice(logits, targets)
        lsk = self.skel_recall(logits, targets, skeleton)
        lb = self.boundary(logits, targets)
        total = self.w_focal_dice * lfd + self.w_skel * lsk + self.w_boundary * lb
        return {
            "total": total,
            "focal_dice": lfd.detach() if hasattr(lfd, "detach") else lfd,
            "skel_recall": lsk.detach() if hasattr(lsk, "detach") else lsk,
            "boundary": lb.detach() if hasattr(lb, "detach") else lb,
        }
