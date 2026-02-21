# Pipeline: V3 (src_v3/) — See PIPELINES.md
"""
Training loop for Vesuvius Challenge Surface Detection — Pipeline V3.

Multi-scale fusion model with CompositeLossV3 on fused output.

WARNING: This is Pipeline V3. Do NOT import from src/ or src_v2/.
See PIPELINES.md in the project root.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import (
    VesuviusDataset,
    collate_volume_batches,
    get_scroll_splits,
    get_deprecated_ids,
)
from .features import GPUFeatureExtractor
from .losses import CompositeLossV3
from .model import get_model, count_parameters


PIPELINE_BANNER = """
╔══════════════════════════════════════════════════╗
║  Pipeline V3 — Multi-Scale Learned Fusion       ║
║  WARNING: Do NOT mix with src/ or src_v2/        ║
╚══════════════════════════════════════════════════╝
"""


def train_one_epoch(
    model: nn.Module,
    feat_extractor: GPUFeatureExtractor,
    loader: DataLoader,
    loss_fn: CompositeLossV3,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
) -> dict:
    model.train()
    running = {"total": 0.0, "focal_dice": 0.0, "skel_recall": 0.0, "boundary": 0.0}
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (raw_images, labels, skeletons) in enumerate(loader):
        raw_images = raw_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        skeletons = skeletons.to(device, non_blocking=True)

        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                features = feat_extractor(raw_images)
                logits = model(features)
                losses = loss_fn(logits, labels, skeletons)
                loss = losses["total"] / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            features = feat_extractor(raw_images)
            logits = model(features)
            losses = loss_fn(logits, labels, skeletons)
            loss = losses["total"] / grad_accum_steps
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

        if batch_idx % 20 == 0:
            print(
                f"  [V3] Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={losses['total'].item():.4f} "
                f"(focal_dice={losses['focal_dice'].item():.4f}, "
                f"skel_recall={losses['skel_recall'].item():.4f}, "
                f"boundary={losses['boundary'].item():.4f})"
            )

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    feat_extractor: GPUFeatureExtractor,
    loader: DataLoader,
    loss_fn: CompositeLossV3,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    model.eval()
    running = {"total": 0.0, "focal_dice": 0.0, "skel_recall": 0.0, "boundary": 0.0}
    n_batches = 0
    surface_tp = 0
    surface_fp = 0
    surface_fn = 0

    for raw_images, labels, skeletons in loader:
        raw_images = raw_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        skeletons = skeletons.to(device, non_blocking=True)

        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                features = feat_extractor(raw_images)
                logits = model(features)
                losses = loss_fn(logits, labels, skeletons)
        else:
            features = feat_extractor(raw_images)
            logits = model(features)
            losses = loss_fn(logits, labels, skeletons)

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

        preds = logits.argmax(dim=1)
        pred_surf = preds == 1
        gt_surf = labels == 1
        surface_tp += (pred_surf & gt_surf).sum().item()
        surface_fp += (pred_surf & ~gt_surf).sum().item()
        surface_fn += (~pred_surf & gt_surf).sum().item()

    avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
    dice_denom = 2 * surface_tp + surface_fp + surface_fn
    avg_losses["surface_voxel_dice"] = 2 * surface_tp / max(dice_denom, 1)
    return avg_losses


def main(args):
    print(PIPELINE_BANNER)
    print("=== Vesuvius Surface Detection Training — Pipeline V3 ===")
    print(f"Config: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    os.makedirs(args.output_dir, exist_ok=True)

    feat_extractor = GPUFeatureExtractor().to(device)
    feat_extractor.eval()
    print("[V3] GPU feature extractor initialized")

    deprecated_ids = get_deprecated_ids(args.data_dir)
    train_ids, val_ids = get_scroll_splits(
        csv_path=os.path.join(args.data_dir, "train.csv"),
        val_scroll_ids=args.val_scrolls.split(","),
        deprecated_ids=set(deprecated_ids),
    )
    print(f"[V3] Train: {len(train_ids)} samples, Val: {len(val_ids)} samples")

    train_dataset = VesuviusDataset(
        data_dir=args.data_dir,
        sample_ids=train_ids,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume,
        surface_bias=args.surface_bias,
        augment=True,
        skeleton_dilation=args.skeleton_dilation,
    )
    val_dataset = VesuviusDataset(
        data_dir=args.data_dir,
        sample_ids=val_ids,
        patch_size=args.patch_size,
        patches_per_volume=2,
        surface_bias=0.8,
        augment=False,
        skeleton_dilation=args.skeleton_dilation,
    )

    print(f"[V3] Train: {len(train_dataset)} volumes -> "
          f"{len(train_dataset) * args.patches_per_volume} patches/epoch")
    print(f"[V3] Val: {len(val_dataset)} volumes -> {len(val_dataset) * 2} patches/epoch")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_volume_batches,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_volume_batches,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    model = get_model(
        in_channels=6,
        num_classes=3,
        base_filters=args.base_filters,
    ).to(device)
    print(f"[V3] Model: MultiScaleFusionUNet, params={count_parameters(model):,}")

    loss_fn = CompositeLossV3(
        w_focal_dice=0.3,
        w_skel=0.3,
        w_boundary=0.2,
        class_weights=(0.3, 3.0, 0.3),
        focal_gamma=args.focal_gamma,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"[V3] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n--- [V3] Epoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f}) ---")

        train_metrics = train_one_epoch(
            model, feat_extractor, loader=train_loader,
            loss_fn=loss_fn, optimizer=optimizer, scaler=scaler,
            device=device, epoch=epoch, use_amp=use_amp,
        )

        val_metrics = validate(
            model, feat_extractor, loader=val_loader,
            loss_fn=loss_fn, device=device, use_amp=use_amp,
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  [V3] Train loss: {train_metrics['total']:.4f} | "
            f"Val loss: {val_metrics['total']:.4f} | "
            f"Val surface Dice: {val_metrics['surface_voxel_dice']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "val": val_metrics,
            "time_s": elapsed,
        }
        history.append(record)

        is_best = val_metrics["total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "pipeline": "v3",
                },
                os.path.join(args.output_dir, "best_model_v3.pth"),
            )
            print(f"  [V3] ** New best model saved (val_loss={best_val_loss:.4f}) **")

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "pipeline": "v3",
                },
                os.path.join(args.output_dir, f"checkpoint_v3_epoch{epoch}.pth"),
            )

        with open(os.path.join(args.output_dir, "history_v3.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

    print("\n=== [V3] Training complete ===")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {args.output_dir}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Vesuvius Surface Detection Training — Pipeline V3"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs_v3")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-volume", type=int, default=4)
    parser.add_argument("--surface-bias", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--skeleton-dilation", type=int, default=2)
    parser.add_argument("--val-scrolls", type=str, default="26002")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
