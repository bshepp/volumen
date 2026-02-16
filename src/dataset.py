# Pipeline: V1 (src/) — See PIPELINES.md
"""
PyTorch Dataset for Vesuvius Challenge Surface Detection.

Key optimizations:
  - Volume-grouped batching: each batch draws all patches from ONE volume,
    so only 1 volume is loaded from disk per batch (not batch_size volumes).
  - Returns raw 1-channel patches (feature computation happens on GPU)
  - Surface-biased sampling (70% patches contain surface voxels)
  - Data augmentation: flips, 90-degree rotations, intensity perturbation
  - Scroll-aware train/val splitting
"""

import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import tifffile


class VesuviusDataset(Dataset):
    """3D patch-based dataset for surface segmentation.

    Each __getitem__ takes a volume index, loads the volume ONCE, and extracts
    `patches_per_volume` random patches. Returns a stacked batch.

    Returns:
      - images: float32 tensor (patches_per_volume, 1, pz, py, px)
      - labels: long tensor (patches_per_volume, pz, py, px)
    """

    def __init__(
        self,
        data_dir: str,
        sample_ids: List[str],
        patch_size: int = 128,
        patches_per_volume: int = 4,
        surface_bias: float = 0.7,
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.sample_ids = sample_ids
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.surface_bias = surface_bias
        self.augment = augment

        self.image_dir = os.path.join(data_dir, "train_images")
        self.label_dir = os.path.join(data_dir, "train_labels")

        self._surface_coords: Dict[str, np.ndarray] = {}
        self._vol_shapes: Dict[str, Tuple[int, ...]] = {}
        self._preload_surface_coords()

    def _preload_surface_coords(self) -> None:
        """Cache surface voxel coordinates for biased patch sampling."""
        max_coords_per_vol = 5000
        for i, sid in enumerate(self.sample_ids):
            lbl_path = os.path.join(self.label_dir, f"{sid}.tif")
            lbl = tifffile.imread(lbl_path)
            self._vol_shapes[sid] = lbl.shape
            coords = np.argwhere(lbl == 1)
            if len(coords) > max_coords_per_vol:
                indices = np.random.choice(len(coords), max_coords_per_vol, replace=False)
                coords = coords[indices]
            self._surface_coords[sid] = coords.astype(np.int16)
            del lbl
            if (i + 1) % 100 == 0:
                print(f"  Loaded surface coords: {i + 1}/{len(self.sample_ids)}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load one volume and extract patches_per_volume patches.

        Returns stacked tensors of shape:
          images: (patches_per_volume, 1, pz, py, px) float32
          labels: (patches_per_volume, pz, py, px)   int64
        """
        sid = self.sample_ids[idx]
        ps = self.patches_per_volume

        img = tifffile.imread(os.path.join(self.image_dir, f"{sid}.tif"))
        lbl = tifffile.imread(os.path.join(self.label_dir, f"{sid}.tif"))
        vol_shape = img.shape

        img_patches = []
        lbl_patches = []

        for _ in range(ps):
            origin = self._sample_patch_origin(sid, vol_shape)
            pz, py, px = origin
            p = self.patch_size
            ip = img[pz:pz+p, py:py+p, px:px+p]
            lp = lbl[pz:pz+p, py:py+p, px:px+p]

            raw = ip.astype(np.float32) / 255.0
            raw = raw[np.newaxis]  # (1, Z, Y, X)

            if self.augment:
                raw, lp = self._augment(raw, lp)

            img_patches.append(np.ascontiguousarray(raw))
            lbl_patches.append(np.ascontiguousarray(lp))

        img_stack = np.stack(img_patches)  # (ps, 1, Z, Y, X)
        lbl_stack = np.stack(lbl_patches)  # (ps, Z, Y, X)

        img_tensor = torch.from_numpy(img_stack).float()
        lbl_tensor = torch.from_numpy(lbl_stack).long()

        return img_tensor, lbl_tensor

    def _sample_patch_origin(
        self, sid: str, vol_shape: Tuple[int, ...]
    ) -> Tuple[int, int, int]:
        """Sample a patch origin, biased toward surface voxels."""
        ps = self.patch_size
        max_z = vol_shape[0] - ps
        max_y = vol_shape[1] - ps
        max_x = vol_shape[2] - ps

        if max_z < 0 or max_y < 0 or max_x < 0:
            return (0, 0, 0)

        surface_coords = self._surface_coords[sid]
        use_surface = (
            random.random() < self.surface_bias and len(surface_coords) > 0
        )

        if use_surface:
            coord_idx = random.randint(0, len(surface_coords) - 1)
            cz, cy, cx = surface_coords[coord_idx]
            half = ps // 2
            pz = int(np.clip(cz - half, 0, max_z))
            py = int(np.clip(cy - half, 0, max_y))
            px = int(np.clip(cx - half, 0, max_x))
        else:
            pz = random.randint(0, max_z)
            py = random.randint(0, max_y)
            px = random.randint(0, max_x)

        return (pz, py, px)

    def _augment(
        self, raw: np.ndarray, lbl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to raw patch and label."""
        for spatial_ax in range(3):
            if random.random() < 0.5:
                raw = np.flip(raw, axis=spatial_ax + 1)
                lbl = np.flip(lbl, axis=spatial_ax)

        k = random.randint(0, 3)
        if k > 0:
            raw = np.rot90(raw, k=k, axes=(2, 3))
            lbl = np.rot90(lbl, k=k, axes=(1, 2))

        if random.random() < 0.3:
            k2 = random.choice([1, 3])
            raw = np.rot90(raw, k=k2, axes=(1, 2))
            lbl = np.rot90(lbl, k=k2, axes=(0, 1))

        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.05, 0.05)
            raw = raw * scale + shift

        return raw, lbl


def collate_volume_batches(batch):
    """Custom collate that flattens the volume-batch dimension.

    Input: list of (patches_per_vol, 1, Z, Y, X) and (patches_per_vol, Z, Y, X)
    Output: (total_patches, 1, Z, Y, X) and (total_patches, Z, Y, X)
    """
    images = torch.cat([b[0] for b in batch], dim=0)
    labels = torch.cat([b[1] for b in batch], dim=0)
    return images, labels


def get_scroll_splits(
    csv_path: str,
    val_scroll_ids: Optional[List[str]] = None,
    deprecated_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split sample IDs into train/val by scroll ID."""
    if val_scroll_ids is None:
        val_scroll_ids = ["26002"]
    if deprecated_ids is None:
        deprecated_ids = set()
    else:
        deprecated_ids = set(deprecated_ids)

    train_ids = []
    val_ids = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["id"]
            scroll = row["scroll_id"]
            if sid in deprecated_ids:
                continue
            if scroll in val_scroll_ids:
                val_ids.append(sid)
            else:
                train_ids.append(sid)

    return train_ids, val_ids


def get_deprecated_ids(data_dir: str) -> List[str]:
    """Get IDs in the deprecated directory."""
    dep_dir = os.path.join(data_dir, "deprecated_train_images")
    if not os.path.exists(dep_dir):
        return []
    return [f.replace(".tif", "") for f in os.listdir(dep_dir) if f.endswith(".tif")]
