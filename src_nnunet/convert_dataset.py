# Pipeline: nnU-Net (src_nnunet/) — See PIPELINES.md
"""
Convert Vesuvius Challenge data to nnU-Net v2 dataset format.

Creates the required directory structure, symlinks to original TIF files
(to avoid doubling disk usage), spacing JSON files, and dataset.json.

nnU-Net v2 expects:
    nnUNet_raw/Dataset011_Vesuvius/
    ├── dataset.json
    ├── imagesTr/
    │   ├── vesuvius_001_0000.tif   (+  vesuvius_001.json)
    │   └── ...
    └── labelsTr/
        ├── vesuvius_001.tif        (+  vesuvius_001.json)
        └── ...

Usage:
    python -m src_nnunet.convert_dataset \\
        --data-dir /path/to/vesuvius-challenge-surface-detection \\
        --output-dir /path/to/nnUNet_raw/Dataset011_Vesuvius \\
        --spacing 7.91
"""

import argparse
import csv
import json
import os
import sys


def read_train_ids(data_dir: str) -> list:
    """Read sample IDs from train.csv, excluding deprecated samples."""
    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: train.csv not found at {csv_path}")
        sys.exit(1)

    deprecated_dir = os.path.join(data_dir, "deprecated_train_images")
    deprecated_ids = set()
    if os.path.isdir(deprecated_dir):
        for f in os.listdir(deprecated_dir):
            if f.endswith(".tif"):
                deprecated_ids.add(os.path.splitext(f)[0])

    ids = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["id"]
            if sample_id not in deprecated_ids:
                ids.append(sample_id)

    return sorted(ids)


def write_spacing_json(path: str, spacing: float):
    """Write a spacing JSON file for a TIF (required by nnU-Net's Tiff3DIO)."""
    with open(path, "w") as f:
        json.dump({"spacing": [spacing, spacing, spacing]}, f)


def write_dataset_json(output_dir: str, num_training: int):
    """Write dataset.json for nnU-Net v2."""
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "surface": 1,
            "interior": 2
        },
        "numTraining": num_training,
        "file_ending": ".tif",
    }
    path = os.path.join(output_dir, "dataset.json")
    with open(path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Wrote {path}")


def create_link(src: str, dst: str, use_copy: bool = False):
    """Create a symlink or copy from src to dst."""
    if os.path.exists(dst):
        return
    if use_copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def convert(data_dir: str, output_dir: str, spacing: float, use_copy: bool = False):
    """Convert Vesuvius data to nnU-Net v2 format."""
    images_src = os.path.join(data_dir, "train_images")
    labels_src = os.path.join(data_dir, "train_labels")

    if not os.path.isdir(images_src):
        print(f"ERROR: train_images directory not found at {images_src}")
        sys.exit(1)
    if not os.path.isdir(labels_src):
        print(f"ERROR: train_labels directory not found at {labels_src}")
        sys.exit(1)

    images_dst = os.path.join(output_dir, "imagesTr")
    labels_dst = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    sample_ids = read_train_ids(data_dir)
    print(f"Found {len(sample_ids)} training samples")

    skipped = 0
    linked = 0
    for i, sample_id in enumerate(sample_ids):
        img_src = os.path.join(images_src, f"{sample_id}.tif")
        lbl_src = os.path.join(labels_src, f"{sample_id}.tif")

        if not os.path.exists(img_src):
            print(f"  SKIP {sample_id}: image not found")
            skipped += 1
            continue
        if not os.path.exists(lbl_src):
            print(f"  SKIP {sample_id}: label not found")
            skipped += 1
            continue

        case_id = f"vesuvius_{i:04d}"

        # Image: {case_id}_0000.tif (single channel)
        create_link(img_src, os.path.join(images_dst, f"{case_id}_0000.tif"), use_copy)
        write_spacing_json(os.path.join(images_dst, f"{case_id}.json"), spacing)

        # Label: {case_id}.tif
        create_link(lbl_src, os.path.join(labels_dst, f"{case_id}.tif"), use_copy)
        write_spacing_json(os.path.join(labels_dst, f"{case_id}.json"), spacing)

        linked += 1
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_ids)}")

    print(f"Done: {linked} samples linked, {skipped} skipped")

    write_dataset_json(output_dir, linked)

    # Write an ID mapping file for reference
    mapping_path = os.path.join(output_dir, "id_mapping.csv")
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nnunet_id", "original_id"])
        for i, sample_id in enumerate(sample_ids):
            writer.writerow([f"vesuvius_{i:04d}", sample_id])
    print(f"Wrote ID mapping to {mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Vesuvius data to nnU-Net v2 format"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to vesuvius-challenge-surface-detection/ directory"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output path (e.g. nnUNet_raw/Dataset011_Vesuvius)"
    )
    parser.add_argument(
        "--spacing", type=float, default=7.91,
        help="Isotropic voxel spacing in micrometers (default: 7.91)"
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of creating symlinks"
    )
    args = parser.parse_args()
    convert(args.data_dir, args.output_dir, args.spacing, args.copy)


if __name__ == "__main__":
    main()
