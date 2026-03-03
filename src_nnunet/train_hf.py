# Pipeline: nnU-Net (src_nnunet/) — See PIPELINES.md
"""
nnU-Net training script for Hugging Face Jobs.

Handles the full pipeline:
  1. Download competition data from Kaggle
  2. Convert to nnU-Net format
  3. Plan and preprocess
  4. Train for 200 epochs
  5. Upload model checkpoint to Hugging Face Hub

Usage (via HF Jobs):
    hf jobs run \\
        --flavor a10g-small \\
        --timeout 12h \\
        --secrets HF_TOKEN KAGGLE_USERNAME KAGGLE_KEY \\
        pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \\
        bash -c "pip install nnunetv2 kaggle huggingface_hub && python train_hf.py"
"""

import csv
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ID = 11
DATASET_NAME = "Dataset011_Vesuvius"
CONFIGURATION = "3d_fullres"
FOLD = "all"
VOXEL_SPACING = 7.91
NUM_EPOCHS = 200

WORK_DIR = "/home/user/nnunet_work"
NNUNET_RAW = os.path.join(WORK_DIR, "nnUNet_raw")
NNUNET_PREPROCESSED = os.path.join(WORK_DIR, "nnUNet_preprocessed")
NNUNET_RESULTS = os.path.join(WORK_DIR, "nnUNet_results")
DATA_DIR = os.path.join(WORK_DIR, "data")

HF_REPO_ID = os.environ.get("HF_REPO_ID", "vesuvius-nnunet-model")


def run(cmd, **kwargs):
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)


def setup_environment():
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = NNUNET_RESULTS

    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS, DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"nnUNet_raw:          {NNUNET_RAW}")
    print(f"nnUNet_preprocessed: {NNUNET_PREPROCESSED}")
    print(f"nnUNet_results:      {NNUNET_RESULTS}")


def download_data():
    """Download competition data from Kaggle."""
    print("\n" + "=" * 60)
    print("Step 1: Download competition data from Kaggle")
    print("=" * 60)

    # Support both auth methods: KAGGLE_API_TOKEN (new) and KAGGLE_USERNAME+KAGGLE_KEY (legacy)
    api_token = os.environ.get("KAGGLE_API_TOKEN", "")
    kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
    kaggle_key = os.environ.get("KAGGLE_KEY", "")

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    if api_token:
        # New-format token: write to ~/.kaggle/kaggle_api_token
        token_path = os.path.join(kaggle_dir, "kaggle_api_token")
        with open(token_path, "w") as f:
            f.write(api_token)
        os.chmod(token_path, 0o600)
        print("Kaggle API token configured (new format)")
    elif kaggle_user and kaggle_key:
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
        with open(kaggle_json, "w") as f:
            json.dump({"username": kaggle_user, "key": kaggle_key}, f)
        os.chmod(kaggle_json, 0o600)
        print("Kaggle credentials configured (legacy format)")
    else:
        print("WARNING: No Kaggle credentials found. Set KAGGLE_API_TOKEN or KAGGLE_USERNAME+KAGGLE_KEY")

    t0 = time.time()
    run("apt-get update -qq && apt-get install -y -qq unzip > /dev/null 2>&1")
    run("df -h /home /tmp / 2>/dev/null || true")
    run(f"kaggle competitions download vesuvius-challenge-surface-detection -p {DATA_DIR}")
    run(f"cd {DATA_DIR} && unzip -q -o '*.zip'")

    # Free ~24GB by removing the zip
    import glob
    for zf in glob.glob(os.path.join(DATA_DIR, "*.zip")):
        os.remove(zf)
        print(f"Removed {zf} to free disk space")

    run("df -h /home /tmp / 2>/dev/null || true")
    print(f"Data download complete in {time.time() - t0:.0f}s")


def convert_data():
    """Convert to nnU-Net format."""
    print("\n" + "=" * 60)
    print("Step 2: Convert data to nnU-Net format")
    print("=" * 60)

    dataset_dir = os.path.join(NNUNET_RAW, DATASET_NAME)
    images_dst = os.path.join(dataset_dir, "imagesTr")
    labels_dst = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "train.csv")
    sample_ids = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_ids.append(row["id"])

    deprecated_dir = os.path.join(DATA_DIR, "deprecated_train_images")
    deprecated_ids = set()
    if os.path.isdir(deprecated_dir):
        for fn in os.listdir(deprecated_dir):
            if fn.endswith(".tif"):
                deprecated_ids.add(os.path.splitext(fn)[0])

    sample_ids = sorted([s for s in sample_ids if s not in deprecated_ids])
    print(f"Training samples: {len(sample_ids)}")

    linked = 0
    for i, sid in enumerate(sample_ids):
        img_src = os.path.join(DATA_DIR, "train_images", f"{sid}.tif")
        lbl_src = os.path.join(DATA_DIR, "train_labels", f"{sid}.tif")

        if not os.path.exists(img_src) or not os.path.exists(lbl_src):
            continue

        case_id = f"vesuvius_{i:04d}"

        img_dst = os.path.join(images_dst, f"{case_id}_0000.tif")
        lbl_dst = os.path.join(labels_dst, f"{case_id}.tif")

        if not os.path.exists(img_dst):
            os.symlink(os.path.abspath(img_src), img_dst)
        if not os.path.exists(lbl_dst):
            os.symlink(os.path.abspath(lbl_src), lbl_dst)

        for target_dir in [images_dst, labels_dst]:
            json_path = os.path.join(target_dir, f"{case_id}.json")
            with open(json_path, "w") as f:
                json.dump({"spacing": [VOXEL_SPACING] * 3}, f)

        linked += 1

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "surface": 1, "interior": 2},
        "numTraining": linked,
        "file_ending": ".tif",
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Linked {linked} samples")


def register_trainer():
    """Register the custom 200-epoch trainer."""
    import nnunetv2

    trainer_code = (
        "import torch\n"
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n"
        "class nnUNetTrainer_200epochs(nnUNetTrainer):\n"
        "    def __init__(self, plans, configuration, fold, dataset_json,\n"
        '                 device=torch.device("cuda")):\n'
        "        super().__init__(plans, configuration, fold, dataset_json, device)\n"
        "        self.num_epochs = 200\n"
    )

    trainer_dir = os.path.join(
        os.path.dirname(nnunetv2.__path__[0]),
        "nnunetv2", "training", "nnUNetTrainer", "variants", "training_length",
    )
    os.makedirs(trainer_dir, exist_ok=True)
    trainer_path = os.path.join(trainer_dir, "nnUNetTrainer_200epochs.py")
    with open(trainer_path, "w") as f:
        f.write(trainer_code)
    print(f"Registered nnUNetTrainer_200epochs at {trainer_path}")


def plan_and_preprocess():
    print("\n" + "=" * 60)
    print("Step 3: Plan and preprocess")
    print("=" * 60)

    run(f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} "
        f"--verify_dataset_integrity -c {CONFIGURATION}")


def train():
    print("\n" + "=" * 60)
    print("Step 4: Train")
    print("=" * 60)

    run(f"nnUNetv2_train {DATASET_ID} {CONFIGURATION} {FOLD} "
        f"-tr nnUNetTrainer_200epochs --npz")


def upload_model():
    """Upload trained model to Hugging Face Hub."""
    print("\n" + "=" * 60)
    print("Step 5: Upload model to Hugging Face Hub")
    print("=" * 60)

    model_dir = os.path.join(
        NNUNET_RESULTS, DATASET_NAME,
        f"nnUNetTrainer_200epochs__nnUNetPlans__{CONFIGURATION}",
    )

    if not os.path.isdir(model_dir):
        print(f"WARNING: Model directory not found at {model_dir}")
        print("Listing nnUNet_results:")
        run(f"find {NNUNET_RESULTS} -type f | head -50")
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("WARNING: HF_TOKEN not set, skipping upload")
        print(f"Model saved at: {model_dir}")
        return

    username = os.environ.get("HF_USERNAME", "briansheppard")
    repo_id = f"{username}/{HF_REPO_ID}"

    run(f'huggingface-cli upload {repo_id} "{model_dir}" . '
        f'--repo-type model --token "{hf_token}"')

    print(f"\nModel uploaded to: https://huggingface.co/{repo_id}")


def main():
    t_start = time.time()

    setup_environment()
    download_data()
    convert_data()
    register_trainer()
    plan_and_preprocess()

    # Free ~33GB by removing raw data (no longer needed after preprocessing)
    import shutil
    for d in [DATA_DIR, os.path.join(NNUNET_RAW, DATASET_NAME)]:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"Removed {d} to free disk space")
    run("df -h / 2>/dev/null || true")

    train()
    upload_model()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Total time: {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
