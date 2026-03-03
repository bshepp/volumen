# nnU-Net Pipeline for Vesuvius Challenge Surface Detection

Self-configuring 3D segmentation using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), the framework used by the **1st place team** in this competition.

## Why nnU-Net

The 1st place team's single nnU-Net model scored **0.577/0.614** (public/private) at 250 epochs — far above our custom UNet3DDeepSup at **0.405/0.426**. nnU-Net automatically configures preprocessing, architecture, patch size, batch size, and augmentation from the dataset properties.

## Setup

```bash
pip install -r src_nnunet/requirements.txt
```

Set the three required environment variables:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Usage

### 1. Convert data to nnU-Net format

```bash
python -m src_nnunet.convert_dataset \
    --data-dir /path/to/vesuvius-challenge-surface-detection \
    --output-dir /path/to/nnUNet_raw/Dataset011_Vesuvius \
    --spacing 7.91
```

This creates symlinks (not copies) to the original TIF files, generates the required spacing JSON files, and writes `dataset.json`.

### 2. Plan and preprocess

```bash
nnUNetv2_plan_and_preprocess -d 011 --verify_dataset_integrity
```

### 3. Train

```bash
# Train on all data (no cross-validation fold), 200 epochs
nnUNetv2_train 011 3d_fullres all -tr nnUNetTrainer --npz
```

To resume from a checkpoint:

```bash
nnUNetv2_train 011 3d_fullres all -tr nnUNetTrainer --npz --c
```

### 4. Inference

```bash
python -m src_nnunet.predict \
    --input-dir /path/to/test_images \
    --output-dir /path/to/predictions \
    --model-dir /path/to/nnUNet_results/Dataset011_Vesuvius/nnUNetTrainer__nnUNetPlans__3d_fullres
```

This runs nnU-Net inference followed by the 1st place post-processing pipeline.

## Training on Kaggle (recommended)

Use `notebooks/nnunet_training.ipynb`. The competition data is already available on Kaggle, so no upload is needed. The notebook handles installation, data conversion, and training in a single session.

## Training on Hugging Face Jobs

The `launch_hf_job.py` script launches a remote GPU training job on Hugging Face. It embeds `train_hf.py` (the full training pipeline: data download, conversion, preprocessing, training, model upload) into the job command via base64 encoding.

```bash
# With Kaggle API token
python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx

# Or with Kaggle username/key
python -m src_nnunet.launch_hf_job --kaggle-username USER --kaggle-key KEY

# Custom flavor or timeout
python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx --flavor a10g-small --timeout 86400
```

Monitor a running job:

```bash
python -m src_nnunet.monitor_hf_job JOB_ID --logs

# Quick status + tail logs
python -m src_nnunet.check_job 60
```

The trained model is uploaded to `huggingface.co/bshepp/vesuvius-nnunet` automatically on completion.

## Files

| File | Purpose |
|------|---------|
| `convert_dataset.py` | Convert TIF data to nnU-Net v2 format |
| `predict.py` | Inference with 1st place post-processing |
| `train_hf.py` | End-to-end training script for HF Jobs (download → train → upload) |
| `launch_hf_job.py` | Launch a training job on Hugging Face Jobs |
| `monitor_hf_job.py` | Monitor job status and stream logs |
| `check_job.py` | Quick job status check and log tail |
| `requirements.txt` | Pipeline-specific dependencies |

## References

- [nnU-Net v2 GitHub](https://github.com/MIC-DKFZ/nnUNet)
- [1st Place Solution Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)
- [Baseline Training Notebook (jirkaborovec)](https://www.kaggle.com/code/jirkaborovec/surface-train-inference-3d-segm-gpu-augment)
