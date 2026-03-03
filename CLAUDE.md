# CLAUDE.md — Agent Instructions for Vesuvius Challenge

## Project Overview

Vesuvius Challenge surface detection: 3D segmentation of papyrus sheet surfaces from micro-CT scans. The goal is to produce a binary surface mask for each volume.

## Multi Pipeline Architecture

This project has **four completely independent pipelines**. Read `PIPELINES.md` before making any changes.

| Pipeline | Directory | Status | Description |
|----------|-----------|--------|-------------|
| V1       | `src/`    | Frozen (completed 200 epochs) | CE+Dice, clDice, Boundary loss, standard UNet3D |
| V2       | `src_v2/` | Frozen (completed 200 epochs) | Focal+Dice, Skeleton Recall, Boundary loss, UNet3D with Deep Supervision |
| V3       | `src_v3/` | Paused (epoch 53/200, NaN at ~55) | Multi-scale learned fusion: 32³ + 64³ + 128³ UNets with learned fusion |
| nnU-Net  | `src_nnunet/` | **Active** (training on HF Jobs) | nnU-Net v2 — self-configuring 3D segmentation (1st place architecture) |

### Training Results (as of Feb 27, 2026)

| Pipeline | Best Val Loss | Best Surface Dice | Epochs Completed |
|----------|--------------|-------------------|------------------|
| V1       | 1.3639       | 0.1162            | 200/200          |
| V2       | **0.6728**   | 0.2538            | 200/200          |
| V3       | 0.7016       | 0.2258            | ~53/200 (NaN'd)  |

### Kaggle Submission Results (Feb 2026)

| Version | Description | Public LB | Private LB |
|---------|-------------|-----------|------------|
| V5 | V2 + original postproc, TTA=True | 0.390 | 0.409 |
| V8 | V2 + 1st-place postproc (late) | — | — |

Scored submission: **0.390 public / 0.409 private**, ~1240th on private LB. Private score higher than public — good generalization. Late submissions (V6-V9, with 1st-place post-processing and cuda/amp) completed but are not scored.

### 1st Place Post-Processing (implemented in V2)

V2's `src_v2/postprocess.py` implements the post-processing pipeline from the 1st place Vesuvius Challenge solution: binary closing (spherical footprint), height-map patching with interpolation, LUT-based 1-voxel hole plugging, and global `binary_fill_holes`. See `PIPELINES.md` for full details and the [writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su) for the source technique.

### Rules

1. **Never import across pipelines.** `src/`, `src_v2/`, `src_v3/`, and `src_nnunet/` share zero code at the import level. File duplication is intentional.
2. **Every source file has a pipeline tag** on line 1: `# Pipeline: V1 (src/) — See PIPELINES.md`, etc. Check this before editing.
3. **Checkpoints are not interchangeable.** V1 saves `best_model.pth`; V2 saves `best_model_v2.pth`; V3 saves `best_model_v3.pth`; nnU-Net uses its own checkpoint format. Architectures differ.
4. **V1 and V2 are frozen.** Do not modify files in `src/` or `src_v2/` unless explicitly asked to.
5. **nnU-Net pipeline is active.** `src_nnunet/` is the current focus. Uses nnU-Net v2's native training/inference framework with our 1st place post-processing.

## Saved Checkpoints

All checkpoints are stored locally in `outputs_aws/`:

```
outputs_aws/
├── v1/
│   ├── best_model_v1.pth       (309 MB, val_loss=1.3639)
│   └── history_v1.json
├── v2/
│   ├── best_model_v2.pth       (309 MB, val_loss=0.6728) ← best overall
│   ├── checkpoint_v2_epoch25.pth
│   ├── checkpoint_v2_epoch200.pth
│   └── history_v2.json
└── v3/
    ├── best_model_v3.pth       (174 MB, val_loss=0.7016)
    ├── checkpoint_v3_epoch25.pth
    ├── checkpoint_v3_epoch50.pth  ← use this to resume (pre-NaN)
    └── history_v3.json
```

Checkpoints are also backed up in S3: `s3://vesuvius-challenge-training-290318/`

## Key Files

- **`PIPELINES.md`** — Primary reference: full documentation of all pipelines, architecture diagrams, training commands, and isolation rules. Read this before making any pipeline changes.
- **`src_nnunet/README.md`** — nnU-Net pipeline setup, training (local and HF Jobs), and inference guide.
- **`adaptive_architecture_research_path.md`** — Future research roadmap for learnable architecture parameters. Not currently in progress.
- **`../vesuvius_sinogram/`** — Sibling project: future research on sinogram-domain surface detection (raw CT projections). Pre-development.
- `requirements.txt` — Shared dependencies (V2 and V3 require `scikit-image`)
- `src_nnunet/requirements.txt` — nnU-Net pipeline dependencies (includes `nnunetv2`)
- `smoke_test.py` — V1 integration test
- `smoke_test_v2.py` — V2 integration test
- `smoke_test_v3.py` — V3 integration test

## Training Commands

```bash
# Pipeline V1
python -m src.train --data-dir /path/to/data --output-dir outputs --epochs 200 --amp

# Pipeline V2
python -m src_v2.train --data-dir /path/to/data --output-dir outputs_v2 --epochs 200 --amp

# Pipeline V3 (resume from pre-NaN checkpoint)
python -m src_v3.train --data-dir /path/to/data --output-dir outputs_v3 --epochs 200 --amp \
    --resume outputs_aws/v3/checkpoint_v3_epoch50.pth

# nnU-Net pipeline — local (see src_nnunet/README.md for full setup)
python -m src_nnunet.convert_dataset --data-dir /path/to/data --output-dir /path/to/nnUNet_raw/Dataset011_Vesuvius
nnUNetv2_plan_and_preprocess -d 011 --verify_dataset_integrity
nnUNetv2_train 011 3d_fullres all -tr nnUNetTrainer_200epochs --npz

# nnU-Net pipeline — Hugging Face Jobs (recommended)
python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx
python -m src_nnunet.check_job 60  # monitor progress
```

## AWS

- All training instances have been **terminated** (Feb 27, 2026).
- V2 ran on g4dn.xlarge (T4), V3 ran on g5.xlarge (A10G).
- See `AWS_TRAINING.md` for setup guide and `docs/archive/AWS_RUN1_REPORT.md` for V1 run history.

## V3 NaN Issue (Fixed)

V3 hit NaN train loss around epoch 55. The model diverged due to FP16/AMP instability — the `(1-p_t)^gamma` term in Focal Loss and `log_softmax` can overflow FP16's range (~65504), producing NaN that cascades through `.backward()`.

**Two guards are now implemented in both V2 and V3 (Feb 2026):**

1. **FP32 loss computation** — All loss `forward()` methods use `@torch.amp.custom_fwd(cast_inputs=torch.float32)`, so logits are upcast to FP32 before softmax/log_softmax. The convolutions (where the compute time is) stay in FP16.
2. **`isfinite` guard in training loop** — If the loss is NaN/Inf, `.backward()` is skipped entirely, the batch is logged as skipped, and the running average is not poisoned. Matches the guard in V1's `src/train.py`.

To resume V3, use `checkpoint_v3_epoch50.pth`. These guards should prevent recurrence.

## Data

- Location on Kaggle: `vesuvius-challenge-surface-detection/`
- Volumes: 704 training samples (~60MB each, 3D TIFF)
- Labels: 3-class (0=background, 1=surface, 2=interior)
- Validation split: scroll ID `26002`
