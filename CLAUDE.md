# CLAUDE.md — Agent Instructions for Vesuvius Challenge

## Project Overview

Vesuvius Challenge surface detection: 3D segmentation of papyrus sheet surfaces from micro-CT scans. The goal is to produce a binary surface mask for each volume.

## Multi Pipeline Architecture

This project has **three completely independent pipelines**. Read `PIPELINES.md` before making any changes.

| Pipeline | Directory | Status | Description |
|----------|-----------|--------|-------------|
| V1       | `src/`    | Frozen (completed 200 epochs) | CE+Dice, clDice, Boundary loss, standard UNet3D |
| V2       | `src_v2/` | Frozen (completed 200 epochs) | Focal+Dice, Skeleton Recall, Boundary loss, UNet3D with Deep Supervision |
| V3       | `src_v3/` | Paused (epoch 53/200, NaN at ~55) | Multi-scale learned fusion: 32³ + 64³ + 128³ UNets with learned fusion |

### Training Results (as of Feb 27, 2026)

| Pipeline | Best Val Loss | Best Surface Dice | Epochs Completed |
|----------|--------------|-------------------|------------------|
| V1       | 1.3639       | 0.1162            | 200/200          |
| V2       | **0.6728**   | 0.2538            | 200/200          |
| V3       | 0.7016       | 0.2258            | ~53/200 (NaN'd)  |

### Rules

1. **Never import across pipelines.** `src/`, `src_v2/`, and `src_v3/` share zero code at the import level. File duplication is intentional.
2. **Every source file has a pipeline tag** on line 1: `# Pipeline: V1 (src/) — See PIPELINES.md`, etc. Check this before editing.
3. **Checkpoints are not interchangeable.** V1 saves `best_model.pth`; V2 saves `best_model_v2.pth`; V3 saves `best_model_v3.pth`. Architectures differ.
4. **V1 and V2 are frozen.** Do not modify files in `src/` or `src_v2/` unless explicitly asked to.

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
- **`adaptive_architecture_research_path.md`** — Future research roadmap for learnable architecture parameters. Not currently in progress.
- `requirements.txt` — Shared dependencies (V2 and V3 require `scikit-image`)
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
```

## AWS

- All training instances have been **terminated** (Feb 27, 2026).
- V2 ran on g4dn.xlarge (T4), V3 ran on g5.xlarge (A10G).
- See `AWS_TRAINING.md` for setup guide and `AWS_RUN1_REPORT.md` for V1 run history.

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
