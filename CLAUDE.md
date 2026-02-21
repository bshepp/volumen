# CLAUDE.md — Agent Instructions for Vesuvius Challenge

## Project Overview

Vesuvius Challenge surface detection: 3D segmentation of papyrus sheet surfaces from micro-CT scans. The goal is to produce a binary surface mask for each volume.

## Multi Pipeline Architecture

This project has **three completely independent pipelines**. Read `PIPELINES.md` before making any changes.

| Pipeline | Directory | Status | Description |
|----------|-----------|--------|-------------|
| V1       | `src/`    | Frozen (training on AWS) | CE+Dice, clDice, Boundary loss, standard UNet3D |
| V2       | `src_v2/` | Active development | Focal+Dice, Skeleton Recall, Boundary loss, UNet3D with Deep Supervision |
| V3       | `src_v3/` | Active development | Multi-scale learned fusion: 32³ + 64³ + 128³ UNets with learned fusion |

### Rules

1. **Never import across pipelines.** `src/`, `src_v2/`, and `src_v3/` share zero code at the import level. File duplication is intentional.
2. **Every source file has a pipeline tag** on line 1: `# Pipeline: V1 (src/) — See PIPELINES.md`, etc. Check this before editing.
3. **Checkpoints are not interchangeable.** V1 saves `best_model.pth`; V2 saves `best_model_v2.pth`; V3 saves `best_model_v3.pth`. Architectures differ.
4. **V1 is frozen.** Do not modify files in `src/` unless explicitly asked to.

## Key Files

- **`PIPELINES.md`** — Primary reference: full documentation of all pipelines, architecture diagrams, training commands, and isolation rules. Read this before making any pipeline changes.
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

# Pipeline V3
python -m src_v3.train --data-dir /path/to/data --output-dir outputs_v3 --epochs 200 --amp
```

## AWS

- Instance type used: `g4dn.xlarge` (Tesla T4, 16GB VRAM, on-demand)
- User data script: `aws/user-data.sh`
- See `AWS_RUN1_REPORT.md` for Run 1 details

## Data

- Location on Kaggle: `vesuvius-challenge-surface-detection/`
- Volumes: 704 training samples (~60MB each, 3D TIFF)
- Labels: 3-class (0=background, 1=surface, 2=interior)
- Validation split: scroll ID `26002`
