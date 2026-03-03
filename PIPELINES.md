# Vesuvius Challenge — Multi Pipeline Documentation

## Overview

This project contains **four completely independent** training/inference pipelines for the Vesuvius Challenge surface detection task. They share **no code** at the Python import level. Any pipeline can be deleted without affecting the others.

---

## ⚠️  CRITICAL: DO NOT MIX PIPELINES

- **Never** import from `src/` inside `src_v2/`, `src_v3/`, or `src_nnunet/`, or vice versa.
- **Never** load a V1 checkpoint into a V2/V3/nnU-Net model or vice versa. The architectures differ.
- **Never** share a `DataLoader` between pipelines. V2 and V3 return 3-tuples `(image, label, skeleton)`; V1 returns 2-tuples `(image, label)`; nnU-Net manages its own data loading.
- Outputs are saved with different filenames (`best_model.pth`, `best_model_v2.pth`, `best_model_v3.pth`, nnU-Net's `checkpoint_final.pth`) to avoid accidental overwrites.

---

## Pipeline V1 — `src/`

| Component  | File             | Description                                           |
|------------|------------------|-------------------------------------------------------|
| Model      | `src/model.py`   | `UNet3D` — standard 3D U-Net, 6→3 channels           |
| Losses     | `src/losses.py`  | `CompositeLoss`: 0.4×(CE+Dice) + 0.3×clDice + 0.3×Boundary |
| Dataset    | `src/dataset.py` | Returns `(image, label)` 2-tuples, volume-grouped batching |
| Features   | `src/features.py`| `GPUFeatureExtractor`: 6-channel features on GPU      |
| Training   | `src/train.py`   | Standard training loop with AMP + cosine annealing    |
| Inference  | `src/inference.py`| Sliding window + TTA                                 |
| Postproc   | `src/postprocess.py`| CC filtering, bridge removal, hole filling, spacing |
| Evaluation | `src/evaluate.py`| Competition metric (SurfaceDice + VOI + TopoScore)    |

### How to train V1

```bash
python -m src.train --data-dir /path/to/data --output-dir outputs --epochs 200 --amp
```

### Architecture

```
Input (B, 1, Z, Y, X)
  → GPUFeatureExtractor → (B, 6, Z, Y, X)
  → UNet3D → (B, 3, Z, Y, X)   [logits]
  → CompositeLoss(CE+Dice, clDice, Boundary)
```

### Status

**Frozen.** Completed 200 epochs on AWS (g4dn.xlarge). Best val_loss: 1.3639, surface Dice: 0.1162. Do not modify these files. Checkpoint: `outputs_aws/v1/best_model_v1.pth`.

---

## Pipeline V2 — `src_v2/`

| Component  | File                | Description                                           |
|------------|---------------------|-------------------------------------------------------|
| Model      | `src_v2/model.py`   | `UNet3DDeepSup` — U-Net with 3 auxiliary decoder heads|
| Losses     | `src_v2/losses.py`  | `DeepSupCompositeLoss`: Focal+Dice + SkeletonRecall + Boundary at 4 scales |
| Dataset    | `src_v2/dataset.py` | Returns `(image, label, skeleton)` 3-tuples           |
| Features   | `src_v2/features.py`| `GPUFeatureExtractor` (own copy, identical to V1)     |
| Training   | `src_v2/train.py`   | Training loop adapted for deep supervision + skeleton |
| Inference  | `src_v2/inference.py`| Sliding window + TTA (model returns logits in eval)  |
| Postproc   | `src_v2/postprocess.py`| 1st place post-processing pipeline (see below)      |
| Evaluation | `src_v2/evaluate.py`| Same as V1 (own copy)                                 |

### How to train V2

```bash
python -m src_v2.train --data-dir /path/to/data --output-dir outputs_v2 --epochs 200 --amp
```

### Architecture

```
Input (B, 1, Z, Y, X)
  → GPUFeatureExtractor → (B, 6, Z, Y, X)
  → UNet3DDeepSup (training) →
      {"logits": (B,3,Z,Y,X), "aux_1": (B,3,Z/2,Y/2,X/2),
       "aux_2": (B,3,Z/4,Y/4,X/4), "aux_3": (B,3,Z/8,Y/8,X/8)}
  → DeepSupCompositeLoss(FocalDice + SkeletonRecall + Boundary, multi-scale)

UNet3DDeepSup (eval) →  (B, 3, Z, Y, X)   [main logits only]
```

### Three improvements over V1

1. **Focal Loss** replaces Cross-Entropy
   - `FL(p_t) = -α_t (1-p_t)^γ log(p_t)` with γ=2.0
   - Down-weights easy background voxels, focuses on hard boundary cases
   - Better for the severe class imbalance (surface ~6% of volume)

2. **Deep Supervision**
   - Auxiliary 1×1×1 classification heads at each decoder stage (2×, 4×, 8× downsampled)
   - Weighted sum: `loss = main + 0.5×aux1 + 0.25×aux2 + 0.125×aux3`
   - Provides gradient signal to all decoder stages, preventing vanishing gradients
   - During inference (`model.eval()`), only the main head output is returned

3. **Skeleton Recall Loss** replaces clDice
   - Uses precomputed skeletons from `skimage.morphology.skeletonize` + 2px dilation
   - Skeleton computed on CPU in DataLoader workers (~5ms per patch)
   - Loss = `1 - recall` where `recall = Σ(pred × skel_gt) / Σ(skel_gt)`
   - ~90% cheaper than clDice's iterative soft skeletonization on GPU
   - Better gradient signal for thin structure preservation

### Composite loss formula

```
L = 0.3 × (Focal + Dice) + 0.3 × SkeletonRecall + 0.2 × Boundary
```

At each auxiliary scale, the same formula is applied to downsampled targets.

### Post-processing (1st place solution)

V2's `postprocess.py` implements the post-processing pipeline from the **1st place solution** for the Vesuvius Challenge Surface Detection competition ([writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)).

The pipeline runs five steps:

1. **Connected component filtering** — remove components smaller than `min_component_size` (default 500 voxels)
2. **Per-sheet binary closing** — `scipy.ndimage.binary_closing` with a spherical footprint of radius 3, applied to each connected component individually (prevents merging nearby sheets)
3. **Height-map patching** — for each sheet:
   - Project to 2D along the axis giving the largest projected area
   - Build height map (mean depth) and thickness map from the projection
   - Identify internal gaps (inside the filled 2D projection, missing from the sheet)
   - Fill gaps by linear interpolation along rows and columns separately
   - Average the two fills weighted by inverse distance to gap edge
   - Reconstruct 3D voxels from the patched height/thickness maps
   - Discard the patch if it introduced new internal holes (safety check)
4. **1-voxel hole plugging** — 256-entry lookup table over 2×2×2 neighborhoods that detects face-diagonal gaps (two foreground voxels at opposite corners of a face with both intermediate gap voxels empty) and adds one bridging voxel per gap for 6-connected watertightness
5. **Global `binary_fill_holes`** — `scipy.ndimage.binary_fill_holes` for remaining enclosed cavities of any size

The legacy functions (`remove_bridges`, `fill_small_holes`, `spacing_validation`) are retained as importable for backward compatibility but are not part of the default pipeline.

### Kaggle submission results

| Version | Description | Public LB | Private LB | Notes |
|---------|-------------|-----------|------------|-------|
| V5 | Original postproc, TTA=True | 0.390 | 0.409 | Scored (before deadline) |
| V8 | 1st-place postproc (late) | — | — | After deadline, not scored |

Scored submission: **0.390 public / 0.409 private**, ~1240th on private LB. The model scored higher on the unseen 80% private data than on the 20% public test, indicating good generalization.

### Status

**Frozen.** Completed 200 epochs on AWS (g4dn.xlarge, T4 16GB). Best val_loss: **0.6728**, surface Dice: **0.2538** — the best-performing pipeline overall. Post-processing updated to 1st place pipeline (Feb 2026). Checkpoints: `outputs_aws/v2/best_model_v2.pth`, `checkpoint_v2_epoch25.pth`, `checkpoint_v2_epoch200.pth`.

---

## Pipeline V3 — `src_v3/`

| Component  | File                | Description                                           |
|------------|---------------------|-------------------------------------------------------|
| Model      | `src_v3/model.py`   | `MultiScaleFusionUNet` — 3 UNets (32³, 64³, 128³) + learned fusion |
| Losses     | `src_v3/losses.py`  | `CompositeLossV3`: Focal+Dice + SkeletonRecall + Boundary on fused output |
| Dataset    | `src_v3/dataset.py` | Returns `(image, label, skeleton)` 3-tuples, patch_size=128 |
| Features   | `src_v3/features.py`| `GPUFeatureExtractor` (own copy)                      |
| Training   | `src_v3/train.py`   | Training loop for multi-scale fusion model            |
| Inference  | `src_v3/inference.py`| Sliding window 128³, model handles multi-scale internally |
| Postproc   | `src_v3/postprocess.py`| Same as V1/V2 (own copy)                           |
| Evaluation | `src_v3/evaluate.py`| Same as V1/V2 (own copy)                              |

### How to train V3

```bash
python -m src_v3.train --data-dir /path/to/data --output-dir outputs_v3 --epochs 200 --amp
```

### Architecture

The model learns to fuse predictions from multiple window sizes (32³, 64³, 128³). Center crops are extracted from the 128³ input; each scale gets its own UNet; outputs are upsampled and fused via a 1×1×1 conv.

```
Input (B, 6, 128, 128, 128)
  → Branch 1: full 128³ → UNet → logits_128
  → Branch 2: center 64³ → UNet → logits_64 → upsample to 128³
  → Branch 3: center 32³ → UNet → logits_32 → upsample to 128³
  → Concat → Conv3d(9, 3, 1) → fused logits (B, 3, 128, 128, 128)
```

The fusion layer learns when to trust fine vs coarse scale. Checkpoints include `"pipeline": "v3"`.

### Status

**Paused.** Trained ~53 epochs on AWS (g5.xlarge, A10G 24GB) before hitting NaN loss at epoch ~55 due to FP16/AMP instability. Best val_loss: 0.7016, surface Dice: 0.2258 (from before NaN divergence). **NaN guards now implemented:** FP32 loss computation (`@torch.amp.custom_fwd(cast_inputs=torch.float32)`) and `isfinite` skip in the training loop. To resume, use `checkpoint_v3_epoch50.pth`. Checkpoints: `outputs_aws/v3/best_model_v3.pth`, `checkpoint_v3_epoch25.pth`, `checkpoint_v3_epoch50.pth`.

---

## Pipeline nnU-Net — `src_nnunet/`

| Component  | File                       | Description                                           |
|------------|----------------------------|-------------------------------------------------------|
| Data Conv  | `src_nnunet/convert_dataset.py` | Convert TIF data to nnU-Net v2 format (symlinks + spacing JSONs) |
| Inference  | `src_nnunet/predict.py`    | nnU-Net prediction + 1st place post-processing         |
| HF Train   | `src_nnunet/train_hf.py`   | End-to-end training script for HF Jobs (download → convert → preprocess → train → upload) |
| HF Launch  | `src_nnunet/launch_hf_job.py` | Launch a training job on Hugging Face Jobs          |
| HF Monitor | `src_nnunet/monitor_hf_job.py` | Monitor job status and stream logs                 |
| HF Check   | `src_nnunet/check_job.py`  | Quick job status check and log tail                    |
| Training   | `notebooks/nnunet_training.ipynb` | Kaggle training notebook (installs nnunetv2, converts data, trains) |
| Submission | `notebooks/submission_nnunet.ipynb` | Kaggle submission notebook                          |
| Deps       | `src_nnunet/requirements.txt` | Pipeline-specific dependencies (includes `nnunetv2`)  |

### Why nnU-Net

The 1st place team used nnU-Net v2. Their single model at 250 epochs scored **0.577/0.614** (public/private), compared to our custom UNet3DDeepSup at **0.390/0.409**. The architecture difference alone accounts for ~0.2 in Dice score.

### How to train nnU-Net

```bash
# 1. Convert data
python -m src_nnunet.convert_dataset \
    --data-dir /path/to/vesuvius-challenge-surface-detection \
    --output-dir /path/to/nnUNet_raw/Dataset011_Vesuvius

# 2. Plan and preprocess (auto-configures patch size, batch size, normalization)
nnUNetv2_plan_and_preprocess -d 011 --verify_dataset_integrity

# 3. Train (200 epochs, all data, no cross-validation)
nnUNetv2_train 011 3d_fullres all -tr nnUNetTrainer_200epochs --npz
```

Or use `notebooks/nnunet_training.ipynb` on Kaggle (free GPU, data already available).

**Training on Hugging Face Jobs** (recommended for full training runs):

```bash
python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx
python -m src_nnunet.check_job 60  # monitor progress
```

See `src_nnunet/README.md` for details on flavors, timeouts, and monitoring.

### Architecture

nnU-Net v2 automatically configures the entire segmentation pipeline based on dataset properties:

```
Input: 3D TIF volume (raw CT)
  → nnU-Net auto-preprocessing (normalization, resampling)
  → nnU-Net 3D U-Net (auto-configured patch size, batch size, architecture)
  → Softmax → 3-class segmentation (background, surface, interior)
  → 1st place post-processing (binary closing, height-map patching, hole plugging, fill holes)
  → Binary surface mask
```

Key differences from V1/V2/V3:
- **No custom feature extractor** — nnU-Net handles preprocessing natively
- **Auto-configured** — patch size, batch size, network depth, and augmentation are determined from dataset statistics
- **Built-in TTA** — mirroring-based test-time augmentation
- **nnU-Net's own checkpoint format** — not compatible with V1/V2/V3 models

### Post-processing

Uses the same 1st place post-processing pipeline as V2 (inlined in `src_nnunet/predict.py` and `notebooks/submission_nnunet.ipynb` for pipeline isolation):
1. Connected component filtering (min 500 voxels)
2. Per-sheet binary closing (spherical footprint, radius 3)
3. Height-map patching (projection + interpolation)
4. LUT-based 1-voxel hole plugging
5. Global `binary_fill_holes`

### Training strategy (matching 1st place)

The 1st place team's approach:
1. **Baseline:** Patch size 128, batch size 2, 4000 epochs (but 250 epochs was nearly identical in score)
2. **Fine-tune:** Larger patch sizes (192, 256) for 250 epochs each
3. **Ensemble:** Weighted combination of 4 models

Our initial plan: Train for **200 epochs** at default auto-configured settings. The 1st place team's 4000-epoch model scored 0.613 private vs 0.614 private at 250 epochs — diminishing returns beyond ~200 epochs.

### Status

**Active — training in progress.** Pipeline code is complete. Training running on Hugging Face Jobs (A10G GPU, 200 epochs). Model will be uploaded to `huggingface.co/bshepp/vesuvius-nnunet` on completion.

### References

- [nnU-Net v2 GitHub](https://github.com/MIC-DKFZ/nnUNet)
- [1st Place Solution Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)
- [Baseline Notebook (jirkaborovec)](https://www.kaggle.com/code/jirkaborovec/surface-train-inference-3d-segm-gpu-augment)

---

## File duplication is intentional

`src_v2/`, `src_v3/`, and `src_nnunet/` each have their own copies of shared code (e.g., post-processing). This is **by design**: it ensures complete isolation. Modify one pipeline without risk of breaking another.

**Note:** V2's `postprocess.py` has diverged from V1's — it implements the 1st place post-processing pipeline. V1 and V3 retain the original post-processing (CC filtering, bridge removal, hole filling, spacing validation). The nnU-Net pipeline's `predict.py` inlines the same 1st place post-processing code.

---

## Checkpoint format

All pipelines save checkpoints with the same dict structure:

```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict": dict,
    "best_val_loss": float,
    "val_metrics": dict,       # only in best_model checkpoints
    "pipeline": "v2" | "v3",   # V2/V3 only — V1 checkpoints lack this key
}
```

V2 checkpoints include `"pipeline": "v2"`; V3 includes `"pipeline": "v3"`. V1 checkpoints do not have this key.

---

## Dependencies

All pipelines require the packages listed in `requirements.txt`. V2 and V3 additionally require `scikit-image` (for `skimage.morphology.skeletonize`). The nnU-Net pipeline has its own `src_nnunet/requirements.txt` which includes `nnunetv2`.
