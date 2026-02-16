# Vesuvius Challenge — Dual Pipeline Documentation

## Overview

This project contains **two completely independent** training/inference pipelines for the Vesuvius Challenge surface detection task. They share **no code** at the Python import level. Either pipeline can be deleted without affecting the other.

---

## ⚠️  CRITICAL: DO NOT MIX PIPELINES

- **Never** import from `src/` inside `src_v2/` or vice versa.
- **Never** load a V1 checkpoint into a V2 model or vice versa. The architectures differ (V2 has auxiliary heads).
- **Never** share a `DataLoader` between pipelines. V2's dataset returns 3-tuples `(image, label, skeleton)`; V1's returns 2-tuples `(image, label)`.
- Outputs are saved with different filenames (`best_model.pth` vs `best_model_v2.pth`, `history.json` vs `history_v2.json`) to avoid accidental overwrites.

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

**Frozen.** This pipeline was the first run trained on AWS (g4dn.xlarge). Do not modify these files.

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
| Postproc   | `src_v2/postprocess.py`| Same as V1 (own copy)                              |
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

---

## File duplication is intentional

`src_v2/features.py`, `src_v2/postprocess.py`, and `src_v2/evaluate.py` are exact copies of their `src/` counterparts. This is **by design**: it ensures complete isolation between the two pipelines. A future agent or developer can confidently modify V2 files without any risk of breaking V1.

---

## Checkpoint format

Both pipelines save checkpoints with the same dict structure:

```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict": dict,
    "best_val_loss": float,
    "val_metrics": dict,       # only in best_model checkpoints
    "pipeline": "v2",          # V2 only — V1 checkpoints lack this key
}
```

V2 checkpoints include a `"pipeline": "v2"` key. V1 checkpoints do not have this key. This can be used to detect which pipeline produced a checkpoint.

---

## Dependencies

Both pipelines require the packages listed in `requirements.txt`. V2 additionally requires `scikit-image` (for `skimage.morphology.skeletonize`).
