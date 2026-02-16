# Vesuvius Challenge - Surface Detection

## Competition Summary
- **Sponsor:** Curious Cases Inc.
- **Platform:** Kaggle (Code Competition)
- **Prize Pool:** $200,000 (1st: $60k, 2nd: $40k, 3rd: $30k, ... 10th: $5k)
- **Deadline:** February 27, 2026 (Final Submission)
- **License:** CC BY-NC 4.0

## Objective
Detect the **recto papyrus surface** in 3D CT scans of carbonized Herculaneum scrolls.
The recto surface is the side facing the scroll's center (umbilicus), composed of horizontal fibers.
Output feeds directly into the Vesuvius Challenge virtual unwrapping pipeline.

## Key Requirements
- **Avoid topological mistakes:** No artificial mergers between sheets; no holes splitting entities.
- **Surface continuity:** Keep surfaces as single, clean sheets.
- Detecting approximate sheet position (recto or verso) is acceptable.

## Evaluation Metric
**Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score**

All terms are in [0, 1], higher is better.

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| SurfaceDice@τ (τ=2.0) | 0.35 | Surface boundary proximity within tolerance |
| VOI_score (α=0.3) | 0.35 | Instance split/merge via Variation of Information |
| TopoScore | 0.30 | Topological features (Betti numbers: components, tunnels, cavities) |

### Practical Implications
- Slight boundary misplacement (≤ τ) is tolerated
- Bridges between layers are **heavily penalized** (VOI merge + TopoScore)
- Splits within a single wrap are penalized (VOI split + TopoScore k=0)
- Spurious holes/handles are penalized (TopoScore k=1/k=2)

## Submission Format
- **Code Competition** (Kaggle Notebooks)
- Submit `submission.zip` containing one `.tif` per test image
- Each mask named `[image_id].tif`, matching source dimensions exactly
- Same dtype as train masks (uint8)
- CPU ≤ 9h or GPU ≤ 9h runtime, no internet
- Max 3 submissions/day, select up to 2 final submissions

## Dataset Structure

```
vesuvius-challenge-surface-detection/
├── train.csv              # 806 entries (id, scroll_id)
├── test.csv               # 1 entry (id=1407735, scroll_id=26002)
├── train_images/          # 786 .tif files (3D volumes, uint8)
├── train_labels/          # 786 .tif files (3D masks, uint8, values {0,1,2})
├── deprecated_train_images/ # 20 .tif files (older labels, values {0,1})
├── deprecated_train_labels/ # 20 .tif files
└── test_images/           # 1 .tif file (1407735.tif)
```

Note: 806 entries in CSV but only 786 files in train dirs (20 are in deprecated).

### Volume Shapes
| Shape | Count |
|-------|-------|
| 320×320×320 | 738 |
| 256×256×256 | 47 |
| 384×384×384 | 1 |

### Scroll Distribution
| Scroll ID | Count |
|-----------|-------|
| 34117 | 382 |
| 35360 | 176 |
| 26010 | 130 |
| 26002 | 88 |
| 44430 | 17 |
| 53997 | 13 |

### Label Semantics (Current Train Labels)
Labels are uint8 with values {0, 1, 2}:
- **Label 0** (~25-39% of voxels): Likely papyrus material / interior
- **Label 1** (~4-9% of voxels): Likely the **recto surface** (target)
- **Label 2** (~52-71% of voxels): Likely background/air

Observations:
- At volume borders (Z=0, Z=319), everything is label 2 → consistent with background
- Label 1 is thin (surface-like), distributed in the interior of the volume
- Deprecated labels had only {0, 1} (binary), suggesting older simpler annotation

### Image Properties
- dtype: uint8 (0-255)
- LZW compressed TIF stacks
- Mean intensity ~71-143 depending on region
- Total train images: ~25.9 GB
- Total train labels: ~0.7 GB (much smaller due to compression of sparse labels)

## Competition Rules Summary
- Max team size: 5
- External data: allowed if publicly available and free
- Pre-trained models: allowed
- Must cite "EduceLab-Scrolls" in publications
- Winners must open-source under CC-BY-NC 4.0
- Host may release additional (less curated) labeled data during competition

## Baseline
- Modified nnUNet: 0.543 LB raw, 0.562 with post-processing

## Key References
- EduceLab-Scrolls paper: https://doi.org/10.48550/arXiv.2304.02084
- Vesuvius Challenge: https://scrollprize.org
- Metric notebook: https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo
- Unwrapping tutorial: https://www.youtube.com/watch?v=yHbpVcGD06U
