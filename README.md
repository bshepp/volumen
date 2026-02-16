# volumen

3D surface segmentation of ancient Herculaneum papyrus scrolls from micro-CT scans. Built for the [Vesuvius Challenge](https://scrollprize.org/) [$200K Surface Detection competition](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) on Kaggle.

The Herculaneum scrolls were carbonized by the eruption of Mount Vesuvius in 79 AD. This project detects the recto papyrus surface in volumetric CT data — a critical step in the virtual unwrapping pipeline that is enabling these ancient texts to be read for the first time in two millennia.

## Approach

The core architecture is a **3D U-Net** that takes raw CT volumes and produces per-voxel surface predictions across three classes (background, surface, interior). A **GPU-accelerated feature extractor** computes 6 periodicity-aware channels (raw CT, LoG at two scales, Hessian trace, gradient magnitude, fiber anisotropy) using separable 3D convolutions with fixed kernels, replacing a scipy/CPU bottleneck with a 10-50x speedup.

Training uses **volume-grouped batching** — each DataLoader item loads a single TIFF volume and extracts multiple patches, reducing disk I/O from O(batch_size) to O(1) per batch. Combined with mixed-precision training and surface-biased sampling, this saturates the GPU on even modest hardware.

The project maintains two fully independent pipelines:

### Pipeline V1 (`src/`)

| Component | Description |
|-----------|-------------|
| Model | `UNet3D` — standard 3D U-Net (27M params) |
| Loss | 0.4 × (CE + Dice) + 0.3 × clDice + 0.3 × Boundary |
| Dataset | Returns `(image, label)` |

### Pipeline V2 (`src_v2/`)

| Component | Description |
|-----------|-------------|
| Model | `UNet3DDeepSup` — U-Net with auxiliary decoder heads at 2x, 4x, 8x |
| Loss | 0.3 × (Focal + Dice) + 0.3 × Skeleton Recall + 0.2 × Boundary, at 4 scales |
| Dataset | Returns `(image, label, skeleton)` with precomputed skeletonization |

V2 improvements:
- **Focal Loss** replaces cross-entropy, down-weighting easy background voxels to focus on hard boundary cases
- **Deep Supervision** provides gradient signal to all decoder stages via auxiliary 1x1x1 heads
- **Skeleton Recall Loss** replaces clDice — uses precomputed skeletons from `skimage` (~90% cheaper than differentiable skeletonization)

The pipelines share zero imports. See [PIPELINES.md](PIPELINES.md) for full architecture documentation.

## Competition Metric

**Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score**

The metric rewards topological correctness (no mergers between sheets, no spurious holes) alongside geometric accuracy. Post-processing includes connected component filtering, bridge detection/removal, hole filling, and physics-based sheet spacing validation.

## Setup

```bash
git clone https://github.com/bshepp/volumen.git
cd volumen
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA-capable GPU (tested on Tesla T4 16GB).

**Data:** Download from the [Kaggle competition page](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data) and place in `vesuvius-challenge-surface-detection/`.

## Training

```bash
# Pipeline V1
python -m src.train --data-dir vesuvius-challenge-surface-detection --output-dir outputs --epochs 200 --amp

# Pipeline V2
python -m src_v2.train --data-dir vesuvius-challenge-surface-detection --output-dir outputs_v2 --epochs 200 --amp
```

Key arguments:
- `--patch-size 128` — cubic patch side length
- `--patches-per-volume 4` — patches extracted per volume load (effective batch size)
- `--base-filters 32` — first encoder stage width
- `--depth 4` — encoder/decoder stages
- `--focal-gamma 2.0` — focal loss focusing parameter (V2 only)
- `--skeleton-dilation 2` — skeleton dilation radius in voxels (V2 only)
- `--resume checkpoint.pth` — resume from checkpoint

## Inference

```python
from src_v2.inference import run_inference

run_inference(
    model_path="outputs_v2/best_model_v2.pth",
    test_volume_path="vesuvius-challenge-surface-detection/test_images/1407735.tif",
    output_path="submission/1407735.tif",
    use_tta=True,           # 8-flip test-time augmentation
    use_postprocess=True,   # topology-aware post-processing
)
```

## Smoke Tests

```bash
python smoke_test.py      # Pipeline V1
python smoke_test_v2.py   # Pipeline V2
```

## Project Structure

```
volumen/
├── src/                  # Pipeline V1 (frozen)
│   ├── model.py          #   UNet3D
│   ├── losses.py         #   CE+Dice, clDice, Boundary
│   ├── dataset.py        #   Volume-grouped batching
│   ├── features.py       #   GPU feature extractor (6-channel)
│   ├── train.py          #   Training loop
│   ├── inference.py      #   Sliding window + TTA
│   ├── postprocess.py    #   Topology-aware post-processing
│   └── evaluate.py       #   Competition metric
├── src_v2/               # Pipeline V2 (active)
│   ├── model.py          #   UNet3DDeepSup (deep supervision)
│   ├── losses.py         #   Focal+Dice, Skeleton Recall, Boundary
│   ├── dataset.py        #   + skeleton precomputation
│   ├── train.py          #   Adapted for deep supervision
│   └── ...               #   Own copies of shared modules
├── aws/                  # EC2 launch scripts & IAM policies
├── configs/              # Training configuration
├── notebooks/            # Kaggle submission notebook
├── PIPELINES.md          # Dual pipeline documentation
├── CLAUDE.md             # Agent instructions
├── COMPETITION_NOTES.md  # Competition rules & dataset analysis
└── requirements.txt
```

## Citation

This project uses the EduceLab-Scrolls dataset:

> S. Parker, C. Parsons, and W. B. Seales. "EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT." arXiv:2304.02084, 2023.

## License

[CC BY-NC 4.0](LICENSE) — as required by the Vesuvius Challenge competition rules.
