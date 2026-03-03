# volumen

3D surface segmentation of ancient Herculaneum papyrus scrolls from micro-CT scans. Built for the [Vesuvius Challenge](https://scrollprize.org/) [$200K Surface Detection competition](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) on Kaggle.

The Herculaneum scrolls were carbonized by the eruption of Mount Vesuvius in 79 AD. This project detects the recto papyrus surface in volumetric CT data — a critical step in the virtual unwrapping pipeline that is enabling these ancient texts to be read for the first time in two millennia.

## Approach

The core architecture is a **3D U-Net** that takes raw CT volumes and produces per-voxel surface predictions across three classes (background, surface, interior). A **GPU-accelerated feature extractor** computes 6 periodicity-aware channels (raw CT, LoG at two scales, Hessian trace, gradient magnitude, fiber anisotropy) using separable 3D convolutions with fixed kernels, replacing a scipy/CPU bottleneck with a 10-50x speedup.

Training uses **volume-grouped batching** — each DataLoader item loads a single TIFF volume and extracts multiple patches, reducing disk I/O from O(batch_size) to O(1) per batch. Combined with mixed-precision training and surface-biased sampling, this saturates the GPU on even modest hardware.

The project maintains **four fully independent pipelines**. See [PIPELINES.md](PIPELINES.md) for full architecture documentation.

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

V2 improvements: Focal Loss, Deep Supervision, Skeleton Recall Loss (see [PIPELINES.md](PIPELINES.md)).

### Pipeline V3 (`src_v3/`)

| Component | Description |
|-----------|-------------|
| Model | `MultiScaleFusionUNet` — 3 UNets (32³, 64³, 128³) + learned fusion |
| Loss | Focal+Dice + SkeletonRecall + Boundary on fused output |
| Dataset | Returns `(image, label, skeleton)`, patch_size=128 |

V3: Model learns to fuse predictions from multiple window sizes; fusion layer selects when to trust fine vs coarse scale. Pipelines share zero imports.

### Pipeline nnU-Net (`src_nnunet/`)

| Component | Description |
|-----------|-------------|
| Model | nnU-Net v2 — self-configuring 3D U-Net (1st place architecture) |
| Training | Automated via Hugging Face Jobs (A10G GPU) |
| Post-proc | 1st place pipeline (binary closing, height-map patching, hole plugging) |

The 1st place team's single nnU-Net scored **0.577/0.614** (public/private) at 250 epochs. Training is currently in progress on Hugging Face Jobs.

### Results

| Pipeline | Best Val Loss | Surface Dice | Epochs | GPU |
|----------|--------------|-------------|--------|-----|
| V1       | 1.3639       | 0.1162      | 200/200 | T4 |
| **V2**   | **0.6728**   | **0.2538**  | 200/200 | T4 |
| V3       | 0.7016       | 0.2258      | ~53/200 | A10G |
| nnU-Net  | —            | —           | training | A10G (HF Jobs) |

V2 is the best-performing custom pipeline. V3 training paused due to NaN divergence at epoch ~55 (FP16/AMP instability). nnU-Net training is in progress on Hugging Face Jobs. See [PIPELINES.md](PIPELINES.md) for details.

### Kaggle Submission Results

| Version | Date | Description | Public LB | Private LB |
|---------|------|-------------|-----------|------------|
| V3 | Feb 27 | First attempt | — | — |
| V5 | Feb 28 | V2 + original postproc, TTA=True | 0.390 | 0.409 |
| V6 | Feb 28 | V2 + TTA=True | — | — |
| V7 | Mar 1 | V2 + cuda, amp=True | — | — |
| V8 | Mar 1 | V2 + 1st-place post-processing | — | — |
| V9 | Mar 1 | V2 + latest | — | — |
| **V10** | **Mar 2** | **V2 + 1st-place postproc + tuned settings** | **0.405** | **0.426** |
| V11 | Mar 2 | Latest iteration | — | — |
| V12 | Mar 3 | Latest iteration | — | — |

Best submission: **0.405 public / 0.426 private** (V10, ~1240th on private LB), up from 0.390/0.409 (V5). 11 total submissions, 2 scored. The 1st-place post-processing added +0.017 on private LB. The model scored *higher* on the unseen 80% private data than on the 20% public test in both scored runs, indicating strong generalization.

## Competition Metric

**Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score**

The metric rewards topological correctness (no mergers between sheets, no spurious holes) alongside geometric accuracy.

## Post-Processing

V2 uses a post-processing pipeline adapted from the **1st place solution** for the Vesuvius Challenge Surface Detection competition ([writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)):

1. **Connected component filtering** — remove small fragments
2. **Per-sheet binary closing** — `scipy.ndimage.binary_closing` with a spherical footprint (radius 3), applied per connected component to avoid merging nearby sheets
3. **Height-map patching** — project each sheet to 2D along its best axis, build height/thickness maps, fill gaps via linear interpolation in both row and column directions with distance-weighted averaging, reconstruct 3D voxels, and discard patches that introduce new holes
4. **1-voxel hole plugging** — 256-entry lookup table over 2×2×2 neighborhoods that detects face-diagonal gaps and adds bridging voxels for 6-connected watertightness
5. **Global binary fill holes** — `scipy.ndimage.binary_fill_holes` for remaining enclosed cavities

See `src_v2/postprocess.py` for the implementation.

## Setup

```bash
git clone https://github.com/bshepp/volumen.git
cd volumen
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA-capable GPU (tested on Tesla T4 16GB, NVIDIA A10G 24GB).

**Data:** Download from the [Kaggle competition page](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data) and place in `vesuvius-challenge-surface-detection/`.

**Compute:** See [COMPUTE_OPTIONS.md](COMPUTE_OPTIONS.md) for HF Pro, Colab Pro, Kaggle, and AWS — when to use each for training and inference.

## Training

```bash
# Pipeline V1
python -m src.train --data-dir vesuvius-challenge-surface-detection --output-dir outputs --epochs 200 --amp

# Pipeline V2
python -m src_v2.train --data-dir vesuvius-challenge-surface-detection --output-dir outputs_v2 --epochs 200 --amp

# Pipeline V3 (multi-scale learned fusion)
python -m src_v3.train --data-dir vesuvius-challenge-surface-detection --output-dir outputs_v3 --epochs 200 --amp

# Pipeline nnU-Net (via Hugging Face Jobs)
python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx
python -m src_nnunet.check_job 60  # monitor progress
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
python smoke_test_v3.py   # Pipeline V3
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
├── src_v2/               # Pipeline V2 (completed, best model)
│   ├── model.py          #   UNet3DDeepSup (deep supervision)
│   ├── losses.py         #   Focal+Dice, Skeleton Recall, Boundary
│   ├── dataset.py        #   + skeleton precomputation
│   ├── train.py          #   Adapted for deep supervision
│   └── ...               #   Own copies of shared modules
├── src_v3/               # Pipeline V3 (paused, multi-scale fusion)
│   ├── model.py          #   MultiScaleFusionUNet (32³, 64³, 128³)
│   ├── losses.py         #   CompositeLossV3 on fused output
│   ├── dataset.py        #   (image, label, skeleton), patch_size=128
│   └── ...               #   Own copies of shared modules
├── src_nnunet/           # Pipeline nnU-Net (active, 1st place architecture)
│   ├── convert_dataset.py#   Convert TIF data to nnU-Net format
│   ├── predict.py        #   Inference + 1st place post-processing
│   ├── train_hf.py       #   End-to-end training for HF Jobs
│   ├── launch_hf_job.py  #   Launch training on Hugging Face Jobs
│   └── monitor_hf_job.py #   Monitor job status and logs
├── aws/                  # EC2 launch scripts & IAM policies
├── configs/              # Training configuration
├── notebooks/            # Kaggle submission notebook
├── PIPELINES.md          # Multi-pipeline documentation
├── CLAUDE.md             # Agent instructions
├── COMPETITION_NOTES.md  # Competition rules & dataset analysis
├── requirements.txt
└── ../vesuvius_sinogram/ # Future: sinogram-domain surface detection
```

## Related Projects

- **[vesuvius_sinogram](../vesuvius_sinogram/)** — Future research direction: surface detection directly from raw CT sinogram (projection) data, bypassing the reconstruction-then-segment pipeline. Hypothesis: raw projections contain discriminative signal lost during filtered back-projection.

## Citation

This project uses the EduceLab-Scrolls dataset:

> S. Parker, C. Parsons, and W. B. Seales. "EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT." arXiv:2304.02084, 2023.

Post-processing adapted from the 1st place solution:

> "1st Place Solution for the Vesuvius Challenge Surface Detection." Kaggle Competition Writeup, 2024. [Link](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)

## License

[CC BY-NC 4.0](LICENSE) — as required by the Vesuvius Challenge competition rules.
