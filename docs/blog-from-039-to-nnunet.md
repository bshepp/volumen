# From 0.39 to nnU-Net: A Messy Multi-Pipeline Journey on the Vesuvius Challenge

*The honest dev log — what broke, what worked, and why we're still running*

---

We started at **0.390 public / 0.409 private** on the [Vesuvius Challenge Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) and iterated to **0.405 / 0.426** — roughly 1240th on the private leaderboard. The winners hit **0.577 / 0.614**. That gap is still massive. This post is the real story: four pipelines, eleven submissions, a lot of failures, and what we learned the hard way.

## Pipelines at a Glance[[pipelines]]

| Pipeline | Architecture | Loss | Status | Val Loss | Surface Dice | Kaggle |
|----------|--------------|------|--------|----------|--------------|--------|
| V1 | 3D U-Net, 6→3 ch | CE+Dice, clDice, Boundary | Frozen | 1.36 | 0.116 | — |
| V2 | UNet3DDeepSup, deep supervision | Focal+Dice, SkeletonRecall, Boundary | Frozen | **0.67** | **0.254** | **0.405 / 0.426** |
| V3 | Multi-scale fusion (32³+64³+128³) | Focal+Dice, SkeletonRecall, Boundary | Paused @ ep53 | 0.70 | 0.226 | — |
| nnU-Net | nnU-Net v2 (1st place) | Native | Training on HF Jobs | — | — | — |

All pipelines use the same validation split (scroll ID 26002). V2 integrates the [1st place post-processing pipeline](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su); V1 and V3 use the original pipeline.

---

## The Task[[task]]

Ancient Herculaneum papyrus scrolls were carbonized by Vesuvius in 79 AD. To read them, we need to detect the recto (front) surface in 3D micro-CT scans — thin sheets buried in noise. The competition metric blends geometric accuracy (SurfaceDice), topology (no spurious holes, no sheet mergers), and region overlap (VOI). It's a 3D segmentation problem with extreme class imbalance: surface voxels are ~6% of the volume.

---

## Pipeline Evolution[[evolution]]

We ended up with **four completely independent pipelines**. No shared imports. Each can be deleted without touching the others. It's intentionally messy — we wanted to test competing strategies in isolation.

### V1: The Baseline That Taught Us Everything Wrong

Standard 3D U-Net, CE+Dice, clDice, boundary loss. We ran it on AWS (g4dn.xlarge, T4). **60 minutes per epoch** because the Laplacian-of-Gaussian and Hessian features were computed on CPU with scipy. The GPU sat idle 90% of the time. We hit NaN at epoch 29. We hit OOM when the DataLoader forked and duplicated 15GB of preloaded coordinates. We ran 200 epochs and got **surface Dice 0.116**. Best val loss: 1.36. Not great.

**What we fixed:** Label smoothing to avoid `log(0)` in CE. Skip non-finite batches. Moved feature computation to GPU (10–50× speedup). Switched to volume-grouped batching. V1 is frozen now — we don't touch it.

### V2: The One That Actually Worked

Focal loss, deep supervision (auxiliary heads at 2×, 4×, 8×), skeleton recall instead of clDice (cheaper, better gradient for thin structures). Same T4, ~57 min/epoch. Completed 200 epochs. **Val loss 0.67, surface Dice 0.25**. Our best-performing custom model.

We also adopted the **1st place post-processing pipeline**: per-sheet binary closing, height-map patching with interpolation, LUT-based hole plugging, global fill. That pipeline is worth a lot — but it wasn't available during the competition. We integrated it after the deadline, once the winners published their writeups. Our first scored submission (V5) used original post-processing: 0.390/0.409. After several failed attempts (V6–V9 all timed out or errored), Version 10 with 1st-place post-processing landed at **0.405/0.426** — a +0.015/+0.017 improvement from post-processing alone.

### V3: The Promising One That NaNa'd

Multi-scale fusion: three UNets (32³, 64³, 128³) with a learned fusion layer. Hit **NaN at epoch 55** on an A10G. Cause: FP16 + Focal loss. The $(1-p_t)^\gamma$ term and log_softmax can overflow FP16. We added FP32 loss casts and `isfinite` guards. Checkpoint at epoch 50 is intact. We'll resume eventually. V3 had learned faster per epoch than V2 — the multi-scale architecture seems to have better inductive bias. But we didn't get to finish.

### nnU-Net: Copying the Winners

The 1st place team used nnU-Net v2. One model, 250 epochs, 0.577/0.614. We're training nnU-Net for 200 epochs on **Hugging Face Jobs**. That journey had its own comedy of errors.

---

## The HF Jobs Saga[[hf-jobs]]

We wanted fire-and-forget training. HF Jobs + pre-paid credits seemed ideal. Here's what broke:

| Failure | Cause | Fix |
|---------|-------|-----|
| SSL errors | `hf` CLI / `huggingface_hub` intermittent failures | Raw `requests` with retry logic |
| 404 on job launch | Wrong endpoint | `POST /api/jobs/{namespace}` not `/api/jobs` |
| 400 Bad Request | `secrets={"HF_TOKEN": True}` | Pass actual token string |
| `unzip` not found | PyTorch Docker image | `apt-get install -y unzip` before extract |
| OOM @ 73% preprocessing | Container 110GB, zip+raw+preprocessed > 110GB | Delete zip after extract, delete raw after preprocessing |

We embedded the training script as base64 in the job command (no `train_hf.sh` to upload). It works now. Launch:

```bash
python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx
python -m src_nnunet.check_job 60   # monitor
```

The nnU-Net run is in progress. Model uploads to the Hub on completion.

---

## The Kaggle Submission Comedy[[kaggle]]

| Issue | Cause | Fix |
|-------|-------|-----|
| Patch size 128 | OOM on Kaggle's T4 | Dropped to 64 |
| TTA = True | 8 full sliding-window passes | Stride 48, batch size 2 — finally finished |
| `imagecodecs` | Submission kernels have no internet | Add via Dependency Manager, not `pip install` |
| V6–V9 all failed | Timeouts and errors | Iterating on stride, batch size, cuda, amp settings |
| V10 succeeded | 1st-place postproc + tuned settings | **0.405/0.426** — post-processing worth +0.017 |

---

## What We'd Do Differently[[lessons]]

| Priority | Lesson | Why |
|----------|--------|-----|
| 1 | Study winning post-processing early | The 1st place pipeline was a huge lever. We only had access after the competition closed. In future competitions, study prior winners' post-processing from day one. |
| 2 | Benchmark TTA cost early | 8× inference is brutal. Try 4 flips (Y and X only) or no TTA for speed. |
| 3 | nnU-Net from the start | Custom architectures closed some of the gap. The winners' architecture closed most of it. Sometimes adopt, not invent. |
| 4 | Disk math before launch | 110GB sounds like a lot. Zip + raw + preprocessed did not fit. Do the arithmetic upfront. |

---

## Where We Are Now[[status]]

- **nnU-Net** — Training on HF Jobs. Model will upload to the Hub when done.
- **V2 / V3** — Kept as testbeds for systematically trying techniques from top solutions (patch schedules, ensemble weights, multi-threshold). We're not done experimenting.
- **Compute stack** — HF Pro (Jobs for training, ZeroGPU for demos) + Colab Pro (linked for extended HF quota) + Kaggle (submissions, 30 hrs/week).

---

## Reproducibility[[reproducibility]]

| Resource | Location |
|----------|----------|
| Code | [github.com/bshepp/volumen](https://github.com/bshepp/volumen) |
| V2 weights (Kaggle) | [briansheppard/vesuvius-v2-weights](https://www.kaggle.com/models/briansheppard/vesuvius-v2-weights) |
| nnU-Net (on completion) | [huggingface.co/bshepp/vesuvius-nnunet](https://huggingface.co/bshepp/vesuvius-nnunet) |
| Competition data | [Kaggle: vesuvius-challenge-surface-detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data) |
| Validation split | Scroll ID `26002` held out |

**Inference (V2):**

```python
from src_v2.inference import run_inference

run_inference(
    model_path="best_model_v2.pth",
    test_volume_path="path/to/volume.tif",
    output_path="submission.tif",
    use_tta=True,
    use_postprocess=True,
)
```

**Submission settings (Kaggle):** `PATCH_SIZE=64`, `STRIDE=48`, `BATCH_SIZE=2`, `USE_TTA=True`, `USE_POSTPROCESS=True`.

---

## Glossary[[glossary]]

| Term | Definition |
|------|------------|
| TTA | Test-time augmentation: 8 flip combinations (Z×Y×X axes), average predictions. 8× inference cost. |
| Skeleton recall | Loss = 1 − recall over precomputed skeletons. Replaces clDice; ~90% cheaper, better gradient for thin structures. |
| 1st place post-processing | Binary closing, height-map patching, LUT hole plugging, global fill. See [writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su). |
| Volume-grouped batching | One volume per DataLoader item, multiple patches extracted. O(1) disk I/O per batch. |

---

## The Honest Takeaway[[takeaway]]

We started at 0.39 and iterated to 0.405. The leaders shipped 0.61. The gap is real. But we have four pipelines, eleven submissions (seven of which failed), a solid post-processing stack, and a running nnU-Net training job. The messy part — the OOMs, the NaNs, the wrong API calls, the disk space math — is the actual work. This post is that.

Code: [github.com/bshepp/volumen](https://github.com/bshepp/volumen)
