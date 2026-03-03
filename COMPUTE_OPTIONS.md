# Compute Options — Vesuvius Challenge Training & Inference

This project uses multiple compute platforms. This document summarizes what's available with **HF Pro**, **Colab Pro**, **Kaggle**, and **AWS**, and when to use each.

---

## Platform Summary

| Platform      | Cost                    | GPU Options           | Best for                    |
|---------------|-------------------------|------------------------|-----------------------------|
| **HF Jobs**   | Pre-paid credits        | A10G, T4, A100, H200  | Long training (nnU-Net)     |
| **HF Sessions** | 30 hrs/week + Colab Pro link | T4 x2, etc.         | Interactive notebooks, V2/V3 experiments |
| **Colab Pro** | ~$10/mo (100 units)     | T4, L4, A100          | Extended runs, background execution |
| **Kaggle**    | Free (30 hrs/week)      | T4, P100               | Submissions, quick tests    |
| **AWS**       | Pay-per-hour            | T4, A10G, A100         | Full control (see AWS_TRAINING.md) |

---

## Hugging Face Pro ($9/month)

### What You Get

| Feature | Benefit |
|---------|---------|
| **Jobs** | Run CPU/GPU jobs on HF infra. Pre-paid credits required. Fire-and-forget training. |
| **ZeroGPU Spaces** | 8× daily quota (up to 25 min H200). Demos, quick inference — not long training. |
| **ZeroGPU priority** | Highest queue position |
| **Spaces Dev Mode** | SSH/VS Code into running Space. Hot reload, live debugging. |
| **Inference Providers** | 20× included credits vs free (2M vs 100k requests) |
| **Storage** | 10× private (1TB), 2× public |
| **Dataset Viewer** | On private datasets |
| **Blog posts** | Publish on your HF profile |

### Jobs Usage (nnU-Net)

```bash
python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx
python -m src_nnunet.check_job 60
```

Model uploads to `huggingface.co/bshepp/vesuvius-nnunet` on completion.

---

## Colab Pro (~$10/month)

### What You Get

| Feature | Benefit |
|---------|---------|
| **Compute units** | 100/month (Pro) or 500/month (Pro+) |
| **GPUs** | T4, L4, A100 — better than free Colab |
| **Memory** | More RAM than free tier |
| **Background execution** | Pro+ only — long runs survive disconnect |

### Integration: Link to HF for More Quota

HF GPU sessions (e.g. JupyterLab on the Hub) have a **30-hour weekly quota**. Linking your Colab Pro subscription extends this — your Colab Pro compute units unlock additional GPU time on HF's platform. Use the "Link to Colab Pro for more Quota" option in Session/Accelerator settings when you hit the limit.

---

## When to Use What

| Use Case | Platform | Why |
|----------|----------|-----|
| **nnU-Net training (~10 hrs)** | HF Jobs | Fire-and-forget, auto-upload. Pre-paid credits. |
| **V2/V3 experiments, notebooks** | HF Sessions (linked Colab Pro) | 30 hrs + Colab Pro boost. T4 x2. Good for iterative dev. |
| **Kaggle submissions** | Kaggle | Data already there, scoring. 30 hrs/week GPU. |
| **Heavy multi-day runs** | Colab Pro (native) | 100 compute units, background on Pro+. |
| **Model hosting / inference** | HF Inference | PRO gives 20× credits. |
| **Full control, custom infra** | AWS | See `AWS_TRAINING.md`. Instances terminated; can relaunch. |

---

## Recommendations

### Leverage Now

1. **HF Jobs** — Keep using for nnU-Net and future long training. Add pre-paid credits when needed.
2. **HF Sessions + Colab Pro link** — For V2/V3 as testbeds. Link Colab Pro if you exceed 30 hrs/week.
3. **Spaces Dev Mode** — Try it for V2/V3 iteration: VS Code over SSH, hot reload, no Docker rebuilds.
4. **ZeroGPU** — Good for quick demos or inference tests (25 min/day H200).

### Consider Later

1. **Blog post on HF** — Write up your multi-pipeline approach, 1st place post-processing integration, nnU-Net adoption. PRO includes publishing. Good portfolio piece.
2. **Dataset** — If you create derived data (e.g. preprocessed patches, skeletons), consider publishing on the Hub. PRO gives 1TB private storage.
3. **Colab Pro+** — If 100 units aren't enough, 500 units + background execution helps for very long runs.

### Do Not Prioritize

- **AWS** — Instances are terminated. HF/Colab/Kaggle cover current needs. Relaunch only if you need full control or spot pricing.
- **ZeroGPU for training** — Daily minutes, not hours. Use Jobs or Colab instead.

---

## Publishing Opportunities (HF Pro)

| Opportunity | Effort | Value |
|-------------|--------|-------|
| **Blog: "From 0.39 to nnU-Net: A Multi-Pipeline Journey on the Vesuvius Challenge"** | Medium | Showcases systematic testing of top solutions. Good for portfolio. |
| **Dataset: Precomputed skeletons** | Low | V2/V3 use skeletonized labels. Could help others. |
| **Model: vesuvius-nnunet** | Done | nnU-Net model auto-uploads from HF Jobs. |
| **Model: V2 weights on HF** | Low | Already on Kaggle; mirror to Hub for discoverability. |

---

## Links

- [HF Pro benefits](https://huggingface.co/pro)
- [HF Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)
- [Colab Pro](https://colab.research.google.com/signup)
- `AWS_TRAINING.md` — Full AWS setup (instances terminated; guide preserved)
