# AWS Training Guide for Vesuvius Surface Detection

## Upload code to S3 (before the next run)

The instance gets code from S3 at launch (see `aws/user-data.sh`). To deploy your latest local code so the **next** run uses it:

**From repo root (PowerShell):**
```powershell
.\aws\upload-code-to-s3.ps1
```

**Or manually:**
```powershell
aws s3 sync src/ s3://vesuvius-challenge-training-290318/code/src/ --delete
aws s3 cp requirements.txt s3://vesuvius-challenge-training-290318/code/requirements.txt
```

- **New instance:** Launch as usual; user-data will `aws s3 sync` this code onto the instance.
- **Existing instance:** SSH in and run `aws s3 sync s3://vesuvius-challenge-training-290318/code/ /home/ubuntu/vesuvius/code/ --region us-east-1`, then restart training if needed.

---

## Quick Start

### 1. Launch Instance
- **Current:** g5.xlarge (A10G 24GB, ~$1.01/hr on-demand, ~$0.35/hr spot)
- **Budget option:** g4dn.xlarge (T4 16GB, ~$0.53/hr on-demand)
- **AMI:** Deep Learning AMI (Ubuntu 22.04) — comes with PyTorch, CUDA, etc.
- **Storage:** 150 GB gp3 (dataset is ~27 GB, outputs ~2 GB)

### 2. Upload Data
```bash
# Option A: Upload from local (slow)
scp -r vesuvius-challenge-surface-detection/ ubuntu@<IP>:/home/ubuntu/data/

# Option B: Download from Kaggle directly on instance
pip install kaggle
kaggle competitions download -c vesuvius-challenge-surface-detection
unzip vesuvius-challenge-surface-detection.zip -d /home/ubuntu/data/vesuvius-challenge-surface-detection
```

### 3. Upload Code
```bash
scp -r src/ configs/ requirements.txt run_training.sh ubuntu@<IP>:/home/ubuntu/vesuvius/
```

### 4. Install Dependencies
```bash
ssh ubuntu@<IP>
cd /home/ubuntu/vesuvius
pip install -r requirements.txt
```

### 5. Run Training

The `user-data.sh` script automatically launches V2 training on instance boot. To run manually or switch pipelines:

```bash
# Pipeline V2 (default — Focal + DeepSup + SkeletonRecall, ~8-12 hrs on A10G)
python -m src_v2.train \
    --data-dir /home/ubuntu/vesuvius/data/vesuvius-challenge-surface-detection \
    --output-dir /home/ubuntu/vesuvius/outputs_v2 \
    --epochs 200 --patch-size 128 --base-filters 32 --depth 4 \
    --focal-gamma 2.0 --skeleton-dilation 2 --amp

# Pipeline V3 (multi-scale fusion, ~12-18 hrs on A10G)
python -m src_v3.train \
    --data-dir /home/ubuntu/vesuvius/data/vesuvius-challenge-surface-detection \
    --output-dir /home/ubuntu/vesuvius/outputs_v3 \
    --epochs 200 --patch-size 128 --base-filters 16 \
    --focal-gamma 2.0 --skeleton-dilation 2 --amp
```

To launch V3 instead of V2 on boot, set `PIPELINE=v3` in `user-data.sh`.

### 6. Download Results
```bash
scp ubuntu@<IP>:/home/ubuntu/vesuvius/outputs_v2/best_model_v2.pth ./outputs_aws/
scp ubuntu@<IP>:/home/ubuntu/vesuvius/outputs_v2/history_v2.json ./outputs_aws/
```

## Estimated Costs

| Instance | GPU | On-Demand $/hr | Spot $/hr | V2 Training (200 ep) | V3 Training (200 ep) |
|----------|-----|----------------|-----------|----------------------|----------------------|
| g5.xlarge | A10G 24GB | ~$1.01 | ~$0.35 | ~8-12 hrs | ~12-18 hrs |
| g4dn.xlarge | T4 16GB | ~$0.53 | ~$0.21 | ~20-25 hrs | ~30-40 hrs |

## Training Configuration

See `configs/default.yaml` and `configs/default_v3.yaml`. Key settings per pipeline:

| Setting | V1 (frozen) | V2 (active) | V3 (active) |
|---------|------------|-------------|-------------|
| Patch size | 128³ | 128³ | 128³ |
| Base filters | 32 | 32 | 16 |
| Depth | 4 | 4 | 4/4/3 |
| Params | 27M | ~27M | ~15M |
| Loss | CE+Dice+clDice+Boundary | Focal+Dice+Skel+Boundary | Focal+Dice+Skel+Boundary |
| Optimizer | AdamW, lr=1e-3, cosine | AdamW, lr=1e-3, cosine | AdamW, lr=1e-3, cosine |
| Validation | scroll 26002 | scroll 26002 | scroll 26002 |

## Performance Optimizations (V2/V3)

Both V2 and V3 training scripts enable three performance flags on CUDA:

- **TF32 matmul precision** (`torch.set_float32_matmul_precision("medium")`): Uses Tensor Float 32 for matmuls on Ampere+ GPUs (A10G, A100). ~30-50% faster with negligible accuracy impact.
- **cuDNN benchmark** (`torch.backends.cudnn.benchmark = True`): Auto-tunes convolution algorithms for fixed input sizes. ~5-15% faster.
- **torch.compile** (`mode="reduce-overhead"`): JIT-compiles the model graph with kernel fusion. ~20-40% faster.

These are applied automatically in `main()` when running on CUDA. No flags needed.

## After Training: Kaggle Submission

1. Upload `best_model.pth` as a Kaggle Dataset (e.g., "vesuvius-model-weights")
2. Create a new Kaggle Notebook from `notebooks/submission.ipynb`
3. Add both datasets: competition data + your model weights
4. Update `MODEL_DIR` path in the notebook config cell
5. Enable GPU, commit and submit

## Local Validation Results (5 epochs, reduced model)

From our test with a tiny model (1.6M params, 5 epochs):
- Score: 0.26 (raw) / 0.26 (post-processed)
- This is expected to be low — full training with 27M params for 200 epochs
  on the complete dataset should approach/exceed the 0.543 baseline.
