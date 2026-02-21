# AWS Training Run 1 - Report

**Date**: February 14-15, 2026
**Instance**: g4dn.xlarge (Tesla T4 16GB, 4 vCPU, 16GB RAM) — on-demand
**Region**: us-east-1b
**Instance ID**: i-0e404fe2060674794
**Cost Rate**: ~$0.53/hr
**Total Runtime**: ~15.5 hours (Feb 14 17:48 UTC → Feb 15 ~09:15 UTC)
**Estimated Cost**: ~$8.20

---

## What Was Accomplished

- 14 complete training epochs (+ partway through epoch 15) out of 200
- Best model checkpoint saved: `best_model.pth` (310 MB, from epoch 1, val_loss=1.365)
- Full training log saved: `training.log`
- 27M parameter 3D U-Net with 6-channel periodicity-exploitation features
- 704 training samples, 82 validation samples (scroll 26002 held out)

## Training Curve (14 epochs)

| Epoch | Train Loss | Val Loss | Val Surface Dice | Time (s) |
|-------|-----------|----------|-----------------|----------|
| 1     | 1.5020    | 1.3652   | 0.0000          | 4151     |
| 2     | 1.4676    | 1.3985   | 0.0014          | 3544     |
| 3     | 1.4594    | 1.3639   | 0.0337          | 3563     |
| 4     | 1.4504    | 1.3921   | 0.0650          | 3567     |
| 5     | 1.4340    | 1.3835   | 0.0599          | 3557     |
| 6     | 1.4373    | 1.3890   | 0.1130          | 3579     |
| 7     | 1.4320    | 1.4121   | 0.0690          | 3601     |
| 8     | 1.4231    | 1.3845   | 0.0985          | 3639     |
| 9     | 1.4137    | 1.3704   | 0.0690          | 3602     |
| 10    | 1.4059    | 1.3998   | 0.1162          | 3590     |
| 11    | 1.4052    | 1.4252   | 0.0976          | 3568     |
| 12    | 1.3857    | 1.4240   | 0.0783          | 3611     |
| 13    | 1.3932    | 1.4617   | 0.0995          | 3617     |
| 14    | 1.3786    | 1.4219   | 0.0762          | 3605     |

**Observations**:
- Train loss is steadily decreasing (1.50 → 1.38), confirming the model IS learning
- Val loss plateaued around 1.36-1.46, with some oscillation
- Surface Dice emerged from 0 to ~0.06-0.12 range, but noisy — early signal
- Best val loss was actually epoch 3 (1.3639), suggesting possible mild overfitting by epoch 14

## Issues Encountered and Fixes Applied

### 0. NaN training loss (epochs 29+)
- **Cause**: FP16 (AMP) + cross-entropy. When the model is very confident and wrong, softmax gives prob ≈ 0 for the correct class → `log(0) = -inf`; in FP16 this propagates to NaN. Occasional batches can also trigger NaNs in clDice (iterative soft skeletonization in FP16).
- **Fixes applied** (in `src/train.py` and `src/losses.py`):
  - **Label smoothing** (0.01) in `WeightedCEDiceLoss` so CE never sees exact 0/1 and avoids `log(0)`.
  - **Skip backward/step** when `loss` is not finite so a bad batch does not corrupt the model.
  - **Exclude non-finite batches** from the epoch running average so the printed "Train loss" is not NaN.
- These changes apply to future runs (e.g. after syncing code to AWS). The current long-running job was started before the fix; best checkpoint is unchanged.

### 1. Spot Instance Quota = 0 for G-type
- **Problem**: G and VT spot instance quota was 0 vCPUs
- **Fix**: Used on-demand g4dn.xlarge (~$0.53/hr vs ~$0.21/hr spot)
- **Future**: Request a quota increase via AWS Service Quotas console

### 2. OOM Kill (16 GB RAM)
- **Problem**: Original dataset pre-loaded surface coordinates for all 786 volumes using int64 numpy arrays. With num_workers=4, fork() duplicated ~15 GB of Python process memory, exceeding 16 GB RAM
- **Fix**: 
  - Changed surface coords from int64 to int16 (75% memory savings)
  - Capped to 5000 coords per volume
  - Reduced num_workers from 4 to 2
  - Added 8 GB swap file as safety net
- **Result**: RAM usage dropped from 15.3 GB to ~3 GB

### 3. Missing Training Images (5 of 786)
- **Problem**: S3 sync from local missed 5 files (850790699, 865516044, 86701140, 867889560, 871773282)
- **Fix**: Manually uploaded and synced the 5 missing .tif files

### 4. `python` Not Found on DL AMI
- **Problem**: Ubuntu Deep Learning AMI uses `python3`, not `python`
- **Fix**: Created symlink: `sudo ln -sf /usr/bin/python3 /usr/bin/python`

### 5. Windows CRLF Line Endings
- **Problem**: Scripts created on Windows had `\r` characters causing `bash: $'\r': command not found`
- **Fix**: Sent commands directly via SSH instead of piping scripts

## The Core Performance Problem

**Each epoch takes ~60 minutes** — far too slow for 200 epochs.

**Root cause**: CPU-bound feature computation (Laplacian of Gaussian, Hessian trace, gradient magnitude) in `compute_features_for_patch()` is the bottleneck, not the GPU. The GPU is idle 90%+ of the time waiting for data.

At this rate: 200 epochs × 60 min = **200 hours = 8.3 days = ~$106**

---

## Suggestions for Next Run

### Priority 1: Fix the Data Pipeline Bottleneck (HIGHEST IMPACT)

**Option A — Move Feature Computation to GPU (Recommended)**
Replace scipy-based CPU feature computation with PyTorch GPU operations:
```python
# Instead of scipy.ndimage.gaussian_laplace on CPU:
# Use torch conv3d with precomputed Gaussian/LoG kernels on GPU
# This moves the bottleneck from CPU to GPU where there's spare capacity
```
- Expected speedup: 10-50x for feature computation
- Epoch time: ~3-6 minutes instead of 60
- Total training: ~10-20 hours instead of 200

**Option B — Pre-compute Features Offline**
Run a one-time feature computation job that saves 6-channel features as .npy files:
- Each volume: 320³ × 6 × float32 = ~786 MB → use float16 = ~393 MB
- Total: 786 volumes × 393 MB = ~309 GB (large but feasible on 150 GB EBS with streaming)
- Alternative: pre-compute for just the patches most likely to be sampled

**Option C — Reduce Per-Epoch Work (Quick Fix)**
- Set `patches_per_volume=1` (instead of 4): cuts epoch from 60 to 15 min
- Set `epochs=800` to compensate: 800 × 15 min = 200 hours (same total, but more frequent checkpoints)
- Or accept fewer total gradient steps with `patches_per_volume=2, epochs=200`: ~100 hours

### Priority 2: Use a CPU-Heavier Instance

| Instance | vCPUs | RAM | GPU | Spot Price | On-Demand |
|----------|-------|-----|-----|------------|-----------|
| g4dn.xlarge | 4 | 16 GB | T4 16GB | $0.21 | $0.53 |
| **g4dn.2xlarge** | **8** | **32 GB** | **T4 16GB** | **$0.24** | **$0.75** |
| g4dn.4xlarge | 16 | 64 GB | T4 16GB | $0.36 | $1.20 |

The g4dn.2xlarge doubles the CPU cores and RAM for only 40% more cost, and would allow num_workers=4-6, roughly halving data loading time.

### Priority 3: Request G-Type Spot Quota Increase
- Go to AWS Console → Service Quotas → EC2 → "All G and VT Spot Instance Requests"
- Request increase from 0 to at least 8 vCPUs
- This would cut costs from ~$0.53/hr to ~$0.21/hr (60% savings)

### Priority 4: Model/Training Improvements
- **Learning rate**: May need warmup — the best val loss was at epoch 1/3, suggesting initial LR may be too aggressive after the first few epochs
- **Loss weights**: clDice loss is consistently high (~0.7-0.9) and not dropping much; may need tuning
- **Resume training**: The checkpoint can be loaded to resume from epoch 14 instead of starting over

---

## Files and Artifacts

### On S3 (`s3://vesuvius-challenge-training-290318/`)
- `code/` — Full training codebase
- `data/` — Complete 25.4 GB dataset (786 train images, 786 labels, test data)
- `outputs/best_model.pth` — Best model checkpoint (epoch 1, 310 MB)
- `outputs/training.log` — Full training log

### Local (`F:\kaggle\vesuvius_challenge\`)
- `outputs_aws/best_model.pth` — Downloaded best model
- `outputs_aws/training.log` — Downloaded training log
- `aws/` — Launch scripts, IAM policies, user-data script

### AWS Resources Created
- **S3 Bucket**: `vesuvius-challenge-training-290318`
- **IAM Role**: `vesuvius-training-role` (S3 + SSM access)
- **Instance Profile**: `vesuvius-training-profile`
- **Key Pair Used**: `3body-compute`
- **Security Group Used**: `sg-01db2d9932427a00a` (3body-sg, SSH open)

### AWS Resources to Clean Up (if not needed)
- S3 bucket contains ~25 GB of data — costs ~$0.58/month to keep
- IAM role and instance profile — no cost but clutter
- The EC2 instance has been terminated (no ongoing cost)

---

## Quick-Start for Next Run

```bash
# 1. Request G-type spot quota increase (do this NOW, takes 1-2 days)
# AWS Console → Service Quotas → EC2 → "All G and VT Spot Instance Requests" → Request increase to 8

# 2. Implement GPU-based features (Priority 1A), then re-upload code:
aws s3 sync src/ s3://vesuvius-challenge-training-290318/code/src/

# 3. Launch (after implementing fixes):
aws ec2 run-instances \
  --image-id ami-0fe59b4f6e7e66c3e \
  --instance-type g4dn.xlarge \
  --key-name 3body-compute \
  --security-group-ids sg-01db2d9932427a00a \
  --iam-instance-profile Name=vesuvius-training-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://aws/user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vesuvius-training},{Key=Project,Value=vesuvius-challenge}]'
```
