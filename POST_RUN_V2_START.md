# After V1 Run Completes: Download Best Model & Start V2

When the current V1 training run reaches epoch 200 and finishes, use this procedure to save the best model, clear the instance for V2, and start Pipeline V2 (Focal + DeepSup + Skeleton Recall).

## Prerequisites

1. **Upload code to S3** (including `src_v2/` and `aws/*.sh`) so the instance can sync and run the post-run script:
   ```powershell
   .\aws\upload-code-to-s3.ps1
   ```
   Do this at least once before the run completes (you can run it now so S3 has V2 + scripts).

2. Instance must have **S3 read/write** (instance profile `vesuvius-training-profile` already allows this).

## Procedure (after epoch 200 completes)

From the repo root, run:

```powershell
.\aws\post-run-download-and-start-v2.ps1
```

**Optional arguments:**
- `-InstanceId i-xxxxxxxx` — use if your training instance ID is different (default: the current g4dn).
- `-SkipDownload` — run the instance script (upload to S3, clear, start V2) but do not download the best model to your machine.

## What the script does

1. **Checks** that V1 training has completed (looks for "Training complete" or Epoch 200 in `training.log`). If not, it prompts you to continue anyway.
2. **On the instance** (via SSM):
   - Confirms best V1 model is `outputs/best_model.pth`.
   - Uploads to S3:
     - `outputs/run1_best_model.pth`
     - `outputs/run1_history.json`
     - `outputs/run1_training.log`
   - Removes V1 checkpoint files from the instance to free disk.
   - Syncs code from S3 (gets `src_v2` and `aws/` scripts if not already there).
   - Installs requirements (including scikit-image for V2).
   - Starts V2 training: `python -m src_v2.train ...` with output in `outputs_v2/`.
3. **On your machine:** Downloads from S3 into `outputs_aws/`:
   - `run1_best_model.pth`
   - `run1_history.json`
   - `run1_training.log`

## Manual alternative

If you prefer to run steps yourself:

1. **Verify V1 is done** (e.g. SSM or SSH):
   ```bash
   tail -20 /home/ubuntu/vesuvius/outputs/training.log
   ```
2. **On the instance**, run the post-run script:
   ```bash
   cd /home/ubuntu/vesuvius/code && bash aws/post-run-download-and-start-v2.sh
   ```
   (Ensure code is synced from S3 first so `aws/post-run-download-and-start-v2.sh` exists.)
3. **Download best model** to your machine:
   ```powershell
   aws s3 cp s3://vesuvius-challenge-training-290318/outputs/run1_best_model.pth .\outputs_aws\run1_best_model.pth
   ```

## After V2 starts

- V2 logs: `/home/ubuntu/vesuvius/outputs_v2/training.log`
- Check progress via SSM (e.g. `tail` that file) or SSH.
- V2 saves `best_model_v2.pth` and `history_v2.json` in `outputs_v2/`.
