# AWS Training Guide for Vesuvius Surface Detection

## Quick Start

### 1. Launch Instance
- **Recommended:** p3.2xlarge spot instance (V100 16GB, ~$1/hr spot)
- **Budget option:** g4dn.xlarge (T4 16GB, ~$0.53/hr spot)
- **AMI:** Deep Learning AMI (Ubuntu 22.04) — comes with PyTorch, CUDA, etc.
- **Storage:** 100 GB gp3 (dataset is ~27 GB, outputs ~2 GB)

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
```bash
# Full training (200 epochs, ~10-15 hours on V100)
DATA_DIR=/home/ubuntu/data/vesuvius-challenge-surface-detection \
OUTPUT_DIR=/home/ubuntu/outputs \
bash run_training.sh

# Or with custom settings
python -m src.train \
    --data-dir /home/ubuntu/data/vesuvius-challenge-surface-detection \
    --output-dir /home/ubuntu/outputs \
    --epochs 200 \
    --batch-size 2 \
    --patch-size 128 \
    --base-filters 32 \
    --depth 4 \
    --amp
```

### 6. Download Results
```bash
scp ubuntu@<IP>:/home/ubuntu/outputs/best_model.pth ./outputs_aws/
scp ubuntu@<IP>:/home/ubuntu/outputs/history.json ./outputs_aws/
```

## Estimated Costs

| Instance | GPU | Spot $/hr | Training Time | Total Cost |
|----------|-----|-----------|---------------|------------|
| p3.2xlarge | V100 16GB | ~$1.00 | ~10-15 hrs | ~$10-15 |
| g4dn.xlarge | T4 16GB | ~$0.53 | ~20-25 hrs | ~$10-13 |

## Training Configuration

Default settings in `configs/default.yaml`:
- Patch size: 128^3
- Batch size: 2
- Base filters: 32, Depth: 4 (27M params)
- Loss: 0.4*(CE+Dice) + 0.3*clDice + 0.3*BoundaryLoss
- Optimizer: AdamW, lr=1e-3, cosine annealing
- Validation: scroll 26002 held out (matches test scroll)

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
