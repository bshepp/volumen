#!/bin/bash
# AWS training launch script for Vesuvius Surface Detection
# Target: p3.2xlarge (V100 16GB) or g4dn.xlarge (T4 16GB) spot instance
#
# Usage:
#   1. Upload this repo to the instance (or git clone)
#   2. Install dependencies: pip install -r requirements.txt
#   3. Download/mount the dataset to DATA_DIR
#   4. Run: bash run_training.sh
#
# Estimated training time: 10-15 hours on V100
# Estimated cost: $10-30 (spot pricing)

set -e

# Configuration
DATA_DIR="${DATA_DIR:-/home/ubuntu/data/vesuvius-challenge-surface-detection}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/outputs}"

echo "=== Vesuvius Surface Detection Training ==="
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'None detected')"

# Install dependencies
pip install -r requirements.txt

# Run training
python -m src.train \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 200 \
    --batch-size 2 \
    --patch-size 128 \
    --patches-per-volume 4 \
    --surface-bias 0.7 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --base-filters 32 \
    --depth 4 \
    --cldice-iters 10 \
    --val-scrolls "26002" \
    --num-workers 4 \
    --amp \
    --save-every 25

echo "=== Training complete ==="
echo "Best model saved to: ${OUTPUT_DIR}/best_model.pth"
echo "History saved to: ${OUTPUT_DIR}/history.json"
