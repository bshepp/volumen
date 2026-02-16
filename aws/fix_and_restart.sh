#!/bin/bash
set -ex

# Add 8GB swap space as safety net
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "Swap enabled:"
free -h

# Pull updated code from S3
aws s3 sync s3://vesuvius-challenge-training-290318/code/ /home/ubuntu/vesuvius/code/ --region us-east-1
echo "Code updated"

# Restart training with reduced workers (2 instead of 4 for 16GB RAM)
cd /home/ubuntu/vesuvius/code

nohup python3 -u -m src.train \
    --data-dir /home/ubuntu/vesuvius/data/vesuvius-challenge-surface-detection \
    --output-dir /home/ubuntu/vesuvius/outputs \
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
    --num-workers 2 \
    --amp \
    --save-every 25 \
    > /home/ubuntu/vesuvius/outputs/training.log 2>&1 &

echo "TRAINING_PID: $!"
echo "Training restarted with num_workers=2"
