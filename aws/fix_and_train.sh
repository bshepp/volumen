#!/bin/bash
set -ex

# Fix python symlink
sudo ln -sf /usr/bin/python3 /usr/bin/python

# Verify CUDA
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Check data is present
ls /home/ubuntu/vesuvius/data/vesuvius-challenge-surface-detection/

# Start training with nohup
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
    --num-workers 4 \
    --amp \
    --save-every 25 \
    > /home/ubuntu/vesuvius/outputs/training.log 2>&1 &

echo "Training PID: $!"
echo $! > /home/ubuntu/vesuvius/outputs/train.pid
echo "Training started! Check with: tail -f /home/ubuntu/vesuvius/outputs/training.log"
