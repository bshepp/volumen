#!/bin/bash
set -ex
exec > /var/log/vesuvius-setup.log 2>&1
echo "=== Vesuvius training setup started at $(date) ==="

S3_BUCKET="vesuvius-challenge-training-290318"
WORK_DIR="/home/ubuntu/vesuvius"
DATA_DIR="${WORK_DIR}/data/vesuvius-challenge-surface-detection"
CODE_DIR="${WORK_DIR}/code"
OUTPUT_DIR="${WORK_DIR}/outputs"

sleep 10

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"
chown -R ubuntu:ubuntu "${WORK_DIR}"

echo "=== Adding swap space ==="
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

echo "=== Downloading code from S3 ==="
aws s3 sync "s3://${S3_BUCKET}/code/" "${CODE_DIR}/" --region us-east-1

echo "=== Downloading data from S3 ==="
mkdir -p "${DATA_DIR}"
aws s3 sync "s3://${S3_BUCKET}/data/vesuvius-challenge-surface-detection/" "${DATA_DIR}/" --region us-east-1

echo "=== Data download complete ==="
ls -la "${DATA_DIR}/"

echo "=== Setting up Python ==="
cd "${CODE_DIR}"
ln -sf /usr/bin/python3 /usr/bin/python
pip install --upgrade pip
pip install -r requirements.txt

echo "=== GPU Status ==="
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Starting training at $(date) ==="
chown -R ubuntu:ubuntu "${WORK_DIR}"

su - ubuntu -c "cd ${CODE_DIR} && nohup python3 -u -m src.train \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
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
    --val-scrolls 26002 \
    --num-workers 4 \
    --amp \
    --save-every 25 \
    > ${OUTPUT_DIR}/training.log 2>&1 &"

echo "=== Setup complete at $(date) ==="
