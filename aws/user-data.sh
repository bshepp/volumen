#!/bin/bash
set -ex
exec > /var/log/vesuvius-setup.log 2>&1
echo "=== Vesuvius training setup started at $(date) ==="

S3_BUCKET="vesuvius-challenge-training-290318"
WORK_DIR="/home/ubuntu/vesuvius"
DATA_DIR="${WORK_DIR}/data/vesuvius-challenge-surface-detection"
CODE_DIR="${WORK_DIR}/code"
OUTPUT_V2="${WORK_DIR}/outputs_v2"

# PIPELINE controls which pipeline trains on launch.
# Set to "v2" or "v3". V1 is frozen (see PIPELINES.md).
PIPELINE="${PIPELINE:-v2}"

sleep 10

mkdir -p "${WORK_DIR}" "${OUTPUT_V2}"
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

echo "=== Starting Pipeline ${PIPELINE} training at $(date) ==="
chown -R ubuntu:ubuntu "${WORK_DIR}"

if [ "${PIPELINE}" = "v3" ]; then
    OUTPUT_DIR="${WORK_DIR}/outputs_v3"
    mkdir -p "${OUTPUT_DIR}"
    chown ubuntu:ubuntu "${OUTPUT_DIR}"
    su - ubuntu -c "cd ${CODE_DIR} && nohup python3 -u -m src_v3.train \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --epochs 200 \
        --patch-size 128 \
        --patches-per-volume 4 \
        --surface-bias 0.7 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --base-filters 16 \
        --focal-gamma 2.0 \
        --skeleton-dilation 2 \
        --val-scrolls 26002 \
        --num-workers 4 \
        --amp \
        --save-every 25 \
        > ${OUTPUT_DIR}/training.log 2>&1 &"
else
    su - ubuntu -c "cd ${CODE_DIR} && nohup python3 -u -m src_v2.train \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_V2} \
        --epochs 200 \
        --patch-size 128 \
        --patches-per-volume 4 \
        --surface-bias 0.7 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --base-filters 32 \
        --depth 4 \
        --focal-gamma 2.0 \
        --skeleton-dilation 2 \
        --val-scrolls 26002 \
        --num-workers 4 \
        --amp \
        --save-every 25 \
        > ${OUTPUT_V2}/training.log 2>&1 &"
fi

echo "=== Setup complete at $(date) ==="
