#!/bin/bash
# Run this ON THE INSTANCE after V1 training completes (epoch 200).
# Via SSM: aws ssm send-command --instance-ids <ID> ... --parameters '{"commands":["cd /home/ubuntu/vesuvius/code && bash aws/post-run-download-and-start-v2.sh"]}'
# Or run steps manually. Requires: code and src_v2 synced from S3, instance profile with S3 read/write.

set -e
S3_BUCKET="vesuvius-challenge-training-290318"
WORK_DIR="/home/ubuntu/vesuvius"
DATA_DIR="${WORK_DIR}/data/vesuvius-challenge-surface-detection"
CODE_DIR="${WORK_DIR}/code"
OUTPUT_DIR="${WORK_DIR}/outputs"
OUTPUT_V2="${WORK_DIR}/outputs_v2"

echo "=== Post-run: save best V1, clear, start V2 ==="

# 1) Confirm V1 training has finished
if pgrep -f "src.train" >/dev/null 2>&1; then
    echo "ERROR: V1 training (src.train) is still running. Wait for it to complete."
    exit 1
fi
if [ ! -f "${OUTPUT_DIR}/training.log" ]; then
    echo "ERROR: No training.log found."
    exit 1
fi
if ! grep -q "Training complete\|Epoch 200/200" "${OUTPUT_DIR}/training.log" 2>/dev/null; then
    echo "WARNING: training.log does not show 'Training complete' or 'Epoch 200/200'. Proceed anyway? Last lines:"
    tail -3 "${OUTPUT_DIR}/training.log"
fi

# 2) Identify best model (V1 saves only when val loss improves)
BEST_PTH="${OUTPUT_DIR}/best_model.pth"
if [ ! -f "${BEST_PTH}" ]; then
    echo "ERROR: best_model.pth not found. Cannot save best model."
    exit 1
fi
echo "Best V1 model: ${BEST_PTH}"

# 3) Upload best model and logs to S3 for download
aws s3 cp "${BEST_PTH}" "s3://${S3_BUCKET}/outputs/run1_best_model.pth" --region us-east-1
[ -f "${OUTPUT_DIR}/history.json" ] && aws s3 cp "${OUTPUT_DIR}/history.json" "s3://${S3_BUCKET}/outputs/run1_history.json" --region us-east-1
aws s3 cp "${OUTPUT_DIR}/training.log" "s3://${S3_BUCKET}/outputs/run1_training.log" --region us-east-1
echo "Uploaded run1_best_model.pth, run1_history.json, run1_training.log to S3."

# 4) Clear disk: remove V1 checkpoints (keep best in S3; remove from instance to free space)
rm -f "${OUTPUT_DIR}"/checkpoint_epoch*.pth
rm -f "${OUTPUT_DIR}/best_model.pth"
echo "Cleared V1 .pth files from instance."

# 5) Ensure latest code (including src_v2) is present
aws s3 sync "s3://${S3_BUCKET}/code/" "${CODE_DIR}/" --region us-east-1
chown -R ubuntu:ubuntu "${CODE_DIR}"
echo "Synced code (src + src_v2) from S3."

# 6) Install deps (V2 needs scikit-image)
cd "${CODE_DIR}"
pip install -q -r requirements.txt
echo "Requirements OK."

# 7) Create V2 output dir and start V2 training
mkdir -p "${OUTPUT_V2}"
chown ubuntu:ubuntu "${OUTPUT_V2}"

echo "Starting V2 training (Focal + DeepSup + SkeletonRecall) ..."
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

echo "V2 training started. Log: ${OUTPUT_V2}/training.log"
echo "=== Post-run script done ==="
