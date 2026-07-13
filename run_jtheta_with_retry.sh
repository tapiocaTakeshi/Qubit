#!/bin/bash
# J/θ専用JGLUE学習は数時間ネットワークストリーミングに依存するため、
# 一時的なDNS/接続エラーでプロセスが落ちても自動的に再起動する。
cd /root/Qubit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_RETRIES=40
BASE_CKPT=qbnn100m_v2_phase2_best.pt
CKPT_NAME=qbnn100m_v2_jtheta_jglue.pt

for i in $(seq 1 $MAX_RETRIES); do
    if [ -f "$CKPT_NAME" ]; then
        RESUME="$CKPT_NAME"
    else
        RESUME="$BASE_CKPT"
    fi

    echo "=== 試行 $i/$MAX_RETRIES: resume-from=$RESUME ===" >> logs/train_j_theta_jglue.log

    python3 -u train_j_theta_jglue.py \
        --resume-from "$RESUME" \
        --target-tokens 20000000 \
        --micro-batch-size 8 --grad-accum 4 \
        --lr 1e-4 --lambda-entangle 0.5 \
        --n-val-examples 300 --eval-every-steps 200 --save-every-steps 200 \
        --ckpt-name "$CKPT_NAME" \
        >> logs/train_j_theta_jglue.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== 正常終了 ===" >> logs/train_j_theta_jglue.log
        exit 0
    fi
    echo "=== 異常終了 (exit=$EXIT_CODE) — 15秒後に再試行 ===" >> logs/train_j_theta_jglue.log
    sleep 15
done
echo "=== 最大リトライ回数到達、諦め ===" >> logs/train_j_theta_jglue.log
exit 1
