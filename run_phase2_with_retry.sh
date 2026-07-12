#!/bin/bash
# Phase 2はネットワーク経由のストリーミングに数時間依存するため、
# 一時的なDNS/接続エラーでプロセス全体が落ちても自動的に再起動する。
# 既に保存済みチェックポイントがあればそこから再開する。
cd /root/Qubit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_RETRIES=40
for i in $(seq 1 $MAX_RETRIES); do
    if [ -f qbnn100m_v2_phase2.pt ]; then
        RESUME=qbnn100m_v2_phase2.pt
    else
        RESUME=qbnn100m_v2_phase1_best.pt
    fi

    echo "=== 試行 $i/$MAX_RETRIES: resume-from=$RESUME ===" >> logs/pretrain_fresh100m_phase2.log

    python3 -u pretrain_fresh_100m.py --phase 2 --dataset-mix phase2_base \
        --target-tokens 1100000000 \
        --micro-batch-size 8 --grad-accum 8 \
        --lr 8e-5 --j-lr-ratio 0.1 \
        --lambda-mode warmup --lambda-warmup-target 0.03 --lambda-warmup-start 0.35 --lambda-warmup-end 0.65 \
        --lr-schedule cosine --lr-warmup-frac 0.02 --lr-min-ratio 0.1 \
        --freeze-j-theta \
        --eval-every-steps 300 --save-every-steps 300 \
        --n-val-articles 400 \
        --resume-from "$RESUME" \
        --ckpt-name qbnn100m_v2_phase2.pt \
        >> logs/pretrain_fresh100m_phase2.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== 正常終了 ===" >> logs/pretrain_fresh100m_phase2.log
        exit 0
    fi
    echo "=== 異常終了 (exit=$EXIT_CODE) — 15秒後に再試行 ===" >> logs/pretrain_fresh100m_phase2.log
    sleep 15
done
echo "=== 最大リトライ回数到達、諦め ===" >> logs/pretrain_fresh100m_phase2.log
exit 1
