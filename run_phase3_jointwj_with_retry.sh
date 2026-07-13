#!/bin/bash
# Phase 3: qbnn100m_v2_phase2_best.pt から継続し、W/Jを同時学習する。
# 一般コーパス上の通常の次トークン予測でJ(もつれテンソル)をWと共に学習させ、
# 専用のPlannerデータやJGLUE単独学習は使わない。
cd /root/Qubit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_RETRIES=40
BASE_CKPT=qbnn100m_v2_phase2_best.pt
CKPT_NAME=qbnn100m_v2_phase3_jointwj.pt

for i in $(seq 1 $MAX_RETRIES); do
    if [ -f "$CKPT_NAME" ]; then
        RESUME="$CKPT_NAME"
    else
        RESUME="$BASE_CKPT"
    fi

    echo "=== 試行 $i/$MAX_RETRIES: resume-from=$RESUME ===" >> logs/pretrain_fresh100m_phase3.log

    python3 -u pretrain_fresh_100m.py --phase 3 --dataset-mix phase2_base \
        --target-tokens 500000000 \
        --micro-batch-size 8 --grad-accum 8 \
        --lr 8e-5 --j-lr-ratio 0.1 \
        --lambda-mode warmup --lambda-warmup-target 0.05 --lambda-warmup-start 0.0 --lambda-warmup-end 0.3 \
        --rho 0.2 --j-reg-alpha 1e-5 \
        --lr-schedule cosine --lr-warmup-frac 0.02 --lr-min-ratio 0.1 \
        --freeze-j-theta \
        --eval-every-steps 300 --save-every-steps 300 \
        --n-val-articles 400 \
        --resume-from "$RESUME" \
        --ckpt-name "$CKPT_NAME" \
        >> logs/pretrain_fresh100m_phase3.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== 正常終了 ===" >> logs/pretrain_fresh100m_phase3.log
        exit 0
    fi
    echo "=== 異常終了 (exit=$EXIT_CODE) — 15秒後に再試行 ===" >> logs/pretrain_fresh100m_phase3.log
    sleep 15
done
echo "=== 最大リトライ回数到達、諦め ===" >> logs/pretrain_fresh100m_phase3.log
exit 1
