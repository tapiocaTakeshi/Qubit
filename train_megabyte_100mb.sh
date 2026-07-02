#!/bin/bash
# 100MBモデル学習用スクリプト（B200推奨）
# VRAM最適化設定を含む

set -e

# ============================================
# メモリ断片化対策
# ============================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# 学習実行
# ============================================
python train_hf_dataset.py \
  --dataset-id "llm-jp/databricks-dolly-15k-ja" \
  --model-size megabyte_100mb \
  --batch-size 1 \
  --max-seq-len 512 \
  --gradient-accumulation-steps 8 \
  --use-bf16 \
  --gradient-checkpointing \
  --epochs 3 \
  --split train \
  "$@"
