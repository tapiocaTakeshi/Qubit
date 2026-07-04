#!/bin/bash
echo "🚀 NeuroQuantum 学習進捗監視"
echo "================================"
echo ""

while true; do
  echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
  
  # チェックポイントファイル
  if [ -f "neuroq_small_checkpoint.pt" ]; then
    size=$(ls -lh neuroq_small_checkpoint.pt | awk '{print $5}')
    echo "✅ メインチェックポイント: $size"
  fi
  
  # 中間チェックポイント
  ckpt_count=$(find checkpoints -name "*.pt" 2>/dev/null | wc -l)
  if [ "$ckpt_count" -gt 0 ]; then
    echo "📁 中間チェックポイント: $ckpt_count 個"
  fi
  
  # トレーニングログ
  if [ -f "training_log_main.txt" ]; then
    lines=$(wc -l < training_log_main.txt)
    echo "📝 ログ行数: $lines"
    tail -3 training_log_main.txt | grep -E "Loss|Batch" | tail -1
  fi
  
  # ラッパープロセス確認
  if pgrep -f "train_all_japanese_datasets" > /dev/null 2>&1; then
    echo "🔄 学習プロセス: 実行中"
  else
    echo "⏹️  学習プロセス: 完了"
    break
  fi
  
  echo "---"
  sleep 5
done

echo ""
echo "================================"
echo "✅ 学習完了"
ls -lh neuroq_small_*
