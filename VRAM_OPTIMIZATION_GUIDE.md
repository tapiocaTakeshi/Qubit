# VRAM最適化ガイド

## 概要

100MBモデルの学習時に発生するVRAM不足エラーを解決するための最適化ガイドです。

## エラーの原因

- **主因**: `max_seq_len=16384` のままの巨大なAttention活性化値と勾配がVRAMを占有
- Attention由来のメモリは `seq_len^2` に比例するため、16384→512で約**1024分の1**に削減可能

## 解決手順

### 1. 環境変数の設定（必須）

メモリ断片化を防止するため、以下の環境変数を設定してから実行します：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 2. 100MBモデルのパラメータ設定

`megabyte_100mb` の設定値：
- `max_seq_len`: **512** （推奨）または 1024
- `batch_size`: **1** （推奨）
- `gradient_checkpointing`: **True**
- `use_bf16`: **True**
- `gradient_accumulation_steps`: **8**

### 3. 実行例

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_hf_dataset.py \
  --dataset-id "llm-jp/databricks-dolly-15k-ja" \
  --model-size megabyte_100mb \
  --batch-size 1 \
  --max-seq-len 512 \
  --gradient-accumulation-steps 8 \
  --use-bf16 \
  --gradient-checkpointing \
  --epochs 3 \
  --split train
```

### 4. パラメータの説明

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `max_seq_len` | 512 | シーケンス最大長。Attention由来のメモリは seq_len^2 に比例 |
| `batch_size` | 1 | バッチサイズ。VRAMを最小化 |
| `gradient_accumulation_steps` | 8 | 勾配蓄積ステップ。VRAMを増やさずに実効バッチサイズを増加 |
| `use_bf16` | True | BF16混合精度学習でメモリ使用量を削減 |
| `gradient_checkpointing` | True | 勾配チェックポイント（再計算トレードオフ）でメモリを削減 |

### 5. メモリ計算

100MBモデル（megabyte_100mb）の推定メモリ使用量：

```
前: max_seq_len=16384
  Attention活性化: ~100 MB × (16384²/512²) = ~100 MB × 1024 = ~100 GB
  合計: ~170 GB

後: max_seq_len=512
  Attention活性化: ~100 MB × 1 = ~100 MB
  合計: ~5 GB
```

→ **B200（178 GB VRAM）で余裕を持って学習可能**

## トラブルシューティング

### OOM エラーが発生する場合

1. **`max_seq_len` をさらに削減**
   ```bash
   --max-seq-len 256
   ```

2. **バッチサイズをさらに削減**
   ```bash
   --batch-size 1  # (既に最小値)
   ```

3. **モデルサイズを削減**
   ```bash
   --model-size small  # (100MBより小さいモデル)
   ```

### 学習が遅い場合

実効バッチサイズを増やす（VRAMを増やさない方法）：

```bash
--gradient-accumulation-steps 16  # 8から16に増加
```

## 関連ファイル

- `neuroquantum_layered.py`: `get_model_config_by_size()` 関数でモデル設定を定義
- `train_hf_dataset.py`: トレーニングスクリプト
  - 環境変数設定による自動メモリ最適化
  - BF16混合精度学習対応
  - 勾配蓄積対応

## 参考資料

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
