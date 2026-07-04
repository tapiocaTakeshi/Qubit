# NeuroQuantum 統合学習ガイド

複数の日本語データセットでNeuroQuantumモデルを統合学習するための完全なガイドです。

## セットアップ

### 1. 必要なパッケージのインストール

```bash
# 基本パッケージ
pip install -r requirements_training.txt

# または個別にインストール
pip install torch torchvision torchaudio
pip install datasets sentencepiece transformers
pip install huggingface-hub
```

### 2. HuggingFace Hubへのアップロード（オプション）

HuggingFace Hubに学習済みモデルをアップロードする場合、認証が必要です：

```bash
huggingface-cli login
# または環境変数で設定
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

## 学習の実行

### 基本的な使用方法

```bash
# シンプルな実行（小規模モデル）
python train_all_japanese_datasets.py --model-size small

# 中規模モデル
python train_all_japanese_datasets.py --model-size medium

# 大規模モデル
python train_all_japanese_datasets.py --model-size large
```

### 詳細なオプション

```bash
# サンプル数を制限（テスト用）
python train_all_japanese_datasets.py \
  --model-size small \
  --max-samples 5000 \
  --epochs 1

# ウォームアップスケジューラーを有効化
python train_all_japanese_datasets.py \
  --model-size medium \
  --enable-warmup \
  --warmup-steps 1000

# 勾配クリッピングのカスタマイズ
python train_all_japanese_datasets.py \
  --model-size medium \
  --max-grad-norm 1.5 \
  --gradient-accumulation-steps 16

# HuggingFace Hubにアップロード
HF_TOKEN=hf_xxx python train_all_japanese_datasets.py \
  --model-size large \
  --upload \
  --repo-id tapiocatakeshi/Qubit

# 特定のデータセットだけ学習
python train_all_japanese_datasets.py \
  --model-size small \
  --dataset-filter wikipedia

# 既存チェックポイントから再開
python train_all_japanese_datasets.py \
  --model-size small \
  --resume \
  --ckpt-name neuroq_small_checkpoint.pt
```

## 学習対象データセット

自動的に以下の日本語データセットから学習します（エポック数1）：

| # | データセット | 説明 | サンプル数 |
|---|--|--|--|
| 1 | llm-jp/databricks-dolly-15k-ja | 日本語指示データセット | 15K |
| 2 | wikimedia/wikipedia (ja) | Wikipedia日本語版 | 数百万 |
| 3 | mc4 (ja) | MC4日本語コーパス | 数百万 |
| 4 | oscar-corpus/OSCAR-2301 (ja) | OSCAR 2301日本語版 | 数十億 |

### データセットのフィルタリング

```bash
# Wikipediaのみ
--dataset-filter wikipedia

# Dollyのみ
--dataset-filter dolly

# MCからのみ
--dataset-filter mc4
```

## モデルサイズの選択

| サイズ | embed_dim | hidden_dim | num_layers | 推奨GPU |
|--------|-----------|-----------|-----------|--------|
| small | 128 | 256 | 4 | 4GB+ |
| medium | 256 | 512 | 4 | 8GB+ |
| large | 512 | 1024 | 6 | 16GB+ |
| xlarge | 768 | 2048 | 12 | 24GB+ |

## トレーニング設定

### バッチサイズと勾配蓄積

```bash
# メモリが限られている場合
python train_all_japanese_datasets.py \
  --model-size small \
  --batch-size 2 \
  --gradient-accumulation-steps 16

# 十分なメモリがある場合
python train_all_japanese_datasets.py \
  --model-size large \
  --batch-size 8 \
  --gradient-accumulation-steps 4
```

### 学習率の調整

```bash
# 低い学習率（より安定的）
python train_all_japanese_datasets.py \
  --model-size medium \
  --lr 1e-4

# 高い学習率（より高速）
python train_all_japanese_datasets.py \
  --model-size medium \
  --lr 1e-3
```

## チェックポイント管理

学習中に自動的にチェックポイントが保存されます：

```
checkpoints/
├── neuroq_small_checkpoint_epoch000_batch000100.pt
├── neuroq_small_checkpoint_epoch000_batch000200.pt
└── ...
neuroq_small_checkpoint.pt  # メインチェックポイント
neuroq_small_tokenizer.model  # トークナイザー
```

### チェックポイントからの再開

```bash
python train_all_japanese_datasets.py \
  --model-size small \
  --resume \
  --ckpt-name neuroq_small_checkpoint.pt
```

## トラブルシューティング

### メモリ不足エラー

```bash
# バッチサイズを削減
--batch-size 1

# 勾配蓄積ステップを増加
--gradient-accumulation-steps 32

# より小さいモデルを使用
--model-size small
```

### 遅い学習

```bash
# 勾配蓄積ステップを削減
--gradient-accumulation-steps 4

# より大きいバッチサイズ
--batch-size 8

# ウォームアップを有効化（実際には高速化）
--enable-warmup --warmup-steps 500
```

### データセット読み込みエラー

```bash
# 特定のデータセットをフィルタ
--dataset-filter wikipedia

# サンプル数を制限
--max-samples 10000
```

## パフォーマンス最適化

### 推奨設定（バランス型）

```bash
python train_all_japanese_datasets.py \
  --model-size medium \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --enable-warmup \
  --warmup-steps 1000 \
  --max-grad-norm 1.0 \
  --use-bf16 \
  --save-every 100
```

### 推奨設定（高速型）

```bash
python train_all_japanese_datasets.py \
  --model-size small \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --lr 1e-3 \
  --use-bf16 \
  --save-every 50
```

### 推奨設定（メモリ効率型）

```bash
python train_all_japanese_datasets.py \
  --model-size small \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --gradient-checkpointing \
  --use-bf16 \
  --save-every 200
```

## 出力ファイル

学習完了後、以下のファイルが生成されます：

```
neuroq_small_checkpoint.pt      # モデルチェックポイント
neuroq_small_tokenizer.model    # トークナイザー
neuroq_small_tokenizer.vocab    # 語彙ファイル
training_log.json               # トレーニングログ
```

## HuggingFace Hubへのアップロード

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxx python train_all_japanese_datasets.py \
  --model-size medium \
  --upload \
  --repo-id <your-username>/neuroquantum-ja
```

アップロード後、以下のURLでモデルにアクセスできます：
```
https://huggingface.co/<your-username>/neuroquantum-ja
```

## ログ出力の読み方

```
=== Loading ... ===
Dataset size: 10000 / using ALL 10000 samples

=== トークナイザー構築 ===
Vocabulary size: 32000

=== 学習開始 ===
Epoch 1/1, Batch 25, Loss: 4.5234, LR: 5.00e-04
Epoch 1/1, Batch 50, Loss: 4.1234, LR: 5.02e-04
...
```

## 参考資料

- [neuroquantum_layered.py](./neuroquantum_layered.py) - NeuroQuantumモデルの実装
- [train_hf_dataset.py](./train_hf_dataset.py) - 単一データセット学習スクリプト
- [train_all_japanese_datasets.py](./train_all_japanese_datasets.py) - 統合学習スクリプト
