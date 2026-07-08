# megabyte_100mb モデル 多言語学習ガイド

## 概要

このガイドでは、**embedding-gemma-300m** (日本語対応テキストエンベディング) を使用して、**megabyte_100mb** NeuroQuantumモデルを複数の大規模データセットで学習する方法を説明します。

### モデル仕様

```
Model: megabyte_100mb
─────────────────────────────
Embedding Dimension:  1024
Hidden Dimension:     2048
Number of Heads:      16
Number of Layers:     10
Max Sequence Length:  512
Batch Size:           1-4 (推奨)
Estimated Parameters: ~100M
```

### 学習データセット（15種類）

| # | Dataset | 説明 | 規模 |
|---|---------|------|------|
| 1 | wikimedia/wikipedia | 日本語Wikipedia | 数百万ページ |
| 2 | google/wiki40b | 40言語Wikipedia | 数百万ページ |
| 3 | allenai/c4 | CommonCrawl (英語) | 数十億トークン |
| 4 | mc4 | 多言語CommonCrawl | 数十億トークン |
| 5 | Open-Orca/OpenOrca | 指示フォロー | 655K例 |
| 6 | HuggingFaceH4/ultrachat_200k | チャット対話 | 200K例 |
| 7 | open-thoughts/OpenThoughts-114k | 思考過程 | 114K例 |
| 8 | argilla/ultrafeedback-binarized | 品質フィードバック | 178K例 |
| 9 | bigcode/the-stack-v2 | プログラミングコード | 数百億トークン |
| 10 | ise-uiuc/Magicoder-OSS-Instruct | コード指示 | 75K例 |
| 11 | WizardLMTeam/WizardCoder-Python | Pythonコード | 34K例 |
| 12 | HuggingFaceH4/CodeAlpaca_20K | コードアルパカ | 20K例 |
| 13 | meta-math/MetaMathQA | 数学問題 | 395K例 |
| 14 | AI-MO/NuminaMath-CoT | 数学推論 | 765K例 |
| 15 | gsm8k | 数学応用問題 | 8.8K例 |

### 埋め込みモデル: embedding-gemma-300m

Google提供の **embedding-gemma-300m** は以下の特徴があります：

- **言語対応**: 日本語を含む100以上の言語
- **次元数**: 300次元 (メモリ効率的)
- **専門性**: テキスト検索・分類に最適化
- **APIベース**: Google Generative AI APIを使用

**note**: より高度な機能にはGoogle API keyが必要です。

---

## セットアップ手順

### 1. Google API Key の取得

embedding-gemma-300m を使用するには Google API key が必要です。

```bash
# Google AI Studio から API key を取得
# https://makersuite.google.com/app/apikey

# 環境変数に設定
export GOOGLE_API_KEY='your-api-key-here'

# 確認
echo $GOOGLE_API_KEY
```

**APIキーの設定方法:**

```bash
# Option A: 一時的に設定（現在のセッションのみ）
export GOOGLE_API_KEY='sk-...'

# Option B: 永続的に設定
echo "export GOOGLE_API_KEY='sk-...'" >> ~/.bashrc
source ~/.bashrc

# Option C: .envファイル経由（推奨）
echo "GOOGLE_API_KEY='sk-...'" > .env
source .env
```

### 2. 依存関係のインストール

```bash
# PyTorch (GPU推奨)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Google Generative AI
pip install google-generativeai>=0.3.0

# Hugging Face Datasets
pip install datasets>=2.14.0

# その他
pip install sentencepiece tqdm
```

### 3. PyTorch確認

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 学習実行

### 方法A: シェルスクリプト（推奨）

```bash
# 基本的な実行
./quick_train_megabyte_100mb.sh

# パラメータ指定実行
EPOCHS=3 BATCH_SIZE=4 MAX_SAMPLES_PER_DATASET=10000 ./quick_train_megabyte_100mb.sh
```

**環境変数:**

```bash
EPOCHS=1                    # 学習エポック数
BATCH_SIZE=2                # バッチサイズ（GPU容量に応じて調整）
MAX_SAMPLES_PER_DATASET=5000 # データセット当たりの最大サンプル数
SAVE_DIR=./checkpoints      # チェックポイント保存先
```

### 方法B: Pythonスクリプト直接実行

```bash
python3 train_megabyte_100mb_multilingual.py \
    --epochs 3 \
    --batch-size 4 \
    --max-samples-per-dataset 10000 \
    --save-dir ./checkpoints \
    --use-gemma-300m
```

**オプション:**

```
--epochs N                      # エポック数（デフォルト: 1）
--batch-size N                  # バッチサイズ（デフォルト: 2）
--max-samples-per-dataset N     # 1データセットの最大サンプル数（デフォルト: 5000）
--save-dir PATH                 # チェックポイント保存先（デフォルト: ./checkpoints）
--vocab-size N                  # 語彙サイズ（デフォルト: 32000）
--use-gemma-300m               # embedding-gemma-300m を使用（デフォルト: 有効）
--no-gemma-300m                # embedding-gemma-300m を不使用
```

### 方法C: train_hf_dataset.py を使用（単一データセット）

```bash
# 単一データセットで学習
python3 train_hf_dataset.py \
    --dataset-id "wikimedia/wikipedia" \
    --split "20220301.ja" \
    --model-size megabyte_100mb \
    --epochs 1 \
    --max-samples 5000
```

---

## 計算リソース要件

| GPU/Device | Recommended | Supported |
|-----------|------------|-----------|
| NVIDIA A100 (80GB) | ✅ 最適 | バッチサイズ 8+ |
| NVIDIA A10 (24GB) | ✅ 最適 | バッチサイズ 2-4 |
| NVIDIA RTX 4090 (24GB) | ✅ 良好 | バッチサイズ 2 |
| NVIDIA RTX 3090 (24GB) | ⚠️ 可能 | バッチサイズ 1-2 |
| CPU (64GB RAM以上) | ⚠️ 遅い | バッチサイズ 1 |

### 推定学習時間

- **全15データセット、100Kサンプル/セット、GPU(A100)**: ~24-48時間
- **5データセット、5Kサンプル/セット、GPU(RTX 4090)**: ~6-12時間
- **単一小規模データセット、CPU**: ~24-48時間

### メモリ使用量

```
Model Parameters: ~100M
Model Size: ~400MB (FP32) / ~200MB (FP16/BF16)
Activation Memory (batch_size=2): ~2-3GB
Optimizer State (AdamW): ~1-1.5GB
Total: ~4-5GB (GPU VRAM必要)
```

---

## 学習中の監視

### ログ出力例

```
Device: cuda
📚 Loading all datasets...

📂 Loading wikimedia/wikipedia...
  Processing 0/5000...
  Processing 1000/5000...
✓ Loaded 4800 texts from wikimedia/wikipedia
  Total sequences so far: 12000

🚀 Starting training for 3 epochs...

━━━ Epoch 1/3 ━━━
  Epoch 1, Batch 0: Loss = 8.5234
  Epoch 1, Batch 10: Loss = 7.8912
  Epoch 1, Batch 20: Loss = 7.2341
  Average Loss: 7.4521
  ✓ Checkpoint saved to ./checkpoints/megabyte_100mb_multilingual_epoch1.pt
```

### 学習を中断・再開

```bash
# Ctrl+C で安全に中断
# チェックポイントは自動保存される

# 中断したポイントから再開（現在未実装、今後対応予定）
python3 train_megabyte_100mb_multilingual.py \
    --resume \
    --checkpoint ./checkpoints/megabyte_100mb_multilingual_epoch1.pt
```

---

## トラブルシューティング

### エラー: "GOOGLE_API_KEY environment variable not set"

```bash
# APIキーを設定
export GOOGLE_API_KEY='your-key-here'

# または embedding-gemma-300m を不使用
python3 train_megabyte_100mb_multilingual.py --no-gemma-300m
```

### エラー: "CUDA out of memory"

バッチサイズを削減:

```bash
./quick_train_megabyte_100mb.sh BATCH_SIZE=1
```

またはサンプル数を削減:

```bash
./quick_train_megabyte_100mb.sh MAX_SAMPLES_PER_DATASET=2000
```

### エラー: "Failed to load dataset"

- ネットワーク接続を確認
- データセットが利用可能か確認（https://huggingface.co/datasets）
- HuggingFace CLI認証: `huggingface-cli login`

### エラー: "Google Embedding API error"

1. APIキーが正しいか確認
2. APIクォータを確認（https://makersuite.google.com/）
3. API が有効か確認

---

## 出力ファイル

学習完了後、以下のファイルが生成されます：

```
checkpoints/
├── megabyte_100mb_multilingual_epoch1.pt   # エポック1のチェックポイント
├── megabyte_100mb_multilingual_epoch2.pt   # エポック2のチェックポイント
└── megabyte_100mb_multilingual_epoch3.pt   # エポック3のチェックポイント
```

### チェックポイント使用方法

```python
import torch
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig

# モデル作成
config = NeuroQuantumConfig(...)
model = NeuroQuantum(config=config, use_google_embedding=True)

# チェックポイント読込
checkpoint = torch.load("./checkpoints/megabyte_100mb_multilingual_epoch1.pt")
model.load_state_dict(checkpoint)

# 推論
model.eval()
output = model(token_ids)
```

---

## 高度な設定

### カスタムデータセットの追加

`train_megabyte_100mb_multilingual.py` の `DATASETS` リストを編集:

```python
DATASETS = [
    ("your-dataset-id", "split-name"),
    ("wikimedia/wikipedia", "20220301.ja"),
    # ... 他のデータセット
]
```

### モデルサイズの変更

```bash
# megabyte_300mb を使用
python3 train_hf_dataset.py \
    --dataset-id "wikimedia/wikipedia" \
    --model-size megabyte_300mb \
    --epochs 1

# billion_1b を使用
python3 train_hf_dataset.py \
    --dataset-id "openwebtext" \
    --model-size billion_1b \
    --epochs 1
```

### カスタムエンベディング

embedding-gemma-300m の代わりに別のエンベディングを使用:

```python
model = NeuroQuantum(
    config=config,
    use_google_embedding=True,
    google_model="models/text-embedding-004",  # 768次元
    tokenizer=tokenizer
)
```

---

## 参考資料

- [Google Generative AI API](https://ai.google.dev/)
- [embedding-gemma-300m Documentation](https://ai.google.dev/api/embeddings)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [NeuroQuantum GitHub](https://github.com/tapiocaTakeshi/Qubit)

---

## サポート

質問や問題が発生した場合:

1. GitHub Issues: https://github.com/tapiocaTakeshi/Qubit/issues
2. Discussions: https://github.com/tapiocaTakeshi/Qubit/discussions
3. Email: higuchiyuya.riddle@gmail.com

---

**Last Updated**: 2026-07-07
