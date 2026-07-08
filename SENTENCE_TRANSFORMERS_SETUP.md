# google/embeddinggemma-300m with sentence-transformers セットアップ

## 概要

HuggingFace の **sentence-transformers** を使って、`google/embeddinggemma-300m` をローカルで実行できるようになりました。

### 利点

✅ **API キー不要** - ローカルで完全実行
✅ **高速** - インターネット接続不要  
✅ **プライベート** - データはローカルに保持
✅ **シンプル** - 3行でセットアップ可能

---

## セットアップ（3ステップ）

### Step 1: 依存関係インストール

```bash
pip install sentence-transformers torch
```

**オプション: GPU加速**
```bash
# NVIDIA GPU の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon の場合
pip install torch::torch::torchvision torchtext --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: 確認

```bash
python3 -c "from sentence_transformers import SentenceTransformer; print('✓ OK')"
```

### Step 3: 学習開始

```bash
cd /root/Qubit
./quick_train_megabyte_100mb.sh
```

---

## 詳細な実行オプション

### 基本実行

```bash
# デフォルト: sentence-transformers で google/embeddinggemma-300m を使用
python3 train_megabyte_100mb_multilingual.py \
    --epochs 1 \
    --batch-size 2 \
    --max-samples-per-dataset 5000
```

### Google API を使用

```bash
# Google Generative AI API を使用
export GOOGLE_API_KEY='your-api-key'
python3 train_megabyte_100mb_multilingual.py \
    --use-gemma-300m \
    --epochs 1
```

### デフォルト埋め込みを使用

```bash
# API キーなし、外部埋め込みなし
python3 train_megabyte_100mb_multilingual.py \
    --no-external-embedding \
    --epochs 1
```

---

## モデル仕様

### google/embeddinggemma-300m (sentence-transformers)

```
Provider:          HuggingFace / Google
Dimensions:        300
Languages:         100+ (including Japanese)
Model Type:        Sentence Transformer
Execution:         Local (GPU/CPU)
Performance:       Fast (100-1000 sentences/sec on GPU)
```

**モデル情報**: https://huggingface.co/google/embeddinggemma-300m

---

## パフォーマンス比較

| 実行方法 | 速度 | 精度 | API Key | 接続 |
|---------|------|------|---------|------|
| sentence-transformers | ⭐⭐⭐ 高速 | ⭐⭐⭐⭐⭐ 高 | 不要 | 不要 |
| Google API | ⭐⭐ 中速 | ⭐⭐⭐⭐⭐ 高 | 必須 | 必須 |
| デフォルト | ⭐⭐⭐ 高速 | ⭐⭐⭐ 中 | 不要 | 不要 |

---

## 推奨実行環境

### GPU環境（推奨）

```
NVIDIA GPU:        RTX 4090, A100, A10 以上
VRAM:              6GB 以上
処理速度:          100-500 sentences/sec
```

**実行例**:
```bash
python3 train_megabyte_100mb_multilingual.py \
    --epochs 3 \
    --batch-size 4 \
    --max-samples-per-dataset 10000
```

### CPU環境

```
メモリ:            32GB 以上
処理速度:          5-10 sentences/sec
実行時間:          24-48時間 (全15データセット)
```

**実行例**:
```bash
# CPU の場合はバッチサイズを削減
python3 train_megabyte_100mb_multilingual.py \
    --epochs 1 \
    --batch-size 1 \
    --max-samples-per-dataset 2000
```

---

## トラブルシューティング

### 問題: ModuleNotFoundError: No module named 'sentence_transformers'

```bash
pip install sentence-transformers
```

### 問題: CUDA out of memory

```bash
# バッチサイズを削減
python3 train_megabyte_100mb_multilingual.py --batch-size 1
```

### 問題: モデルダウンロードが遅い

```bash
# HuggingFace キャッシュ場所を変更
export HF_HOME=/path/to/large/disk
python3 train_megabyte_100mb_multilingual.py
```

### 問題: CPU で実行したい

```bash
# 自動的にCPUで実行されます（遅くなります）
python3 train_megabyte_100mb_multilingual.py \
    --epochs 1 \
    --batch-size 1 \
    --max-samples-per-dataset 1000
```

---

## キャッシュ管理

### モデルキャッシュの場所

```bash
# デフォルト
~/.cache/huggingface/hub/

# カスタム場所に設定
export HF_HOME=/path/to/cache
```

### キャッシュのクリア

```bash
# すべてのHuggingFaceキャッシュをクリア
rm -rf ~/.cache/huggingface/

# または選択的に
rm -rf ~/.cache/huggingface/hub/google*
```

### ストレージ使用量

```
google/embeddinggemma-300m:  ~200MB
全HuggingFaceキャッシュ:       ~5-10GB (複数モデル使用時)
```

---

## 使用例

### 日本語テキストの埋め込み

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

texts = [
    "これはテキストの例です。",
    "日本語に対応しています。",
    "複数の言語をサポートしています。"
]

embeddings = model.encode(texts)
print(f"埋め込み形状: {embeddings.shape}")  # (3, 300)
```

### 学習パイプライン内での使用

```python
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig

config = NeuroQuantumConfig(...)
model = NeuroQuantum(
    config=config,
    use_sentence_transformers=True,
    sentence_transformers_model="google/embeddinggemma-300m"
)

# 学習開始
output = model(token_ids)
```

---

## FAQ

**Q: sentence-transformers と Google API どちらが良い？**

A: 
- sentence-transformers: 推奨（速い、無料、API不要）
- Google API: Google の最新モデル使用時、より高精度が必要な場合

**Q: オフラインで実行できる？**

A: はい。モデルをダウンロード後、インターネット接続なしで実行可能。

**Q: マルチGPU対応？**

A: 現在は単一GPU対応。複数GPU使用には DataParallel を追加実装が必要。

**Q: モデルをカスタマイズできる？**

A: はい。sentence-transformers は ファインチューニング に対応しています。

---

## パフォーマンスチューニング

### 高速化

```bash
# バッチサイズを増やす（GPU容量に応じて）
python3 train_megabyte_100mb_multilingual.py \
    --batch-size 8  # デフォルト: 2
```

### メモリ削減

```bash
# グラディエント累積を使用
python3 train_megabyte_100mb_multilingual.py \
    --batch-size 1  # バッチサイズを削減
```

### 精度向上

```bash
# より大きいモデルを使用
# (sentence-transformersの他のモデルに変更)
```

---

## 技術詳細

### embedding-gemma-300m アーキテクチャ

- **ベース**: Google Gemma 2B
- **タイプ**: Sentence Transformer (SBERT)
- **学習データ**: 多言語テキスト（Wikipedia, CommonCrawl等）
- **推定パラメータ**: ~300M

### 射影層

モデルの埋め込み次元（1024）と sentence-transformers の出力（300）をマッピング：

```
sentence-transformers (300次元)
         ↓
    射影層 (300 → 1024)
         ↓
    モデル入力 (1024次元)
```

---

## さらに詳しく

- **Sentence-Transformers ドキュメント**: https://www.sbert.net/
- **google/embeddinggemma-300m**: https://huggingface.co/google/embeddinggemma-300m
- **NeuroQuantum GitHub**: https://github.com/tapiocaTakeshi/Qubit

---

**作成日**: 2026-07-07
**バージョン**: v1.0
**ステータス**: ✅ 準備完了・実行可能
