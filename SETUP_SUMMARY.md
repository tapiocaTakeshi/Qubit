# megabyte_100mb + embedding-gemma-300m セットアップ完了

## 実施内容

### 1. ✅ neuroquantum_layered.py の修正

**変更内容**: `GoogleEmbeddingWrapper` クラスにおいて、デフォルトモデルを `models/embedding-gemma-300m` に更新しました。

- **場所**: `/root/Qubit/neuroquantum_layered.py` (行1160)
- **変更**:
  ```python
  # Before: model: str = "models/text-embedding-004"
  # After:  model: str = "models/embedding-gemma-300m"
  ```
- **特徴**:
  - 日本語を含む100以上の言語対応
  - 300次元のコンパクトな埋め込み
  - 自動次元射影により、モデルの埋め込み次元との不一致に対応

---

### 2. ✅ 多言語・多分野学習スクリプト作成

**ファイル**: `/root/Qubit/train_megabyte_100mb_multilingual.py`

#### 対応データセット（15種類）

```
1.  wikimedia/wikipedia          → 日本語Wikipedia
2.  google/wiki40b              → 40言語Wikipedia
3.  allenai/c4                  → CommonCrawl (英語)
4.  mc4                         → 多言語CommonCrawl
5.  Open-Orca/OpenOrca          → 指示フォロー (655K)
6.  HuggingFaceH4/ultrachat_200k → チャット (200K)
7.  open-thoughts/OpenThoughts-114k → 思考過程 (114K)
8.  argilla/ultrafeedback-binarized → フィードバック (178K)
9.  bigcode/the-stack-v2        → プログラミングコード
10. ise-uiuc/Magicoder-OSS-Instruct → コード指示 (75K)
11. WizardLMTeam/WizardCoder-Python → Pythonコード (34K)
12. HuggingFaceH4/CodeAlpaca_20K → コードアルパカ (20K)
13. meta-math/MetaMathQA        → 数学問題 (395K)
14. AI-MO/NuminaMath-CoT        → 数学推論 (765K)
15. gsm8k                       → 数学応用問題 (8.8K)
```

#### 主な機能

- ✅ embedding-gemma-300m を使用したテキストエンベディング
- ✅ 複数データセットの自動ロード
- ✅ 柔軟なテキスト抽出（対話形式・テキスト形式に対応）
- ✅ 自動トークン化とシーケンス構築
- ✅ エポックごとのチェックポイント保存
- ✅ GPU/CPU 両対応
- ✅ bfloat16 自動キャスト対応

---

### 3. ✅ クイックスタートシェルスクリプト

**ファイル**: `/root/Qubit/quick_train_megabyte_100mb.sh`

```bash
# 基本的な実行
./quick_train_megabyte_100mb.sh

# パラメータ指定
EPOCHS=3 BATCH_SIZE=4 ./quick_train_megabyte_100mb.sh
```

#### 機能

- 依存関係の自動確認
- Google API Key チェック
- 環境変数による柔軟な設定
- 見やすいログ出力

---

### 4. ✅ 包括的なドキュメント

**ファイル**: `/root/Qubit/MEGABYTE_100MB_TRAINING_GUIDE.md`

**内容**:
- モデル仕様の詳細説明
- セットアップ手順（Google API Key 取得・依存関係インストール）
- 3つの実行方法
- リソース要件と推定学習時間
- トラブルシューティング
- 高度な設定例

---

### 5. ✅ 設定テンプレート

**ファイル**: `/root/Qubit/training_config.json`

訓練パラメータ、データセット設定、最適化オプションを一元管理できるJSON形式の設定ファイル。

---

## クイックスタート

### Step 1: Google API Key 取得

```bash
# https://makersuite.google.com/app/apikey から取得

export GOOGLE_API_KEY='your-api-key-here'
```

### Step 2: 依存関係のインストール

```bash
pip install google-generativeai datasets torch sentencepiece
```

### Step 3: 学習開始

**方法A（推奨）:**
```bash
cd /root/Qubit
./quick_train_megabyte_100mb.sh
```

**方法B（Pythonスクリプト直接）:**
```bash
python3 train_megabyte_100mb_multilingual.py \
    --epochs 1 \
    --batch-size 2 \
    --max-samples-per-dataset 5000
```

---

## モデル仕様

```
Name:                megabyte_100mb
Architecture:        NeuroQuantum (QBNN-based)
Embedding Model:     embedding-gemma-300m
Total Parameters:    ~100M
─────────────────────────────
Embedding Dim:       1024 (projection from 300)
Hidden Dim:          2048
Attention Heads:     16
Num Layers:          10
Max Seq Length:      512
Vocab Size:          32000
Batch Size:          1-4
─────────────────────────────
FP32 Size:           ~400MB
FP16/BF16 Size:      ~200MB
GPU Memory (bs=2):   ~4-5GB
```

---

## 学習パイプライン

```
データセット読込
    ↓
テキスト抽出（形式自動検出）
    ↓
トークン化（BOS/EOS/BOF/EOF マーカー付与）
    ↓
embedding-gemma-300m で埋め込み生成
    ↓
モデルフォーワードパス
    ↓
Cross-Entropy Loss 計算
    ↓
勾配クリップ & バックプロパゲーション
    ↓
AdamW オプティマイザー更新
    ↓
チェックポイント保存
```

---

## ファイル構成

```
/root/Qubit/
├── neuroquantum_layered.py          ← 修正済み（embedding-gemma-300m対応）
├── train_megabyte_100mb_multilingual.py  ← 新規作成
├── quick_train_megabyte_100mb.sh         ← 新規作成
├── MEGABYTE_100MB_TRAINING_GUIDE.md      ← 新規作成
├── training_config.json                  ← 新規作成
├── SETUP_SUMMARY.md                      ← このファイル
└── train_hf_dataset.py          ← 既存（単一データセット学習）
```

---

## 推奨実行条件

### GPU環境（推奨）
```
- NVIDIA A100 (80GB):  最適 (バッチサイズ 8+)
- NVIDIA A10 (24GB):   最適 (バッチサイズ 2-4)
- NVIDIA RTX 4090:     良好 (バッチサイズ 2)
- NVIDIA RTX 3090:     可能 (バッチサイズ 1)
```

### CPU環境
```
- RAM 64GB以上:  可能 (非常に遅い)
- バッチサイズ: 1
- 学習時間:     数日～数週間
```

---

## 主な特徴

### 1. embedding-gemma-300m を使用
- ✅ 日本語対応
- ✅ 多言語対応（100+言語）
- ✅ メモリ効率的（300次元）
- ✅ Google Generative AI API

### 2. 多分野学習
- ✅ テキスト (Wikipedia)
- ✅ 指示フォロー (OpenOrca)
- ✅ チャット対話 (UltraChat)
- ✅ プログラミングコード (TheStack, Magicoder)
- ✅ 数学推論 (MetaMath, NuminaMath)

### 3. 自動形式検出
- 対話形式の自動抽出
- テキスト列の柔軟な検出
- 複数の列名形式に対応

### 4. 効率的な学習
- Gradient clipping
- Layer normalization
- Residual connections
- bfloat16 自動キャスト

---

## 注意事項

### 1. Google API Key の必須性

```bash
# embedding-gemma-300m を使用する場合は必須
export GOOGLE_API_KEY='your-key'

# 設定しない場合は、デフォルト埋め込みを使用
python3 train_megabyte_100mb_multilingual.py --no-gemma-300m
```

### 2. ネットワーク接続

- 初回実行時にデータセットをダウンロード（数GBの帯域幅が必要）
- Google API呼び出し（インターネット接続が必要）

### 3. ディスク容量

```
- チェックポイント保存: ~400MB × エポック数
- キャッシュ: ~10GB (Hugging Face datasets)
- 合計: ~30-50GB推奨
```

---

## トラブルシューティング

### よくあるエラー

**1. CUDA Out of Memory**
```bash
# バッチサイズを削減
BATCH_SIZE=1 ./quick_train_megabyte_100mb.sh

# またはサンプル数を削減
MAX_SAMPLES_PER_DATASET=2000 ./quick_train_megabyte_100mb.sh
```

**2. Google API Key not set**
```bash
export GOOGLE_API_KEY='your-key'

# または embedding-gemma-300m を不使用
python3 train_megabyte_100mb_multilingual.py --no-gemma-300m
```

**3. Dataset loading failed**
```bash
# Hugging Face キャッシュをクリア
rm -rf ~/.cache/huggingface/datasets

# インターネット接続を確認
```

---

## 次のステップ

1. **Google API Key を取得**
   ```bash
   # https://makersuite.google.com/app/apikey
   ```

2. **依存関係をインストール**
   ```bash
   pip install google-generativeai datasets
   ```

3. **学習を開始**
   ```bash
   cd /root/Qubit
   ./quick_train_megabyte_100mb.sh
   ```

4. **結果を監視**
   ```bash
   # チェックポイントが ./checkpoints に保存される
   ls -lh ./checkpoints/
   ```

---

## 技術仕様

### embedding-gemma-300m

**提供者**: Google AI (Gemini)

**特徴**:
- **次元数**: 300
- **言語**: 100+言語（日本語含む）
- **タスク型**: RETRIEVAL_DOCUMENT (デフォルト)
- **API**: Google Generative AI

**射影層**:
- embedding-gemma-300m (300次元) → モデル埋め込み層 (1024次元)
- 自動的に線形射影層が追加される
- パラメータ数: 300 × 1024 ≈ 307K

### NeuroQuantum Architecture

**QBNN (Quantum-Bit Neural Network)**:
- Entanglement テンソル J による補正
- 学習可能な λ (もつれ強度)
- Attention-Free デザイン (FFNのみ)

**最適化**:
- AdamW (weight_decay=0.01)
- Gradient Clipping (norm=1.0)
- Layer Normalization
- Residual Connections

---

## 今後の改善予定

- [ ] チェックポイントから再開機能
- [ ] 分散学習対応
- [ ] WandB統合
- [ ] TensorBoard ログ
- [ ] より多くのエンベディングモデル対応
- [ ] 日本語特化チューニング

---

## サポート・フィードバック

- **GitHub**: https://github.com/tapiocaTakeshi/Qubit
- **Email**: higuchiyuya.riddle@gmail.com
- **Issues**: https://github.com/tapiocaTakeshi/Qubit/issues

---

**作成日**: 2026-07-07
**ステータス**: ✅ セットアップ完了・実行可能
**バージョン**: NeuroQuantum megabyte_100mb v1.0
