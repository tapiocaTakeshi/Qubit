# NeuroQuantum Layered GPU Training Guide
## Kaggle日本語データセットでの学習

このガイドでは、Kaggleの複数の日本語データセットを使用して、`neuroquantum_layered.py`をGPUで学習する方法を説明します。

## 対象データセット

以下のKaggleデータセットを使用します：

1. **wikimedia/wikipedia** [20231101.ja]
   - 日本語Wikipedia記事
   - 高品質な百科事典テキスト

2. **mc4** [ja]
   - Common Crawlの日本語テキスト
   - ウェブテキストの多様性

3. **oscar-corpus/OSCAR-2301** [ja]
   - 日本語の大規模テキストコーパス
   - 自然言語テキストの多様性

4. **globis-university/aozorabunko-clean**
   - 青空文庫（日本の古典文学）
   - 文学的なテキスト

5. **shunk031/JGLUE** [JNLI, JCommonsenseQA, JaQuAD]
   - 日本語言語理解ベンチマーク
   - QAとテキスト推論データ

## 前提条件

### ハードウェア要件

- **GPU**: NVIDIA CUDA対応GPU（推奨: 10GB以上のVRAM）
- **システムRAM**: 最低64GB（推奨: 128GB以上）
- **ストレージ**: 最低500GB（データセット + モデル）

### ソフトウェア要件

```bash
- Python 3.8+
- CUDA 11.8+ (GPU使用時)
- PyTorch 2.0+
- Kaggle CLI
```

## セットアップ

### Step 1: Kaggle認証情報の準備

1. Kaggleアカウントにログイン: https://www.kaggle.com/settings/account
2. "API"セクションまでスクロール
3. "Create New API Token"をクリック
4. ダウンロードされた`kaggle.json`を`~/.kaggle/`に配置

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: 環境セットアップ

```bash
# セットアップスクリプトを実行
bash setup_kaggle_training.sh

# または手動でインストール
pip install kaggle torch sentencepiece tqdm numpy psutil
```

## 使用方法

### シンプルな訓練

```bash
# 基本的な実行
python train_kaggle_japanese_datasets.py

# または高度な機能を使用
python train_kaggle_advanced.py
```

### オプション付き実行

#### 基本スクリプト

```bash
python train_kaggle_japanese_datasets.py \
    --datasets-dir ./kaggle_datasets \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --max-samples 50000 \
    --model-size auto
```

#### 高度なスクリプト

```bash
python train_kaggle_advanced.py \
    --datasets-dir ./kaggle_datasets \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --max-samples 50000 \
    --gradient-accumulation-steps 2 \
    --use-amp \
    --seed 42
```

### パラメータ説明

| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `--datasets-dir` | `./kaggle_datasets` | データセット保存ディレクトリ |
| `--batch-size` | auto | バッチサイズ（自動検出） |
| `--num-epochs` | 3 | 訓練エポック数 |
| `--learning-rate` | 1e-4 | 学習率 |
| `--max-samples` | 50000 | 読み込む最大サンプル数 |
| `--model-size` | auto | モデルサイズ（auto/large/medium/small） |
| `--skip-download` | - | ダウンロードをスキップ |
| `--gradient-accumulation-steps` | 1 | グラデーション累積ステップ |
| `--use-amp` | True | 混合精度訓練 |
| `--seed` | 42 | ランダムシード |

## GPU別の推奨設定

### A100 (40GB VRAM)

```bash
python train_kaggle_advanced.py \
    --batch-size 8 \
    --num-epochs 5 \
    --learning-rate 1e-4 \
    --max-samples 100000
```

### V100 (32GB VRAM)

```bash
python train_kaggle_advanced.py \
    --batch-size 6 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --max-samples 50000
```

### RTX 4090 (24GB VRAM)

```bash
python train_kaggle_advanced.py \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --max-samples 50000 \
    --gradient-accumulation-steps 2
```

### A6000 (48GB VRAM)

```bash
python train_kaggle_advanced.py \
    --batch-size 8 \
    --num-epochs 5 \
    --learning-rate 1e-4 \
    --max-samples 100000 \
    --gradient-accumulation-steps 1
```

## 訓練の流れ

### 自動実行フロー

1. **GPU検出**: GPU性能に基づいてモデル設定を自動調整
2. **データセット準備**: Kaggle APIからダウンロード（初回のみ）
3. **前処理**: テキストデータのトークン化と正規化
4. **モデル初期化**: GPU対応コンフィグでNeuroQuantumを初期化
5. **訓練ループ**: 指定エポック数まで訓練
6. **チェックポイント保存**: 各エポック後にモデルを保存

### 出力ファイル

```
./
├── kaggle_datasets/          # ダウンロードされたデータセット
│   ├── wikipedia/
│   ├── mc4/
│   ├── oscar/
│   ├── aozorabunko/
│   └── jglue/
├── checkpoints/              # 訓練済みモデル
│   ├── model_best.pth
│   ├── checkpoint_epoch0.pth
│   ├── checkpoint_epoch1.pth
│   └── checkpoint_epoch2.pth
├── logs/
│   └── training.log         # 訓練ログ
└── training.log             # 詳細ログ
```

## 訓練の監視

### ログの確認

```bash
# リアルタイムログ表示
tail -f training.log

# 全ログ確認
cat training.log

# GPU使用率の監視
nvidia-smi -l 1  # 1秒ごとに更新
```

### チェックポイントの確認

```bash
# 最新のチェックポイント
ls -lh checkpoints/

# モデルサイズ
du -sh checkpoints/model_best.pth
```

## 訓練の再開

### チェックポイントから再開

```bash
python train_kaggle_advanced.py \
    --resume-from checkpoints/checkpoint_epoch0.pth \
    --num-epochs 5
```

### カスタム設定で再開

```bash
python train_kaggle_advanced.py \
    --resume-from checkpoints/model_best.pth \
    --num-epochs 10 \
    --learning-rate 5e-5 \
    --batch-size 2
```

## トラブルシューティング

### メモリ不足エラー

```
RuntimeError: CUDA out of memory
```

**解決策:**
1. バッチサイズを削減
2. グラデーション累積を増加
3. 最大サンプル数を削減

```bash
python train_kaggle_advanced.py \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-samples 25000
```

### Kaggle認証エラー

```
401 Unauthorized
```

**解決策:**
```bash
# Kaggle認証情報の確認
cat ~/.kaggle/kaggle.json

# 再認証
kaggle auth --username YOUR_USERNAME
```

### データセット ダウンロード エラー

```
403 Access Denied
```

**解決策:**
1. Kaggleアカウントでデータセットに同意
2. Kaggle CLI認証情報を更新
3. `--skip-download` フラグで既存データを使用

## パフォーマンス最適化

### 混合精度訓練（デフォルト）

```bash
# AMP有効（デフォルト）
python train_kaggle_advanced.py --use-amp
```

### グラデーション累積

少ないVRAMでも大きなバッチサイズを実現：

```bash
python train_kaggle_advanced.py \
    --batch-size 2 \
    --gradient-accumulation-steps 4
# 実質的なバッチサイズ = 2 * 4 = 8
```

### マルチプロセッシング

```bash
# より多くのワーカーでデータ読み込みを並列化
# スクリプト内の num_workers を増加
```

## モデルの用途

訓練済みモデルは以下に使用できます：

1. **日本語テキスト生成**
2. **要約生成**
3. **質問回答**
4. **テキスト分類**
5. **言語理解タスク**

## 例: モデルの推論

```python
import torch
from neuroquantum_layered import NeuroQuantum

# 訓練済みモデルを読み込み
checkpoint = torch.load('checkpoints/model_best.pth')
model = NeuroQuantum(config_dict=checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推論
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# テキスト入力をトークン化
input_ids = torch.tensor([[...]], device=device)

# 推論実行
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs

# 次のトークンの確率分布
probs = torch.softmax(logits[:, -1, :], dim=-1)
```

## ベストプラクティス

1. **段階的な学習率削減**
   - 最初の1エポック: 1e-4
   - その後: 5e-5

2. **定期的なチェックポイント保存**
   - デフォルトで100バッチごと

3. **ログの確認**
   - 損失が安定して減少しているか確認

4. **メモリ効率**
   - GPUで利用可能なVRAM量に応じて設定を調整

## 参考資料

- [Kaggle API ドキュメント](https://github.com/Kaggle/kaggle-api)
- [PyTorch公式ドキュメント](https://pytorch.org/)
- [混合精度訓練](https://pytorch.org/docs/stable/amp.html)
- [勾配累積](https://pytorch.org/docs/stable/notes/amp_examples.html)

## ライセンス

このコードはMITライセンスの下で公開されています。

## サポート

問題が発生した場合は、以下を確認してください：

1. Python/PyTorchのバージョン
2. GPU ドライバのバージョン
3. Kaggle認証情報
4. ディスク容量（最低500GB必要）

---

**Last Updated**: 2024-07
