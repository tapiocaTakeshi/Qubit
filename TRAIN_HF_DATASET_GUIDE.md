# Hugging Face データセット学習ガイド

`train_hf_dataset.py` は任意の Hugging Face データセットIDを指定して、NeuroQuantum モデルを学習するユニバーサルスクリプトです。

## 主な特徴

- **自動列検出**: instruction/input/output/context/response など様々な列名形式に自動対応
- **すべてのデータで学習**: デフォルトでは指定データセットのすべてのデータを使用
- **複数形式対応**: 対話形式やテキスト形式を自動判定
- **Split 自動検出**: データセットから利用可能な split を自動検出
- **チェックポイント管理**: 学習の再開や追加学習に対応
- **HF Hub アップロード**: 学習済みモデルを HuggingFace に直接アップロード可能

## 基本的な使い方

### 1. 最もシンプルな使用方法（全データで学習）

```bash
python train_hf_dataset.py --dataset-id "llm-jp/databricks-dolly-15k-ja" --epochs 3
```

- データセット: `llm-jp/databricks-dolly-15k-ja`
- すべてのデータを使用
- 3 エポック学習
- チェックポイント: `neuroq_small_dolly_15k_ja_checkpoint.pt`

### 2. Split を明示的に指定

```bash
python train_hf_dataset.py --dataset-id "openwebtext" --split "train" --epochs 5
```

### 3. テスト実行（少数のサンプルで動作確認）

```bash
python train_hf_dataset.py --dataset-id "wikitext" --split "train" \
    --max-samples 1000 --epochs 1 --vocab-size 4000
```

### 4. HF Hub にアップロード

```bash
HF_TOKEN=hf_xxx python train_hf_dataset.py \
    --dataset-id "llm-jp/databricks-dolly-15k-ja" \
    --epochs 3 --upload --repo-id tapiocatakeshi/Qubit
```

### 5. 既存チェックポイントから追加学習

```bash
python train_hf_dataset.py --dataset-id "llm-jp/databricks-dolly-15k-ja" \
    --resume --ckpt-name neuroq_small_oasst_ja_checkpoint.pt \
    --reset-epochs --epochs 3
```

- `--resume`: 既存チェックポイントから再開
- `--reset-epochs`: 前のエポック数を無視し、新しいデータセットで 0 から学習

## 対応するデータセット例

### 日本語データセット

- `llm-jp/databricks-dolly-15k-ja` - 日本語指示追従データセット（15k件）
- `tapiocatakeshi/oasst-ja` - OASST 日本語データセット
- `elyza/ELYZA-tasks-100` - ELYZA 日本語タスクデータセット

### 英語データセット

- `openwebtext` - ウェブテキストコーパス（OpenAI WaJ）
- `wikitext` - Wikipedia テキストデータセット
- `bookcorpus` - 書籍コーパス
- `glue` - GLUE ベンチマークデータセット

### マルチリンガルデータセット

- `wikipedia` - Wikipedia（多言語）
- `common_voice` - 音声認識データセット（多言語）

## オプション一覧

```
--dataset-id TEXT
    Hugging Face データセット ID（必須）
    例: "llm-jp/databricks-dolly-15k-ja", "openwebtext"

--split TEXT
    読み込む split（デフォルト: 自動検出）
    例: "train", "validation", "test"

--max-samples INT
    使用する最大サンプル数（デフォルト: 0 = 全件）
    0 または負値ですべてのデータを使用

--epochs INT
    学習エポック数（デフォルト: 3）

--lr FLOAT
    学習率（デフォルト: 5e-4）

--vocab-size INT
    語彙サイズ（デフォルト: 32000）

--batch-size INT
    バッチサイズ（デフォルト: small モデル設定の 4）

--max-seq-len INT
    最大シーケンス長（デフォルト: small モデル設定の 4096）

--ckpt-name TEXT
    チェックポイント名（デフォルト: neuroq_small_<dataset-name>.pt）

--save-every INT
    N バッチごとに中間チェックポイント保存（デフォルト: 500）

--resume
    既存チェックポイントから学習を再開する

--reset-epochs
    --resume 時に完了済みエポック数をリセット
    別データセットで事前学習したモデルを新しいデータで追加学習する場合に使用

--upload
    学習済みモデルを HuggingFace Hub にアップロード

--repo-id TEXT
    アップロード先リポジトリID（デフォルト: tapiocatakeshi/Qubit）

--hf-token TEXT
    HuggingFace トークン（デフォルト: 環境変数 HF_TOKEN または HUGGING_FACE_HUB_TOKEN）

--tokenizer-prefix TEXT
    トークナイザーのファイルプリフィックス
    （デフォルト: neuroq_small_<dataset-name>_tokenizer）
```

## 実行例

### 例 1: 日本語データセットで基本学習

```bash
python train_hf_dataset.py \
    --dataset-id "llm-jp/databricks-dolly-15k-ja" \
    --epochs 3 \
    --vocab-size 32000
```

### 例 2: 英語 Wikipedia で学習

```bash
python train_hf_dataset.py \
    --dataset-id "wikitext" \
    --split "train" \
    --epochs 2 \
    --batch-size 8 \
    --max-seq-len 2048
```

### 例 3: 小規模データセットで高精度学習

```bash
python train_hf_dataset.py \
    --dataset-id "glue" \
    --split "train" \
    --max-samples 50000 \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 16
```

### 例 4: 既存モデルを新しいデータで微調整

```bash
python train_hf_dataset.py \
    --dataset-id "elyza/ELYZA-tasks-100" \
    --resume \
    --ckpt-name "neuroq_small_oasst_ja_checkpoint.pt" \
    --reset-epochs \
    --epochs 5
```

### 例 5: 学習後 HF Hub にアップロード

```bash
HF_TOKEN=hf_xxxxxxxxxxxx python train_hf_dataset.py \
    --dataset-id "llm-jp/databricks-dolly-15k-ja" \
    --epochs 3 \
    --upload \
    --repo-id "tapiocatakeshi/Qubit"
```

## 自動検出メカニズム

### 列名の自動検出順序

1. **対話形式**の検出：
   - instruction/prompt/question/query → output/response/answer/completion
   - 形式: `<USER> {instruction}\n{context}\n<ASSISTANT> {output}`

2. **テキスト列**の検出：
   - text, document, content, passage, article, sentence, body, description, summary

3. **フォールバック**：
   - 行内の最初の非空文字列値（4 文字以上）

### Split の自動検出

スクリプトは以下の順序でデータセットから split を自動検出します：
1. 指定された split（`--split`）を使用
2. データセットの最初の split を使用

## トラブルシューティング

### メモリ不足エラー

バッチサイズまたはシーケンス長を減らしてください：

```bash
python train_hf_dataset.py --dataset-id "..." \
    --batch-size 2 \
    --max-seq-len 1024
```

### テキストが抽出されない

データセットの列構造を確認してください。データセットビューアで列名を確認し、
`extract_generic_text` 関数に列名を追加することができます。

### HF Hub へのアップロードエラー

HuggingFace トークンを確認してください：

```bash
# トークンを環境変数で設定
export HF_TOKEN=hf_xxxxxxxxxxxx

# またはコマンドラインで指定
python train_hf_dataset.py --dataset-id "..." --hf-token "hf_xxxxxxxxxxxx"
```

## GPU/CPU の選択

GPU が利用可能な場合は自動的に使用されます。
CPU のみで学習する場合も動作しますが、時間がかかります。

## チェックポイント管理

各エポック完了後にチェックポイントが自動保存されます：

```
neuroq_small_<dataset-name>_checkpoint.pt  # モデルと optimizer の状態
neuroq_small_<dataset-name>_tokenizer.model  # トークナイザー
neuroq_small_<dataset-name>_tokenizer.vocab  # 語彙ファイル
```

## 詳細な使用例

### マルチステップ学習パイプライン

```bash
# ステップ 1: OASST で事前学習
python train_hf_dataset.py \
    --dataset-id "tapiocatakeshi/oasst-ja" \
    --epochs 5

# ステップ 2: dolly-15k-ja で追加学習
python train_hf_dataset.py \
    --dataset-id "llm-jp/databricks-dolly-15k-ja" \
    --resume \
    --ckpt-name "neuroq_small_oasst_ja_checkpoint.pt" \
    --reset-epochs \
    --epochs 3

# ステップ 3: HF Hub にアップロード
HF_TOKEN=hf_xxx python train_hf_dataset.py \
    --dataset-id "llm-jp/databricks-dolly-15k-ja" \
    --resume \
    --ckpt-name "neuroq_small_dolly_15k_ja_checkpoint.pt" \
    --epochs 0 \
    --upload \
    --repo-id "tapiocatakeshi/Qubit"
```

## ライセンスと引用

このスクリプトは Qubit プロジェクトの一部です。
使用するデータセットのライセンス条項に従ってください。
