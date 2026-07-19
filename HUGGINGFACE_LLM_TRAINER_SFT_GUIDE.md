# Qubit megabyte_100mb - Hugging Face LLM Trainer Skill SFT トレーニング ガイド

このガイドでは、**Hugging Face LLM Trainer Skill** を使用して、**megabyte_100mb NeuroQuantum モデル**を SFT（教師あり微調整）でトレーニングする方法を説明します。

---

## 📚 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [セットアップ](#セットアップ)
4. [SFT トレーニング](#sft-トレーニング)
5. [データセット](#データセット)
6. [実装例](#実装例)
7. [トラブルシューティング](#トラブルシューティング)
8. [次のステップ](#次のステップ)

---

## 概要

### Hugging Face LLM Trainer Skill とは

**Hugging Face LLM Trainer Skill** は、大規模言語モデルをファインチューニングするための AI Agent 用スキルです。以下の特徴があります：

- ✅ **簡単な API**: 複雑な設定なしにトレーニング可能
- ✅ **TRL ライブラリベース**: SFT、DPO、GRPO に対応
- ✅ **自動最適化**: GPU メモリと学習速度を自動調整
- ✅ **Hugging Face Hub 統合**: トレーニング済みモデルを直接保存

### megabyte_100mb モデル

**megabyte_100mb** は、Qubit AI のコアモデルです：

```
モデル仕様
─────────────────────────────
Embedding Dimension:  1024
Hidden Dimension:     2048
Number of Heads:      16
Number of Layers:     10
Max Sequence Length:  512
推定パラメータ数:      ~100M
量子インスパイア:      APQB (Adjustable Pseudo Quantum Bit)
```

### SFT（教師あり微調整）とは

**SFT** は、ラベル付きデータセット（質問と回答のペア）を使用してモデルを微調整します。

**処理フロー:**
```
入力データ: {"text": "質問", "output": "望ましい回答"}
           ↓
SFT トレーニング: モデルを微調整
           ↓
出力: 質問に対して良い回答を生成するモデル
```

**メリット:**
- 特定のタスクやドメインに最適化
- 回答品質の向上
- 推論コストの削減（小さなモデルで高性能）

---

## 前提条件

### 1. 必須環境

- **Python**: 3.9 以上
- **PyTorch**: 2.0 以上
- **GPU**: 16GB VRAM 以上（推奨：A100 80GB または RTX 4090 24GB）
- **Hugging Face アカウント**: https://huggingface.co

### 2. Hugging Face 認証

```bash
# Hugging Face CLI でログイン
huggingface-cli login

# トークンを入力（Settings → Access Tokens から取得）
# https://huggingface.co/settings/tokens
```

### 3. 環境変数設定

```bash
# Hugging Face API キーを設定
export HF_TOKEN="your-hugging-face-token"

# (オプション) CUDA デバイス指定
export CUDA_VISIBLE_DEVICES="0"  # GPU ID
```

---

## セットアップ

### 1. Skill のインストール

Claude Code で Hugging Face LLM Trainer Skill をインストール：

```bash
# Skills マーケットプレイスを登録
/plugin marketplace add huggingface/skills

# LLM Trainer Skill をインストール
/plugin install huggingface-llm-trainer@huggingface/skills
```

または手動でリポジトリからコピー：

```bash
# リポジトリクローン
git clone https://github.com/huggingface/skills.git
cd skills/skills/huggingface-llm-trainer

# Skill を使用可能にする
cp -r . ~/.agents/skills/huggingface-llm-trainer
```

### 2. 依存パッケージのインストール

```bash
# Hugging Face Transformers と TRL
pip install transformers>=4.36.0
pip install trl>=0.7.0
pip install datasets>=2.14.0
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# オプション
pip install bitsandbytes  # 8-bit quantization 用
pip install peft          # LoRA adaptation 用
pip install pynvml        # GPU 監視用
```

### 3. モデル確認

```python
# Python で モデル設定を確認
from transformers import AutoConfig

model_name = "tapiocaTakeshi/megabyte_100mb"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

print(f"Model: {model_name}")
print(f"Hidden Size: {config.hidden_size}")
print(f"Num Layers: {config.num_hidden_layers}")
print(f"Max Position: {config.max_position_embeddings}")
```

---

## SFT トレーニング

### 基本的なトレーニングフロー

#### 1️⃣ **データセット準備**

SFT トレーニングには、以下の形式のデータが必要です：

**形式 A: JSON Lines (.jsonl)**
```jsonl
{"text": "質問: 東京とは何ですか?", "output": "東京は日本の首都です。"}
{"text": "質問: Python とは?", "output": "Python はプログラミング言語です。"}
```

**形式 B: Hugging Face Dataset**
```python
from datasets import load_dataset

dataset = load_dataset("elyza/ELYZA-tasks-100")
# 必須カラム: "instruction" または "text", "output"
```

**形式 C: CSV**
```csv
text,output
"質問: 東京とは?","東京は日本の首都です。"
"質問: Python とは?","Python はプログラミング言語です。"
```

#### 2️⃣ **Hugging Face LLM Trainer Skill を使用したトレーニング**

Claude Code で以下を実行：

```
Use the HF model trainer Skill to fine-tune megabyte_100mb with SFT on the ELYZA-tasks-100 dataset
```

Claude が自動的に以下を実行します：
- ✅ データセット準備
- ✅ モデル読み込み
- ✅ トレーニングループ実行
- ✅ チェックポイント保存
- ✅ Hugging Face Hub へアップロード

#### 3️⃣ **トレーニング結果**

```
Output:
  Training complete!
  Model saved: huggingface.co/username/megabyte_100mb-sft
  Evaluation Metrics:
    - Final Loss: 2.34
    - BLEU Score: 0.42
    - ROUGE-L: 0.51
```

---

### 詳細なトレーニング設定

#### CLI コマンド例

```bash
# 基本的な SFT トレーニング
python -m trl.sft.sft_trainer \
    --model_name_or_path tapiocaTakeshi/megabyte_100mb \
    --dataset_name elyza/ELYZA-tasks-100 \
    --output_dir ./sft_megabyte_100mb \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --warmup_steps 100 \
    --bf16 \
    --gradient_accumulation_steps 2
```

#### Python スクリプト例

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. モデル・トークナイザー読み込み
model_name = "tapiocaTakeshi/megabyte_100mb"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 2. データセット準備
dataset = load_dataset("elyza/ELYZA-tasks-100")

# 3. トレーニング設定
sft_config = SFTConfig(
    output_dir="./sft_megabyte_100mb",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
    bf16=True,
    gradient_accumulation_steps=2,
    max_seq_length=512,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
)

# 4. トレーナー作成
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("test"),
)

# 5. トレーニング実行
trainer.train()

# 6. モデル保存
trainer.save_model("./sft_megabyte_100mb_final")
```

---

## データセット

### 推奨データセット

| # | Dataset | 説明 | 形式 | サイズ | 推奨用途 |
|---|---------|------|------|--------|--------|
| 1 | **elyza/ELYZA-tasks-100** | 日本語タスク | Q&A | 100例 | 🎯 **推奨** (クイック試験) |
| 2 | elyza/ELYZA-tasks-1500 | 日本語タスク拡張 | Q&A | 1.5K | 日本語最適化 |
| 3 | Open-Orca/OpenOrca | 指示フォロー | Instruction | 655K | 高品質学習 |
| 4 | HuggingFaceH4/ultrachat_200k | チャット対話 | Chat | 200K | 会話最適化 |
| 5 | meta-math/MetaMathQA | 数学問題 | Math QA | 395K | 推論タスク |
| 6 | bigcode/the-stack-v2 | コード | Code | 数B トークン | コード生成 |

### データセット準備（カスタム例）

#### 例 1: JSONL ファイルから直接トレーニング

```bash
# custom_data.jsonl を作成
cat > custom_data.jsonl << 'EOF'
{"text": "Q: Qubit AI とは?", "output": "Qubit AI は量子インスパイアニューラルネットワークライブラリです。"}
{"text": "Q: SFT の利点は?", "output": "特定タスクへの最適化と性能向上が期待できます。"}
EOF

# トレーニング実行
python train_sft.py \
    --model_name_or_path tapiocaTakeshi/megabyte_100mb \
    --train_file custom_data.jsonl \
    --output_dir ./sft_custom \
    --num_train_epochs 3
```

#### 例 2: CSV から Hugging Face Dataset へ変換

```python
from datasets import Dataset
import pandas as pd

# CSV を読み込み
df = pd.read_csv("training_data.csv")

# Hugging Face Dataset に変換
dataset = Dataset.from_dict({
    "text": df["question"].tolist(),
    "output": df["answer"].tolist()
})

# トレーニング実行
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

---

## 実装例

### シナリオ 1: クイック試験（ELYZA-tasks-100）

```python
# クイック SFT トレーニング
# 所要時間: ~15 分 (RTX 4090)

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル準備
model_name = "tapiocaTakeshi/megabyte_100mb"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# データセット準備
dataset = load_dataset("elyza/ELYZA-tasks-100")

# トレーニング設定（軽量）
config = SFTConfig(
    output_dir="./sft_quick",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-4,
    max_seq_length=512,
    bf16=True,
)

# トレーニング
trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=config, train_dataset=dataset["train"])
trainer.train()

print("✅ トレーニング完了！")
print("モデルが保存されました: ./sft_quick")
```

### シナリオ 2: 本格的なトレーニング（Open-Orca）

```python
# 本格的な SFT トレーニング
# 所要時間: ~8-12 時間 (A100 80GB)

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name = "tapiocaTakeshi/megabyte_100mb"

# LoRA アダプタ設定（VRAM 削減）
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

# モデル準備
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", load_in_8bit=True)
model = get_peft_model(model, peft_config)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# データセット準備（分割学習）
dataset = load_dataset("Open-Orca/OpenOrca", split="train[:100000]")  # 最初の 100K 例

# トレーニング設定（本格的）
config = SFTConfig(
    output_dir="./sft_orca",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    warmup_steps=500,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    max_seq_length=512,
    bf16=True,
    logging_steps=10,
    push_to_hub=True,
    hub_model_id="username/megabyte_100mb-sft-orca",
    hub_strategy="every_save",
)

# トレーニング
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()

# LoRA チェックポイントをマージして保存
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./sft_orca_merged")
tokenizer.save_pretrained("./sft_orca_merged")
```

### シナリオ 3: 日本語特化トレーニング（ELYZA 拡張版）

```python
# 日本語タスク最適化
# 所要時間: ~4 時間 (RTX 4090)

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tapiocaTakeshi/megabyte_100mb"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 日本語データセット準備
dataset_1500 = load_dataset("elyza/ELYZA-tasks-1500")

# 日本語特化設定
config = SFTConfig(
    output_dir="./sft_elyza_jp",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_seq_length=512,
    bf16=True,
    gradient_accumulation_steps=2,
    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=20,
)

# トレーニング
trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=config, train_dataset=dataset_1500["train"])
trainer.train()

# 日本語推論テスト
model.eval()
prompt = "質問: Qubit AI の特徴は何ですか?\n回答:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## トラブルシューティング

### エラー 1: CUDA メモリ不足

**症状:**
```
RuntimeError: CUDA out of memory
```

**解決策:**

```bash
# バッチサイズを削減
--per_device_train_batch_size 1

# グラディエント蓄積を増加
--gradient_accumulation_steps 8

# 8-bit 量子化を有効化
--load_in_8bit

# LoRA アダプタを使用（VRAM ~60% 削減）
```

### エラー 2: トークナイザーエラー

**症状:**
```
ValueError: Trying to decode the token in the middle of a byte!
```

**解決策:**

```python
# トークナイザーを明示的に指定
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False  # 高速トークナイザーを無効化
)

# パッディングトークンを設定
tokenizer.pad_token = tokenizer.eos_token
```

### エラー 3: データセット読込エラー

**症状:**
```
ConnectionError: Failed to load dataset
```

**解決策:**

```bash
# Hugging Face 認証を確認
huggingface-cli login

# キャッシュをクリア
rm -rf ~/.cache/huggingface/datasets

# オフラインモードを確認
export HF_DATASETS_OFFLINE=0
```

### エラー 4: トレーニング中の NaN 損失

**症状:**
```
Loss: NaN
```

**解決策:**

```python
# 学習率を低下
--learning_rate 1e-4

# グラディエント正規化を有効化
--gradient_checkpointing true

# 勾配クリッピングを有効化
--max_grad_norm 1.0
```

---

## 次のステップ

### 1️⃣ **モデル評価**

```python
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# テストセットで評価
test_dataset = load_dataset("elyza/ELYZA-tasks-100", split="test")
test_loader = DataLoader(test_dataset, batch_size=4)

model.eval()
total_loss = 0
for batch in test_loader:
    inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    total_loss += outputs.loss.item()

avg_loss = total_loss / len(test_loader)
print(f"Test Loss: {avg_loss:.4f}")
```

### 2️⃣ **DPO（直接選好最適化）での改善**

```bash
# SFT 後に DPO でさらに改善
Use the HF model trainer Skill to apply DPO on megabyte_100mb-sft 
with the Ultrafeedback dataset
```

### 3️⃣ **Hugging Face Hub へアップロード**

```python
# トレーニング済みモデルをアップロード
trainer.push_to_hub("username/megabyte_100mb-sft-elyza")

# Token は環境変数から自動取得
# export HF_TOKEN="your_token"
```

### 4️⃣ **ローカル推論**

```bash
# GGUF 形式に変換
python export_gguf.py \
    --input ./sft_elyza_jp/final_model \
    --output megabyte_100mb_sft.gguf \
    --quantization Q4_K_M

# Ollama で実行
ollama pull tapiocaTakeshi/megabyte_100mb-sft
ollama run tapiocaTakeshi/megabyte_100mb-sft
```

---

## Hugging Face Jobs での実行（クラウド）

ローカルハードウェアがない場合、**Hugging Face Jobs** を使用してクラウドで直接トレーニングできます。

### クイック実行

```bash
# デフォルト設定（medium モデル、A10 GPU）
./scripts/train_qubit_hfjobs.sh

# small モデルで低コスト実行
./scripts/train_qubit_hfjobs.sh small --epochs 5

# large モデルで高性能実行
./scripts/train_qubit_hfjobs.sh large --epochs 10
```

### 主な利点

- ✅ **ローカル環境不要**: GPU なしでも実行可能
- ✅ **自動最適化**: メモリと速度を自動調整
- ✅ **複数 GPU サイズ**: A10 small / A40 large / マルチ GPU 対応
- ✅ **自動 GGUF 変換**: トレーニング完了後、自動的に GGUF 形式に変換
- ✅ **Hub 自動アップロード**: 完成したモデルを自動的にアップロード

### 詳細ガイド

完全な詳細ガイドは以下を参照：

📖 **[HUGGINGFACE_JOBS_TRAINING_GUIDE.md](./HUGGINGFACE_JOBS_TRAINING_GUIDE.md)**

- セットアップと認証
- スクリプト実行方法
- モデルサイズガイド
- トラブルシューティング
- 料金見積もり

---

## リソース

### Hugging Face Skills ドキュメント
- [Hugging Face LLM Trainer Skill](https://github.com/huggingface/skills/tree/main/skills/huggingface-llm-trainer)
- [TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl)

### Qubit AI ドキュメント
- [README.md](./README.md)
- [MEGABYTE_100MB_TRAINING_GUIDE.md](./MEGABYTE_100MB_TRAINING_GUIDE.md)
- [DPO_GUIDE.md](./DPO_GUIDE.md)

### 参考資料
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)

---

## サポート

質問や問題が発生した場合：

1. **GitHub Issues**: https://github.com/tapiocaTakeshi/Qubit/issues
2. **Discussions**: https://github.com/tapiocaTakeshi/Qubit/discussions
3. **Email**: higuchiyuya.riddle@gmail.com
4. **Hugging Face Hub**: https://huggingface.co/tapiocaTakeshi

---

**Last Updated**: 2026-07-19
**Author**: tapiocaTakeshi
**License**: MIT
