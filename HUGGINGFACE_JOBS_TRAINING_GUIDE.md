# Qubit SFT トレーニング - Hugging Face Jobs 実行ガイド

このガイドでは、**Hugging Face Jobs** を使用して、**Qubit (QBNN) モデル**を **SFT（教師あり微調整）**でリモートトレーニングする方法を説明します。

---

## 📚 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [セットアップ](#セットアップ)
4. [スクリプト実行](#スクリプト実行)
5. [トレーニング設定](#トレーニング設定)
6. [モデルサイズガイド](#モデルサイズガイド)
7. [結果確認とトラブルシューティング](#結果確認とトラブルシューティング)
8. [ベストプラクティス](#ベストプラクティス)

---

## 概要

### Hugging Face Jobs とは

**Hugging Face Jobs** は、Hugging Face インフラで直接トレーニングジョブを実行できるサービスです：

- ✅ **クラウド実行**: ローカルハードウェア不要
- ✅ **複数 GPU オプション**: small (1x A10), large (1x A40), multiGPU 対応
- ✅ **自動管理**: セットアップ不要、すぐに実行
- ✅ **課金体系**: 利用した GPU 時間に応じた課金
- ✅ **シークレット管理**: API トークンを安全に管理

### このスクリプトの特徴

`scripts/train_qubit_hfjobs.sh` が提供する機能：

- 🎯 **複数モデルサイズ対応**: small / medium / large / xlarge
- 📊 **日本語最適化**: `kunishou/databricks-dolly-15k-ja` データセット
- 🔧 **完全カスタマイズ**: エポック、学習率、バッチサイズを柔軟に指定
- 🚀 **自動最適化**: メモリ効率とトレーニング速度を自動調整
- 💾 **GGUF 変換**: トレーニング後、自動的に GGUF 形式に変換
- 🌐 **Hub アップロード**: 完成したモデルを自動的にアップロード
- ⚠️ **エラーハンドリング**: ログと詳細なエラーメッセージ

---

## 前提条件

### 1. Hugging Face アカウント

- ✅ Hugging Face アカウント作成: https://huggingface.co
- ✅ 支払い方法登録（クレジットカード）
- ✅ 十分な残高確保（GPU 時間に応じた課金）

### 2. 認証トークン

```bash
# Hugging Face CLI でログイン
huggingface-cli login

# トークン取得: https://huggingface.co/settings/tokens
# 必要な権限: write (モデルアップロード用)
```

### 3. ローカル環境

```bash
# 必須: Hugging Face Hub CLI
pip install huggingface-hub

# 確認
huggingface-cli whoami
```

### 4. Qubit モデルへのアクセス

```bash
# Qubit モデルの利用条件に同意
# https://huggingface.co/tapiocaTakeshi/Qubit
```

---

## セットアップ

### 1. リポジトリクローン

```bash
git clone https://github.com/tapiocaTakeshi/Qubit.git
cd Qubit
```

### 2. 認証確認

```bash
# Hugging Face CLI で認証
huggingface-cli login

# トークンを入力（Settings → Access Tokens から取得）
# https://huggingface.co/settings/tokens
```

### 3. スクリプト権限設定

```bash
chmod +x scripts/train_qubit_hfjobs.sh
```

---

## スクリプト実行

### 基本的な実行方法

#### 1️⃣ **デフォルト設定で実行（推奨）**

```bash
# medium モデル、20 エポック、デフォルト設定
./scripts/train_qubit_hfjobs.sh

# または明示的に指定
./scripts/train_qubit_hfjobs.sh medium
```

**実行内容:**
- モデルサイズ: `medium`
- エポック: 20
- 学習率: 3e-5
- バッチサイズ: 4
- GPU: A10 small（1x A10 GPU）
- 出力: GGUF + Hub アップロード

#### 2️⃣ **異なるモデルサイズで実行**

```bash
# small モデル（クイック試験、低コスト）
./scripts/train_qubit_hfjobs.sh small

# large モデル（高性能）
./scripts/train_qubit_hfjobs.sh large

# xlarge モデル（最高性能）
./scripts/train_qubit_hfjobs.sh xlarge
```

#### 3️⃣ **トレーニングパラメータをカスタマイズ**

```bash
# エポック数を変更
./scripts/train_qubit_hfjobs.sh medium --epochs 10

# 学習率を調整
./scripts/train_qubit_hfjobs.sh large --lr 1e-4

# バッチサイズを指定
./scripts/train_qubit_hfjobs.sh medium --batch-size 8 --grad-accum 2

# 複数パラメータを指定
./scripts/train_qubit_hfjobs.sh xlarge \
  --epochs 5 \
  --lr 5e-5 \
  --batch-size 2 \
  --jobs-flavor a40-large
```

#### 4️⃣ **Hub アップロードをスキップ**

```bash
# ローカル結果のみ保存、アップロードなし
./scripts/train_qubit_hfjobs.sh medium --no-upload
```

#### 5️⃣ **高性能 GPU で実行**

```bash
# A40 GPU（より高性能、高コスト）
./scripts/train_qubit_hfjobs.sh large --jobs-flavor a40-large

# マルチ GPU（複数 GPU 環境）
./scripts/train_qubit_hfjobs.sh xlarge --jobs-flavor multi-gpu-large
```

### スクリプトオプション一覧

```bash
./scripts/train_qubit_hfjobs.sh [MODEL_SIZE] [OPTIONS]

位置指定引数:
  MODEL_SIZE              small / medium / large / xlarge
                          デフォルト: medium

オプション:
  --epochs N              エポック数 (デフォルト: 20)
  --batch-size N          バッチサイズ (デフォルト: 4)
  --lr RATE               学習率 (デフォルト: 3e-5)
  --grad-accum N          勾配蓄積ステップ (デフォルト: 4)
  --quantization Q        量子化形式 (デフォルト: Q4_K_M)
  --upload-repo REPO      アップロード先リポジトリ
  --jobs-flavor FLAVOR    GPU タイプ (デフォルト: a10g-small)
  --jobs-timeout TIME     タイムアウト時間 (デフォルト: 6h)
  --no-upload             Hub アップロードをスキップ
  --help                  ヘルプを表示
```

---

## トレーニング設定

### モデルサイズ別の推奨設定

#### small モデル（クイック試験）

```bash
# コスト: 最小、速度: 最速
./scripts/train_qubit_hfjobs.sh small \
  --epochs 5 \
  --batch-size 8 \
  --jobs-flavor a10g-small

# 推定時間: 30 分
# 推定コスト: $1-2
```

#### medium モデル（標準、推奨）

```bash
# コスト: 低、速度: 高速
./scripts/train_qubit_hfjobs.sh medium \
  --epochs 20 \
  --batch-size 4 \
  --jobs-flavor a10g-small

# 推定時間: 2-3 時間
# 推定コスト: $5-10
```

#### large モデル（高性能）

```bash
# コスト: 中、速度: 中程度
./scripts/train_qubit_hfjobs.sh large \
  --epochs 10 \
  --batch-size 4 \
  --jobs-flavor a40-large \
  --lr 2e-5

# 推定時間: 4-6 時間
# 推定コスト: $15-25
```

#### xlarge モデル（最高性能）

```bash
# コスト: 高、速度: 完全精度
./scripts/train_qubit_hfjobs.sh xlarge \
  --epochs 5 \
  --batch-size 2 \
  --jobs-flavor multi-gpu-large \
  --lr 1e-5

# 推定時間: 8-12 時間
# 推定コスト: $30-50
```

### データセット情報

**`kunishou/databricks-dolly-15k-ja`**

- 📊 **概要**: 日本語版 Databricks Dolly データセット
- 🎯 **最適用途**: 日本語 QA タスク、汎用会話
- 📝 **データ型**: 質問・指示・出力 形式
- 📈 **サイズ**: ~15,000 サンプル
- 🌍 **言語**: 日本語（100% 日本語対応）
- ✅ **品質**: 高品質、手動キュレーション

**フォーマット例:**

```json
{
  "instruction": "以下の文を日本語に翻訳してください",
  "context": "Hello, world!",
  "response": "こんにちは、世界!"
}
```

### 学習率の調整ガイド

```
学習率の選択:
  - 1e-3 以上:     学習が不安定（推奨されません）
  - 1e-4 ～ 5e-4:  標準的な学習（大きなモデル向け）
  - 1e-5 ～ 5e-5:  fine-tuning（推奨、多くの場合）
  - 1e-6 以下:     超小規模調整（最小限の変更）
```

デフォルト設定:

```bash
# small/medium: 3e-5（推奨）
./scripts/train_qubit_hfjobs.sh medium

# large: より低い学習率を推奨
./scripts/train_qubit_hfjobs.sh large --lr 2e-5

# xlarge: さらに低い学習率
./scripts/train_qubit_hfjobs.sh xlarge --lr 1e-5
```

---

## モデルサイズガイド

### パラメータ数と性能

| モデルサイズ | パラメータ | VRAM | 推論速度 | 精度 | 推奨用途 |
|:-----------|:-------:|:----:|:-----:|:---:|:---------|
| **small** | ~10M | 2GB | 高速 | 低 | プロトタイプ、テスト |
| **medium** | ~50M | 4GB | 高速 | 中 | **推奨**、汎用 |
| **large** | ~200M | 12GB | 中 | 高 | 高精度が必要な場合 |
| **xlarge** | ~900M | 40GB | 遅い | 最高 | 最大性能が必要 |

### 推奨 GPU と設定

| モデルサイズ | 推奨 GPU | バッチサイズ | 推定時間 | 推定コスト |
|:-----------|:---------|:--------:|:-----:|:-------:|
| small | a10g-small | 8 | 30 分 | $1-2 |
| medium | a10g-small | 4 | 2-3h | $5-10 |
| large | a40-large | 4 | 4-6h | $15-25 |
| xlarge | multi-gpu | 2 | 8-12h | $30-50 |

---

## 結果確認とトラブルシューティング

### トレーニング進捗の確認

#### 1️⃣ **Job 実行状況をリアルタイム監視**

```bash
# Hugging Face Hub ダッシュボードで確認
# https://huggingface.co/account/billing/overview

# または CLI で確認
huggingface-cli whoami  # 認証確認

# ジョブ一覧表示（今後対応予定）
# hf jobs list
```

#### 2️⃣ **Job ログを確認**

Hugging Face Jobs のダッシュボード:
- https://huggingface.co/docs/hub/jobs
- Job 実行状況、ログ、エラーメッセージを確認

#### 3️⃣ **アップロード完了を確認**

```bash
# モデルが Hub にアップロードされたか確認
huggingface-cli repo info tapiocaTakeshi/qubit-medium-sft-q4km

# または Web で確認
# https://huggingface.co/tapiocaTakeshi/qubit-medium-sft-q4km
```

### 一般的なエラーと解決策

#### エラー 1: "HF_TOKEN not found"

**症状:**
```
ERROR: HF_TOKEN environment variable not set
```

**解決策:**

```bash
# 認証を確認
huggingface-cli login

# または環境変数を設定
export HF_TOKEN="hf_xxxxxxxxxxxx"

# 確認
huggingface-cli whoami
```

#### エラー 2: "Job timed out"

**症状:**
```
ERROR: Job exceeded timeout limit
```

**解決策:**

```bash
# タイムアウト時間を延長
./scripts/train_qubit_hfjobs.sh medium --jobs-timeout 12h

# または、より小さいモデルまたは少ないエポックで実行
./scripts/train_qubit_hfjobs.sh small --epochs 5
```

#### エラー 3: "Dataset not found"

**症状:**
```
ERROR: Failed to load dataset kunishou/databricks-dolly-15k-ja
```

**解決策:**

```bash
# データセットが存在するか確認
huggingface-cli repo info kunishou/databricks-dolly-15k-ja --repo-type dataset

# または、別のデータセットを使用
# (カスタムスクリプト必要)
```

#### エラー 4: "CUDA out of memory"

**症状:**
```
RuntimeError: CUDA out of memory
```

**解決策:**

```bash
# バッチサイズを削減
./scripts/train_qubit_hfjobs.sh medium --batch-size 2

# または、勾配蓄積を増加
./scripts/train_qubit_hfjobs.sh medium --grad-accum 8

# または、より小さいモデルを選択
./scripts/train_qubit_hfjobs.sh small
```

#### エラー 5: "Upload failed"

**症状:**
```
ERROR: Failed to upload model to Hub
```

**解決策:**

```bash
# アップロードをスキップして再実行（手動アップロード）
./scripts/train_qubit_hfjobs.sh medium --no-upload

# トークンの権限を確認
huggingface-cli token-info

# 手動でアップロード
huggingface-cli upload \
  tapiocaTakeshi/qubit-medium-sft-manual \
  gguf_models/qbnn_medium_Q4_K_M.gguf
```

---

## ベストプラクティス

### 1️⃣ **初回実行時は small モデルで試験**

```bash
# コスト最小、速度最速で動作確認
./scripts/train_qubit_hfjobs.sh small --epochs 1 --no-upload

# 成功後、本番サイズで実行
./scripts/train_qubit_hfjobs.sh medium --epochs 20
```

### 2️⃣ **学習率は控えめに設定**

```bash
# デフォルトより低い学習率から開始
./scripts/train_qubit_hfjobs.sh medium --lr 1e-5

# 必要に応じて段階的に増加
```

### 3️⃣ **複数回トレーニングで性能向上**

```bash
# Phase 1: Medium モデルで基礎学習
./scripts/train_qubit_hfjobs.sh medium --epochs 20

# Phase 2: Large モデルで高度な学習
./scripts/train_qubit_hfjobs.sh large --epochs 10 --lr 2e-5
```

### 4️⃣ **コスト管理**

```bash
# 推定コストを事前に把握
# small: $1-2, medium: $5-10, large: $15-25

# 予算に応じてモデルサイズを選択
# テスト: small, 本番: medium/large
```

### 5️⃣ **定期的なモニタリング**

```bash
# Hub ダッシュボードで GPU 使用時間を確認
# https://huggingface.co/account/billing/overview

# 不要な Job は削除
# 自動クリーンアップ（30日後に自動削除）
```

---

## トレーニング完了後

### 1️⃣ **モデルの確認**

Hub にアップロードされたモデルを確認：

```bash
# Web で確認
# https://huggingface.co/tapiocaTakeshi/qubit-medium-sft-q4km

# CLI で確認
huggingface-cli repo info tapiocaTakeshi/qubit-medium-sft-q4km
```

### 2️⃣ **ローカルで推論テスト**

```bash
# モデルをダウンロード
huggingface-cli download \
  tapiocaTakeshi/qubit-medium-sft-q4km \
  --local-dir ./qubit-sft

# Ollama で実行
ollama create qubit-sft -f ./Modelfile
ollama run qubit-sft "質問: Qubit AI とは?"
```

### 3️⃣ **DPO で更に改善**

```bash
# SFT 後、DPO（直接選好最適化）で性能向上
# 詳細: HUGGINGFACE_LLM_TRAINER_SFT_GUIDE.md を参照
```

---

## 料金見積もり

### Hugging Face Jobs の料金

| GPU | 時間単価 | 月額上限 |
|:---:|:-----:|:------:|
| A10 small | $0.50/h | 無制限 |
| A40 large | $1.50/h | 無制限 |
| 8x H100 | $20/h | 無制限 |

### トレーニングコスト例

```
small モデル (30 分) × $0.50/h = $0.25
medium モデル (2.5h) × $0.50/h = $1.25
large モデル (5h) × $1.50/h = $7.50
xlarge モデル (10h) × $1.50/h = $15.00
```

---

## リソース

### 公式ドキュメント
- [Hugging Face Hub Jobs](https://huggingface.co/docs/hub/jobs)
- [Hugging Face Hub CLI](https://huggingface.co/docs/hub/security-tokens)

### Qubit ドキュメント
- [README.md](./README.md)
- [HUGGINGFACE_LLM_TRAINER_SFT_GUIDE.md](./HUGGINGFACE_LLM_TRAINER_SFT_GUIDE.md)
- [MEGABYTE_100MB_TRAINING_GUIDE.md](./MEGABYTE_100MB_TRAINING_GUIDE.md)

### 参考資料
- [Databricks Dolly](https://github.com/databrickslabs/dolly)
- [日本語データセット](https://huggingface.co/datasets/kunishou)

---

## サポート

質問や問題が発生した場合：

1. **GitHub Issues**: https://github.com/tapiocaTakeshi/Qubit/issues
2. **Discussions**: https://github.com/tapiocaTakeshi/Qubit/discussions
3. **Email**: higuchiyuya.riddle@gmail.com

---

**Last Updated**: 2026-07-19
**Author**: tapiocaTakeshi
**License**: MIT
