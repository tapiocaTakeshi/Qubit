# GitHub Actions で Qubit SFT トレーニングを実行

このガイドでは、GitHub Actions を使用して、Hugging Face Jobs 上で Qubit モデルの SFT トレーニングを自動実行する方法を説明します。

---

## 📋 概要

### 提供されているワークフロー

#### 1️⃣ **手動トリガーワークフロー** (`train-qubit-hf-jobs.yml`)

GitHub Actions UI から手動でトレーニングを実行できます。

**特徴:**
- ✅ モデルサイズをカスタマイズ（small/medium/large/xlarge）
- ✅ ハイパーパラメータを柔軟に調整
- ✅ GPU タイプを選択（a10g-small/a40-large）
- ✅ Hub アップロードをオン/オフ設定
- ✅ リアルタイムログ表示

#### 2️⃣ **スケジュール実行ワークフロー** (`train-qubit-scheduled.yml`)

毎週自動的にトレーニングを実行できます。

**特徴:**
- ✅ 毎週日曜日 00:00 UTC に自動実行
- ✅ Medium モデル、20 エポックで固定実行
- ✅ 手動トリガーも可能
- ✅ 結果は Hub に自動アップロード

---

## 🔧 セットアップ手順

### 1️⃣ GitHub シークレット設定

Hugging Face API トークンを GitHub シークレットに追加します。

**手順:**

```
GitHub リポジトリ
  → Settings
  → Secrets and variables
  → Actions
  → New repository secret
```

**シークレット情報:**

| Name | Value |
|------|-------|
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxxxx` (your actual token) |

✅ 追加完了

### 2️⃣ ワークフローファイルの確認

ワークフローファイルが `.github/workflows/` に存在することを確認：

```bash
ls -la .github/workflows/
# train-qubit-hf-jobs.yml      (手動トリガー)
# train-qubit-scheduled.yml    (スケジュール実行)
```

### 3️⃣ リポジトリにプッシュ

```bash
git add .github/workflows/
git commit -m "feat: Add GitHub Actions workflows for Qubit training"
git push origin main
```

---

## 🚀 実行方法

### **オプション A: 手動トリガー（推奨）**

GitHub UI から直接実行します。

#### ステップ 1: Actions タブを開く

```
https://github.com/tapiocaTakeshi/Qubit/actions
```

#### ステップ 2: ワークフロー選択

```
"Qubit SFT Training - Hugging Face Jobs" → Run workflow
```

#### ステップ 3: パラメータ入力

| パラメータ | デフォルト | オプション | 説明 |
|-----------|----------|-----------|------|
| **model_size** | medium | small / medium / large / xlarge | モデルサイズ |
| **epochs** | 20 | 1-100 | トレーニングエポック数 |
| **batch_size** | 4 | 1-16 | バッチサイズ |
| **learning_rate** | 3e-5 | - | 学習率 |
| **gpu_flavor** | a10g-small | a10g-small / a40-large / multi-gpu-large | GPU タイプ |
| **timeout** | 6h | 1h-24h | タイムアウト時間 |
| **upload_enabled** | true | true / false | Hub アップロード有無 |

#### ステップ 4: Run workflow をクリック

トレーニングが開始されます。

#### ステップ 5: 実行状況を監視

```
Actions → Qubit SFT Training - Hugging Face Jobs → [Latest Run]
```

リアルタイムでログを確認できます。

---

### **オプション B: 定期実行（スケジュール）**

毎週日曜日に自動実行します。

#### 設定確認

```yaml
# .github/workflows/train-qubit-scheduled.yml
on:
  schedule:
    - cron: "0 0 * * 0"  # 毎週日曜日 00:00 UTC
```

#### スケジュール変更

別の日時で実行したい場合：

```yaml
# 毎日実行
- cron: "0 0 * * *"

# 毎週月曜日
- cron: "0 0 * * 1"

# 毎月1日
- cron: "0 0 1 * *"

# 毎6時間
- cron: "0 */6 * * *"
```

変更後、プッシュしてください：

```bash
git add .github/workflows/train-qubit-scheduled.yml
git commit -m "chore: Update training schedule"
git push origin main
```

---

## 📊 実行例

### 例 1: Small モデルで低コスト試験

**入力:**
```
Model Size: small
Epochs: 5
Batch Size: 8
Learning Rate: 5e-4
GPU Flavor: a10g-small
Timeout: 2h
Upload: true
```

**予想結果:**
- 実行時間: 30 分
- 推定コスト: $1-2
- 出力: `tapiocaTakeshi/qubit-small-sft-q4km`

### 例 2: Large モデルで高性能学習

**入力:**
```
Model Size: large
Epochs: 10
Batch Size: 4
Learning Rate: 2e-5
GPU Flavor: a40-large
Timeout: 8h
Upload: true
```

**予想結果:**
- 実行時間: 4-6 時間
- 推定コスト: $15-25
- 出力: `tapiocaTakeshi/qubit-large-sft-q4km`

### 例 3: XLarge モデルで最高性能

**入力:**
```
Model Size: xlarge
Epochs: 5
Batch Size: 2
Learning Rate: 1e-5
GPU Flavor: multi-gpu-large
Timeout: 12h
Upload: true
```

**予想結果:**
- 実行時間: 8-12 時間
- 推定コスト: $30-50
- 出力: `tapiocaTakeshi/qubit-xlarge-sft-q4km`

---

## 🔍 実行状況の確認

### リアルタイムログ表示

```
GitHub → Actions → [Workflow Name] → [Run Number]
```

### 完了後の確認

#### ✅ 成功時

```
✅ Training job completed successfully!

📊 Job Details:
  - Model Size: medium
  - Epochs: 20
  - GPU: a10g-small

📤 Check uploaded model:
  https://huggingface.co/tapiocaTakeshi/qubit-medium-sft-q4km
```

#### ❌ 失敗時

```
❌ Training job failed!

🔧 Troubleshooting:
  1. Check GitHub Actions logs
  2. Verify HF_TOKEN is set correctly
  3. Check Hugging Face Jobs dashboard
```

---

## 🐛 トラブルシューティング

### エラー 1: "Authentication failed"

**症状:**
```
Error: Not logged in
```

**解決策:**

1. GitHub シークレットを確認
   ```
   Settings → Secrets and variables → Actions → HF_TOKEN
   ```

2. トークンが正しいか確認
   ```
   https://huggingface.co/settings/tokens
   ```

3. 書き込み権限があるか確認
   - トークン詳細で "write" が有効か確認

### エラー 2: "Job timeout"

**症状:**
```
Error: Job exceeded timeout limit
```

**解決策:**

1. タイムアウト時間を延長
   ```
   Timeout: 12h (デフォルト 6h から変更)
   ```

2. またはモデルサイズを削減
   ```
   Model Size: small (デフォルト medium から変更)
   ```

### エラー 3: "CUDA out of memory"

**症状:**
```
RuntimeError: CUDA out of memory
```

**解決策:**

```
- GPU Flavor: a40-large (デフォルト a10g-small から変更)
- または Batch Size: 2 (デフォルト 4 から削減)
```

### エラー 4: "Upload failed"

**症状:**
```
Error: Failed to upload model to Hub
```

**解決策:**

1. トークンの権限を確認
2. または Upload を無効化
   ```
   Upload Enabled: false
   ```
3. 手動でアップロード
   ```bash
   huggingface-cli upload \
     username/model-name \
     gguf_models/ \
     --repo-type model
   ```

---

## 📈 コスト管理

### 月額コスト計算

Hugging Face Jobs の料金：

| GPU | 時間単価 |
|:---:|:------:|
| A10 small | $0.50/h |
| A40 large | $1.50/h |
| Multi-GPU | $20/h |

### 月額予算例

```
Weekly Training (Medium):
  2.5h × $0.50/h × 4 週 = $5/月

Daily Training (Small):
  0.5h × $0.50/h × 30 日 = $7.5/月

Combined (Medium + Large, 週1回):
  (2.5h × $0.50 + 5h × $1.50) × 4 = $35/月
```

---

## 🔒 セキュリティ

### トークンの安全性

- ✅ シークレットは暗号化されて保存
- ✅ ログには表示されない
- ✅ Actions からのみアクセス可能
- ❌ コミット内に含めない

### ベストプラクティス

```bash
# ✅ 環境変数で使用（安全）
export HF_TOKEN='...'

# ❌ コミットに含める（危険）
HF_TOKEN='...' ./scripts/train_qubit_hfjobs.sh
```

---

## 📚 参考資料

### GitHub Actions ドキュメント
- [GitHub Actions 公式ドキュメント](https://docs.github.com/en/actions)
- [Workflows](https://docs.github.com/en/actions/using-workflows)
- [Scheduling workflows](https://docs.github.com/en/actions/using-workflows/scheduling-workflows)

### Qubit ドキュメント
- [HUGGINGFACE_JOBS_TRAINING_GUIDE.md](../HUGGINGFACE_JOBS_TRAINING_GUIDE.md)
- [scripts/train_qubit_hfjobs.sh](../scripts/train_qubit_hfjobs.sh)

### Hugging Face リソース
- [Hugging Face Hub Jobs](https://huggingface.co/docs/hub/jobs)
- [API Tokens](https://huggingface.co/settings/tokens)

---

## ✨ ワークフロー統合図

```
┌─────────────────────────────────────┐
│  GitHub Actions UI                  │
│  (Actions → Run workflow)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  train-qubit-hf-jobs.yml            │
│  (手動トリガー)                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  scripts/train_qubit_hfjobs.sh      │
│  (トレーニング実行)                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Hugging Face Jobs                  │
│  (クラウド実行: GPU トレーニング)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Hugging Face Hub                   │
│  (モデルアップロード)                 │
└─────────────────────────────────────┘
```

---

## 🎯 次のステップ

1. ✅ GitHub シークレット設定
2. ✅ ワークフローファイルをプッシュ
3. ✅ Actions タブから実行
4. 📊 結果を確認
5. 🚀 定期実行を有効化

---

**Last Updated**: 2026-07-19
**Author**: tapiocaTakeshi
**License**: MIT
