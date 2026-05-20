# GitHub Actions Setup Guide
# GitHub ActionsでHugging Faceへの自動アップロード設定

## 概要

GitHub Actionsを使用して、量子化されたNeuroQuantumモデルを自動的にHugging Face（huggingface.co/tapiocatakeshi/Qubit）にアップロードできます。

---

## 🚀 設定手順

### Step 1: Hugging Face API Tokenの取得

1. [Hugging Face](https://huggingface.co) にアクセス
2. **Settings** → **Access Tokens** を開く
3. **New token** をクリック
4. トークンタイプを **write** に設定（モデルアップロードに必要）
5. トークンをコピー（再表示できないので、安全な場所に保存）

**トークンの有効期限:**
- 無制限 or 自由に設定可能

---

### Step 2: GitHub Secretsの設定

1. リポジトリ設定ページを開く
   ```
   GitHub > Settings > Secrets and variables > Actions
   ```

2. **New repository secret** をクリック

3. 以下のシークレットを追加：

| Name | Value |
|------|-------|
| `HF_TOKEN` | Hugging Faceから取得したトークン |
| `HF_USERNAME` | tapiocatakeshi |

**例:**
```
Name: HF_TOKEN
Secret: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### Step 3: ワークフローファイルの確認

ワークフローファイルが以下の場所に存在することを確認：

```
.github/workflows/
├── upload-to-huggingface.yml
└── generate-quantized-models.yml
```

これらのファイルはGitリポジトリに含まれています。

---

## 📋 ワークフロー説明

### Workflow 1: `upload-to-huggingface.yml`

**トリガー:**
- メインブランチへのプッシュ（GGUF or PTファイル変更時）
- 手動トリガー（workflow_dispatch）

**動作:**
1. GGUF and PT ファイルを検出
2. メタデータファイルを準備
3. Hugging Faceにアップロード
4. モデルカードを自動生成

**使用方法:**
```bash
# 自動（メイン へプッシュ時）
git push origin main

# 手動
GitHub Actions > upload-to-huggingface > Run workflow
```

### Workflow 2: `generate-quantized-models.yml`

**トリガー:**
- スケジュール: 毎週金曜日 午後9時（UTC）
- 手動トリガー（workflow_dispatch）

**動作:**
1. モデルを生成（ダミー or チェックポイント）
2. 1-bit, 2-bit, 3-bit に量子化
3. GGUF形式にエクスポート
4. Hugging Faceにアップロード
5. アーティファクトに保存

**スケジュール:**
```yaml
cron: '0 21 * * 5'  # 毎週金曜日 21:00 UTC
```

---

## 💡 使用例

### 例1: ローカルで生成してアップロード

```bash
# ローカルで1-bitモデルを生成
python quantize_neuroquantum_1bit.py checkpoint.pt

# GGUF化
python export_1bit_gguf.py model_1bit.pt

# メインブランチにプッシュ（自動でアップロード）
git add model_1bit.gguf
git commit -m "Add 1-bit quantized model"
git push origin main
```

### 例2: GitHub Actions で自動生成

```
GitHub Actions > generate-quantized-models > Run workflow
```

パラメータ入力:
- quantization_type: all
- model_size: all

### 例3: 特定のモデルをアップロード

```
GitHub Actions > upload-to-huggingface > Run workflow
```

パラメータ入力:
- model_path: model_2bit.gguf
- repo_id: tapiocatakeshi/Qubit（デフォルト）

---

## 📊 アップロード対象ファイル

### 自動アップロード対象

| ファイルタイプ | パターン | 保存先 |
|--------------|---------|--------|
| GGUF | `*.gguf` | リポジトリ直下 |
| PyTorch | `*quantized*.pt` | リポジトリ直下 |
| メタデータ | `*.json` | `metadata/` |
| ドキュメント | `*.md` | リポジトリ直下 |

### ディレクトリ構造（HFリポジトリ）

```
tapiocatakeshi/Qubit/
├── README.md (モデルカード自動生成)
├── models/
│   ├── 1bit/
│   │   ├── model_small_1bit.gguf
│   │   └── model_medium_1bit.gguf
│   ├── 2bit/
│   │   ├── model_small_2bit.gguf
│   │   └── model_medium_2bit.gguf
│   └── 3bit/
│       ├── model_small_3bit.gguf
│       └── model_medium_3bit.gguf
├── checkpoints/
│   ├── 1bit/
│   ├── 2bit/
│   └── 3bit/
├── metadata/
│   └── *.json
├── BINARY_1BIT_QUANTIZATION_GUIDE.md
├── MULTIBIT_QUANTIZATION_COMPARISON.md
└── GGUF_LOADING_TROUBLESHOOTING.md
```

---

## 🔧 トラブルシューティング

### 問題1: 認証エラー

```
Error: Invalid authentication
```

**解決方法:**
1. HF_TOKEN が正しく設定されているか確認
2. Hugging Face でトークンの有効期限を確認
3. GitHub Secrets の値をコピー＆ペーストで再確認

### 問題2: ファイルが見つからない

```
No GGUF or PT files found
```

**解決方法:**
```bash
# ローカルでモデルを生成してコミット
python quantize_neuroquantum_2bit.py checkpoint.pt
git add model_2bit.gguf
git commit -m "Add 2-bit model"
git push origin main
```

### 問題3: アップロード権限エラー

```
Error: Permission denied
```

**解決方法:**
1. Hugging Faceトークンの権限を確認（write必須）
2. リポジトリの所有権を確認

### 問題4: ワークフローが実行されない

**確認事項:**
- `.github/workflows/` ディレクトリが存在
- YAMLファイルの文法が正しい
- ブランチが `main` である

---

## 📈 ワークフロー監視

### 実行状況の確認

1. リポジトリ → **Actions** タブ
2. ワークフロー一覧から選択
3. 実行ログを確認

### ログの見方

```
✅ Checkout repository          [成功]
✅ Set up Python                [成功]
✅ Install dependencies          [成功]
✅ Login to Hugging Face         [成功]
✅ Find GGUF models             [成功]
✅ Upload GGUF models           [成功]
✅ Upload metadata              [成功]
✅ Generate model card          [成功]
✅ Create release               [成功]
```

---

## 🎯 実運用ガイド

### 推奨フロー

```
1. ローカル開発
   └─ python quantize_neuroquantum_2bit.py checkpoint.pt

2. コミット & プッシュ
   └─ git push origin main

3. GitHub Actions 自動実行
   ├─ ファイル検出
   ├─ Hugging Face アップロード
   ├─ メタデータ保存
   └─ リリース作成

4. Hugging Face リポジトリ確認
   └─ https://huggingface.co/tapiocatakeshi/Qubit
```

### スケジュール実行

毎週金曜日（UTC 21:00）に自動実行：
- 最新の量子化モデル生成
- 複数ビット幅（1/2/3-bit）
- 複数サイズ（small/medium）

---

## 🔐 セキュリティ上の注意

### トークン管理

✅ **推奨:**
- GitHub Secrets を使用
- 定期的なトークン更新
- 最小限の権限（write only）

❌ **非推奨:**
- コードに直接埋め込み
- リポジトリに保存
- 複数人での共有

### リポジトリ設定

```
Settings > Actions > General

Action permissions:
✓ Allow all actions and reusable workflows
```

---

## 📱 通知設定

### GitHub 通知

ワークフロー完了時に通知を受け取る：

```
Settings > Notifications > Custom routing
```

### Hugging Face 通知

モデルアップロード完了時に通知：

```
Hugging Face > Settings > Notifications
```

---

## 🚀 次のステップ

### Phase 1: セットアップ完了後

1. **確認テスト**
   ```bash
   # ローカルでモデル生成
   python quantize_neuroquantum_2bit.py checkpoint.pt
   
   # コミット & プッシュ
   git add model_2bit.gguf
   git push origin main
   ```

2. **GitHub Actions 実行確認**
   - Actions タブでワークフロー実行を確認
   - 完了後、Hugging Face でアップロード確認

### Phase 2: 自動化の拡張

```yaml
# スケジュール変更（例：毎日）
cron: '0 9 * * *'  # 毎日 09:00 UTC

# 追加トリガー
on:
  push:
    tags:
      - 'v*'  # タグをプッシュ時に実行
```

### Phase 3: CI/CD パイプライン統合

```yaml
- テスト実行
- 品質チェック
- 自動デプロイ
- 通知
```

---

## 📚 参考ドキュメント

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [Upload Models to Hugging Face](https://huggingface.co/docs/hub/security-tokens)

---

## ✅ チェックリスト

### セットアップ前

- [ ] Hugging Face アカウント作成
- [ ] GitHub リポジトリアクセス確認
- [ ] ローカルで量子化モデル生成テスト

### セットアップ実施

- [ ] Hugging Face トークン取得
- [ ] GitHub Secrets 設定（HF_TOKEN）
- [ ] ワークフローファイル確認
- [ ] テスト実行

### セットアップ確認

- [ ] Workflow 実行ログ確認
- [ ] Hugging Face リポジトリでアップロード確認
- [ ] モデルカード表示確認
- [ ] メタデータ保存確認

---

## 🎉 完成

GitHub Actions による自動アップロード設定が完了しました！

**URL**: https://huggingface.co/tapiocatakeshi/Qubit

```bash
# これ以降、以下のコマンドで自動アップロード：
git push origin main
```

---

**作成日**: 2026-05-20  
**最終更新**: 2026-05-20  
**ステータス**: ✅ 設定完了  
**自動化レベル**: 完全自動化
