# Hugging Face Hub への自動アップロード

このドキュメントは、GitHub ActionsとHugging Face Hubを使用してGGUFモデルを自動生成・アップロードする手順を説明します。

## 前提条件

1. **Hugging Face アカウント**
   - https://huggingface.co に登録
   - Settings → Access Tokens から API トークンを取得

2. **GitHub Secrets 設定**
   - リポジトリの Settings → Secrets and variables → Actions
   - `HF_TOKEN` という名前で Hugging Face トークンを追加

## セットアップ手順

### 1. GitHub Secrets に HF_TOKEN を追加

```
リポジトリ → Settings → Secrets and variables → Actions
→ New repository secret
  - Name: HF_TOKEN
  - Secret: hf_xxxxxxxxxxxx（Hugging Face のトークン）
```

### 2. ワークフローの実行

GitHub Actions ワークフロー `upload-gguf-models.yml` が設定されています。

**方法1: GitHub UIから実行**

```
リポジトリ → Actions → "Generate and Upload GGUF Models to Hugging Face"
→ Run workflow
  - Quantization: Q4_K_M (デフォルト)
  - Sizes: small (またはカンマ区切りで複数)
  - Repository name: username/qubit-q4-k-m
→ Run workflow
```

**方法2: GitHub CLI から実行**

```bash
gh workflow run upload-gguf-models.yml \
  -f quantization=Q4_K_M \
  -f sizes=small \
  -f repo_name=username/qubit-q4-k-m
```

## ワークフロー入力パラメータ

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `quantization` | 量子化形式 | `Q4_K_M`, `Q5_K_M`, `F32` |
| `sizes` | モデルサイズ（カンマ区切り） | `small` または `small,medium,large` |
| `repo_name` | Hugging Face リポジトリ | `username/qubit-q4-k-m` |

## ワークフローの処理内容

1. ✅ **環境セットアップ**
   - Python 3.11 セットアップ
   - 依存パッケージインストール（torch, numpy, gguf, huggingface-hub）

2. ✅ **GGUF モデル生成**
   ```bash
   python generate_gguf_models.py \
     --quantization <quantization> \
     --sizes <sizes>
   ```

3. ✅ **Hugging Face Hub へアップロード**
   - 各モデルを順番にアップロード
   - GitHub Secrets から HF_TOKEN を自動取得
   - モデルカード自動生成
   - マニフェストファイルもアップロード

4. ✅ **成果物保存**
   - `gguf_models/` ディレクトリを GitHub Artifacts として保存
   - 30日間保持

5. ✅ **サマリー作成**
   - ワークフローの実行結果をサマリーに出力
   - Hugging Face リポジトリへのリンク生成

## 実行例

### 例1: Q4_K_M 量子化、小型モデルのみ

```
Quantization: Q4_K_M
Sizes: small
Repository: tapiocaTakeshi/qubit-q4-k-m
```

**結果:**
- `neuroquantum_small_Q4_K_M.gguf` が Hugging Face にアップロード
- URL: https://huggingface.co/tapiocaTakeshi/qubit-q4-k-m

### 例2: Q5_K_M 量子化、複数サイズ

```
Quantization: Q5_K_M
Sizes: small,medium,large
Repository: tapiocaTakeshi/qubit-models
```

**結果:**
- `neuroquantum_small_Q5_K_M.gguf`
- `neuroquantum_medium_Q5_K_M.gguf`
- `neuroquantum_large_Q5_K_M.gguf`

すべて `tapiocaTakeshi/qubit-models` にアップロード

## ローカルでのテスト

ワークフローをローカルで実行したい場合：

```bash
# HF_TOKEN を環境変数で設定
export HF_TOKEN="hf_xxxxxxxxxxxx"

# GGUF 生成
python generate_gguf_models.py \
  --quantization Q4_K_M \
  --sizes small

# Hugging Face にアップロード
python upload_to_huggingface.py \
  --repo-name "username/qubit-q4-k-m" \
  --gguf-path "gguf_models/neuroquantum_small_Q4_K_M.gguf"
```

## トラブルシューティング

### HF_TOKEN エラー

```
ValueError: Hugging Face token not provided
```

**解決:**
- GitHub Secrets に `HF_TOKEN` が正しく設定されているか確認
- リポジトリ Settings → Secrets and variables で確認

### モデルアップロード失敗

```
Failed to upload file
```

**確認項目:**
- HF_TOKEN が有効か（https://huggingface.co/settings/tokens で確認）
- リポジトリ名が正しいか
- GGUF ファイルが生成されたか（ログで確認）

### メモリ不足

大型モデルを生成する場合、GitHub Actions ランナーがメモリ不足になる可能性があります。

**対策:**
- `small` サイズから開始
- `medium`, `large` は単独で実行
- またはセルフホストランナーを使用

## GitHub Actions ログの確認

```
リポジトリ → Actions → ワークフロー実行
→ Generate and Upload GGUF Models to Hugging Face
→ ログを確認
```

## サポート

- 📖 [Hugging Face Hub ドキュメント](https://huggingface.co/docs/hub)
- 🚀 [GitHub Actions ドキュメント](https://docs.github.com/actions)
- 🐛 [Issue を報告](https://github.com/tapiocatakeshi/Qubit/issues)
