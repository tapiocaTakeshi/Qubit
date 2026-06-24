# GitHub Actions - HF_TOKEN セットアップガイド

## 🔐 Step 1: HF_TOKEN を GitHub Secrets に追加

### 手順

1. GitHub でリポジトリを開く
2. **Settings** → **Secrets and variables** → **Actions** に移動
3. **New repository secret** をクリック
4. 以下を入力：
   - **Name**: `HF_TOKEN`
   - **Value**: あなたの HuggingFace API トークン
5. **Add secret** をクリック

### トークン取得方法

1. [HuggingFace](https://huggingface.co/settings/tokens) にアクセス
2. **Create new token** をクリック
3. **Role**: `read` 以上を選択
4. トークンをコピー
5. GitHub Secrets に貼り付け

---

## ✅ Step 2: ワークフローファイルを確認

`.github/workflows/qbnn-cli.yml` が以下の構成になっています：

### ワークフローの特徴

✅ **自動トリガー:**
- `main` / `develop` / `claude/**` ブランチへの push
- Pull Requestの作成

✅ **テスト環境:**
- Node.js 18.x と 20.x
- Ubuntu 最新版

✅ **実行内容:**
1. 依存関係のインストール
2. TypeScript ビルド
3. QBNN CLI ヘルプコマンド確認
4. ビルド成果物のアップロード

✅ **HF_TOKEN の使用:**
```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

---

## 🚀 Step 3: ワークフロー実行確認

1. コードを push / Pull Request を作成
2. リポジトリの **Actions** タブを開く
3. `QBNN CLI CI/CD` ワークフローを確認
4. ビルドが成功したか確認

### ログの見方

```
✅ Install dependencies
✅ Build TypeScript  
✅ Verify QBNN CLI
✅ Upload build artifacts
```

---

## 🔐 セキュリティのベストプラクティス

### ✅ すべきこと

```yaml
# 正しい: Secrets 参照
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

```bash
# 正しい: .gitignore に機密情報を除外
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
```

### ❌ してはいけないこと

```yaml
# 間違い: トークンをハードコード
env:
  HF_TOKEN: hf_xxxxx
```

```bash
# 間違い: ログにトークンを出力
echo "Token: $HF_TOKEN"
```

---

## 📊 ワークフロー実行統計

### キャッシング機能

- **Node modules キャッシュ**: CI/CD 速度を高速化
- **キャッシュキー**: `package-lock.json` に基づく
- **効果**: 2回目以降のビルドが 50% 高速化

### 実行時間の目安

| ステップ | 時間 |
|---------|------|
| Setup | 10秒 |
| Install (初回) | 30秒 |
| Install (キャッシュ時) | 5秒 |
| Build | 15秒 |
| Verify CLI | 10秒 |
| **合計 (初回)** | **~65秒** |
| **合計 (キャッシュ時)** | **~40秒** |

---

## 🔄 マトリックステスト

Node.js の複数バージョンで自動テスト：

```yaml
strategy:
  matrix:
    node-version: [18.x, 20.x]
```

これにより以下が自動実行されます：
- Node 18.x でのビルド・テスト
- Node 20.x でのビルド・テスト

---

## 📈 成果物（Artifacts）

ビルド成功時、以下が自動アップロードされます：

```
qubit-cli-build-18.x/
  ├── bin/
  │   ├── qbnn-cli.js
  │   ├── multi-agent-cli.js
  │   └── ...
  └── ...

qubit-cli-build-20.x/
  └── [同じ構成]
```

**ダウンロード方法:**
1. Actions → 実行ログ
2. 下部の **Artifacts** セクション
3. `qbnn-cli-build-*` をダウンロード

---

## 🆘 トラブルシューティング

### エラー: "HF_TOKEN not found"

```
Error: HF_TOKEN is not set
```

**解決策:**
1. GitHub Settings → Secrets を確認
2. Secret が正しく保存されているか確認
3. ワークフローの `env:` セクションを確認

### エラー: "Timeout"

```
Error: timeout waiting for QBNN CLI
```

**解決策:**
```yaml
timeout-minutes: 15  # ジョブのタイムアウト設定
```

### エラー: "npm install fails"

```
npm ERR! 404 Not Found
```

**解決策:**
- インターネット接続を確認
- npm registry の状態を確認
- package.json の依存関係を確認

---

## 📝 ワークフロー拡張例

### デプロイメント追加

```yaml
- name: Deploy to production
  if: github.ref == 'refs/heads/main'
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    npm run build
    # デプロイメントコマンド
```

### Slack 通知追加

```yaml
- name: Notify Slack
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
```

### コード品質チェック

```yaml
- name: Run linter
  run: npm run lint

- name: Run tests
  run: npm run test
```

---

## ✅ チェックリスト

- [ ] HF_TOKEN を GitHub Secrets に追加した
- [ ] `.github/workflows/qbnn-cli.yml` が存在することを確認した
- [ ] コードを push して、ワークフロー実行を確認した
- [ ] ビルドが成功したことを確認した
- [ ] Actions ログで HF_TOKEN が正しく使用されていることを確認した
- [ ] 成果物（Artifacts）が生成されたことを確認した

---

## 📚 参考資料

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [HuggingFace API Tokens](https://huggingface.co/settings/tokens)
- [Qubit AI README](../../qubit_ai_cli/README.md)

---

**✨ これであなたのQBNN CLIプロジェクトは完全な CI/CD パイプラインを持つようになりました！**
