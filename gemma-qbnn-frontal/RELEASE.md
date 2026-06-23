# Release Guide - gemma-qbnn-frontal

このドキュメントは、GitHub Actionsを使用してnpmにリリースするプロセスを説明します。

## リリース前の準備

### 1. NPM トークンの設定

GitHub Secretsに`NPM_TOKEN`を設定する必要があります。

1. [npm](https://www.npmjs.com)にログイン
2. Account Settings → Tokens → Create Token
3. Token type: **Automation** を選択
4. GitHubリポジトリの Settings → Secrets and variables → Actions
5. `NPM_TOKEN`という名前でシークレットを追加

### 2. バージョンの更新

リリース前に`package.json`のバージョンを更新します：

```bash
cd gemma-qbnn-frontal
npm version patch  # 1.0.0 → 1.0.1
# または
npm version minor  # 1.0.0 → 1.1.0
# または
npm version major  # 1.0.0 → 2.0.0
```

## リリースプロセス

### ステップ1: 変更をコミット

```bash
git add .
git commit -m "feat: Add new features for v1.0.1"
git push origin your-branch
```

### ステップ2: プルリクエストをマージ

main/developブランチにマージします。

### ステップ3: タグを作成してプッシュ

```bash
# バージョンを更新
cd gemma-qbnn-frontal
npm version patch

# タグを作成
git tag gemma-qbnn-frontal@1.0.1

# プッシュ
git push origin main
git push origin gemma-qbnn-frontal@1.0.1
```

### ステップ4: GitHub Actionsが自動実行

タグがプッシュされると、以下が自動で実行されます：

1. ✓ コードをチェックアウト
2. ✓ Node.jsをセットアップ
3. ✓ 依存関係をインストール
4. ✓ ビルド実行
5. ✓ テスト実行（利用可能な場合）
6. ✓ npmに公開
7. ✓ GitHubリリースを作成

## ワークフロー詳細

### npm-publish.yml

タグ`gemma-qbnn-frontal@*`がプッシュされたときにトリガーされます。

**実行内容:**
- コード取得
- Node.js 20.x セットアップ
- 依存関係インストール
- ビルド
- テスト（オプション）
- npm公開
- GitHubリリース作成

**トリガー条件:**
```
タグ形式: gemma-qbnn-frontal@<version>
例: gemma-qbnn-frontal@1.0.0
    gemma-qbnn-frontal@1.0.1
```

### test.yml

プルリクエストと特定のブランチへのプッシュ時にトリガーされます。

**実行内容:**
- Node.js 18.x, 20.x で並列テスト
- Linting
- ビルド
- テスト
- 型チェック
- カバレッジレポート

**トリガー条件:**
```
ブランチ: main, develop, feature/**
パス: gemma-qbnn-frontal/** の変更
```

## バージョニング戦略

### Semantic Versioning

MAJOR.MINOR.PATCH形式を使用します：

- **MAJOR**: 破壊的変更（APIの変更など）
- **MINOR**: 新機能（後方互換性あり）
- **PATCH**: バグ修正

### 例

```bash
# バグ修正 (1.0.0 → 1.0.1)
npm version patch

# 新機能 (1.0.1 → 1.1.0)
npm version minor

# 大きな変更 (1.1.0 → 2.0.0)
npm version major
```

## リリースノート例

GitHub Actionsが自動的に作成するリリースノートの例：

```
## Release Information

This release publishes **gemma-qbnn-frontal** to npm registry.

### Installation
npm install gemma-qbnn-frontal@1.0.1

### Features
- Gemma + QBNN ハイブリッド推論システム
- Language understanding, issue discovery, quantum judgment
- Dynamic response generation (no templates)
- APQB quantum-based decision making

### Documentation
- [README](...)
- [Package on npm](https://www.npmjs.com/package/gemma-qbnn-frontal)
```

## トラブルシューティング

### ワークフローが実行されない

1. **タグの形式を確認:**
   ```bash
   git tag -l
   # 出力: gemma-qbnn-frontal@1.0.0
   ```

2. **タグが正しくプッシュされているか確認:**
   ```bash
   git push origin --tags
   ```

3. **ワークフロー設定を確認:**
   `.github/workflows/npm-publish.yml`のタグパターン確認

### NPM認証エラー

1. **NPM_TOKENが設定されているか確認:**
   - Settings → Secrets and variables → Actions
   - `NPM_TOKEN`が存在するか確認

2. **トークンが有効か確認:**
   ```bash
   npm login
   npm token list
   ```

3. **registry-urlが正しいか確認:**
   - `registry-url: https://registry.npmjs.org/`

### ビルドエラー

1. **ローカルでビルドテスト:**
   ```bash
   cd gemma-qbnn-frontal
   npm install
   npm run build
   ```

2. **TypeScriptエラー確認:**
   ```bash
   npx tsc --noEmit
   ```

## 手動リリース

### オプション1: 自動化スクリプト（推奨）

最も簡単な方法は付属のリリーススクリプトを使用することです：

```bash
cd gemma-qbnn-frontal
npm run release
```

スクリプトは以下の処理を自動実行します：

1. リリースタイプ（Patch/Minor/Major/Custom）を選択
2. ビルドとテストを実行
3. バージョンを更新
4. npm に公開
5. Gitタグを作成してプッシュ（オプション）

**前提条件:**
```bash
npm login
```

### オプション2: 手動コマンド

自動スクリプトを使わずに手動でリリースしたい場合：

```bash
cd gemma-qbnn-frontal

# 1. ビルドとテスト実行
npm run build
npm test

# 2. バージョン更新（選択肢から1つ）
npm version patch   # 1.0.0 → 1.0.1
npm version minor   # 1.0.0 → 1.1.0
npm version major   # 1.0.0 → 2.0.0

# 3. npm に公開
npm publish --access public

# 4. リリースノート作成（オプション、GitHub CLI使用）
gh release create gemma-qbnn-frontal@1.0.1 \
  --title "gemma-qbnn-frontal v1.0.1" \
  --notes "Release notes here"
```

## モニタリング

### リリース後の確認

1. **npmnpm パッケージ確認:**
   - https://www.npmjs.com/package/gemma-qbnn-frontal

2. **GitHub リリース確認:**
   - GitHub → Releases タブ

3. **インストール確認:**
   ```bash
   npm install gemma-qbnn-frontal@latest
   ```

4. **テスト確認:**
   ```typescript
   const { GemmaQBNNEngine } = require("gemma-qbnn-frontal");
   const engine = new GemmaQBNNEngine();
   ```

## チェックリスト

リリース前の確認項目：

- [ ] 変更内容が`CHANGELOG.md`に記載されている
- [ ] `package.json`のバージョンが更新されている
- [ ] すべてのテストがパスしている
- [ ] ビルドが成功している
- [ ] TypeScript型チェックが成功している
- [ ] `NPM_TOKEN`がGitHub Secretsに設定されている
- [ ] README.md が最新である
- [ ] コミットがmain/developにマージされている

## 参考リンク

- [npm CLI Documentation](https://docs.npmjs.com/cli/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)
- [Publishing packages to npm](https://docs.npmjs.com/packages-and-modules/contributing-packages-to-the-registry)
