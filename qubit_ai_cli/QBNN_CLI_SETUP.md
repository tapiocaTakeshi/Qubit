# QBNN-Only CLI - Local Setup & Usage Guide

## 🧠 概要

QBNN-Only CLIは、Quantum-inspired Bidirectional Neural Network（量子的双方向ニューラルネットワーク）を使用した対話型分析システムです。本番環境でHuggingFace APIを使用して、実際のテキスト生成と画像分析を行います。

---

## 📋 事前準備

### 1. HuggingFace API トークンの取得

1. [HuggingFace](https://huggingface.co/) にアクセス
2. アカウントにログイン（なければ作成）
3. Settings → Access Tokens
4. `Create new token` をクリック
5. Role: `read` 以上を選択
6. トークンをコピー

### 2. リポジトリのクローン

```bash
git clone <repository-url>
cd Qubit/qubit_ai_cli
```

### 3. 依存関係のインストール

```bash
npm install
```

### 4. ビルド（TypeScript → JavaScript）

```bash
npm run build
```

---

## 🚀 実行方法

### 方法1: 本番実行（JavaScript版）

```bash
# トークンを環境変数に設定
export HF_TOKEN="your-huggingface-token"

# QBNN CLIを実行
npm run qbnn
```

### 方法2: 開発実行（TypeScript直実行）

```bash
# トークンを設定
export HF_TOKEN="your-huggingface-token"

# TypeScript直実行（変更をリアルタイムに反映）
npm run dev:qbnn
```

### 方法3: 一行で実行

```bash
HF_TOKEN="your-huggingface-token" npm run qbnn
```

---

## 💬 チャット例

### テキストで会話

```
🧠 QBNN-Only Interactive Chat System 🧠

You: AIの仕組みについて説明してください

🧠 Analyzing with quantum-inspired reasoning...

[QBNN Analysis Output...]
```

### 画像を分析

```
You: /analyze /path/to/image.png

🧠 Initiating quantum-inspired image analysis...

[多次元画像分析結果...]
```

---

## 📖 利用可能なコマンド

| コマンド | 説明 |
|---------|------|
| `/analyze <path>` | 画像ファイルをQBNNで分析 |
| `/history` | 会話履歴を表示 |
| `/export` | 会話をJSON形式でエクスポート |
| `/clear` | 会話履歴をクリア |
| `/help` | ヘルプを表示 |
| `/exit` / `/quit` | チャットを終了 |

---

## 🔍 QBNN の特徴

### Logical Decomposition（論理的分解）
- 複雑な問題を構成要素に分解
- 各要素の関係性を明確化

### Bidirectional Analysis（双方向分析）
- 複数の視点から問題を検討
- 因果関係を両方向から分析

### Structured Synthesis（構造化統合）
- 各視点の洞察を統合
- 包括的な結論を導出

### Quantum Superposition Thinking（量子的重ね合わせ思考）
- 複数の有効な解釈を同時に保持
- コンテキストに応じて最適な視点を選別

---

## 📊 画像分析の5次元フレームワーク

### 1. 視覚構造分析
- 主要要素の識別と配置
- 色彩・フォント・レイアウトの特徴
- デザイン原則の適用

### 2. 目的・文脈理解
- デザインの意図
- ターゲットユーザー
- 期待される機能

### 3. UX/UI評価
- ユーザビリティ
- アクセシビリティ
- 心理学的効果

### 4. 改善提案
- 具体的な改善ポイント
- 実装可能性の評価
- 期待される効果

### 5. 技術的実装
- 使用すべき技術スタック
- コード例（CSS/JS/TS）
- パフォーマンス考慮

---

## 📁 ファイル構成

```
qubit_ai_cli/
├── src/
│   ├── bin/
│   │   ├── qbnn-cli.ts          # QBNN-Only CLIメイン実装
│   │   ├── multi-agent-cli.ts   # マルチエージェントCLI
│   │   └── ...
│   └── ...
├── dist/                         # コンパイル後のJavaScript
├── package.json
├── tsconfig.json
└── QBNN_CLI_SETUP.md             # このファイル
```

---

## 🔧 トラブルシューティング

### エラー: "HF_TOKEN not found"
```bash
# 環境変数が正しく設定されているか確認
echo $HF_TOKEN

# 設定されていなければ設定
export HF_TOKEN="your-token"
```

### エラー: "timeout"
- ネットワーク接続を確認
- HuggingFaceサーバーの状態を確認
- タイムアウト設定を確認（デフォルト: 120秒）

### エラー: "Invalid token"
- HuggingFaceサイトでトークンが有効か確認
- トークンに必要な権限があるか確認（read以上必須）

### 会話履歴が保存されない
- `.qubit-qbnn-history/` ディレクトリの権限を確認
- ディスク容量を確認

---

## 📈 パフォーマンス情報

- **初回レスポンス時間**: 2-8秒（モデルの初期化含む）
- **通常のレスポンス**: 1-3秒
- **画像分析**: 3-5秒（複雑度による）
- **最大トークン数**: 1000トークン
- **温度設定**: 0.4（分析的で確定的）

---

## 💾 会話の永続化

### 自動保存
```
.qubit-qbnn-history/
├── qbnn-1702000000000.json
├── qbnn-1702000100000.json
└── ...
```

### 手動エクスポート
```
You: /export
✅ Conversation exported to qbnn-chat-2024-06-24.json
```

---

## 🎯 使用シーン例

### シーン1: 技術的問題の分析
```
You: Reactでのパフォーマンス最適化について教えてください

🧠 QBNN Analysis:
【問題分解】
  ✓ レンダリング最適化
  ✓ バンドルサイズ削減
  ✓ メモリ管理

【双方向分析】
  → 原因から結果へ: 不要な再レンダリング → パフォーマンス低下
  ← 結果から原因へ: 遅延検出 → React.memoやuseMemoで最適化

[詳細な分析結果...]
```

### シーン2: 画像デザイン分析
```
You: /analyze ./ui-mockup.png

🧠 QBNN Image Analysis:
【視覚構造分析】
  • レイアウト: グリッド型（3列）
  • 色彩: Material Design色体系
  • フォント: San-serif + Monospace

【UX評価】
  ✓ 視認性: 高（コントラスト比 7:1）
  ✗ アクセシビリティ: 色覚異常への対応不足
  
【改善提案】
  1. アイコンを追加（色以外の情報伝達）
  2. フォーカスインジケータを強調
  3. 小画面対応のレスポンシブ調整

[実装例とコード...]
```

---

## 📚 参考資料

- [HuggingFace Documentation](https://huggingface.co/docs)
- [Qubit AI Repository](https://github.com/tapiocaTakeshi/Qubit)
- [qubit_ai npm Package](https://www.npmjs.com/package/qubit_ai)

---

## ⚖️ ライセンス

MIT License - 自由に使用・改変・配布が可能です

---

## 📞 サポート

問題が発生した場合：
1. `.qubit-qbnn-history/` の会話ログを確認
2. エラーメッセージを記録
3. GitHubのIssuesで報告

---

**Happy QBNN Chatting! 🧠✨**
