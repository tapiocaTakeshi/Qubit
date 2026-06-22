# MCP Server Deployment Guide

QBNN Frontal Engine MCP Server のデプロイメント手順

## 概要

このプロジェクトには、QBNN量子ニューラルネットワークを利用した「Frontal Engine」という判断エンジンを提供する MCP（Model Context Protocol）サーバーが含まれています。

**Frontal Engine** は以下の機能を提供します：
- 意思決定、リスク評価、品質判定、倫理的判断などの判断タスク
- Yes/No判定とスコア（0-100）を返す
- QBNN モデルを使用した高度な判断ロジック

## ローカルでの実行

### 前提条件

- Python 3.11以上
- PyTorch 2.4.0
- 必要なパッケージ（requirements.txtで自動インストール）

### 実行手順

```bash
# 方法1: スクリプトを使用（推奨）
./run_mcp_server.sh

# 方法2: 直接Pythonで実行
python3 -u frontal_engine_mcp_server.py
```

MCPサーバーが起動し、標準入出力でJSONRPCプロトコルを使用してクライアントからのリクエストを待機します。

### テスト

```bash
python3 test_mcp_client.py
```

このコマンドは以下をテストします：
1. 必要なモジュールのインポート
2. MCPサーバー設定ファイルの妥当性
3. MCPサーバーの起動と通信

## Docker でのデプロイメント

### ローカルビルド

```bash
docker build -f docker/Dockerfile.mcp -t qubit-mcp-server:latest .
```

### ローカル実行

```bash
docker run -it qubit-mcp-server:latest
```

## Container Registry へのデプロイ

プロジェクトには GitHub Actions ワークフロー（`.github/workflows/deploy-mcp-server.yml`）が含まれており、以下を自動化します：

1. **自動ビルド・プッシュ**: main ブランチに MCP関連のファイルが変更されたときに自動実行
2. **イメージ管理**: GitHub Container Registry (ghcr.io) にコンテナイメージを保存
3. **テスト**: 各ビルド後にMCPサーバーの起動テストを実行

### イメージのプル

```bash
# GitHub Container Registryからプル
docker pull ghcr.io/tapiocatakeshi/qubit/mcp-server:latest
```

### ワークフロー トリガー

ワークフローは以下の場合に自動実行されます：

- `frontal_engine_mcp_server.py` が変更された
- `frontal_engine.mcp.json` が変更された
- `neuroquantum_layered.py` または `qbnn_layered.py` が変更された
- `docker/Dockerfile.mcp` が変更された
- `requirements.txt` が変更された

手動トリガーも可能です（GitHub UI または API 経由）。

## MCPサーバーの仕様

### ツール: judge

判断タスクを実行し、Yes/No判定、スコア、根拠説明を返します。

**入力スキーマ:**
```json
{
  "context": "判断の背景情報・文脈（必須）",
  "judgment_request": "何を判断するか・判断内容（必須）",
  "criteria": {...},
  "options": [...],
  "strict_mode": false
}
```

**出力スキーマ:**
```json
{
  "decision": "Yes or No",
  "score": 0-100,
  "reasoning": "判断の根拠説明",
  "confidence": "high, medium, or low",
  "key_factors": ["要因1", "要因2", ...],
  "timestamp": "ISO形式の時刻"
}
```

### 使用例

```bash
# JSONRPCリクエスト
cat << 'EOF' | python3 frontal_engine_mcp_server.py
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}

{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "judge",
    "arguments": {
      "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。",
      "judgment_request": "このプロジェクトをリリースしても安全か？"
    }
  }
}
EOF
```

## 本番環境での利用

### Kubernetes へのデプロイ

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: frontal-engine-mcp
spec:
  containers:
  - name: mcp-server
    image: ghcr.io/tapiocatakeshi/qubit/mcp-server:latest
    ports:
    - containerPort: 8000
    env:
    - name: PYTHONUNBUFFERED
      value: "1"
    resources:
      requests:
        memory: "2Gi"
        cpu: "1000m"
      limits:
        memory: "4Gi"
        cpu: "2000m"
```

### 環境変数

MCPサーバーは以下の環境変数をサポートしています：

| 変数 | 説明 | デフォルト |
|------|------|----------|
| `PYTHONUNBUFFERED` | Python出力のバッファリング無効化 | `1` |
| `MODEL_DIR` | モデルファイルのディレクトリ | `/app` |

## トラブルシューティング

### MCPサーバーが起動しない

1. Python のバージョンを確認：`python3 --version` （3.11以上が必要）
2. 依存パッケージをインストール：`pip install -r requirements.txt`
3. トークナイザーファイルが存在するか確認：`ls -la neuroq_tokenizer.*`

### Docker ビルドが失敗する

1. ベースイメージが入手可能か確認：`docker pull python:3.11-slim`
2. 全てのファイルがチェックアウトされているか確認
3. ディスク容量を確認

### メモリ不足エラー

MCPサーバーは GPU に対応しているため、CUDA メモリを確認：

```bash
nvidia-smi
```

QBNN モデルの読み込みには CPU メモリが必要な場合があります。

## セキュリティ考慮事項

- MCPサーバーは信頼できるクライアントからのみアクセスを許可してください
- 本番環境では TLS/SSL エンクリプションを使用してください
- 定期的にセキュリティアップデートを適用してください
- ログをモニタリングして異常なアクセスを検出してください

## パフォーマンス最適化

### キャッシング

MCPサーバーは以下をキャッシュします：
- QBNN モデルと重み
- トークナイザー

### スケーリング

複数の MCPサーバーインスタンスを実行する場合：
- ロードバランサーを使用
- 状態を保持しないことを確認
- 定期的なモニタリングを実装

## ドキュメント

- MCP仕様: https://modelcontextprotocol.io/
- QBNN理論: `README.md` を参照
- MCPサーバー実装: `frontal_engine_mcp_server.py` を参照

## ライセンス

MIT License - 詳細は LICENSE ファイルを参照

## サポート

問題が発生した場合は、GitHub Issues で報告してください。
