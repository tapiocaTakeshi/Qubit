# QBNN Frontal Engine - デプロイメントガイド

このガイドでは、QBNN Frontal Engine MCP サーバーを3つの異なる方法でデプロイする手順を説明します。

---

## 目次
1. [Claude Code 統合](#claude-code-統合)
2. [スタンドアロンサービス](#スタンドアロンサービス)
3. [Docker コンテナ](#docker-コンテナ)
4. [Systemd サービス](#systemd-サービス)
5. [トラブルシューティング](#トラブルシューティング)

---

## Claude Code 統合

Claude Code で QBNN Frontal Engine を MCP サーバーとして使用します。

### セットアップ

1. **設定ファイルを確認**
   ```bash
   cat .claude/settings.json
   ```

2. **Claude Code を再起動**
   - Claude Code クライアント（CLI、Web、IDE拡張）を再起動します
   - MCP サーバーが自動的に登録されます

3. **確認**
   Claude Code の MCP ツールパネルで `judge` ツールが利用可能になります。

### 使用例

```python
# Claude Code 内で以下のように使用
result = await use_mcp_tool("judge", {
    "context": "プロジェクトは完了し、すべてのテストに合格しました。",
    "judgment_request": "リリースしても安全か？"
})
```

### トラブルシューティング

- **MCP サーバーが起動しない**
  ```bash
  python frontal_engine_mcp_server.py
  ```
  で直接実行してエラーを確認

- **パッケージが不足**
  ```bash
  pip install -r requirements.txt
  ```

---

## スタンドアロンサービス

システムサービスとして独立して実行します。

### セットアップ

1. **依存パッケージをインストール**
   ```bash
   pip install -r requirements.txt
   ```

2. **スクリプトで起動**
   ```bash
   ./run_frontal_engine.sh start
   ```

3. **ステータス確認**
   ```bash
   ./run_frontal_engine.sh status
   ```

4. **ログを確認**
   ```bash
   ./run_frontal_engine.sh logs
   ```

### コマンドリファレンス

```bash
./run_frontal_engine.sh start      # 起動
./run_frontal_engine.sh stop       # 停止
./run_frontal_engine.sh restart    # 再起動
./run_frontal_engine.sh status     # ステータス表示
./run_frontal_engine.sh logs       # ログ表示
./run_frontal_engine.sh test       # テスト実行
```

### プロセス管理

- **PID ファイル**: `/tmp/qbnn-frontal-engine.pid`
- **ログファイル**: `/tmp/qbnn-frontal-engine.log`
- **自動リスタート**: 最大3回まで自動的に再起動

### 統合例

他のアプリケーションから stdio経由でアクセス：

```python
import json
import subprocess

# MCP サーバーを起動
process = subprocess.Popen(
    ["python", "frontal_engine_mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# JSON-RPC リクエストを送信
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "judge",
        "arguments": {
            "context": "...",
            "judgment_request": "..."
        }
    }
}

process.stdin.write(json.dumps(request) + "\n")
response = process.stdout.readline()
result = json.loads(response)
```

---

## Docker コンテナ

コンテナ化されたサービスとして実行します。

### イメージのビルド

```bash
./run_frontal_engine.sh docker-build
```

または直接：

```bash
docker build -f Dockerfile.mcp -t qbnn-frontal-engine:latest .
```

### コンテナの実行

#### インタラクティブ実行（開発・テスト用）

```bash
./run_frontal_engine.sh docker-run
```

または：

```bash
docker run -it --rm qbnn-frontal-engine:latest
```

#### バックグラウンド実行（本番用）

```bash
./run_frontal_engine.sh docker-daemon
```

または：

```bash
docker run -d \
  --name qbnn-frontal-engine \
  --restart unless-stopped \
  -v $(pwd):/app \
  qbnn-frontal-engine:latest
```

### コンテナ管理

```bash
# ステータス確認
docker ps | grep qbnn-frontal-engine

# ログ表示
docker logs -f qbnn-frontal-engine

# コンテナ停止
docker stop qbnn-frontal-engine

# コンテナ削除
docker rm qbnn-frontal-engine
```

### Docker Compose での運用

`docker-compose.yml` を作成：

```yaml
version: '3.8'

services:
  frontal-engine:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    container_name: qbnn-frontal-engine
    restart: unless-stopped
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONPATH=/app
      - MODEL_DIR=/app
    healthcheck:
      test: ["CMD", "python", "-c", "from frontal_engine_mcp_server import judge"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
```

実行：

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Systemd サービス

Linux システムでシステムサービスとして登録します。

### セットアップ

1. **サービスファイルをコピー**
   ```bash
   sudo cp qbnn-frontal-engine.service /etc/systemd/system/
   ```

2. **デーモンをリロード**
   ```bash
   sudo systemctl daemon-reload
   ```

3. **サービスを有効化**
   ```bash
   sudo systemctl enable qbnn-frontal-engine
   ```

4. **サービスを起動**
   ```bash
   sudo systemctl start qbnn-frontal-engine
   ```

### コマンドリファレンス

```bash
# ステータス確認
sudo systemctl status qbnn-frontal-engine

# ログ表示
sudo journalctl -u qbnn-frontal-engine -f

# 再起動
sudo systemctl restart qbnn-frontal-engine

# 停止
sudo systemctl stop qbnn-frontal-engine

# 無効化
sudo systemctl disable qbnn-frontal-engine
```

### ポート設定

デフォルトでは stdio で通信します。HTTP インターフェースが必要な場合は、別途ラッパーを作成してください。

### トラブルシューティング

- **サービスが起動しない**
  ```bash
  sudo journalctl -u qbnn-frontal-engine -n 50
  ```

- **Python パスの問題**
  `qbnn-frontal-engine.service` の `ExecStart` と `WorkingDirectory` を確認

- **メモリ不足**
  `MemoryLimit=4G` を調整

---

## トラブルシューティング

### 一般的な問題

#### 1. PyTorch がインストールされていない

```bash
pip install torch
```

#### 2. MCP パッケージが不足

```bash
pip install mcp
```

#### 3. トークナイザーが見つからない

```bash
python train_tokenizer.py 8000 . 20000
```

#### 4. ポート競合

```bash
# ポート使用状況を確認
lsof -i :3000  # または対象ポート

# プロセスを強制終了
kill -9 <PID>
```

#### 5. メモリ不足

- Docker メモリ制限を増やす
- Systemd の `MemoryLimit` を増やす
- `max_seq_len` を削減

### デバッグモード

```bash
# Python ログレベルを設定
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
python frontal_engine_mcp_server.py
```

### ログの確認

```bash
# スタンドアロン
tail -f /tmp/qbnn-frontal-engine.log

# Docker
docker logs -f qbnn-frontal-engine

# Systemd
sudo journalctl -u qbnn-frontal-engine -f
```

---

## パフォーマンス最適化

### リソース使用量の削減

1. **モデルサイズの縮小**
   - `embed_dim`: 512 → 256
   - `hidden_dim`: 1024 → 512
   - `num_layers`: 6 → 4

2. **バッチサイズの調整**
   - 同時判断数を制限

3. **キャッシング**
   - 同じ判断リクエストの結果をキャッシュ

### スケーリング

複数インスタンスの実行：

```bash
# 3つのインスタンスを起動
for i in 1 2 3; do
  PORT=300$i ./run_frontal_engine.sh start
done
```

---

## セキュリティ考慮事項

1. **認証**: MCP サーバーの前にリバースプロキシを配置して認証を追加
2. **入力検証**: `context` と `judgment_request` のサイズを制限
3. **レート制限**: DDoS 対策としてレート制限を実装
4. **監視**: ログを中央監視システムに送信

---

## サポートとフィードバック

問題が発生した場合や改善提案がある場合は、GitHub Issues で報告してください。

```
Repository: tapiocaTakeshi/Qubit
Branch: claude/qbnn-frontal-engine-server-lj9iz3
```
