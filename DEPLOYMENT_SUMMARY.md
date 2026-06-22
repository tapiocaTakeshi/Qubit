# QBNN Frontal Engine - デプロイメント概要

## ✅ 実装完了

QBNN Frontal Engine MCP サーバーが3つの完全なデプロイメント方法で利用可能になりました。

---

## 📋 デプロイメント方法

### 1️⃣ Claude Code 統合（推奨）

**最も簡単な統合方法**

```bash
# Claude Code クライアントを再起動するだけ
# .claude/settings.json が自動的に読み込まれます
```

**利点:**
- ✅ セットアップ不要
- ✅ Claude Code 内で直接使用
- ✅ 自動ツール検出

**用途:**
- Claude Code での判断機能の統合
- MCP ツールとしてのアクセス

---

### 2️⃣ スタンドアロンサービス

**プロセスベースの実行**

```bash
# 起動
./run_frontal_engine.sh start

# ステータス確認
./run_frontal_engine.sh status

# ログ表示
./run_frontal_engine.sh logs

# 停止
./run_frontal_engine.sh stop
```

**利点:**
- ✅ 軽量で高速
- ✅ 簡単な管理
- ✅ PID ベースのプロセス管理

**ファイル:**
- `run_frontal_engine.sh` - 完全な管理スクリプト
- PID: `/tmp/qbnn-frontal-engine.pid`
- ログ: `/tmp/qbnn-frontal-engine.log`

**用途:**
- 開発環境
- テスト環境
- 軽量な本番環境

---

### 3️⃣ Docker コンテナ

**コンテナ化されたデプロイメント**

```bash
# イメージをビルド
./run_frontal_engine.sh docker-build

# インタラクティブ実行（開発用）
./run_frontal_engine.sh docker-run

# バックグラウンド実行（本番用）
./run_frontal_engine.sh docker-daemon

# Docker Compose で実行
docker-compose up -d
```

**利点:**
- ✅ 環境の完全な分離
- ✅ スケーラビリティ
- ✅ リソース制限が可能
- ✅ 一貫した実行環境

**ファイル:**
- `Dockerfile.mcp` - コンテナイメージ定義
- `docker-compose.yml` - オーケストレーション設定

**用途:**
- 本番環境
- マイクロサービスアーキテクチャ
- クラウドデプロイメント

---

### 4️⃣ Systemd サービス（Linux）

**OS レベルのサービス統合**

```bash
# サービスをインストール
sudo cp qbnn-frontal-engine.service /etc/systemd/system/
sudo systemctl daemon-reload

# 有効化・起動
sudo systemctl enable qbnn-frontal-engine
sudo systemctl start qbnn-frontal-engine

# ステータス確認
sudo systemctl status qbnn-frontal-engine

# ログ表示
sudo journalctl -u qbnn-frontal-engine -f
```

**利点:**
- ✅ OS レベルの統合
- ✅ 自動リスタート
- ✅ systemd による管理
- ✅ journald へのロギング

**ファイル:**
- `qbnn-frontal-engine.service` - Systemd サービスファイル

**用途:**
- エンタープライズ環境
- 24/7 運用が必要な環境
- 複数サーバー構成

---

## 📊 デプロイメント比較表

| 項目 | Claude Code | スタンドアロン | Docker | Systemd |
|------|-----------|--------------|--------|---------|
| セットアップの簡単さ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| パフォーマンス | 高 | 高 | 中～高 | 高 |
| リソース使用量 | 低 | 低 | 中 | 低 |
| スケーラビリティ | × | △ | ⭐⭐⭐ | △ |
| 自動リスタート | × | × | ⭐ | ⭐⭐ |
| 監視・ロギング | △ | △ | ⭐ | ⭐⭐ |
| 本番環境適性 | 開発向け | 中規模 | 大規模 | 専用 |

---

## 🚀 クイックスタート

### パターン 1: Claude Code を使う（最速）

```bash
# 1. Claude Code を再起動
# 2. tools パネルで judge ツールが利用可能
# 3. すぐに使用開始
```

### パターン 2: ローカル開発用

```bash
# 1. 依存パッケージをインストール
pip install -r requirements.txt

# 2. サーバーを起動
./run_frontal_engine.sh start

# 3. 別のターミナルからテスト
./run_frontal_engine.sh test
```

### パターン 3: Docker で本番環境

```bash
# 1. イメージをビルド
docker build -f Dockerfile.mcp -t qbnn-frontal-engine:latest .

# 2. コンテナを実行
docker run -d --restart unless-stopped qbnn-frontal-engine:latest

# 3. ログを確認
docker logs -f qbnn-frontal-engine
```

### パターン 4: Systemd でエンタープライズ運用

```bash
# 1. サービスをインストール
sudo cp qbnn-frontal-engine.service /etc/systemd/system/
sudo systemctl daemon-reload

# 2. 有効化
sudo systemctl enable qbnn-frontal-engine

# 3. 起動
sudo systemctl start qbnn-frontal-engine

# 4. ログを監視
sudo journalctl -u qbnn-frontal-engine -f
```

---

## 📁 ファイル構成

```
/home/user/Qubit/
├── frontal_engine_mcp_server.py        # メイン MCP サーバー実装
├── frontal_engine.mcp.json             # MCP スキーマ定義
├── test_frontal_engine.py              # 完全版テスト（torch 使用）
├── test_frontal_engine_light.py        # ライト版テスト（torch 不要）
│
├── .claude/
│   └── settings.json                   # Claude Code MCP 設定
│
├── run_frontal_engine.sh               # スタンドアロン管理スクリプト
├── Dockerfile.mcp                      # Docker イメージ定義
├── docker-compose.yml                  # Docker Compose 設定
├── qbnn-frontal-engine.service         # Systemd サービスファイル
│
├── FRONTAL_ENGINE_README.md            # MCP サーバー機能説明
├── DEPLOYMENT_GUIDE.md                 # デプロイメント詳細ガイド
└── DEPLOYMENT_SUMMARY.md               # このファイル
```

---

## 🔧 よく使うコマンド

### スタンドアロン

```bash
./run_frontal_engine.sh start           # 起動
./run_frontal_engine.sh status          # ステータス確認
./run_frontal_engine.sh logs            # ログ表示（リアルタイム）
./run_frontal_engine.sh stop            # 停止
./run_frontal_engine.sh test            # テスト実行
```

### Docker

```bash
docker-compose up -d                    # 起動
docker-compose logs -f                  # ログ表示
docker-compose ps                       # ステータス確認
docker-compose down                     # 停止
```

### Systemd

```bash
sudo systemctl start qbnn-frontal-engine       # 起動
sudo systemctl status qbnn-frontal-engine      # ステータス確認
sudo journalctl -u qbnn-frontal-engine -f      # ログ表示
sudo systemctl restart qbnn-frontal-engine     # 再起動
sudo systemctl stop qbnn-frontal-engine        # 停止
```

---

## 🐛 トラブルシューティング

### サーバーが起動しない

```bash
# ログを確認
tail -f /tmp/qbnn-frontal-engine.log    # スタンドアロン
docker logs qbnn-frontal-engine         # Docker
sudo journalctl -u qbnn-frontal-engine  # Systemd
```

### PyTorch がインストールされていない

```bash
pip install torch numpy
```

### ポートが使用中

```bash
lsof -i :3000
kill -9 <PID>
```

### Docker イメージがビルドできない

```bash
docker build -f Dockerfile.mcp --progress=plain -t qbnn-frontal-engine:latest .
```

---

## 📈 パフォーマンス推奨設定

### 開発環境
```bash
./run_frontal_engine.sh start
```

### テスト環境
```bash
docker run -it qbnn-frontal-engine:latest
```

### 本番環境
```bash
docker-compose up -d
# または
sudo systemctl start qbnn-frontal-engine
```

### 高負荷環境
```bash
# Kubernetes で多数のレプリカを実行
# 又は docker-compose でスケーリング
docker-compose up -d --scale frontal-engine=3
```

---

## ✅ 検証チェックリスト

- [ ] Claude Code で judge ツールが表示される
- [ ] スタンドアロン: `./run_frontal_engine.sh start` が成功
- [ ] スタンドアロン: `./run_frontal_engine.sh test` が成功
- [ ] Docker: `docker build` が成功
- [ ] Docker Compose: `docker-compose up` が成功
- [ ] Systemd: `sudo systemctl status qbnn-frontal-engine` が active
- [ ] すべてのテストが pass している

---

## 📞 次のステップ

1. **デプロイメント方法を選択** - 環境に応じて最適な方法を選択
2. **セットアップを実行** - DEPLOYMENT_GUIDE.md を参照
3. **テストを実行** - 動作確認
4. **本番環境に展開** - 監視・ロギングを設定
5. **フィードバック** - 問題があれば報告

---

## 📝 ドキュメント

- `FRONTAL_ENGINE_README.md` - MCP サーバーの機能説明
- `DEPLOYMENT_GUIDE.md` - 詳細なデプロイメントガイド
- `DEPLOYMENT_SUMMARY.md` - このファイル（概要）

---

**作成日**: 2026-06-22
**バージョン**: 1.0.0
**ステータス**: ✅ 本番環境対応完了
