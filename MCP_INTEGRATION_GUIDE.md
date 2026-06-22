# Claude DesktopおよびCursor統合ガイド

QBNN Frontal Engine MCPサーバーをClaude DesktopとCursor IDEに統合する手順です。

## 前提条件

- Python 3.11以上がインストール済み
- MCPパッケージがインストール済み: `pip install 'mcp>=0.5.0'`
- QBNN Frontal Engineプロジェクトをローカルに配置

## 1. Claude Desktop統合

### 設定ファイルの場所

**macOS/Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

### 設定内容

以下の設定を `claude_desktop_config.json` の `mcpServers` セクションに追加します：

```json
{
  "mcpServers": {
    "qbnn-frontal-engine": {
      "command": "python",
      "args": ["/path/to/Qubit/frontal_engine_mcp_server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**パスを置き換えてください:**
- `/path/to/Qubit` をQubitプロジェクトの実際のパスに変更
  - macOS/Linux: `/Users/username/Qubit`
  - Windows: `C:\Users\username\Qubit`

### 完全な設定例

```json
{
  "mcpServers": {
    "qbnn-frontal-engine": {
      "command": "python",
      "args": ["/Users/username/Qubit/frontal_engine_mcp_server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## 2. Cursor IDE統合

### 設定ファイルの場所

**macOS/Linux:**
```
~/.cursor/settings.json
```

**Windows:**
```
%APPDATA%\.cursor\settings.json
```

### 設定内容

Cursorの設定ファイルに以下を追加します：

```json
{
  "mcp": {
    "servers": {
      "qbnn-frontal-engine": {
        "command": "python",
        "args": ["/path/to/Qubit/frontal_engine_mcp_server.py"],
        "env": {
          "PYTHONUNBUFFERED": "1"
        }
      }
    }
  }
}
```

### 完全な設定例

```json
{
  "mcp": {
    "servers": {
      "qbnn-frontal-engine": {
        "command": "python",
        "args": ["/Users/username/Qubit/frontal_engine_mcp_server.py"],
        "env": {
          "PYTHONUNBUFFERED": "1"
        }
      }
    }
  }
}
```

## 3. 統合確認

### Claude Desktop

1. 設定ファイルを保存
2. Claude Desktopを再起動
3. MCP Tools セクションで `qbnn-frontal-engine` が表示されることを確認
4. `judge` ツールが利用可能か確認

### Cursor IDE

1. 設定ファイルを保存
2. Cursorを再起動
3. MCPサーバーが接続されたことを確認
4. Copilotチャットで `@judge` を使用可能か確認

## 4. 使用方法

### Claude Desktop

```
前頭葉として、以下の判断を行ってください：

Context: プロジェクトは予定通り進行し、テスト成功率95%です
Question: このプロジェクトをリリースしても安全ですか？
```

Claudeが自動的に `judge` ツールを使用します。

### Cursor IDE

Copilotチャットで：

```
@judge プロジェクトをリリースしても安全か判断してください。
背景: テスト成功率95%、品質基準すべて満たし、リスク要因なし
```

## トラブルシューティング

### サーバーが起動しない場合

1. Pythonパスが正しいか確認：
```bash
which python  # macOS/Linux
where python  # Windows
```

2. 必要なパッケージがインストールされているか確認：
```bash
pip list | grep mcp
```

3. サーバーを手動テスト：
```bash
python /path/to/frontal_engine_mcp_server.py
```

### ツールが表示されない場合

1. ログを確認（Claude Desktop）: メニュー > Logs
2. 設定ファイルのJSON形式が正しいか確認
3. ファイルパスが正しいか確認（相対パスではなく絶対パスを使用）

## 高度な設定

### カスタム環境変数

```json
{
  "env": {
    "PYTHONUNBUFFERED": "1",
    "DEBUG": "1"
  }
}
```

### 複数のMCPサーバー

他のMCPサーバーを同時に使用可能です：

```json
{
  "mcpServers": {
    "qbnn-frontal-engine": {...},
    "other-mcp-server": {...}
  }
}
```

## セキュリティに関する注意

- 設定ファイルはローカルマシンにのみ保存してください
- パスに機密情報を含めないでください
- 本番環境では環境変数を使用してパスを管理することを推奨

## 参考リンク

- [Claude API 公式ドキュメント](https://claude.ai)
- [MCP Protocol 仕様](https://modelcontextprotocol.io)
- FRONTAL_ENGINE_README.md（このプロジェクト内）
