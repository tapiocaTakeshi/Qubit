# Qubit Chat — TUI チャットボット CLI

[Textual](https://textual.textualize.io/) 製のフルスクリーン・ターミナルチャット
インターフェースです。Qubit の **NeuroQuantum** 推論エンジンと対話できます。

![chat bubbles](https://img.shields.io/badge/UI-Textual-blueviolet)

## インストール

```bash
pip install textual          # もしくは pip install -r ../requirements.txt
```

NeuroQuantum / Quantum / Echo バックエンドは pure Python なので追加依存はありません。
学習済みモデル（torch）を使う場合のみ `torch` とチェックポイントが必要です。

## 起動

```bash
# プロジェクトルートから
python -m tui_chat.app
# もしくはランチャー
python qubit_chat.py

# バックエンドを指定
python -m tui_chat.app --backend neuroquantum         # 既定 (会話エンジン, pure Python)
python -m tui_chat.app --backend neuroquantum-model   # 学習済みモデル (torch + checkpoint)
python -m tui_chat.app --backend quantum              # 判断エンジン (pure Python)
python -m tui_chat.app --backend echo                 # 動作確認用
```

引数を省略すると、利用可能な最良のバックエンドを自動選択します
（`neuroquantum` → `quantum` → `echo`）。

## 使い方

メッセージを入力して **Enter** で送信。応答は UI をブロックしないよう
別スレッドで生成されます。

### スラッシュコマンド

| コマンド | 説明 |
| --- | --- |
| `/help` | コマンド一覧を表示 |
| `/clear` | 会話履歴をクリア |
| `/backend <name>` | バックエンドを切替 (`neuroquantum`, `neuroquantum-model`, `quantum`, `echo`) |
| `/quit` | 終了 |

### キーバインド

| キー | 動作 |
| --- | --- |
| `Enter` | メッセージ送信 |
| `Ctrl+L` | 会話をクリア |
| `Ctrl+C` | 終了 |

## 構成

```
tui_chat/
├── app.py       # Textual アプリ本体 (UI / 入力 / ワーカー)
├── backend.py   # バックエンド抽象化と各エンジン実装
└── __init__.py
```

`backend.py` の `Backend` プロトコルは `reply(message, history) -> str` のみ。
新しいエンジンを追加するには、このメソッドを実装したクラスを `_BACKENDS` に
登録するだけです。
