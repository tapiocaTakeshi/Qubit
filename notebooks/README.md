# 学習用ノートブック (Notebooks)

このディレクトリには Qubit プロジェクトを理解するための学習用 Jupyter ノートブックを置きます。

## 一覧

| ファイル | 内容 |
|----------|------|
| `01_reinforcement_learning_tutorial.ipynb` | 強化学習の入門ハンズオン。多腕バンディット → Q学習 → 方策勾配 と進み、最後にこのリポジトリの DPO (`dpo_utils.py`) を実際に動かして RLHF との繋がりを学びます。 |

## 実行方法

```bash
# リポジトリのルートで依存関係をインストール
pip install numpy torch matplotlib jupyter

# ノートブックを起動
jupyter notebook notebooks/01_reinforcement_learning_tutorial.ipynb
```

> 後半の DPO セルはリポジトリの `dpo_utils.py` を import します。ノートブックは
> `sys.path` に一つ上の階層（リポジトリのルート）を追加するので、`notebooks/` 内から
> 起動しても動作します。`matplotlib` が無い環境ではグラフ部分は自動でスキップされます。
