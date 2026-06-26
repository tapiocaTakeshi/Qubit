# wikimedia/wikipedia 全データ学習ガイド

`train_wikipedia.py` は HuggingFace の [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
データセットの **全記事** を 3 エポック学習し、NeuroQuantum モデルのチェックポイントを生成する。

## 特徴

- **全データ対応**: 全記事をメモリに載せず、`datasets` の streaming モードで逐次読み込み。
  一定サイズのバッファ（シャード）単位で学習する「ストリーミング・シャード学習」方式。
  日本語版（約140万記事）/ 英語版（約640万記事）でもメモリ使用量を一定に保てる。
- **3 エポック**: データ全体を 3 回通過する（環境変数で変更可）。
- **エポックごとのチェックポイント保存**: 途中で中断しても再開可能。
- トークナイザ（SentencePiece）は先頭サンプル記事から構築し、学習自体は全データで行う。

## 実行

```bash
# 日本語版・全データ・3エポック（デフォルト）
python train_wikipedia.py

# 英語版で学習
WIKI_CONFIG=20231101.en python train_wikipedia.py

# 日英両方を学習
WIKI_CONFIG=20231101.ja,20231101.en python train_wikipedia.py

# 動作確認（各 config から 5000 記事だけ）
WIKI_MAX_SAMPLES=5000 python train_wikipedia.py
```

> **Note**: フル Wikipedia 学習は GPU 必須で、相応の計算時間がかかる。
> RunPod / Modal などの GPU 環境での実行を想定している（`requirements.txt` の
> torch / datasets / sentencepiece が必要）。

## 環境変数

| 変数 | デフォルト | 説明 |
| --- | --- | --- |
| `WIKI_CONFIG` | `20231101.ja` | 学習対象の config 名。カンマ区切りで複数指定可 |
| `WIKI_DATASET` | `wikimedia/wikipedia` | データセット ID |
| `WIKI_EPOCHS` | `3` | エポック数 |
| `WIKI_MAX_SAMPLES` | `0`（=全データ） | 各 config から使う最大記事数（デバッグ用） |
| `WIKI_TOKENIZER_SAMPLES` | `200000` | トークナイザ語彙構築に使う記事数 |
| `WIKI_BUFFER_SIZE` | `50000` | 1 シャードあたりの学習シーケンス数 |

## 生成物

| ファイル | 内容 |
| --- | --- |
| `neuroq_wikipedia_checkpoint.pt` | 学習済みモデルチェックポイント |
| `neuroq_wikipedia_tokenizer.model` / `.vocab` | SentencePiece トークナイザ |
| `training_history_wikipedia.json` | 学習履歴（loss / 記事数 / パラメータ数など） |
