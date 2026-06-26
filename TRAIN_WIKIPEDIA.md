# wikimedia/wikipedia 全データ学習ガイド

`train_wikipedia.py` は HuggingFace の [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
データセットの **全記事** を 3 エポック学習し、NeuroQuantum モデルのチェックポイントを生成する。

## 特徴

- **バッチ処理（map-style）がデフォルト**: `wikimedia/wikipedia` は Parquet/Arrow 形式で、
  `load_dataset` が Arrow ファイルをメモリマップする。全記事（日本語版約140万・英語版約640万）を
  RAM に載せず、ディスクから逐次読みつつ **バッチ単位でトークナイズして即学習・破棄**するため、
  メモリ使用量は 1 バッチ分で一定。
- **3 エポック**: データ全体を 3 回通過する（環境変数で変更可）。
  バッチ処理なら初回 DL 1 回でキャッシュを再利用でき、全記事を完全シャッフルできる。
- **エポックごとのチェックポイント保存**: 途中で中断しても再開可能。
- トークナイザ（SentencePiece）は先頭サンプル記事から構築し、学習自体は全データで行う。
- **streaming フォールバック**: ディスク容量が厳しい場合は `WIKI_STREAMING=1` で
  ストリーミング読み込みに切り替えられる（毎エポック再取得・バッファ内シャッフルになる）。

### バッチ処理 vs streaming

| | streaming | バッチ処理 (デフォルト) |
| --- | --- | --- |
| メモリ | 一定 | 一定（Arrow メモリマップ） |
| 3 エポックの DL | 毎エポック再取得（3回） | 初回 1 回 DL → キャッシュ再利用 |
| シャッフル | バッファ内のみ | 全記事を完全シャッフル |
| ディスク使用量 | 小 | 大（全データを DL） |

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

# ディスク節約（streaming フォールバック）
WIKI_STREAMING=1 python train_wikipedia.py
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
| `WIKI_STREAMING` | `0` | `1` で streaming フォールバック（デフォルトはバッチ処理） |

## 生成物

| ファイル | 内容 |
| --- | --- |
| `neuroq_wikipedia_checkpoint.pt` | 学習済みモデルチェックポイント |
| `neuroq_wikipedia_tokenizer.model` / `.vocab` | SentencePiece トークナイザ |
| `training_history_wikipedia.json` | 学習履歴（loss / 記事数 / パラメータ数など） |
