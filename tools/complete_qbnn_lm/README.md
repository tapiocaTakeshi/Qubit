# 完全QBNN 文字言語モデル（学習・評価・ブラウザ実行）

FFN のニューロンを完全ブロッホ球にした文字単位の GPT を、日本語の文章・会話・コードで学習するためのスクリプト群。
学習したモデルはブラウザ用に書き出し、`browser/` のチャット画面（観測室）で動かせる。CPU だけで動く。

## 構成

| ファイル | 役割 |
|---|---|
| `scale_exp.py` | モデル本体（`GPT`、FFN の種類 `gelu` / `tanh` / `complete` / `nq`、大きさ `XS`〜`XL`）と、同じ条件でニューロンの種類を比べる実験 |
| `chat_data.py` | 会話データ（oasst 形式 `{"conversations": [{"from", "value"}]}`）を `<bos>ユーザー<sep>応答<eos>` の列にする |
| `sft.py`, `sft_chat.py` | 指示・会話データでの追加学習（初期の M モデル用） |
| `cont.py` | 文章（損失は全文字）と会話・指示（損失は応答と `<eos>` だけ）を半々に混ぜた学習。300 ステップごとに保存し、再実行で続きから再開する |
| `cont2.py` | `cont.py` に青空文庫・Wikipedia 話題の多段会話・llm-japanese-dataset を加えたもの |
| `cont3.py` | `cont2.py` に Magpie-Tanuki・Nemotron 会話・hh-rlhf 日本語・JaQuAD・Wikinews を加えたもの |
| `prep_new.py`, `prep_llmjp.py`, `prep_new2.py` | 上の追加データを、語彙内の文字だけ・文脈 256 文字に収まるものに絞って保存する |
| `eval_iq.py` | 「賢さ」テスト。選択肢ごとの尤度で答えを選ばせる（JCommonsenseQA・首都・県庁所在地・足し算・助詞） |
| `eval_metrics.py` | 損失以外の指標（top-k 正解率・較正誤差・生成の多様性など） |
| `infer_xl3.py` | 2 つのモデルに同じ質問・同じ乱数で答えさせて比べる |
| `export_gpt.py` | 学習済みモデルを fp16 にしてブラウザ用に書き出す（`meta_gpt.json`、12 MB ごとの `weights_k.wasm`、PyTorch の参照出力 `ref_gpt.json`） |
| `browser/engine_gpt.js` | ブラウザ・Node.js 用の推論エンジン（KV キャッシュ、文脈が満ちたら半分を残してずらす） |
| `browser/template_chat_bin.html`, `browser/build_gpt_bin.py` | チャット画面（観測室）。生成中の完全ブロッホ球ニューロンの状態を表示する |
| `browser/test_gpt.js` | エンジンの出力が PyTorch と一致するかを確かめる |

## データの置き場所

学習データとモデルは環境変数 `QBNN_LM_DATA` のディレクトリに置く（既定は `tools/complete_qbnn_lm/data/`）。
`prep_*.py` が読む Hugging Face のダウンロードは `QBNN_LM_DOWNLOADS`（既定は `$QBNN_LM_DATA/downloads`）。

使ったファイルは次のとおり。

- 文章: `jawiki*.txt`（日本語版 Wikipedia）、`fineweb_train.txt` / `fineweb_val.txt`（FineWeb-2 日本語）、`aozora.txt` / `news_aozora.txt`（青空文庫・Wikinews）、`code.txt`（Python）
- 会話: `oasst1-chat-44k-ja.json`、`oasst2-chat-68k-ja.json`、`conv_wiki.json`、`chat_new3.json`
- 指示: `instruct_raw.json`（dolly-ja・amenokaku-code・japanese_alpaca）、`llmjp_mix.json`、`instr_new3.json`、`jaquad_val.json`
- 語彙: `vocab.json`（Wikipedia で多い 4,000 文字）、`sft_meta.json`（語彙に `<bos>` `<sep>` `<eos>` `<pad>` を足したもの）

## 使い方

```bash
export QBNN_LM_DATA=/path/to/data

# ニューロンの種類の比較（M, 6000 ステップ）
python scale_exp.py M complete 6000 0 4

# 文章＋会話の学習: STEPS INIT LM_ROWS CHAT_ROWS LR TAG 文章ファイル SIZE WARMUP
python cont3.py 40000 model_M_complete_cont_XL260000.pt 10 6 2e-4 _XL3 fineweb_train.txt+jawiki3.txt XL 300

# 賢さテスト（:sft は特殊トークン入りの語彙と文脈 256）
python eval_iq.py "XL=XL:complete:model_M_complete_cont_XL340000.pt:sft"

# ブラウザ用に書き出して観測室を組み立てる
python export_gpt.py $QBNN_LM_DATA/model_M_complete_cont_XL340000.pt XL out $QBNN_LM_DATA/sft_meta.json
cp browser/* out/ && cd out
node test_gpt.js                                  # PyTorch との一致を確認
python build_gpt_bin.py . "学習データの説明" "2,740万"   # observatory_L.html ができる
```

`observatory_L.html` は同じディレクトリの `weights_0.wasm` … を読み込む。中身は fp16 の重みで、配信先が `.bin` を扱えないので拡張子を `.wasm` にしている。

## 結果（CPU 4 コア）

**ニューロンの種類の比較**（M = 幅 256・4 層、約 420 万パラメータ、6,000 ステップ、Wikipedia の検証損失、bits/文字）

| FFN のニューロン | 損失 |
|---|---|
| GELU | 4.186 |
| 完全ブロッホ球 | 4.200 |
| NeuroQuantum 型 QBNN | 4.347 |

パラメータ数をそろえると、完全ブロッホ球は GELU とほぼ同じで、NeuroQuantum 型より良い。

**大きさと学習量**（文脈 256、bits/文字、低いほど良い）

| モデル | パラメータ | 学習 | Web | Wikipedia | コード | 会話 | 指示 |
|---|---|---|---|---|---|---|---|
| L | 1,229 万 | 計 11 万ステップ | 3.809 | 3.376 | 1.379 | 2.726 | 2.451 |
| XL | 2,740 万 | 4 万ステップ | 3.979 | 3.566 | 1.518 | 2.897 | 2.666 |
| XL | 2,740 万 | 計 10 万ステップ | 3.748 | 3.328 | 1.394 | 2.667 | 2.412 |
| XL | 2,740 万 | 計 14 万ステップ（`cont3.py` のデータを追加） | **3.668** | **3.262** | **1.356** | **2.618** | **2.336** |

XL は 4 万ステップでは L に負けたが、大きさに見合う量（10 万ステップ）を学習させると全体で L を上回った。

**賢さテスト**（選択肢から尤度で選ぶ。偶然に当たる率は JCommonsenseQA・首都・県庁所在地 20%、足し算 5%、助詞 14%）

| モデル | JCommonsenseQA | 首都 | 県庁所在地 | 足し算 | 助詞 |
|---|---|---|---|---|---|
| L 計 11 万 | 25.0% | 20.0% | 38.3% | 4% | 77.2% |
| XL 計 10 万 | 27.3% | 22.5% | 61.7% | 8% | 78.5% |
| XL 計 14 万 | 26.5% | 22.5% | **66.0%** | 8% | **79.2%** |

クイズデータ（llm-japanese-dataset）には首都・県庁所在地を問う問題が含まれうるので、その 2 つの点には似た問題を学習で見た効果が混ざっている可能性がある。

**推論の例**（XL 計 14 万）

- 「日本の首都はどこですか？」→「日本の首都は東京です。」（乱数によっては誤る）
- 「こんにちは！」→「こんにちは！私はオープンアシスタントです。何かお手伝いしましょうか？」
- 「おすすめの勉強方法を教えてください。」→「もちろんです！以下におすすめの勉強方法を紹介します。…」

文体・答え方の形はよく覚えるが、個々の事実（富士山・作家など）はまだ誤りが多く、会話の 2 往復目以降は話の筋を保てないことが多い。

**ブラウザ実行**: `test_gpt.js` で次の文字の対数確率の差は最大 6.8×10⁻⁴、200 文字の貪欲生成は PyTorch と同じ。XL で 1 文字あたり約 53 ms。
