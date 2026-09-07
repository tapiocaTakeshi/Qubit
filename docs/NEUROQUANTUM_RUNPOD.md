# NeuroQuantumをRunpod GPU Podで学習する

既存のRunpod GPU Podに入り、CUDA版PyTorch 2.4以上とPython 3.10以上を使用します。
この手順は単一GPU向けです。`train_runpod.py` のServerlessジョブ送信とは別の、Pod内で直接動かす学習入口です。

## 準備と起動

RunpodのPyTorchテンプレートを使い、成果物を残すボリュームをマウントしてください。
Podを削除した後も残したい場合はNetwork Volumeを使用します。
以下はマウント先が `/workspace`、リポジトリが `/workspace/Qubit` の例です。
`NQ_RUN_DIR` を実際のマウント先に合わせて設定してください。

```bash
cd /workspace/Qubit
python -m pip install -r requirements-runpod.txt
export NQ_RUN_DIR=/workspace/neuroquantum-dolly
bash scripts/train_neuroquantum_runpod.sh \
  --dataset-id llm-jp/databricks-dolly-15k-ja \
  --epochs 3
```

既に学習したモデルへ追加学習する場合は、そのモデルのチェックポイントと対応する
トークナイザーを指定してください。未指定の初回実行は新規モデルの学習です。
同じ実験の再開には同じ `NQ_RUN_DIR`、モデルサイズ、語彙サイズ、系列長、Attentionウィンドウを使います。

```bash
bash scripts/train_neuroquantum_runpod.sh \
  --dataset-id llm-jp/databricks-dolly-15k-ja \
  --resume --epochs 6
```

再開処理は既存の `train_hf_dataset.py` に従います。完了エポック数から再開し、
中断したエポックの途中のデータ位置・乱数状態までは復元しません。
別モデル・別データセットの実験には別の `NQ_RUN_DIR` を指定してください。

| 設定 | 起動スクリプトの既定値 | 目的 |
| --- | --- | --- |
| モデル | `small` | GPU容量だけで巨大モデルを自動選択しない |
| マイクロバッチ | `1` | ピークVRAMを抑える |
| 勾配蓄積 | `8` | 少量ずつ処理して学習を更新する |
| 最大系列長 | `1024` | 位置埋め込みと入力の上限 |
| Attentionウィンドウ | `256` | 過去255トークンと自身を参照する |
| Gradient checkpointing | 有効 | 逆伝播時に再計算し、中間テンソルの保持を減らす |
| 混合精度 | 対応CUDA GPUではBF16、その他はFP32 | GPU能力に合わせる |
| 出力 | `$NQ_RUN_DIR/checkpoint.pt`、`tokenizer.model` / `.vocab` | マウントしたボリュームに保存する |

VRAM使用量はモデル・語彙数・系列長に依存します。この設定は全GPUでの動作保証ではありません。
余裕があれば `--batch-size 2`、メモリをさらに抑えるなら `--max-seq-len 512` を指定します。
`--attention-window` はモデルが参照できる文脈も変えるため、既存モデルでは学習時の値に合わせます。
`--no-gradient-checkpointing` と `--no-use-bf16` で各機能を無効にできます。
モデル変更は `NQ_MODEL_SIZE=medium` などで指定できます。

## アルゴリズムの変更

- 長い系列のLocalAttentionを、幅Wのクエリと最大2W−1個のK/Vへ分割しました。
  同じ因果ウィンドウの計算を維持し、Attentionの計算量は系列長Nに対してO(NW)です。
  通常のforwardではN×Nのマスクを作らず、系列長ごとのマスクキャッシュも保持しません。
  外部マスクは非ゼロが参照可能という従来仕様を維持します。パディングマスクには
  `(batch, 1, 1, sequence)` を使えます。利用者が渡すN×Nマスクの確保は削減対象外です。
- ウィンドウ以下の系列で外部マスクがなければ、PyTorch SDPAの `is_causal=True` を使います。
  GPUとdtypeに応じてPyTorchがカーネルを選ぶため、Flash Attentionの使用を強制しません。
- 設定だけ存在していたgradient checkpointingを通常の学習forwardに接続しました。
  再計算時のDropout乱数を保持し、同じ勾配を計算します。計算時間とメモリのトレードオフがあります。
- QBNN補正強度を学習時と同じ式に統一しました。旧 `call_count` バッファはロード互換性のため
  残しますが、呼び出し順による推論結果の変動とGPUからのスカラー読み出しをなくしました。
  パラメータ名・形状は維持しています。旧版の2回目以降の推論の揺らぎは再現しません。
- 勾配蓄積は、パディングを除く予測対象トークン数で正規化します。
  長さが異なるマイクロバッチや最後の端数も、同じ論理バッチでのトークン平均損失に対応します。
  ログの平均損失もトークン平均へ変更しました。

通常の学習・推論経路がメモリ削減の対象です。各層のテンソルを保存する
`forward_with_details` は診断用で、従来どおり明示的なマスクや中間出力を保持します。
モデル本体の変更はServerless側から同じクラスを使う場合にも反映されますが、
今回の学習ループ・BF16設定の変更対象は `train_hf_dataset.py` です。

## Runpod上での検証

```bash
python -m pip install pytest
python -m pytest -q tests/test_neuroquantum_regressions.py tests/test_neuroquantum_training.py
python scripts/benchmark_neuroquantum_attention.py --device cuda --length 4096 --window 256
```

テストにはCUDAでのFP16/BF16 forward・backwardが含まれます。CUDAがない環境では
この2ケースはskipされます。BF16非対応GPUではBF16ケースのみskipされます。
ベンチマークは同じ入力のdense参照計算と出力が一致することを確認し、処理時間と
PyTorchの追加ピーク割当メモリを表示します。実行にデータセットや学習済みモデルは不要です。
これはAttention単体の測定で、モデル全体の必要VRAMや文章生成品質を測るものではありません。

参考: [Runpod Pod overview](https://docs.runpod.io/pods/overview)、
[Runpod storage](https://docs.runpod.io/pods/storage/types)、
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、
[PyTorch checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)。
