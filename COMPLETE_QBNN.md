# 完全ブロッホ球と完全QBNN

実装: `complete_qbnn.py`　検証: `tests/test_complete_qbnn.py`

## 完全ブロッホ球

**完全ブロッホ球**は、二葉双曲面と2つの半球体からなる。

- 二葉双曲面 H = {q = (u, v, ζ) : ζ² − u² − v² = 1}。上の葉 H₊ (ζ ≥ 1) と下の葉 H₋ (ζ ≤ −1) がある。
- 射影 **P(q) = (u, v, 1)/ζ**（CB1）は、上の葉を北半球（z > 0、|0⟩ 側）に、下の葉を南半球（z < 0、|1⟩ 側）に移す。
  - 葉の頂点 (0, 0, ±1) は両極 |0⟩・|1⟩（n = 0, 4）に移る。
  - 葉の無限遠は赤道（n = 2、η = 1）に移る。
  - 同じ (u, v) を持つ反対側の葉の点 q̃ = (u, v, −ζ) は、対蹠点 P(q̃) = −P(q) に移る。
- 2つの半球体: 偏極 m = tanh c ∈ (−1, 1) を加え、**b = m·P(q)**（CB4）とする。m > 0 なら北の半球体に、m < 0 なら南の半球体に入る。2つの半球体で、ブロッホ球体全体（任意の1量子ビット密度行列。赤道面は極限として到達）を覆う。

| 量 | 完全ブロッホ球での形 |
|---|---|
| 葉の測地座標 | (u, v) = sinh ρ (cos a, sin a), ζ = ±cosh ρ → b = ±(tanh ρ cos a, tanh ρ sin a, sech ρ)（CB2） |
| 平方根振幅座標 | η = tanh ρ、\|1 − n/2\| = sech ρ。双曲距離 ρ は極からの「コヒーレンスの深さ」 |
| 論文の潜在変数 s（z = tanh s） | sinh s · sinh ρ = 1（CB3） |
| 密度行列 | ρ = (1+m)/2 \|P(q)⟩⟨P(q)\| + (1−m)/2 \|P(q̃)⟩⟨P(q̃)\|、純度 (1 + m²)/2 |
| 測定（葉軸 P(q)） | ±1 が確率 (1 ± m)/2 で出て、P(q) か P(q̃) の純粋状態、すなわちどちらかの葉に収縮する（CB5）。E[b′] = b |
| 逆写像 | (u, v) = (b_x, b_y)/b_z（極からの中心射影）、m = sign(b_z)\|b\|（CB6） |
| 同じ葉の2点の双曲距離 | cosh d = 1 + (1 − b₁·b₂)/(z₁z₂) = 1 + 2(1 − F)/(z₁z₂)（F は純粋状態の忠実度、CB7） |

学習座標 (c, u, v) ∈ ℝ³ には制約がない。それでも |b| = |tanh c| ≤ 1 なので、各ニューロンは常に妥当な密度行列に対応する。学習座標の上で葉をまたぐ連続な経路は、球の中心（最大混合状態）か赤道（無限遠）を通る。

平方根振幅論文の tanh/sech パラメータ化 (r, η) = (tanh s, sech s) は、(sinh s, 1) を正規化したものにあたる。完全ブロッホ球はこれを、双曲面の空間成分 (u, v) を2次元にした形に一般化している。ただし役割は入れ替わる。論文の s は赤道からの距離で、極に到達できない。一方、ρ は極からの距離で、赤道に到達できない。

CB1〜CB7 は、10,000個のランダムな状態について、`apqb_sqrt_amplitude.py` の密度行列・逆写像・回転則と丸め誤差の範囲で照合している。

## 完全QBNN

| クラス | 内容 |
|---|---|
| `CompleteBlochLayer` | 入力 → 線形写像 → 各ニューロンの (c, u, v) → b（CB4）→ 学習可能な1量子ビットゲート R_j ∈ SO(3)（論文 式(8)、恒等で初期化）。出力は `[..., neurons, 3]` |
| `CompleteQBNN` | 隠れ層がすべて完全ブロッホ球ニューロンの順伝播ネット（分類・回帰） |
| `CompleteQBNNLM` | 再帰型の文字言語モデル。各層の記憶は完全ブロッホ球の列で、初期状態は最大混合状態 b = 0 |
| `CompleteQBNNAI` | 語彙・学習・生成・保存をまとめたラッパー |

次の層は、ブロッホベクトルの 3 × neurons 個の成分を読む。読み出しの重みは、ニューロン間の古典的な相互作用 J にあたる。

`measure > 0` を指定すると、各ニューロンを確率 `measure` で測定する。

- `basis='sheet'`: 葉軸で測定する（CB5）。
- `basis='z'`: 計算基底で測定する（論文 式(17)）。

勾配は測定前の b に流す（straight-through）。測定値の期待値は b に等しいので、平方根振幅論文 第7.6節の「測定サンプリング」による不確かさ推定にも使える。

## 使い方

```bash
# 二重らせんの分類
python complete_qbnn.py classify-demo

# 文字言語モデルの学習と生成
python complete_qbnn.py train --text README.md docs/*.md --steps 3000 --out complete_qbnn_lm.pt
python complete_qbnn.py generate --ckpt complete_qbnn_lm.pt --prompt "量子" --samples 3
python complete_qbnn.py generate --ckpt complete_qbnn_lm.pt --prompt "量子" --temperature 0 --measure 1.0 --samples 4
```

```python
from complete_qbnn import CompleteQBNN, describe
net = CompleteQBNN(in_features=2, hidden=[32, 32], out_features=2)
logits = net(x)                 # 測定なし
logits = net(x, measure=1.0)    # 全ニューロンを葉軸で測定
net.diagnostics()               # 層ごとの n, κ, 純度, 上の葉の割合, 双曲深さ, 偏極
```

## 実行結果（CPU）

RESULTS_PLACEHOLDER

## 位置づけと限界

- 完全ブロッホ球は、単一量子ビットの標準的な量子情報（ブロッホ球体と密度行列）を、二葉双曲面の座標で表し直したものである。ニューロンは古典計算機上の状態座標であり、物理的な量子ビットや量子もつれを実現するものではない。量子優位性や高速化は主張しない。
- 赤道面 b_z = 0 は、学習座標の上では無限遠にある。ゲート R_j を通せば任意の向きに回せる。
- 言語モデルは小規模な文字単位の再帰モデルである。NeuroQuantum（約5,430万パラメータ）への組み込みや、既存の QBNN 層との比較は、まだ行っていない。
