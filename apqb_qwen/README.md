# Full APQB Bloch Filter × Qwen3.8
- apqb_filter.py : F=(1+b·u)/2 、tanh/sech写像(n=2(1-tanh s))、Qwen系gated MLPへの差し込み(patch_model)
- test_tiny.py   : 数式検証(φ=0→100%, φ=π→0%)・初期状態で元モデルと同一・学習が進むことを確認済み
- train_qwen38.py: Qwen/Qwen3.8-27B のフィルターのみ学習(GPU必須)
設計: out = down( up · [(1-λ)·SiLU(g) + λ·F(n(g), φ(up), κ, α, β)] )、λは小さく初期化。
検証: 同条件でSiLU/GLUのみ(ベースライン)とperplexityを比較すること。
- infer_qwen3.py : ベースラインとフィルター適用後のモデルで推論を比較(perplexity・次トークンKL・生成文)。`--weights` で学習済みフィルターを読み込む。
  Qwen/Qwen3-0.6B・CPU・未学習フィルター(λ≈4.7%)での結果: perplexity 8.37→10.18、次トークンKL 0.001–0.004、先頭トークンは一致、生成文は途中から分岐。

## 実験結果: Qwen/Qwen3-0.6B (CPU, fp32)
条件: WikiText-2 train先頭3MBで400ステップ, seq 256, lr 2e-3, 両者同じ学習ウィンドウ。評価はWikiText-2 validationの固定16ウィンドウ。

| | 学習パラメータ数 | held-out ppl |
|---|---|---|
| 元のモデル | 0 | 34.07 |
| APQBフィルター(λ初期≈4.7%) 学習前 | — | 36.30 |
| APQBフィルター 学習後 | 602,112 | 25.49 |
| 対照: SiLUにチャネル毎scale+bias (`--control`) | 172,032 | **21.31** |

λは0.0474→0.0483とほぼ動かず。対照の方が少ないパラメータで低いperplexityになったため、この条件ではフィルター版の改善はWikiTextへの適応で説明でき、Bloch filter固有の効果は確認できなかった。
