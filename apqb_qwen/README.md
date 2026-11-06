# Full APQB Bloch Filter × Qwen3.8
- apqb_filter.py : F=(1+b·u)/2 、tanh/sech写像(n=2(1-tanh s))、Qwen系gated MLPへの差し込み(patch_model)
- test_tiny.py   : 数式検証(φ=0→100%, φ=π→0%)・初期状態で元モデルと同一・学習が進むことを確認済み
- train_qwen38.py: Qwen/Qwen3.8-27B のフィルターのみ学習(GPU必須)
設計: out = down( up · [(1-λ)·SiLU(g) + λ·F(n(g), φ(up), κ, α, β)] )、λは小さく初期化。
検証: 同条件でSiLU/GLUのみ(ベースライン)とperplexityを比較すること。
