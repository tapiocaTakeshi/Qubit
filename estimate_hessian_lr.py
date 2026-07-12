#!/usr/bin/env python3
"""
Hessian-vector積を用いたべき乗法で損失関数のヘッセ行列の最大・最小固有値
(λ_max, λ_min) を推定し、二次関数近似における最適学習率

    η_opt = 2 / (λ_max + λ_min)

を求めるスクリプト。

安定条件 0 < η < 2/λ_max も併せて出力する。

使い方:
    python estimate_hessian_lr.py --checkpoint megabyte_100mb_amenokaku_code_finetuned_best_val.pt \
        --dataset-id kunishou/amenokaku-code-instruct --n-samples 8 --power-iters 20
"""

import argparse
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_utils import safe_load_dataset
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer
from finetune_dolly_ja import extract_code_instruct_texts, tokenize_texts


def build_eval_batch(tokenizer, max_seq_len, device, n_samples=8, dataset_id="kunishou/amenokaku-code-instruct"):
    """データセットから小さなバッチを1つ構築する（Hessian推定用のサンプル点）。"""
    ds = safe_load_dataset(dataset_id, split="train")
    texts = extract_code_instruct_texts(ds, n_samples * 3)
    sequences, _ = tokenize_texts(texts[: n_samples * 3], tokenizer, max_seq_len)
    sequences = sequences[:n_samples]

    max_len = min(max(len(s) for s in sequences), max_seq_len)
    padded = []
    for s in sequences:
        s = s[:max_len]
        s = s + [tokenizer.pad_id] * (max_len - len(s))
        padded.append(s)
    return torch.tensor(padded, dtype=torch.long, device=device)


def compute_loss(model, input_ids):
    logits = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0
    )


def hessian_vector_product(loss, params, vector):
    """Pearlmutterのトリックで Hessian-vector積 Hv を計算する。

    フォワードパスで使われない（エイリアス等で勾配が伝播しない）パラメータは
    allow_unused=True で許容し、その成分は0として扱う。
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    dot = torch.sum(flat_grad * vector)
    hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
    hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)]
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def power_iteration(model, params, input_ids, n_iters, shift=0.0, seed=0):
    """|H - shift*I| の最大固有値を推定する（shift=0ならλ_maxそのもの）。"""
    torch.manual_seed(seed)
    n_params = sum(p.numel() for p in params)
    v = torch.randn(n_params, device=input_ids.device)
    v = v / v.norm()

    eigenvalue = None
    for i in range(n_iters):
        model.zero_grad(set_to_none=True)
        loss = compute_loss(model, input_ids)
        hv = hessian_vector_product(loss, params, v)
        if shift != 0.0:
            hv = hv - shift * v
        eigenvalue = torch.dot(v, hv).item()
        norm = hv.norm()
        if norm < 1e-12:
            break
        v = hv / norm
        print(f"    iter {i + 1}/{n_iters}: eigenvalue_est={eigenvalue:.6f} |Hv|={norm.item():.6f}")

    return eigenvalue, v


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-id", default="kunishou/amenokaku-code-instruct")
    parser.add_argument("--n-samples", type=int, default=8, help="Hessian推定に使うサンプル数（バッチサイズ）")
    parser.add_argument("--power-iters", type=int, default=20)
    parser.add_argument("--tokenizer", default="neuroq_tokenizer.model")
    parser.add_argument("--max-seq-len", type=int, default=256, help="計算負荷軽減のため短めを推奨")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("🧮 Hessian固有値推定 → 最適学習率 η_opt = 2/(λ_max+λ_min) の計算")
    print("=" * 80)

    # モデルのアーキテクチャ（position_embeddingサイズ含む）はチェックポイント学習時の
    # max_seq_len=512に固定する。--max-seq-len は計算量削減のための「入力を短く切る」
    # 用途のみに使い、512以下でなければならない。
    model_max_seq_len = 512
    if args.max_seq_len > model_max_seq_len:
        raise ValueError(f"--max-seq-len は{model_max_seq_len}以下にしてください")

    config = NeuroQuantumConfig(
        vocab_size=32000, embed_dim=1024, hidden_dim=2048,
        num_heads=16, num_layers=10, max_seq_len=model_max_seq_len,
        dropout=0.0, lambda_entangle=0.5,
    )
    tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file=args.tokenizer)
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    # 注意: QBNNLayerはeval modeだとcall_countに応じたsin波の動的補正が入り、
    # 呼び出すたびに損失関数が変化してしまう（べき乗法の固定作用素という前提が崩れる）。
    # train modeならdynamic_factor=0.5固定になるため、dropout=0と合わせてtrain modeで決定的にする。
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"\n対象パラメータ数: {n_params:,}")

    print(f"\n📥 サンプルバッチ構築中 ({args.n_samples}件, {args.dataset_id})...")
    input_ids = build_eval_batch(tokenizer, args.max_seq_len, device, args.n_samples, args.dataset_id)
    print(f"  ✓ バッチ形状: {tuple(input_ids.shape)}")

    # scaled_dot_product_attention の効率化カーネル(flash/mem_efficient)は二階微分の
    # backward-of-backward が未実装のため、二階微分に対応した math バックエンドを強制する。
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        print(f"\n🔺 λ_max 推定中 (べき乗法, {args.power_iters}反復)...")
        lambda_max, v_max = power_iteration(model, params, input_ids, args.power_iters, shift=0.0, seed=0)
        print(f"  ✓ λ_max ≈ {lambda_max:.6f}")

        print(f"\n🔻 λ_min 推定中 (シフトべき乗法: λ_max*I - H の最大固有値)...")
        shifted_top, _ = power_iteration(model, params, input_ids, args.power_iters, shift=lambda_max, seed=1)
        # shifted_top = (H - lambda_max I) の最大固有値 = 最も lambda_max から遠い(＝最小の) H固有値 - lambda_max
        lambda_min = shifted_top + lambda_max
        print(f"  ✓ λ_min ≈ {lambda_min:.6f}")

    print("\n" + "=" * 80)
    print("📐 結果")
    print("=" * 80)
    print(f"  λ_max = {lambda_max:.6f}")
    print(f"  λ_min = {lambda_min:.6f}")

    if lambda_max <= 0:
        print("\n⚠️  λ_max <= 0 です。このサンプル点は損失の谷底（凸領域）にない可能性があります。")
        print("    正定値性が崩れているため、教科書的な二次近似の公式は適用できません。")
        return

    eta_stability_bound = 2.0 / lambda_max
    print(f"\n  安定条件: 0 < η < 2/λ_max = {eta_stability_bound:.3e}")

    if lambda_min > 0:
        eta_opt = 2.0 / (lambda_max + lambda_min)
        print(f"  収束最適化: η_opt = 2/(λ_max+λ_min) = {eta_opt:.3e}")
    else:
        print(f"  λ_min <= 0 のため η_opt = 2/(λ_max+λ_min) の公式は使えません"
              f"（損失面が谷底で完全な凸ではない/鞍点近傍の可能性）。")
        print(f"  代わりに安定条件の目安として η ≈ 1/λ_max = {1.0/lambda_max:.3e} を推奨します。")


if __name__ == "__main__":
    main()
