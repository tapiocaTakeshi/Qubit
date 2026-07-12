#!/usr/bin/env python3
"""
ゼロから学習する新規QBNN 100Mモデルの事前学習スクリプト（24時間プラン用）。

アーキテクチャ: embed_dim=544, hidden_dim=1280, num_layers=7, num_heads=8,
max_seq_len=1024, vocab_size=32000 → 実パラメータ数 約100.1M

Phase 1（動作確認）:
    python pretrain_fresh_100m.py --phase 1 --target-tokens 50_000_000 \
        --lambda-mode disabled --lr 3e-4

Phase 2以降は --resume-from で前フェーズのチェックポイントを渡し、
--dataset-mix で構成を変える。
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, QBNNLayer
from progress_logger import ProgressLogger

PRETRAINED_TOKENIZER = "neuroq_tokenizer.model"
ARCH = dict(embed_dim=544, hidden_dim=1280, num_heads=8, num_layers=7, max_seq_len=1024, vocab_size=32000)

progress = ProgressLogger("pretrain_fresh_100m")


# ============================================================
# λ / J / θ 制御
# ============================================================
def set_qbnn_lambda(model, lambda_min, lambda_max):
    n = 0
    for m in model.modules():
        if isinstance(m, QBNNLayer):
            m.lambda_min = lambda_min
            m.lambda_max = lambda_max
            n += 1
    return n


def freeze_j_theta(model, freeze: bool):
    """J(J_source/entangle_op.W_entangle)とθ(lambda_base)をfreeze/unfreezeする。"""
    n_frozen, n_total = 0, 0
    for name, p in model.named_parameters():
        if name.endswith("J_source") or "entangle_op.W_entangle" in name or name.endswith("lambda_base"):
            p.requires_grad = not freeze
            n_total += p.numel()
            if freeze:
                n_frozen += p.numel()
    return n_frozen, n_total


def build_optimizer(model, base_lr, j_lr_ratio, weight_decay=0.1):
    """J/θパラメータは、たとえ現在requires_grad=Falseで凍結中でも常にoptimizerに含める。
    こうしておくと、warmup中に凍結解除(requires_grad=True)しても新たにoptimizerを
    作り直す必要がない（勾配がNoneの間はAdamWが自動的にそのパラメータをスキップする）。
    """
    j_params, other_params = [], []
    for name, p in model.named_parameters():
        is_j = name.endswith("J_source") or "entangle_op.W_entangle" in name or name.endswith("lambda_base")
        if is_j:
            j_params.append(p)
        elif p.requires_grad:
            other_params.append(p)
    groups = [{"params": other_params, "lr": base_lr}]
    if j_params:
        groups.append({"params": j_params, "lr": base_lr * j_lr_ratio})
    return AdamW(groups, weight_decay=weight_decay)


def cosine_warmup_factor(progress, warmup_frac=0.02, min_ratio=0.1):
    """progress: 0.0〜1.0の学習進捗(total_tokens/target_tokens)。線形warmup→cosine減衰。"""
    progress = min(max(progress, 0.0), 1.0)
    if progress < warmup_frac:
        return progress / warmup_frac if warmup_frac > 0 else 1.0
    prog = (progress - warmup_frac) / max(1e-9, 1.0 - warmup_frac)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return min_ratio + (1 - min_ratio) * cos


# ============================================================
# データ: 複数データセットを重み付きでストリーミング混合
# ============================================================
def hf_text_stream(dataset_id, config, split, text_field, seed, skip=0):
    from datasets import load_dataset as _hf_load_dataset

    kwargs = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    ds = _hf_load_dataset(dataset_id, **kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=500)
    if skip:
        ds = ds.skip(skip)
    for row in ds:
        text = (row.get(text_field) or "").strip()
        if len(text) < 100:
            continue
        yield text


def mixed_stream(dataset_specs, seed, skip_val=0):
    """dataset_specs: [(dataset_id, config, split, text_field, weight), ...]
    weight比率に従い各データセットのストリームから確率的にサンプリングする。
    """
    weights = np.array([s[4] for s in dataset_specs], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.RandomState(seed)

    streams = [
        hf_text_stream(ds_id, cfg, split, field, seed + i, skip=skip_val)
        for i, (ds_id, cfg, split, field, _w) in enumerate(dataset_specs)
    ]
    exhausted = [False] * len(streams)

    while not all(exhausted):
        idx = rng.choice(len(streams), p=weights)
        if exhausted[idx]:
            continue
        try:
            text = next(streams[idx])
            yield text, dataset_specs[idx][0]
        except StopIteration:
            exhausted[idx] = True


def article_to_sequences(text, tokenizer, max_seq_len, max_chars=4000):
    content_ids = tokenizer.encode(text[:max_chars], add_special=False)
    max_content = max_seq_len - 4
    if max_content <= 0 or len(content_ids) < 2:
        return []
    if len(content_ids) <= max_content:
        return [[tokenizer.bof_id, tokenizer.bos_id] + content_ids + [tokenizer.eos_id, tokenizer.eof_id]]
    stride = max(max_content // 2, 1)
    starts = list(range(0, len(content_ids) - max_content + 1, stride))
    seqs = []
    for idx, start in enumerate(starts):
        chunk = content_ids[start : start + max_content]
        prefix = [tokenizer.bof_id, tokenizer.bos_id] if idx == 0 else [tokenizer.bos_id]
        suffix = [tokenizer.eos_id, tokenizer.eof_id] if idx == len(starts) - 1 else [tokenizer.eos_id]
        seqs.append(prefix + chunk + suffix)
    return seqs


def pad_batch(seqs, tokenizer, max_seq_len):
    max_len = min(max(len(s) for s in seqs), max_seq_len)
    padded = []
    for s in seqs:
        s = s[:max_len]
        s = s + [tokenizer.pad_id] * (max_len - len(s))
        padded.append(s)
    return torch.tensor(padded, dtype=torch.long)


def build_fixed_val_set(dataset_specs, tokenizer, max_seq_len, n_val, seed):
    """固定検証セットを構築する。ドメイン(データソース)ごとの内訳も同じ1パスで
    追跡し、ドメイン別val loss計測に使えるようにする。
    """
    seqs = []
    domain_seqs = {spec[0]: [] for spec in dataset_specs}
    stream = mixed_stream(dataset_specs, seed=seed)
    count = 0
    for text, src in stream:
        if count >= n_val:
            break
        new_seqs = article_to_sequences(text, tokenizer, max_seq_len)
        seqs.extend(new_seqs)
        domain_seqs[src].extend(new_seqs)
        count += 1
    return seqs, count, domain_seqs


@torch.no_grad()
def eval_on_sequences(model, sequences, tokenizer, device, max_seq_len, batch_size=4):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        if not batch:
            continue
        input_ids = pad_batch(batch, tokenizer, max_seq_len).to(device)
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0
        )
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


DATASET_MIXES = {
    "phase1_verify": [
        ("wikimedia/wikipedia", "20231101.ja", "train", "text", 1.0),
    ],
    "phase2_base": [
        # CulturaXはgatedデータセット(HF_TOKEN + 利用規約同意が必要)のため、
        # 認証不要でアクセスできるCommonCrawlベースの日本語コーパスcc100-jaで代替。
        ("range3/cc100-ja", None, "train", "text", 0.80),
        ("wikimedia/wikipedia", "20231101.ja", "train", "text", 0.15),
        ("globis-university/aozorabunko-clean", None, "train", "text", 0.05),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--dataset-mix", default="phase1_verify", choices=list(DATASET_MIXES.keys()))
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--n-val-articles", type=int, default=200)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--j-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lambda-mode", choices=["disabled", "fixed", "warmup"], default="disabled")
    parser.add_argument("--lambda-fixed", type=float, default=0.02)
    parser.add_argument("--lambda-warmup-target", type=float, default=0.03)
    parser.add_argument("--lambda-warmup-start", type=float, default=0.35,
                         help="学習進捗(0-1)がこの割合に達するまではλ=0を維持")
    parser.add_argument("--lambda-warmup-end", type=float, default=0.65,
                         help="この割合でλがlambda-warmup-targetに到達、以降は維持")
    parser.add_argument("--freeze-j-theta", action="store_true", default=True)
    parser.add_argument("--no-freeze-j-theta", dest="freeze_j_theta", action="store_false")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-warmup-frac", type=float, default=0.02)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--ckpt-name", default="qbnn100m_v2_phase1.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print(f"🧠 QBNN 100M 新規事前学習 — Phase {args.phase} ({args.dataset_mix})")
    print("=" * 80)
    print(f"  アーキテクチャ: {ARCH}")
    print(f"  目標トークン数: {args.target_tokens:,}")
    print(f"  実効バッチ: micro={args.micro_batch_size} x accum={args.grad_accum} = {args.micro_batch_size*args.grad_accum}")
    print(f"  λモード: {args.lambda_mode}")

    tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file=PRETRAINED_TOKENIZER)

    config = NeuroQuantumConfig(
        vocab_size=ARCH["vocab_size"], embed_dim=ARCH["embed_dim"], hidden_dim=ARCH["hidden_dim"],
        num_heads=ARCH["num_heads"], num_layers=ARCH["num_layers"], max_seq_len=ARCH["max_seq_len"],
        dropout=0.1, lambda_entangle=0.5,
    )
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ モデル構築完了: {n_params:,} パラメータ")

    if args.resume_from and os.path.exists(args.resume_from):
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print(f"  ✓ チェックポイント読み込み: {args.resume_from}")
    else:
        print(f"  ✓ ランダム初期化のまま開始（新規学習）")

    dataset_specs = DATASET_MIXES[args.dataset_mix]

    if args.lambda_mode == "disabled":
        n_qbnn = set_qbnn_lambda(model, 0.0, 0.0)
        print(f"  ✓ QBNNLayer {n_qbnn}個の λ=0（通常Transformer相当）")
    elif args.lambda_mode == "fixed":
        n_qbnn = set_qbnn_lambda(model, 0.0, args.lambda_fixed)
        print(f"  ✓ QBNNLayer {n_qbnn}個の λ範囲 [0, {args.lambda_fixed}]")
    # warmup は学習ループ内で徐々に上げる

    if args.freeze_j_theta:
        n_frozen, n_total = freeze_j_theta(model, freeze=True)
        print(f"  ✓ J/θ を凍結: {n_frozen:,} / {n_total:,} パラメータ")

    optimizer = build_optimizer(model, args.lr, args.j_lr_ratio)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Optimizer構築完了 (学習対象 {n_trainable:,} パラメータ)")

    print("\n📊 固定検証セット構築中...")
    val_sequences, n_val_used, domain_val_sequences = build_fixed_val_set(
        dataset_specs, tokenizer, ARCH["max_seq_len"], args.n_val_articles, args.seed
    )
    print(f"  ✓ 検証記事 {n_val_used} 件 → {len(val_sequences)} シーケンス")
    for ds_id, seqs in domain_val_sequences.items():
        print(f"    - {ds_id}: {len(seqs)} シーケンス")

    # 進捗はopt_step数ではなくtotal_tokens/target_tokensで測る。
    # 記事の実長はmax_seq_lenよりかなり短いことが多く、opt_step基準の見積もりは
    # 実際のトークン進捗より大幅に早く「完了」扱いになってしまうため。
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    j_theta_unfrozen = not args.freeze_j_theta

    print("\n🚀 学習開始")
    print("=" * 80)

    stream = mixed_stream(dataset_specs, seed=args.seed, skip_val=args.n_val_articles)
    buffer = []
    total_tokens = 0
    micro_step = 0
    opt_step = 0
    total_loss = 0.0
    accum_count = 0
    start_time = time.time()
    best_val = float("inf")
    best_ckpt = args.ckpt_name.replace(".pt", "_best.pt")
    history = {"start_time": datetime.now(timezone.utc).isoformat(), "log": []}

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for text, src in stream:
        if total_tokens >= args.target_tokens:
            break
        buffer.extend(article_to_sequences(text, tokenizer, ARCH["max_seq_len"]))

        while len(buffer) >= args.micro_batch_size:
            batch_seqs = buffer[: args.micro_batch_size]
            buffer = buffer[args.micro_batch_size :]

            input_ids = pad_batch(batch_seqs, tokenizer, ARCH["max_seq_len"]).to(device)
            total_tokens += int((input_ids != tokenizer.pad_id).sum().item())

            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0
            )
            (loss / args.grad_accum).backward()
            total_loss += loss.item()
            accum_count += 1
            micro_step += 1

            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                opt_step += 1

                prog = min(total_tokens / args.target_tokens, 1.0)

                if args.lr_schedule == "cosine":
                    factor = cosine_warmup_factor(prog, args.lr_warmup_frac, args.lr_min_ratio)
                    for g, base in zip(optimizer.param_groups, base_lrs):
                        g["lr"] = base * factor

                if args.lambda_mode == "warmup":
                    if prog < args.lambda_warmup_start:
                        lam = 0.0
                    elif prog < args.lambda_warmup_end:
                        frac = (prog - args.lambda_warmup_start) / (args.lambda_warmup_end - args.lambda_warmup_start)
                        lam = args.lambda_warmup_target * frac
                    else:
                        lam = args.lambda_warmup_target
                    set_qbnn_lambda(model, 0.0, lam)
                    if not j_theta_unfrozen and prog >= args.lambda_warmup_start:
                        freeze_j_theta(model, freeze=False)
                        j_theta_unfrozen = True
                        print(f"  🔓 [opt_step {opt_step}] J/θ 凍結解除 (progress={prog:.2f})")

                if opt_step % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = total_tokens / max(elapsed, 1e-6)
                    avg_loss = total_loss / micro_step
                    print(f"  [opt_step {opt_step}] tokens={total_tokens:,} avg_loss={avg_loss:.4f} "
                          f"elapsed={elapsed/60:.1f}min rate={rate:.0f}tok/s lr={optimizer.param_groups[0]['lr']:.2e}")

                if opt_step % args.eval_every_steps == 0:
                    val_loss = eval_on_sequences(model, val_sequences, tokenizer, device, ARCH["max_seq_len"])
                    domain_losses = {}
                    for ds_id, seqs in domain_val_sequences.items():
                        if seqs:
                            domain_losses[ds_id] = eval_on_sequences(model, seqs, tokenizer, device, ARCH["max_seq_len"])
                    domain_str = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in domain_losses.items())
                    print(f"  📊 [opt_step {opt_step}] val_loss={val_loss:.4f}  ({domain_str})")
                    history["log"].append({"opt_step": opt_step, "tokens": total_tokens,
                                            "train_loss": total_loss / micro_step, "val_loss": val_loss,
                                            "domain_val_loss": domain_losses,
                                            "lr": optimizer.param_groups[0]["lr"]})
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(model.state_dict(), best_ckpt)
                        print(f"  ✓ ベスト更新 (val_loss={val_loss:.4f}) → {best_ckpt}")

                    # ドメイン別eval(異なる系列長分布を何度も評価)でCUDAアロケータの
                    # 断片化が進み、数百stepおきにOOMする問題への対策。
                    torch.cuda.empty_cache()

                if opt_step % args.save_every_steps == 0:
                    torch.save(model.state_dict(), args.ckpt_name)
                    print(f"  💾 中間保存: {args.ckpt_name}")

    torch.save(model.state_dict(), args.ckpt_name)
    print(f"\n✓ 最終チェックポイント保存: {args.ckpt_name}")
    if best_val < float("inf"):
        print(f"✓ ベストval_loss: {best_val:.4f} → {best_ckpt}")

    history["end_time"] = datetime.now(timezone.utc).isoformat()
    history["total_tokens"] = total_tokens
    history["opt_steps"] = opt_step
    with open(f"training_history_fresh100m_phase{args.phase}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Phase {args.phase} 完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
