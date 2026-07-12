#!/usr/bin/env python3
"""JGLUE(JNLI/JSQuAD派生の instruction/input/output 形式データセット)を使い、
QBNNの「もつれテンソル」J / θ (J_source, entangle_op.W_entangle, lambda_base) だけを
学習するスクリプト。それ以外の全パラメータ(埋め込み・注意機構・FFNの通常重みなど)は凍結する。

想定用途: pretrain_fresh_100m.py で学習したベースモデル(W)はそのままに、
「今どのニューロン群が協力して考えているか」を表す一時的な協調状態(J)を、
JNLI(文の含意関係)やJSQuAD(読解QA)のような言語理解タスクで方向づけるための追加学習。

公式のJGLUEデータセット(shunk031/JGLUE等)はHuggingFace datasetsのスクリプト方式廃止に伴い
ストリーミングロードできなくなっているため、parquet化済みのJumtraミラー
(Jumtra/jglue_jnli, Jumtra/jglue_jsquad, Jumtra/jglue_jsquads_with_input)を代わりに使用する。

使い方:
    python train_j_theta_jglue.py --resume-from qbnn100m_v2_phase2_best.pt \
        --target-tokens 20000000 --lr 1e-4 --lambda-entangle 0.5
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, QBNNLayer

PRETRAINED_TOKENIZER = "neuroq_tokenizer.model"
ARCH = dict(embed_dim=544, hidden_dim=1280, num_heads=8, num_layers=7, max_seq_len=1024, vocab_size=32000)

JGLUE_DATASETS = [
    # (dataset_id, weight)
    ("Jumtra/jglue_jnli", 1.0),
    ("Jumtra/jglue_jsquad", 1.0),
    ("Jumtra/jglue_jsquads_with_input", 1.0),
]


def set_qbnn_lambda(model, lambda_min, lambda_max):
    n = 0
    for m in model.modules():
        if isinstance(m, QBNNLayer):
            m.lambda_min = lambda_min
            m.lambda_max = lambda_max
            n += 1
    return n


def freeze_all_except_j_theta(model):
    """J/θ以外の全パラメータを凍結する(freeze_j_thetaの逆)。"""
    n_trainable, n_total = 0, 0
    for name, p in model.named_parameters():
        is_j = name.endswith("J_source") or "entangle_op.W_entangle" in name or name.endswith("lambda_base")
        p.requires_grad = is_j
        n_total += p.numel()
        if is_j:
            n_trainable += p.numel()
    return n_trainable, n_total


def row_to_text(row):
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("input") or "").strip()
    answer = (row.get("output") or "").strip()
    if not instruction or not answer:
        return None
    parts = [f"<USER> {instruction}"]
    if context:
        parts.append(context)
    parts.append(f"<ASSISTANT> {answer}")
    text = "\n".join(parts).strip()
    return text if len(text) > 4 else None


def jglue_stream(dataset_specs, seed, skip=0):
    from datasets import load_dataset as _hf_load_dataset

    weights = np.array([w for _id, w in dataset_specs], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.RandomState(seed)

    def _one(ds_id, s):
        ds = _hf_load_dataset(ds_id, split="train", streaming=True)
        ds = ds.shuffle(seed=s, buffer_size=500)
        if skip:
            ds = ds.skip(skip)
        for row in ds:
            text = row_to_text(row)
            if text:
                yield text

    streams = [_one(ds_id, seed + i) for i, (ds_id, _w) in enumerate(dataset_specs)]
    exhausted = [False] * len(streams)
    while not all(exhausted):
        idx = rng.choice(len(streams), p=weights)
        if exhausted[idx]:
            continue
        try:
            yield next(streams[idx])
        except StopIteration:
            exhausted[idx] = True


def text_to_sequence(text, tokenizer, max_seq_len):
    content_ids = tokenizer.encode(text, add_special=False)[: max_seq_len - 4]
    if len(content_ids) < 2:
        return None
    return [tokenizer.bof_id, tokenizer.bos_id] + content_ids + [tokenizer.eos_id, tokenizer.eof_id]


def pad_batch(seqs, tokenizer, max_seq_len):
    max_len = min(max(len(s) for s in seqs), max_seq_len)
    padded = []
    for s in seqs:
        s = s[:max_len]
        s = s + [tokenizer.pad_id] * (max_len - len(s))
        padded.append(s)
    return torch.tensor(padded, dtype=torch.long)


def build_val_set(dataset_specs, tokenizer, max_seq_len, n_val, seed):
    seqs = []
    for text in jglue_stream(dataset_specs, seed=seed):
        if len(seqs) >= n_val:
            break
        seq = text_to_sequence(text, tokenizer, max_seq_len)
        if seq:
            seqs.append(seq)
    return seqs


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume-from", required=True, help="J以外のW部分を提供するベースチェックポイント")
    parser.add_argument("--target-tokens", type=int, default=20_000_000)
    parser.add_argument("--n-val-examples", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="J/θ専用の学習率")
    parser.add_argument("--lambda-entangle", type=float, default=0.5,
                         help="config.lambda_entangle相当。各層 lambda_min=x*0.5, lambda_max=x*1.5")
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-name", default="qbnn100m_v2_jtheta_jglue.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print("🧠⚛️ QBNN J/θ 専用学習 (JGLUE: JNLI + JSQuAD)")
    print("=" * 80)
    print(f"  ベースチェックポイント: {args.resume_from}")
    print(f"  目標トークン数: {args.target_tokens:,}")
    print(f"  λ (もつれ強度): entangle={args.lambda_entangle} -> "
          f"[{args.lambda_entangle*0.5}, {args.lambda_entangle*1.5}]")

    tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file=PRETRAINED_TOKENIZER)
    config = NeuroQuantumConfig(
        vocab_size=ARCH["vocab_size"], embed_dim=ARCH["embed_dim"], hidden_dim=ARCH["hidden_dim"],
        num_heads=ARCH["num_heads"], num_layers=ARCH["num_layers"], max_seq_len=ARCH["max_seq_len"],
        dropout=0.1, lambda_entangle=args.lambda_entangle,
    )
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    state = torch.load(args.resume_from, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print(f"  ✓ チェックポイント読み込み完了")

    n_qbnn = set_qbnn_lambda(model, args.lambda_entangle * 0.5, args.lambda_entangle * 1.5)
    print(f"  ✓ QBNNLayer {n_qbnn}個の λ を有効化")

    n_trainable, n_total = freeze_all_except_j_theta(model)
    print(f"  ✓ J/θ 以外を凍結: 学習対象 {n_trainable:,} / 全体 {n_total:,} パラメータ "
          f"({100*n_trainable/n_total:.1f}%)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    print("\n📊 固定検証セット構築中 (JNLI/JSQuAD)...")
    val_sequences = build_val_set(JGLUE_DATASETS, tokenizer, ARCH["max_seq_len"], args.n_val_examples, args.seed)
    print(f"  ✓ 検証シーケンス {len(val_sequences)} 件")

    print("\n🚀 学習開始")
    print("=" * 80)

    stream = jglue_stream(JGLUE_DATASETS, seed=args.seed, skip=args.n_val_examples)
    buffer = []
    total_tokens = 0
    micro_step = 0
    opt_step = 0
    total_loss = 0.0
    accum_count = 0
    start_time = time.time()
    best_val = float("inf")
    best_ckpt = args.ckpt_name.replace(".pt", "_best.pt")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for text in stream:
        if total_tokens >= args.target_tokens:
            break
        seq = text_to_sequence(text, tokenizer, ARCH["max_seq_len"])
        if seq:
            buffer.append(seq)

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
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                opt_step += 1

                if opt_step % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = total_tokens / max(elapsed, 1e-6)
                    avg_loss = total_loss / micro_step
                    print(f"  [opt_step {opt_step}] tokens={total_tokens:,} avg_loss={avg_loss:.4f} "
                          f"elapsed={elapsed/60:.1f}min rate={rate:.0f}tok/s")

                if opt_step % args.eval_every_steps == 0:
                    val_loss = eval_on_sequences(model, val_sequences, tokenizer, device, ARCH["max_seq_len"])
                    print(f"  📊 [opt_step {opt_step}] val_loss={val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(model.state_dict(), best_ckpt)
                        print(f"  ✓ ベスト更新 (val_loss={val_loss:.4f}) → {best_ckpt}")
                    torch.cuda.empty_cache()

                if opt_step % args.save_every_steps == 0:
                    torch.save(model.state_dict(), args.ckpt_name)
                    print(f"  💾 中間保存: {args.ckpt_name}")

    torch.save(model.state_dict(), args.ckpt_name)
    print(f"\n✓ 最終チェックポイント保存: {args.ckpt_name}")
    if best_val < float("inf"):
        print(f"✓ ベストval_loss: {best_val:.4f} → {best_ckpt}")

    print("\n" + "=" * 80)
    print("✅ J/θ JGLUE学習 完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
