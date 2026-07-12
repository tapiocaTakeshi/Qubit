#!/usr/bin/env python3
"""
日本語Wikipediaをストリーミングで読み込み、NeuroQuantum megabyte_100mb を
大規模継続事前学習するスクリプト。

全記事をメモリに載せず、記事を読みながら逐次トークン化・バッチ化して学習する。
エポック数ではなく学習トークン数で進捗を管理する。

使い方:
    python pretrain_wikipedia.py --target-articles 500000 --checkpoint megabyte_100mb_amenokaku_code_finetuned_best_val.pt
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    QBNNLayer,
)
from progress_logger import ProgressLogger

PRETRAINED_TOKENIZER = "neuroq_tokenizer.model"
DEFAULT_CHECKPOINT_NAME = "megabyte_100mb_wikipedia_pretrained.pt"

progress = ProgressLogger("pretrain_wikipedia")


def set_qbnn_lambda_range(model, lambda_min=0.0, lambda_max=0.01):
    count = 0
    for module in model.modules():
        if isinstance(module, QBNNLayer):
            module.lambda_min = lambda_min
            module.lambda_max = lambda_max
            count += 1
    return count


def build_staged_optimizer(model, base_lr, j_lr_ratio=0.1):
    j_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("J_source") or "entangle_op.W_entangle" in name:
            j_params.append(param)
        else:
            other_params.append(param)

    param_groups = [{"params": other_params, "lr": base_lr}]
    if j_params:
        param_groups.append({"params": j_params, "lr": base_lr * j_lr_ratio})

    n_j = sum(p.numel() for p in j_params)
    n_other = sum(p.numel() for p in other_params)
    return AdamW(param_groups, weight_decay=0.01), n_j, n_other


def article_to_sequences(text, tokenizer, max_seq_len):
    """1記事のテキストを1つ以上のトークン化済みシーケンスに変換する。"""
    content_ids = tokenizer.encode(text, add_special=False)
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


def wikipedia_stream(lang, seed, max_chars_per_article, skip=0):
    from datasets import load_dataset as _hf_load_dataset

    config = f"20231101.{lang}"
    ds = _hf_load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    if skip:
        ds = ds.skip(skip)
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < 100:
            continue
        yield text[:max_chars_per_article]


def build_fixed_val_set(tokenizer, max_seq_len, n_val_articles, lang, seed, max_chars_per_article):
    """先頭 n_val_articles 件を固定検証セットとして materialize する。"""
    seqs = []
    stream = wikipedia_stream(lang, seed, max_chars_per_article, skip=0)
    count = 0
    for text in stream:
        if count >= n_val_articles:
            break
        seqs.extend(article_to_sequences(text, tokenizer, max_seq_len))
        count += 1
    return seqs, count


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
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="読み込む既存チェックポイント（フルパスまたはこのディレクトリ内のファイル名）")
    parser.add_argument("--target-articles", type=int, default=1_000_000, help="学習に使う記事数の目標")
    parser.add_argument("--n-val-articles", type=int, default=300, help="固定検証セットに使う記事数")
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--max-chars-per-article", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lambda-max", type=float, default=0.01)
    parser.add_argument("--j-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every-batches", type=int, default=2000)
    parser.add_argument("--save-every-batches", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--patience", type=int, default=3,
                         help="val_lossがこの回数連続で改善しなければ学習を打ち切る（0で無効）")
    parser.add_argument("--min-delta", type=float, default=0.0,
                         help="改善とみなす最小のval_loss低下量")
    parser.add_argument("--lr-schedule", choices=["constant", "exp"], default="constant",
                         help="学習率スケジュール。'exp'は経過時間ベースの指数減衰 lr(t)=lr0*exp(-t/tau)")
    parser.add_argument("--lr-tau", type=float, default=2604.0,
                         help="指数減衰の時定数τ（秒）。実測loss曲線のラプラス変換極から推定した値がデフォルト")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1,
                         help="lr(t)の下限（base_lrに対する比率）。減衰しすぎて学習が止まるのを防ぐ")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 80)
    print("🧠 NeuroQuantum megabyte_100mb - 日本語Wikipedia ストリーミング継続事前学習")
    print("=" * 80)
    print(f"  目標記事数: {args.target_articles:,}")
    print(f"  検証記事数: {args.n_val_articles}")
    print(f"  学習率: {args.lr} (J={args.lr * args.j_lr_ratio:.2e})")
    print(f"  λ上限: {args.lambda_max}")

    max_seq_len = 512
    config = NeuroQuantumConfig(
        vocab_size=32000, embed_dim=1024, hidden_dim=2048,
        num_heads=16, num_layers=10, max_seq_len=max_seq_len,
        dropout=0.1, lambda_entangle=0.5,
    )

    print("\n🔤 トークナイザー初期化中...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file=PRETRAINED_TOKENIZER)

    print("\n📊 固定検証セット構築中...")
    val_sequences, n_val_used = build_fixed_val_set(
        tokenizer, max_seq_len, args.n_val_articles, args.lang, args.seed, args.max_chars_per_article
    )
    print(f"  ✓ 検証記事 {n_val_used} 件 → {len(val_sequences)} シーケンス")

    print("\n🧠 モデル初期化中...")
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)

    ckpt_path = args.checkpoint
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        print(f"  ✓ チェックポイント読み込み: {ckpt_path}")
    else:
        print(f"  ⚠️  チェックポイントが見つかりません: {ckpt_path}（ランダム初期化のまま続行）")

    for param in model.parameters():
        param.requires_grad = True

    n_qbnn = set_qbnn_lambda_range(model, lambda_min=0.0, lambda_max=args.lambda_max)
    print(f"  ✓ QBNNLayer {n_qbnn}個の λ範囲を [0.0, {args.lambda_max}] に固定")

    optimizer, n_j, n_other = build_staged_optimizer(model, base_lr=args.lr, j_lr_ratio=args.j_lr_ratio)
    print(f"  ✓ AdamW 初期化 (J params={n_j:,}, other params={n_other:,})")

    print("\n🚀 ストリーミング学習開始")
    print("=" * 80)

    best_val_loss = float("inf")
    best_ckpt_path = args.ckpt_name.replace(".pt", "_best_val.pt")
    no_improve_streak = 0
    stopped_early = False

    stream = wikipedia_stream(args.lang, args.seed, args.max_chars_per_article, skip=args.n_val_articles)

    buffer = []
    articles_seen = 0
    total_tokens = 0
    n_batches = 0
    total_loss = 0.0
    start_time = time.time()

    history = {"start_time": datetime.now(timezone.utc).isoformat(), "log": []}

    model.train()
    for text in stream:
        if articles_seen >= args.target_articles:
            break
        if stopped_early:
            break
        articles_seen += 1
        buffer.extend(article_to_sequences(text, tokenizer, max_seq_len))

        while len(buffer) >= args.batch_size:
            batch_seqs = buffer[: args.batch_size]
            buffer = buffer[args.batch_size :]

            input_ids = pad_batch(batch_seqs, tokenizer, max_seq_len).to(device)
            total_tokens += int((input_ids != tokenizer.pad_id).sum().item())

            if args.lr_schedule == "exp":
                elapsed_now = time.time() - start_time
                decay = max(np.exp(-elapsed_now / args.lr_tau), args.lr_min_ratio)
                optimizer.param_groups[0]["lr"] = args.lr * decay
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = args.lr * args.j_lr_ratio * decay

            optimizer.zero_grad()
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / n_batches
                rate = total_tokens / max(elapsed, 1e-6)
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [batch {n_batches}] articles={articles_seen:,} tokens={total_tokens:,} "
                    f"avg_loss={avg_loss:.4f} elapsed={elapsed/60:.1f}min rate={rate:.0f}tok/s lr={cur_lr:.2e}"
                )

            if n_batches % args.eval_every_batches == 0:
                val_loss = eval_on_sequences(model, val_sequences, tokenizer, device, max_seq_len, args.batch_size)
                model.train()
                print(f"  📊 [batch {n_batches}] val_loss={val_loss:.4f} (train_avg={total_loss/n_batches:.4f})")
                history["log"].append({
                    "batch": n_batches, "articles": articles_seen, "tokens": total_tokens,
                    "train_loss": total_loss / n_batches, "val_loss": val_loss,
                })
                if val_loss < best_val_loss - args.min_delta:
                    best_val_loss = val_loss
                    no_improve_streak = 0
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"  ✓ ベスト更新 (val_loss={val_loss:.4f}) → {best_ckpt_path}")
                elif args.patience > 0:
                    no_improve_streak += 1
                    print(f"  ⚠️  val_lossが改善しませんでした ({no_improve_streak}/{args.patience}回連続)")
                    if no_improve_streak >= args.patience:
                        print(f"  → Early stopping: val_loss改善なしが{args.patience}回連続のため打ち切ります")
                        stopped_early = True
                        break

            if n_batches % args.save_every_batches == 0:
                torch.save(model.state_dict(), args.ckpt_name)
                print(f"  💾 中間チェックポイント保存: {args.ckpt_name}")

    # 最終保存
    torch.save(model.state_dict(), args.ckpt_name)
    print(f"\n✓ 最終チェックポイント保存: {args.ckpt_name}")
    if best_val_loss < float("inf"):
        print(f"✓ ベストval_loss: {best_val_loss:.4f} → {best_ckpt_path}")

    history["end_time"] = datetime.now(timezone.utc).isoformat()
    history["stopped_early"] = stopped_early
    history["total_articles"] = articles_seen
    history["total_tokens"] = total_tokens
    history["total_batches"] = n_batches
    history["best_val_loss"] = best_val_loss if best_val_loss < float("inf") else None

    with open("training_history_wikipedia_pretrain.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✅ 事前学習完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
