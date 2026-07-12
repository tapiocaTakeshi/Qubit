#!/usr/bin/env python3
"""
NeuroQuantum megabyte_100mb を amenokaku-code-instruct でファインチューニング
既存層を凍結し、新規層のみを学習する。

使い方:
    # 基本的な実行
    python finetune_dolly_ja.py --epochs 3

    # GPU指定
    python finetune_dolly_ja.py --epochs 3 --device cuda:0

    # 小規模テスト（CPU）
    python finetune_dolly_ja.py --max-samples 100 --epochs 1
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    QBNNLayer,
    get_model_config_by_size,
)
from progress_logger import ProgressLogger

DATASET_ID = "kunishou/amenokaku-code-instruct"
PRETRAINED_CHECKPOINT = "/workspace/checkpoints/megabyte_100mb_full_diff_sft_best.pt"
PRETRAINED_TOKENIZER = "neuroq_tokenizer.model"
DEFAULT_CHECKPOINT_NAME = "megabyte_100mb_amenokaku_code_finetuned.pt"

progress = ProgressLogger("finetune_code_instruct")


def _first_str(row, keys):
    """row から keys の最初に見つかった非空文字列を返す。"""
    for k in keys:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_code_instruct_texts(ds, max_samples: int):
    """instruction/output形式のデータセットをテキストへ整形する（汎用）。

    amenokaku-code-instruct, databricks-dolly-15k-ja, gsm8k など
    instruction系/QA系の列名バリエーションに対応する。
    """
    texts = []
    n = len(ds) if max_samples is None or max_samples <= 0 else min(max_samples, len(ds))

    for row in ds.select(range(n)):
        # instruction / prompt / question
        instruction = _first_str(row, ("instruction", "prompt", "question", "input"))
        if not instruction:
            continue

        # context または input
        context = _first_str(row, ("context",))
        if context is None and "input" in row and instruction != _first_str(row, ("input",)):
            context = _first_str(row, ("input",))

        # output / response / code / answer / completion
        answer = _first_str(row, ("output", "response", "code", "answer", "completion"))

        # テキスト整形
        parts = [f"<USER> {instruction}"]
        if context:
            parts.append(context)
        if answer:
            parts.append(f"<ASSISTANT> {answer}")

        combined = "\n".join(parts).strip()
        if len(combined) > 4:
            texts.append(combined)

    return texts


_ROLE_TAG = {
    "human": "<USER>",
    "user": "<USER>",
    "gpt": "<ASSISTANT>",
    "assistant": "<ASSISTANT>",
    "system": "<SYSTEM>",
}


def extract_sharegpt_chat_texts(ds, max_samples: int):
    """ShareGPT形式（conversations: [{from, value}, ...]）のマルチターン会話を
    <USER>/<ASSISTANT>/<SYSTEM> タグ付きテキストへ整形する（oasst1-chat-44k-ja など）。
    """
    texts = []
    n = len(ds) if max_samples is None or max_samples <= 0 else min(max_samples, len(ds))

    for row in ds.select(range(n)):
        conv = row.get("conversations")
        if not conv:
            continue
        parts = []
        for turn in conv:
            role = (turn.get("from") or "").strip().lower()
            value = (turn.get("value") or "").strip()
            if not value:
                continue
            tag = _ROLE_TAG.get(role, "<USER>")
            parts.append(f"{tag} {value}")

        combined = "\n".join(parts).strip()
        if len(combined) > 4:
            texts.append(combined)

    return texts


def tokenize_texts(texts, tokenizer, max_seq_len, group_offset=0, progress_every=1000, label=""):
    """テキストをトークン化する。

    戻り値は (sequences, group_ids)。group_ids は各シーケンスの
    元テキスト（記事/サンプル）を表すID（group_offset起点）で、
    train/val分割時のリーク防止に使う。
    progress_every > 0 の場合、その件数ごとに進捗を表示する。
    """
    sequences = []
    group_ids = []
    n_texts = len(texts)
    start_t = time.time()

    for text_idx, t in enumerate(texts):
        if progress_every and text_idx > 0 and text_idx % progress_every == 0:
            elapsed = time.time() - start_t
            rate = text_idx / max(elapsed, 1e-6)
            eta = (n_texts - text_idx) / max(rate, 1e-6)
            tag = f"[{label}] " if label else ""
            print(f"    {tag}トークン化中: {text_idx}/{n_texts} ({rate:.0f}件/s, 残り約{eta:.0f}秒)")
        content_ids = tokenizer.encode(t, add_special=False)
        max_content = max_seq_len - 4

        if max_content <= 0 or len(content_ids) < 2:
            continue

        gid = group_offset + text_idx

        if len(content_ids) <= max_content:
            seq = (
                [tokenizer.bof_id, tokenizer.bos_id]
                + content_ids
                + [tokenizer.eos_id, tokenizer.eof_id]
            )
            sequences.append(seq)
            group_ids.append(gid)
        else:
            stride = max(max_content // 2, 1)
            starts = list(range(0, len(content_ids) - max_content + 1, stride))

            for idx, start in enumerate(starts):
                chunk = content_ids[start : start + max_content]
                prefix = (
                    [tokenizer.bof_id, tokenizer.bos_id]
                    if idx == 0
                    else [tokenizer.bos_id]
                )
                suffix = (
                    [tokenizer.eos_id, tokenizer.eof_id]
                    if idx == len(starts) - 1
                    else [tokenizer.eos_id]
                )
                sequences.append(prefix + chunk + suffix)
                group_ids.append(gid)

    return sequences, group_ids


def load_wikipedia_texts(max_samples, lang="ja", max_chars_per_article=3000, seed=42):
    """日本語Wikipediaから記事本文を取得する（ストリーミングでN件だけ取得）。"""
    from datasets import load_dataset as _hf_load_dataset

    config = f"20231101.{lang}"
    print(f"   📥 wikimedia/wikipedia ({config}) をストリーミング取得中...")
    ds = _hf_load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    texts = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < 100:
            continue
        texts.append(text[:max_chars_per_article])
        if len(texts) >= max_samples:
            break

    print(f"   ✓ Wikipedia記事 {len(texts)} 件取得")
    return texts


def set_qbnn_lambda_range(model, lambda_min=0.0, lambda_max=0.01):
    """全QBNNLayerのもつれ強度λの範囲を上書きする（過学習抑制のため低く設定）。"""
    count = 0
    for module in model.modules():
        if isinstance(module, QBNNLayer):
            module.lambda_min = lambda_min
            module.lambda_max = lambda_max
            count += 1
    return count


def build_staged_optimizer(model, base_lr, j_lr_ratio=0.1):
    """J (もつれテンソル) は base_lr の j_lr_ratio 倍、それ以外は base_lr で学習するオプティマイザーを構築。"""
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


def freeze_pretrained_layers(model, unfreeze_last_n_blocks=3, num_layers=10):
    """既存層を凍結。ただし transformer_blocks の最後の n 層は学習対象に含める。"""
    frozen_params = 0
    trainable_params = 0
    unfrozen_block_ids = set(range(num_layers - unfreeze_last_n_blocks, num_layers))

    for name, param in model.named_parameters():
        is_unfrozen_block = False
        if name.startswith("transformer_blocks."):
            block_id = int(name.split(".")[1])
            is_unfrozen_block = block_id in unfrozen_block_ids

        if is_unfrozen_block:
            param.requires_grad = True
            trainable_params += param.numel()
        elif "transformer_blocks" in name or "embedding" in name or "token_embedding" in name:
            # 前半の transformer_blocks と embedding は既存層のまま凍結
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()

    return frozen_params, trainable_params


def train_epoch(model, train_loader, optimizer, device, max_grad_norm=1.0):
    """1エポックのトレーニング。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_ids,) in enumerate(train_loader):
        input_ids = input_ids.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        # Shift: 最後のトークン以外を入力、最初のトークン以外を予測
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,  # PAD を無視
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f"  [Batch {batch_idx + 1}] Loss: {avg_loss:.4f}")

    return total_loss / max(num_batches, 1)


def _hessian_vector_product(loss, params, vector):
    """Pearlmutterのトリックで Hessian-vector積 Hv を計算する（未使用パラメータはallow_unused）。"""
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    dot = torch.sum(flat_grad * vector)
    hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
    hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)]
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def _hessian_power_iteration(model, params, input_ids, n_iters, shift=0.0, seed=0):
    """|H - shift*I| の最大固有値をべき乗法で推定する（shift=0ならλ_maxそのもの）。"""
    torch.manual_seed(seed)
    n_params = sum(p.numel() for p in params)
    v = torch.randn(n_params, device=input_ids.device)
    v = v / v.norm()

    eigenvalue = None
    for _ in range(n_iters):
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0
        )
        hv = _hessian_vector_product(loss, params, v)
        if shift != 0.0:
            hv = hv - shift * v
        eigenvalue = torch.dot(v, hv).item()
        norm = hv.norm()
        if norm < 1e-12:
            break
        v = hv / norm

    return eigenvalue


def compute_hessian_lr(model, params, input_ids, n_iters=8, min_ratio=0.1, base_lr_fallback=1e-5):
    """現在のパラメータ位置でλ_max, λ_minを推定し、η_opt=2/(λ_max+λ_min)を返す。

    べき乗法は「固定された線形作用素」を仮定するため、呼び出しごとに損失関数が
    変化してはいけない。2つの乱数源を一時的に潰す:
    - QBNNLayerはeval modeだとcall_countに応じたsin波の動的補正が入るため、
      train modeのまま（dynamic_factor=0.5固定）で推定する。
    - train modeだとDropoutが有効になってしまうため、Dropout層のpを一時的に0にする。
    """
    was_training = model.training
    model.train()

    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    saved_p = [m.p for m in dropout_modules]
    for m in dropout_modules:
        m.p = 0.0

    try:
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            lambda_max = _hessian_power_iteration(model, params, input_ids, n_iters, shift=0.0, seed=0)
            shifted_top = _hessian_power_iteration(model, params, input_ids, n_iters, shift=lambda_max, seed=1)
    finally:
        for m, p in zip(dropout_modules, saved_p):
            m.p = p
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()

    lambda_min = shifted_top + lambda_max

    if lambda_max is None or lambda_max <= 0:
        return base_lr_fallback, lambda_max, lambda_min
    if lambda_min is None or lambda_min <= 0:
        eta = max(1.0 / lambda_max, base_lr_fallback * min_ratio)
    else:
        eta = 2.0 / (lambda_max + lambda_min)
    return eta, lambda_max, lambda_min


def train_n_batches(model, loader_iter, loader, optimizer, device, n_batches, max_grad_norm=1.0,
                     lr_schedule="constant", base_lr=None, j_lr_ratio=0.1, lr_tau=2604.0,
                     lr_min_ratio=0.1, start_time=None, progress_every=100):
    """loader から n_batches 分だけ学習する（尽きたら再度先頭から取得）。

    lr_schedule="exp" の場合、経過時間(start_time からの秒数)に応じて
    lr(t) = base_lr * exp(-t/lr_tau) （下限 base_lr*lr_min_ratio）で学習率を更新する。
    progress_every > 0 の場合、そのバッチ数ごとに進捗（loss/lr/経過時間/ETA）を表示する。
    """
    model.train()
    total_loss = 0.0
    count = 0
    chunk_start = time.time()
    for _ in range(n_batches):
        try:
            (input_ids,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            (input_ids,) = next(loader_iter)

        if lr_schedule == "exp" and base_lr is not None and start_time is not None:
            elapsed_now = time.time() - start_time
            decay = max(math.exp(-elapsed_now / lr_tau), lr_min_ratio)
            optimizer.param_groups[0]["lr"] = base_lr * decay
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = base_lr * j_lr_ratio * decay

        input_ids = input_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if progress_every and count % progress_every == 0:
            elapsed = time.time() - chunk_start
            rate = count / max(elapsed, 1e-6)
            eta = (n_batches - count) / max(rate, 1e-6)
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"    [batch {count}/{n_batches}] avg_loss={total_loss / count:.4f} "
                f"lr={cur_lr:.2e} elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min"
            )

    return total_loss / max(count, 1), loader_iter


@torch.no_grad()
def eval_epoch(model, val_loader, device):
    """検証データに対する平均Lossを計算する（勾配更新なし）。"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for (input_ids,) in val_loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="NeuroQuantum Dolly-ja ファインチューニング")
    parser.add_argument("--epochs", type=int, default=3, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=4, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=1e-4, help="学習率")
    parser.add_argument("--max-samples", type=int, default=-1, help="使用サンプル数（-1で全件）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="検証データの割合")
    parser.add_argument("--full-finetune", action="store_true", help="全層を解凍してフルファインチューニングする")
    parser.add_argument("--staged-safe", action="store_true",
                         help="QBNN向け安全復旧モード: λを低く固定し、Jの学習率を1/10にし、0.5epochごとにval_lossが改善した場合のみ継続する")
    parser.add_argument("--lambda-max", type=float, default=0.01, help="staged-safe時のλ上限")
    parser.add_argument("--j-lr-ratio", type=float, default=0.1, help="staged-safe時のJ学習率比率（base_lrに対する倍率）")
    parser.add_argument("--patience", type=int, default=1, help="staged-safe時、val_lossが何回連続で改善しなければ打ち切るか")
    parser.add_argument("--resume-from", default=None, help="PRETRAINED_CHECKPOINTの代わりにこのチェックポイントから読み込む")
    parser.add_argument("--wiki-samples", type=int, default=0, help="日本語Wikipediaから追加する記事数（0で無効）")
    parser.add_argument("--wiki-max-chars", type=int, default=3000, help="Wikipedia記事1件あたりの最大文字数")
    parser.add_argument("--extra-datasets", default="",
                         help="追加でinstruction形式で混ぜるHFデータセットID（カンマ区切り、'id:config:split'形式も可）")
    parser.add_argument("--extra-max-samples", type=int, default=-1, help="追加データセット1件あたりの最大サンプル数（-1で全件）")
    parser.add_argument("--lr-schedule", choices=["constant", "exp", "hessian"], default="constant",
                         help="学習率スケジュール。'exp'は経過時間ベースの指数減衰、"
                              "'hessian'はChunk毎にHessian固有値からη_opt=2/(λ_max+λ_min)を再計算")
    parser.add_argument("--lr-tau", type=float, default=2604.0,
                         help="指数減衰の時定数τ（秒）。実測loss曲線のラプラス変換極から推定した値がデフォルト")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1,
                         help="lr(t)の下限（base_lrに対する比率）")
    parser.add_argument("--hessian-iters", type=int, default=8,
                         help="hessianスケジュール時のべき乗法反復回数（λ_max/λ_minそれぞれに適用）")

    args = parser.parse_args()

    # シード設定
    torch.manual_seed(args.seed)

    print("="*80)
    print("🧠 NeuroQuantum megabyte_100mb - amenokaku-code-instruct ファインチューニング")
    print("="*80)
    print(f"\n📋 設定:")
    print(f"  チェックポイント: {PRETRAINED_CHECKPOINT}")
    print(f"  デバイス: {args.device}")
    print(f"  エポック: {args.epochs}")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  学習率: {args.lr}")

    device = torch.device(args.device)

    # ========================================
    # 1. データセット読み込み
    # ========================================
    print(f"\n📥 amenokaku-code-instruct データセット読み込み中...")
    try:
        dataset = safe_load_dataset(DATASET_ID, split="train")
        texts = extract_code_instruct_texts(dataset, args.max_samples)
        print(f"  ✓ {len(texts)} サンプル取得")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return

    wiki_texts = []
    if args.wiki_samples > 0:
        print(f"\n📥 日本語Wikipedia読み込み中 ({args.wiki_samples}件)...")
        try:
            wiki_texts = load_wikipedia_texts(
                args.wiki_samples, max_chars_per_article=args.wiki_max_chars, seed=args.seed
            )
        except Exception as e:
            print(f"  ✗ Wikipedia取得エラー: {e}（Wikipediaなしで続行）")
            wiki_texts = []

    extra_texts_list = []
    if args.extra_datasets:
        for spec in args.extra_datasets.split(","):
            spec = spec.strip()
            if not spec:
                continue
            parts = spec.split(":")
            ds_id = parts[0]
            ds_config = parts[1] if len(parts) > 1 and parts[1] else None
            ds_split = parts[2] if len(parts) > 2 and parts[2] else "train"
            print(f"\n📥 追加データセット読み込み中: {spec}")
            try:
                if ds_config:
                    extra_ds = safe_load_dataset(ds_id, split=ds_split, name=ds_config)
                else:
                    extra_ds = safe_load_dataset(ds_id, split=ds_split)
                if "conversations" in extra_ds.column_names:
                    extra_texts = extract_sharegpt_chat_texts(extra_ds, args.extra_max_samples)
                else:
                    extra_texts = extract_code_instruct_texts(extra_ds, args.extra_max_samples)
                print(f"  ✓ {ds_id}: {len(extra_texts)} サンプル取得")
                extra_texts_list.append((ds_id, extra_texts))
            except Exception as e:
                print(f"  ✗ {ds_id} 取得エラー: {e}（このデータセットなしで続行）")

    # ========================================
    # 2. トークナイザー初期化
    # ========================================
    print(f"\n🔤 トークナイザー初期化中...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file=PRETRAINED_TOKENIZER)
    print(f"  ✓ vocab_size: {tokenizer.vocab_size}")

    # ========================================
    # 3. テキストトークン化
    # ========================================
    print(f"\n⚙️  テキストトークン化中...")
    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=32000)
    max_seq_len = 512

    print(f"  amenokaku トークン化開始 ({len(texts)}件)...")
    sequences, group_ids = tokenize_texts(texts, tokenizer, max_seq_len, group_offset=0, label="amenokaku")
    print(f"  ✓ amenokaku: {len(sequences)} シーケンス生成")
    next_group_offset = len(texts)

    if wiki_texts:
        print(f"  wikipedia トークン化開始 ({len(wiki_texts)}件)...")
        wiki_sequences, wiki_group_ids = tokenize_texts(
            wiki_texts, tokenizer, max_seq_len, group_offset=next_group_offset, label="wikipedia"
        )
        print(f"  ✓ wikipedia: {len(wiki_sequences)} シーケンス生成 ({len(wiki_texts)}記事)")
        sequences += wiki_sequences
        group_ids += wiki_group_ids
        next_group_offset += len(wiki_texts)

    for ds_id, extra_texts in extra_texts_list:
        if not extra_texts:
            continue
        print(f"  {ds_id} トークン化開始 ({len(extra_texts)}件)...")
        extra_sequences, extra_group_ids = tokenize_texts(
            extra_texts, tokenizer, max_seq_len, group_offset=next_group_offset, label=ds_id
        )
        print(f"  ✓ {ds_id}: {len(extra_sequences)} シーケンス生成 ({len(extra_texts)}サンプル)")
        sequences += extra_sequences
        group_ids += extra_group_ids
        next_group_offset += len(extra_texts)

    print(f"  ✓ 合計 {len(sequences)} シーケンス")

    # パディング
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_seq_len:
            seq = seq + [tokenizer.pad_id] * (max_seq_len - len(seq))
        else:
            seq = seq[:max_seq_len]
        padded_sequences.append(seq)

    input_ids = torch.tensor(padded_sequences, dtype=torch.long)

    # train/val 分割（記事/サンプル単位でグループ化してリークを防ぐ）
    import random as _random
    from collections import Counter

    n_total = input_ids.size(0)
    group_sizes = Counter(group_ids)
    unique_groups = sorted(group_sizes.keys())
    rng = _random.Random(args.seed)
    rng.shuffle(unique_groups)

    target_val_count = max(1, int(n_total * args.val_ratio)) if n_total > 1 else 0
    val_group_set = set()
    val_count = 0
    for gid in unique_groups:
        if val_count >= target_val_count:
            break
        val_group_set.add(gid)
        val_count += group_sizes[gid]

    val_idx_list, train_idx_list = [], []
    for i, gid in enumerate(group_ids):
        if gid in val_group_set:
            val_idx_list.append(i)
        else:
            train_idx_list.append(i)

    val_idx = torch.tensor(val_idx_list, dtype=torch.long)
    train_idx = torch.tensor(train_idx_list, dtype=torch.long)

    train_dataset = TensorDataset(input_ids[train_idx])
    val_dataset = TensorDataset(input_ids[val_idx]) if len(val_idx) > 0 else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    print(f"  ✓ train/val 分割: train={len(train_idx)} / val={len(val_idx)}")
    print(f"  ✓ データローダー作成: train {len(train_loader)} バッチ"
          + (f" / val {len(val_loader)} バッチ" if val_loader else ""))

    # ========================================
    # 4. モデル初期化
    # ========================================
    print(f"\n🧠 モデル初期化中...")
    config = NeuroQuantumConfig(
        vocab_size=32000,
        embed_dim=1024,
        hidden_dim=2048,
        num_heads=16,
        num_layers=10,
        max_seq_len=512,
        dropout=0.1,
        lambda_entangle=0.5,
    )

    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)

    # ========================================
    # 5. 既存層凍結
    # ========================================
    print(f"\n❄️  既存層を凍結中...")
    load_path = args.resume_from or PRETRAINED_CHECKPOINT
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        print(f"  ✓ チェックポイント読み込み: {load_path}")
    else:
        print(f"  ⚠️  チェックポイント見つかりません: {load_path}")

    if args.full_finetune:
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters())
        frozen_params = 0
        print(f"\n🔓 フルファインチューニング: 全層を解凍")
    else:
        frozen_params, trainable_params = freeze_pretrained_layers(model)

    print(f"\n📊 パラメータ統計:")
    print(f"  凍結層: {frozen_params:,} パラメータ")
    print(f"  学習層: {trainable_params:,} パラメータ")
    print(f"  比率: 凍結 {frozen_params/(frozen_params+trainable_params)*100:.1f}%")

    # ========================================
    # 6. オプティマイザー設定
    # ========================================
    print(f"\n⚙️  オプティマイザー設定中...")
    if args.staged_safe:
        n_qbnn = set_qbnn_lambda_range(model, lambda_min=0.0, lambda_max=args.lambda_max)
        print(f"  ✓ QBNNLayer {n_qbnn}個の λ範囲を [0.0, {args.lambda_max}] に固定")
        optimizer, n_j, n_other = build_staged_optimizer(model, base_lr=args.lr, j_lr_ratio=args.j_lr_ratio)
        print(f"  ✓ AdamW 初期化 (base_lr={args.lr}, J_lr={args.lr * args.j_lr_ratio:.2e}, "
              f"J params={n_j:,}, other params={n_other:,})")
    else:
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params_list, lr=args.lr)
        print(f"  ✓ AdamW 初期化 (lr={args.lr})")

    # ========================================
    # 7. トレーニングループ
    # ========================================
    print(f"\n🚀 トレーニング開始")
    print("="*80)

    history = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "epochs": [],
    }

    best_val_loss = float("inf")
    best_epoch = None
    best_checkpoint_path = DEFAULT_CHECKPOINT_NAME.replace(".pt", "_best_val.pt")

    if args.staged_safe:
        # 0.5 epoch ごとに学習し、val_loss が改善した場合のみ継続する
        half_epoch_batches = max(1, len(train_loader) // 2)
        loader_iter = iter(train_loader)
        chunk = 0
        max_chunks = max(1, round(args.epochs * 2))  # args.epochs を「最大 0.5epoch 単位の回数」の上限として扱う
        no_improve_streak = 0
        lr_start_time = time.time()

        while chunk < max_chunks:
            chunk += 1

            if args.lr_schedule == "hessian":
                print(f"\n🧮 λ_max/λ_min を再計算中（現在のパラメータ位置）...")
                # 学習中に溜まったoptimizer/activationのキャッシュを解放してから
                # 二階微分（通常backwardの約2倍のメモリを要する）を実行する
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                hess_params = [p for p in model.parameters() if p.requires_grad]
                (hess_input_ids,) = next(iter(val_loader))
                hess_input_ids = hess_input_ids[: min(2, hess_input_ids.size(0))].to(device)
                try:
                    eta_opt, lam_max, lam_min = compute_hessian_lr(
                        model, hess_params, hess_input_ids,
                        n_iters=args.hessian_iters, min_ratio=args.lr_min_ratio, base_lr_fallback=args.lr,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"  ⚠️  Hessian計算中にCUDA OOM。前回のlrを維持して続行します。")
                    torch.cuda.empty_cache()
                    eta_opt = optimizer.param_groups[0]["lr"]
                    lam_max = lam_min = float("nan")
                new_lr = max(eta_opt, args.lr * args.lr_min_ratio)
                optimizer.param_groups[0]["lr"] = new_lr
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = new_lr * args.j_lr_ratio
                print(f"  ✓ λ_max≈{lam_max:.4g} λ_min≈{lam_min:.4g} → η_opt≈{eta_opt:.3e} (採用lr={new_lr:.3e})")

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"\n📈 Chunk {chunk} (0.5 epoch 相当, batches={half_epoch_batches}, lr={cur_lr:.2e})")
            avg_loss, loader_iter = train_n_batches(
                model, loader_iter, train_loader, optimizer, device, half_epoch_batches,
                lr_schedule="constant" if args.lr_schedule == "hessian" else args.lr_schedule,
                base_lr=args.lr, j_lr_ratio=args.j_lr_ratio,
                lr_tau=args.lr_tau, lr_min_ratio=args.lr_min_ratio, start_time=lr_start_time,
                progress_every=500,
            )
            print(f"  平均Loss (train): {avg_loss:.4f}")

            val_loss = eval_epoch(model, val_loader, device) if val_loader else None
            if val_loss is not None:
                print(f"  平均Loss (val)  : {val_loss:.4f}")

            history["epochs"].append({"chunk": chunk, "loss": avg_loss, "val_loss": val_loss})

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = chunk
                no_improve_streak = 0
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"  ✓ ベスト更新 (val_loss={val_loss:.4f}) → {best_checkpoint_path}")
                try:
                    import shutil as _shutil
                    _shutil.copy(best_checkpoint_path, f"/workspace/checkpoints/{best_checkpoint_path}")
                    print(f"  ✓ ネットワークボリュームへ即時同期完了")
                except Exception as _e:
                    print(f"  ⚠️  ネットワークボリューム同期失敗: {_e}")
                print(f"  → val_lossが改善したため、次の0.5epochへ継続")
            else:
                no_improve_streak += 1
                print(f"  ⚠️  val_lossが改善しませんでした ({no_improve_streak}/{args.patience}回連続)")
                if no_improve_streak >= args.patience:
                    print(f"  → 学習を打ち切ります (val_loss改善なしが{args.patience}回連続)")
                    break
                print(f"  → patience内のため、次の0.5epochへ継続")
    else:
        for epoch in range(args.epochs):
            print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")
            avg_loss = train_epoch(model, train_loader, optimizer, device)
            print(f"  平均Loss (train): {avg_loss:.4f}")

            val_loss = eval_epoch(model, val_loader, device) if val_loader else None
            if val_loss is not None:
                print(f"  平均Loss (val)  : {val_loss:.4f}")

            history["epochs"].append({"epoch": epoch + 1, "loss": avg_loss, "val_loss": val_loss})

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"  ✓ ベスト更新 (val_loss={val_loss:.4f}) → {best_checkpoint_path}")

    history["end_time"] = datetime.now(timezone.utc).isoformat()
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss if best_epoch else None

    # ========================================
    # 8. チェックポイント保存
    # ========================================
    print(f"\n💾 チェックポイント保存中...")
    checkpoint_path = DEFAULT_CHECKPOINT_NAME
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  ✓ 保存完了 (最終epoch): {checkpoint_path}")
    if best_epoch:
        print(f"  ✓ ベストval_loss checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f}): {best_checkpoint_path}")

    # ネットワークボリュームに同期
    try:
        dest = f"/workspace/checkpoints/{checkpoint_path}"
        import shutil
        shutil.copy(checkpoint_path, dest)
        print(f"  ✓ ネットワークボリュームに同期: {dest}")
        if best_epoch:
            best_dest = f"/workspace/checkpoints/{best_checkpoint_path}"
            shutil.copy(best_checkpoint_path, best_dest)
            print(f"  ✓ ベストcheckpointも同期: {best_dest}")
    except Exception as e:
        print(f"  ⚠️  同期失敗: {e}")

    # 学習履歴保存
    history_file = "training_history_amenokaku_code_finetune.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 学習履歴: {history_file}")

    print("\n" + "="*80)
    print("✅ ファインチューニング完了")
    print("="*80)


if __name__ == "__main__":
    main()
