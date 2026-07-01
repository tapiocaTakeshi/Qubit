#!/usr/bin/env python3
"""
databricks-dolly-15k-ja で QBNN を学習する。

使い方:
    python train_dolly_ja.py --epochs 8 --max-samples 500
    python train_dolly_ja.py --max-samples 100 --epochs 3  # quick test
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import safe_load_dataset
from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)
from progress_logger import ProgressLogger

DATASET_ID = "llm-jp/databricks-dolly-15k-ja"
DEFAULT_CHECKPOINT_NAME = "neuroq_dolly_ja_checkpoint.pt"

progress = ProgressLogger("train_dolly_ja")


def extract_dolly_texts(ds, max_samples: int):
    """databricks-dolly-15k-ja から QA ペアテキストを抽出。"""
    texts = []
    n = len(ds) if max_samples is None or max_samples <= 0 else min(max_samples, len(ds))
    for i, row in enumerate(ds.select(range(n))):
        instruction = (row.get("instruction") or "").strip()
        context = (row.get("context") or "").strip()
        response = (row.get("response") or "").strip()
        
        if not response:
            continue
        
        # Format: instruction + context → response
        parts = []
        if instruction:
            parts.append(f"質問: {instruction}")
        if context:
            parts.append(f"背景: {context}")
        if response:
            parts.append(f"回答: {response}")
        
        text = "\n".join(parts).strip()
        if len(text) > 8:
            texts.append(text)
    return texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    """テキストを学習用シーケンスに変換。"""
    sequences = []
    for t in texts:
        content_ids = tokenizer.encode(t, add_special=False)
        max_content = max_seq_len - 2
        if max_content <= 0 or len(content_ids) < 2:
            continue
        if len(content_ids) <= max_content:
            seq = (
                [tokenizer.bof_id, tokenizer.bos_id]
                + content_ids
                + [tokenizer.eos_id, tokenizer.eof_id]
            )
        else:
            seq = (
                [tokenizer.bof_id, tokenizer.bos_id]
                + content_ids[:max_content]
                + [tokenizer.eos_id, tokenizer.eof_id]
            )
        sequences.append(seq)
    return sequences


def train_epoch(model, sequences, tokenizer, optimizer, scheduler, batch_size=4, device=None, save_every=0, on_save=None):
    """1 エポック学習。"""
    import random

    if device is None:
        device = next(model.parameters()).device

    model.train()
    random.shuffle(sequences)
    losses = []
    n_batches = 0

    for batch_start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[batch_start : batch_start + batch_size]
        if not batch_seqs:
            continue

        max_len = min(max(len(s) for s in batch_seqs), 512)

        input_ids, labels = [], []
        for seq in batch_seqs:
            ids = seq[:max_len]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_id] * pad_len)
            labels.append(ids + [-100] * pad_len)

        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        logits = model(input_tensor)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_tensor[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_batches += 1
        losses.append(loss.item())

        if n_batches % 10 == 0:
            avg = sum(losses[-10:]) / len(losses[-10:])
            progress.info(f"  Batch {n_batches} | Loss: {avg:.4f}")

        if save_every and on_save and n_batches % save_every == 0:
            avg = sum(losses) / len(losses) if losses else 0
            progress.info(f"[checkpoint] batch {n_batches} avg_loss={avg:.4f} を保存中...")
            on_save(epoch=None, batch=n_batches, loss=avg)

    scheduler.step()
    avg_loss = sum(losses) / len(losses) if losses else 0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-samples", type=int, default=None, help="最大サンプル数 (全件なら None)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=None, help="未指定なら small 設定 (4) を使用")
    parser.add_argument("--max-seq-len", type=int, default=None, help="未指定なら small 設定 (4096) を使用")
    parser.add_argument("--ckpt-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--tokenizer-prefix", default="neuroq_dolly_ja_tokenizer")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    progress.info(f"Device: {device}")
    
    # データセット読み込み
    progress.info(f"=== Loading {DATASET_ID} ===")
    ds = safe_load_dataset(DATASET_ID, split="train")
    
    CONFIG = get_model_config_by_size("small", vocab_size=args.vocab_size)
    batch_size = args.batch_size or CONFIG["batch_size"]
    max_seq_len = args.max_seq_len or CONFIG["max_seq_len"]
    
    use_all = args.max_samples is None or args.max_samples <= 0
    effective_max_samples = len(ds) if use_all else min(args.max_samples, len(ds))
    
    progress.info(f"=== Extracting texts ({effective_max_samples} samples) ===")
    texts = extract_dolly_texts(ds, args.max_samples)
    progress.info(f"Extracted {len(texts)} texts")
    
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ckpt_name)
    
    # トークナイザー構築
    progress.info("=== Building SentencePiece tokenizer ===")
    tokenizer_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tokenizer_prefix)
    tokenizer = NeuroQuantumTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(texts, model_prefix=tokenizer_prefix, character_coverage=0.9995)
    actual_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    CONFIG["vocab_size"] = actual_vocab
    progress.info(f"Vocab size: {actual_vocab}")
    
    # モデル構築
    progress.info("=== Building NeuroQuantum (small) ===")
    nq_config = NeuroQuantumConfig(
        vocab_size=actual_vocab,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        max_seq_len=max_seq_len,
        dropout=CONFIG["dropout"],
        lambda_entangle=CONFIG["entangle_strength"],
    )
    model = NeuroQuantum(config=nq_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    progress.info(f"Parameters: {n_params:,}")
    
    # トークナイズ
    progress.info("=== Tokenizing ===")
    sequences = tokenize_texts(texts, tokenizer, max_seq_len)
    progress.info(f"Training sequences: {len(sequences)}")
    if not sequences:
        raise RuntimeError("シーケンスが 0 件。")
    
    # 学習
    progress.info("=== Training ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    def save_checkpoint(epoch=None, batch=None, loss=None, final=False):
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "vocab_size": actual_vocab,
                "embed_dim": CONFIG["embed_dim"],
                "hidden_dim": CONFIG["hidden_dim"],
                "num_heads": CONFIG["num_heads"],
                "num_layers": CONFIG["num_layers"],
                "max_seq_len": max_seq_len,
                "architecture": "neuroquantum",
                "model_size": "small",
            },
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET_ID,
            "max_samples": effective_max_samples,
        }
        torch.save(checkpoint, ckpt_path)
        size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        progress.info(f"Saved: {ckpt_path} ({size_mb:.1f} MB)")
    
    for epoch in range(args.epochs):
        progress.info(f"Epoch {epoch + 1}/{args.epochs}...")
        avg_loss = train_epoch(
            model, sequences, tokenizer, optimizer, scheduler,
            batch_size=batch_size, device=device, save_every=0, on_save=save_checkpoint
        )
        progress.info(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.6f}")
    
    save_checkpoint(final=True)
    progress.info(f"Training complete! Final loss: {avg_loss:.6f}")
    progress.info(f"Checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
