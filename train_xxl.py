#!/usr/bin/env python3
"""
XXL (1.5B) NeuroQuantum モデル学習スクリプト
GPT-2 XL相当: embed=1536, 48層, 24ヘッド
CPU 16GB環境対応: gradient accumulation + メモリ最適化

使用方法:
  python train_xxl.py

HFエンドポイントAPI経由の学習:
  curl -X POST https://<endpoint> -H "Content-Type: application/json" \\
    -d '{"inputs": "train", "action": "train", "dataset_id": "izumi-lab/llm-japanese-dataset",
         "text_column": "output", "split": "train", "max_samples": 2000, "epochs": 10, "lr": 0.0001}'
"""
import os
import sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datetime import datetime, timezone
import json
import random
import math
import gc

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")

# ===== XXL Model Config =====
MODEL_CONFIG = {
    "vocab_size": 32000,
    "embed_dim": 1536,
    "hidden_dim": 4096,
    "num_heads": 24,
    "num_layers": 48,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "entangle_strength": 0.5,
    "architecture": "neuroquantum",
}

# ===== Training Hyperparameters =====
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
MIN_LR_RATIO = 0.1
MAX_SAMPLES_PER_DS = 5000
SAVE_EVERY_STEPS = 200
LOG_EVERY_STEPS = 10

# ===== Datasets (train split) =====
DATASETS = [
    {"id": "kunishou/databricks-dolly-15k-ja", "text_column": "output", "split": "train",
     "max_samples": MAX_SAMPLES_PER_DS},
    {"id": "fujiki/japanese_alpaca_data", "text_column": "output", "split": "train",
     "max_samples": MAX_SAMPLES_PER_DS},
    {"id": "izumi-lab/llm-japanese-dataset", "text_column": "output", "split": "train",
     "max_samples": 3000},
]


def load_all_data():
    all_texts = []
    for ds_info in DATASETS:
        ds_id = ds_info["id"]
        col = ds_info["text_column"]
        split = ds_info["split"]
        max_samples = ds_info["max_samples"]
        print(f"  Loading {ds_id} (split={split}, max={max_samples})...")
        try:
            ds = load_dataset(ds_id, split=split, trust_remote_code=True)
            count = 0
            for row in ds.select(range(min(max_samples, len(ds)))):
                text = row.get(col, "")
                if isinstance(text, str) and len(text.strip()) > 15:
                    all_texts.append(text.strip())
                    count += 1
            print(f"    -> {count} samples")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    random.shuffle(all_texts)
    print(f"\nTotal training texts: {len(all_texts)}")
    return all_texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t, add_special=True)
        if len(ids) < 4:
            continue
        if len(ids) <= max_seq_len:
            sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine_decay)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"=== Building XXL NeuroQuantum (1.5B) ===")

    tokenizer = NeuroQuantumTokenizer(vocab_size=MODEL_CONFIG["vocab_size"], model_file=TOKENIZER_PATH)
    actual_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    MODEL_CONFIG["vocab_size"] = actual_vocab

    nq_config = NeuroQuantumConfig(
        vocab_size=MODEL_CONFIG["vocab_size"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        dropout=MODEL_CONFIG["dropout"],
        lambda_entangle=MODEL_CONFIG["entangle_strength"],
    )
    model = NeuroQuantum(config=nq_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e9:.2f}B)")
    model = model.to(device)

    print(f"\n=== Loading datasets (train split) ===")
    texts = load_all_data()

    print(f"\n=== Tokenizing ===")
    max_seq_len = MODEL_CONFIG["max_seq_len"]
    sequences = tokenize_texts(texts, tokenizer, max_seq_len)
    del texts
    gc.collect()
    print(f"Training sequences: {len(sequences)}")

    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_opt_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM_STEPS
    warmup_steps = int(total_opt_steps * WARMUP_RATIO)
    print(f"\nSteps/epoch: {steps_per_epoch}, Total opt steps: {total_opt_steps}")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}, Warmup: {warmup_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

    print(f"\n{'='*60}")
    print(f"=== Training XXL (1.5B) for {EPOCHS} epochs ===")
    print(f"{'='*60}")
    model.train()
    global_step = 0
    opt_step = 0
    best_loss = float('inf')
    training_log = []

    for epoch in range(EPOCHS):
        random.shuffle(sequences)
        epoch_loss = 0
        epoch_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i + BATCH_SIZE]
            if not batch_seqs:
                continue

            max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
            input_ids, labels = [], []
            for s in batch_seqs:
                ids = s[:max_len]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [tokenizer.pad_id] * pad_len)
                lbl = ids[1:] + [tokenizer.pad_id] * (pad_len + 1)
                labels.append(lbl[:max_len])

            x = torch.tensor(input_ids, dtype=torch.long, device=device)
            y = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                   ignore_index=tokenizer.pad_id)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            epoch_batches += 1
            global_step += 1

            if global_step % GRAD_ACCUM_STEPS == 0:
                opt_step += 1
                lr = get_lr(opt_step, total_opt_steps, warmup_steps, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

                if opt_step % LOG_EVERY_STEPS == 0:
                    avg = epoch_loss / epoch_batches
                    print(f"  E{epoch+1}/{EPOCHS} | S{opt_step}/{total_opt_steps} | "
                          f"Loss: {avg:.4f} | LR: {lr:.2e}")

                if opt_step % SAVE_EVERY_STEPS == 0:
                    _save(model, MODEL_CONFIG, training_log, epoch, epoch_loss/epoch_batches,
                          opt_step, len(sequences))

        avg_loss = epoch_loss / max(epoch_batches, 1)
        training_log.append({"epoch": epoch + 1, "loss": round(avg_loss, 4), "opt_steps": opt_step})
        print(f"\n>>> Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        _save(model, MODEL_CONFIG, training_log, epoch, avg_loss, opt_step, len(sequences))
        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


def _save(model, config, log, epoch, loss, opt_step, n_seqs):
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "training_log": log,
        "datasets": [d["id"] for d in DATASETS],
        "epoch": epoch + 1,
        "opt_step": opt_step,
        "loss": loss,
    }, CKPT_PATH)
    with open(os.path.join(os.path.dirname(__file__), "training_history.json"), "w") as f:
        json.dump({
            "architecture": "neuroquantum",
            "config": config,
            "datasets": [d["id"] for d in DATASETS],
            "total_sequences": n_seqs,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_log": log,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
