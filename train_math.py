#!/usr/bin/env python3
"""
Math training script for NeuroQuantum architecture.
Fine-tunes on HachiML/alpaca_jp_math (Japanese math instruction dataset).
Loads existing checkpoint if available, then continues training.
"""
import os
import sys
import time
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datetime import datetime, timezone
import json

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

# Config — must match existing checkpoint
CONFIG = {
    "vocab_size": 32000,
    "embed_dim": 256,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 512,
    "dropout": 0.1,
    "entangle_strength": 0.5,
}

DATASET_ID = "HachiML/alpaca_jp_math"
DATASET_SPLIT = "v1.0_cleaned"
MAX_SAMPLES = 5000
EPOCHS = 3
LR = 3e-4  # Lower LR for fine-tuning
BATCH_SIZE = 4
MAX_SEQ_LEN = CONFIG["max_seq_len"]


def extract_math_texts(ds, max_samples):
    """Extract instruction+output pairs from math dataset."""
    texts = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")

        # Combine instruction + input + output into training text
        parts = []
        if instruction:
            parts.append(instruction.strip())
        if inp and inp.strip():
            parts.append(inp.strip())
        if output:
            parts.append(output.strip())

        combined = "\n".join(parts)
        if len(combined.strip()) > 10:
            texts.append(combined.strip())
    return texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    """Tokenize texts into training sequences."""
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t, add_special=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                chunk = ids[start:start + max_seq_len]
                sequences.append(chunk)
    return sequences


def train_epoch(model, sequences, tokenizer, optimizer, epoch, device):
    """Train one epoch. Returns (avg_loss, total_tokens, elapsed_seconds)."""
    model.train()
    total_loss = 0
    total_tokens = 0
    n_batches = 0

    random.shuffle(sequences)
    epoch_start = time.time()

    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        if not batch_seqs:
            continue

        max_len = min(max(len(s) for s in batch_seqs), MAX_SEQ_LEN)
        input_ids = []
        labels = []
        for s in batch_seqs:
            ids = s[:max_len]
            lbl = s[:max_len]
            pad_len = max_len - len(ids)
            ids = ids + [tokenizer.pad_id] * pad_len
            lbl = lbl + [-100] * pad_len
            input_ids.append(ids)
            labels.append(lbl)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        total_tokens += (labels_t != -100).sum().item()

        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_t[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 50 == 0:
            elapsed = time.time() - epoch_start
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            avg = total_loss / n_batches
            print(f"  Epoch {epoch+1} | Batch {n_batches} | Avg Loss: {avg:.4f} | {tok_per_sec:.0f} tok/sec")

    elapsed = time.time() - epoch_start
    return total_loss / max(n_batches, 1), total_tokens, elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Load math dataset
    print(f"\n=== Loading {DATASET_ID} (split={DATASET_SPLIT}) ===")
    try:
        ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        print("Trying default train split...")
        ds = load_dataset(DATASET_ID, split="train", trust_remote_code=True)

    texts = extract_math_texts(ds, MAX_SAMPLES)
    print(f"Extracted {len(texts)} math texts")

    # Step 2: Load existing tokenizer or build new one
    print("\n=== Loading tokenizer ===")
    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

    if os.path.isfile(tokenizer_path):
        print(f"Loading existing tokenizer: {tokenizer_path}")
        tokenizer = NeuroQuantumTokenizer(
            vocab_size=CONFIG["vocab_size"],
            model_file=tokenizer_path
        )
    else:
        print("No existing tokenizer found, building new one...")
        tokenizer = NeuroQuantumTokenizer(vocab_size=CONFIG["vocab_size"])
        tokenizer.build_vocab(
            texts,
            model_prefix=os.path.join(os.path.dirname(__file__), "neuroq_tokenizer"),
            character_coverage=0.9995,
        )

    actual_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    CONFIG["vocab_size"] = actual_vocab
    print(f"Vocab size: {actual_vocab}")

    # Step 3: Build model and load checkpoint if available
    print("\n=== Building model ===")
    nq_config = NeuroQuantumConfig(
        vocab_size=actual_vocab,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
        lambda_entangle=CONFIG["entangle_strength"],
    )
    model = NeuroQuantum(config=nq_config).to(device)

    if os.path.isfile(ckpt_path):
        print(f"Loading existing checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state"])
            print("Checkpoint loaded successfully (fine-tuning mode)")
        except RuntimeError as e:
            print(f"Checkpoint mismatch, training from scratch: {e}")
    else:
        print("No checkpoint found, training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Step 4: Tokenize
    print("\n=== Tokenizing ===")
    sequences = tokenize_texts(texts, tokenizer, MAX_SEQ_LEN)
    print(f"Training sequences: {len(sequences)}")

    # Step 5: Train
    print(f"\n=== Training ({EPOCHS} epochs, lr={LR}) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    training_log = []
    grand_total_tokens = 0
    grand_total_time = 0
    for epoch in range(EPOCHS):
        avg_loss, epoch_tokens, epoch_time = train_epoch(model, sequences, tokenizer, optimizer, epoch, device)
        scheduler.step()
        grand_total_tokens += epoch_tokens
        grand_total_time += epoch_time
        tok_sec = epoch_tokens / epoch_time if epoch_time > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f} | {tok_sec:.0f} tok/sec | {epoch_time:.1f}s")
        training_log.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "tokens": epoch_tokens,
            "tok_per_sec": round(tok_sec, 1),
            "elapsed_sec": round(epoch_time, 1)
        })

    overall_tok_sec = grand_total_tokens / grand_total_time if grand_total_time > 0 else 0
    print(f"\nOverall: {grand_total_tokens:,} tokens in {grand_total_time:.1f}s = {overall_tok_sec:.0f} tok/sec")

    # Step 6: Save checkpoint
    print("\n=== Saving checkpoint ===")
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": actual_vocab,
            "embed_dim": CONFIG["embed_dim"],
            "hidden_dim": CONFIG["hidden_dim"],
            "num_heads": CONFIG["num_heads"],
            "num_layers": CONFIG["num_layers"],
            "max_seq_len": CONFIG["max_seq_len"],
            "entangle_strength": CONFIG["entangle_strength"],
            "dropout": CONFIG["dropout"],
            "architecture": "neuroquantum",
        },
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [DATASET_ID],
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved: {ckpt_path} ({os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB)")

    # Step 7: Inference test with math prompts
    print("\n=== Math inference test ===")
    model.eval()
    test_prompts = [
        "1+1=",
        "2×3の答えは",
        "三角形の面積を求める",
        "xの方程式を解いてください",
        "平均値を計算する",
    ]
    total_gen_tokens = 0
    total_gen_time = 0
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)
        gen_count = 0

        infer_start = time.time()
        with torch.no_grad():
            for _ in range(80):
                seq = input_tensor[:, -MAX_SEQ_LEN:]
                logits = model(seq)[:, -1, :] / 0.7
                topk_vals = torch.topk(logits, 40)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= 1.3
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()
                if nxt_id == tokenizer.eos_id:
                    break
                if nxt_id == tokenizer.pad_id:
                    continue
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                gen_count += 1
        infer_elapsed = time.time() - infer_start
        total_gen_tokens += gen_count
        total_gen_time += infer_elapsed
        tok_sec = gen_count / infer_elapsed if infer_elapsed > 0 else 0

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        print(f'  "{prompt}" -> "{generated_text}" ({gen_count} tok, {tok_sec:.1f} tok/sec)')

    avg_infer_tok_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0
    print(f"\nInference avg: {avg_infer_tok_sec:.1f} tok/sec ({total_gen_tokens} tokens in {total_gen_time:.2f}s)")

    # Save training history
    history_path = os.path.join(os.path.dirname(__file__), "training_history.json")
    history_entry = {
        "architecture": "neuroquantum",
        "dataset": DATASET_ID,
        "dataset_split": DATASET_SPLIT,
        "config": CONFIG,
        "total_texts": len(texts),
        "total_sequences": len(sequences),
        "parameters": n_params,
        "training_log": training_log,
        "total_tokens": grand_total_tokens,
        "total_time_sec": round(grand_total_time, 1),
        "avg_train_tok_per_sec": round(overall_tok_sec, 1),
        "avg_inference_tok_per_sec": round(avg_infer_tok_sec, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Append to existing history if present
    existing_history = []
    if os.path.isfile(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    existing_history = existing
                else:
                    existing_history = [existing]
        except Exception:
            pass
    existing_history.append(history_entry)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(existing_history, f, ensure_ascii=False, indent=2)
    print(f"\nTraining history saved: {history_path}")
    print("Done!")


if __name__ == "__main__":
    main()
