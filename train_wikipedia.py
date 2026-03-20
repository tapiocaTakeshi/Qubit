#!/usr/bin/env python3
"""日本語Wikipediaデータで追加学習するスクリプト。"""
import os
import sys
import torch
import torch.nn.functional as F
import math
import random
import gc
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")

# Training params
NUM_CHUNKS = 6
SAMPLES_PER_CHUNK = 3000
EPOCHS_PER_CHUNK = 4
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 3e-5
WARMUP_STEPS = 30
MIN_LR_RATIO = 0.1


def tokenize_texts(texts, tok, max_seq_len):
    sequences = []
    for t in texts:
        ids = tok.encode(t, add_special=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    return sequences


def train_chunk(model, tokenizer, cfg, device, sequences, chunk_idx, total_chunks):
    """Train one chunk of data."""
    if not sequences:
        print(f"  Chunk {chunk_idx+1}: no sequences, skipping")
        return None

    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS_PER_CHUNK) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    max_seq_len = cfg["max_seq_len"]

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS_PER_CHUNK):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i + BATCH_SIZE]
            if not batch_seqs:
                continue

            max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
            input_ids = []
            labels = []
            for s in batch_seqs:
                ids = s[:max_len]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [tokenizer.pad_id] * pad_len)
                labels.append(ids + [-100] * pad_len)

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(input_ids_t)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if n_batches % GRAD_ACCUM == 0:
                if global_step < WARMUP_STEPS:
                    cur_lr = LR * global_step / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    cur_lr = LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if n_batches % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Chunk {chunk_idx+1}/{total_chunks} Epoch {epoch+1}/{EPOCHS_PER_CHUNK} | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    del optimizer
    gc.collect()
    return best_loss


def save_checkpoint(model, cfg, prev_checkpoint, extra_info=""):
    prev_log = prev_checkpoint.get("training_log", [])
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
        "training_log": prev_log + [{"type": "wikipedia", "info": extra_info}],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(set(prev_checkpoint.get("datasets", []) + ["wikipedia-ja"])),
        "qa_training": prev_checkpoint.get("qa_training", False),
    }
    torch.save(new_checkpoint, CKPT_PATH)
    print(f"  Checkpoint saved. {extra_info}")


def main():
    from datasets import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    cfg = checkpoint["config"]
    tokenizer = NeuroQuantumTokenizer(vocab_size=cfg["vocab_size"], model_file=TOKENIZER_PATH)

    nq_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg.get("hidden_dim", cfg["embed_dim"] * 2),
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.1),
        lambda_entangle=cfg.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params")

    # Load Wikipedia data in streaming mode, chunk by chunk
    print(f"\nLoading Japanese Wikipedia (streaming)...")
    print(f"Plan: {NUM_CHUNKS} chunks x {SAMPLES_PER_CHUNK} samples x {EPOCHS_PER_CHUNK} epochs")

    try:
        ds = load_dataset("izumi-lab/wikipedia-ja-20230720", split="train", streaming=True)
    except Exception:
        print("Trying alternative Wikipedia dataset...")
        try:
            ds = load_dataset("singletongue/wikipedia-utils", "passages-c400-jawiki-20230403", split="train", streaming=True)
        except Exception:
            print("Trying range3/wikipedia-ja...")
            ds = load_dataset("range3/wikipedia-ja", split="train", streaming=True)

    max_seq_len = cfg["max_seq_len"]
    total_loaded = 0

    for chunk_idx in range(NUM_CHUNKS):
        print(f"\n--- Chunk {chunk_idx+1}/{NUM_CHUNKS} ---")

        # Collect texts for this chunk
        texts = []
        for row in ds:
            text = row.get("text", row.get("passage", "")).strip()
            if text and len(text) > 50:
                texts.append(text)
                if len(texts) >= SAMPLES_PER_CHUNK:
                    break

        total_loaded += len(texts)
        print(f"  Loaded {len(texts)} texts (total: {total_loaded})")

        if not texts:
            print("  No more data available, stopping.")
            break

        # Tokenize
        sequences = tokenize_texts(texts, tokenizer, max_seq_len)
        print(f"  Sequences: {len(sequences)}")

        # Free text memory
        del texts
        gc.collect()

        # Train
        best_loss = train_chunk(model, tokenizer, cfg, device, sequences, chunk_idx, NUM_CHUNKS)

        # Free sequences
        del sequences
        gc.collect()

        # Save after each chunk
        model.eval()
        save_checkpoint(model, cfg, checkpoint, f"chunk_{chunk_idx+1} loss={best_loss:.4f}")
        # Reload checkpoint for next save
        checkpoint = torch.load(CKPT_PATH, map_location="cpu")
        checkpoint["model_state"] = model.state_dict()

    print(f"\n=== Wikipedia training complete! Total samples: {total_loaded} ===")

    # Test inference
    print("\n=== テスト推論 ===")
    model.eval()
    import api
    api.model = model
    api.tokenizer = tokenizer
    api.config = cfg
    api.device = device
    from api import generate_text

    test_prompts = [
        "質問: ChatGPTについて教えて\n回答:",
        "質問: 日本の首都はどこですか？\n回答:",
        "質問: 富士山について教えてください\n回答:",
    ]
    for prompt in test_prompts:
        result = generate_text(prompt, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3)
        q = prompt.split("\n")[0]
        print(f"\n{q}")
        print(f"回答: {result}")


if __name__ == "__main__":
    main()
