#!/usr/bin/env python3
"""Direct split training script - bypasses API for memory efficiency."""
import os
import sys
import torch
import torch.nn.functional as F
import json
import math
import random
import gc
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")

# Training params
NUM_CHUNKS = 4
MAX_SAMPLES_PER_DS = 200
EPOCHS_PER_CHUNK = 4
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 3e-5
WARMUP_STEPS = 20

# Crafted QA
CRAFTED_QA = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さは？\n回答: 富士山の高さは3776メートルです。",
    "質問: 太陽系で一番大きい惑星は？\n回答: 木星が太陽系で最も大きい惑星です。",
    "質問: 水の化学式は？\n回答: 水の化学式はH2Oです。",
    "質問: 光の速さは？\n回答: 光の速さは秒速約30万キロメートルです。",
    "質問: 日本で一番長い川は？\n回答: 信濃川が日本で最も長い川で、全長367kmです。",
    "質問: 地球の年齢はどれくらい？\n回答: 地球の年齢は約46億年です。",
    "質問: 人間の体で一番大きい臓器は？\n回答: 皮膚が人間の体で最も大きい臓器です。",
]


def format_qa_alpaca(row):
    inst = row.get("instruction", "")
    inp = row.get("input", "")
    out = row.get("output", "")
    if not inst or not out:
        return ""
    q = f"{inst}\n{inp}".strip() if inp else inst
    return f"質問: {q}\n回答: {out}"


def format_qa_conversations(row):
    convs = row.get("conversations", [])
    if not convs or len(convs) < 2:
        return ""
    parts = []
    for c in convs:
        role = c.get("from", "")
        val = c.get("value", "")
        if role == "human":
            parts.append(f"質問: {val}")
        elif role == "gpt":
            parts.append(f"回答: {val}")
    return "\n".join(parts) if parts else ""


def format_qa_izumi(row):
    text = row.get("text", "")
    return text.strip() if text else ""


QA_DATASETS_INFO = [
    {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
    {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "format": "alpaca"},
    {"id": "izumi-lab/llm-japanese-dataset", "format": "izumi"},
]


def load_qa_data_streaming(max_per_ds=200):
    """Load QA data using streaming to minimize memory."""
    import signal
    from datasets import load_dataset
    all_texts = []

    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("Dataset load timed out")

    for ds_info in QA_DATASETS_INFO:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        try:
            print(f"  Loading {ds_id} (streaming, max={max_per_ds})...")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout per dataset
            ds = load_dataset(ds_id, split="train", streaming=True)
            count = 0
            for row in ds:
                if fmt == "alpaca":
                    text = format_qa_alpaca(row)
                elif fmt == "conversations":
                    text = format_qa_conversations(row)
                elif fmt == "izumi":
                    text = format_qa_izumi(row)
                else:
                    continue
                if text and len(text) > 10:
                    all_texts.append(text)
                    count += 1
                    if count >= max_per_ds:
                        break
            signal.alarm(0)
            print(f"    -> {count} samples")
        except TimeoutError:
            signal.alarm(0)
            print(f"    -> Timed out, skipping")
        except Exception as e:
            signal.alarm(0)
            print(f"    -> Failed: {e}")
        gc.collect()

    # Add crafted QA
    for _ in range(40):
        all_texts.extend(CRAFTED_QA)
    print(f"  Total texts: {len(all_texts)} (including {40 * len(CRAFTED_QA)} crafted)")
    return all_texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t)
        if len(ids) > 2:
            sequences.append(ids[:max_seq_len])
    return sequences


def train_chunk(model, config, tokenizer, chunk_texts, chunk_idx, num_chunks, device):
    """Train one chunk."""
    print(f"\n{'='*60}")
    print(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(chunk_texts)} texts")
    print(f"{'='*60}")

    max_seq_len = config["max_seq_len"]
    sequences = tokenize_texts(chunk_texts, tokenizer, max_seq_len)
    print(f"  Sequences: {len(sequences)}")

    if not sequences:
        print("  No sequences, skipping.")
        return None

    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS_PER_CHUNK) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    min_lr_ratio = 0.1

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
                shift_logits.view(-1, config["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if n_batches % GRAD_ACCUM == 0:
                if global_step < WARMUP_STEPS:
                    lr = LR * global_step / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = LR * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
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
        print(f"  Chunk {chunk_idx+1} Epoch {epoch+1}/{EPOCHS_PER_CHUNK} | Loss: {avg_loss:.4f} | Steps: {global_step}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Clean up optimizer to free memory
    del optimizer
    gc.collect()
    return best_loss


def save_checkpoint(model, config, chunk_info):
    """Save model checkpoint."""
    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "chunk_info": chunk_info,
    }
    torch.save(ckpt, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"  Checkpoint saved: {CKPT_PATH} ({size_mb:.1f}MB)")


def main():
    print("=" * 60)
    print("  Split Training - Direct Script")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    device = torch.device("cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        lambda_entangle=cfg.get("entangle_strength", cfg.get("lambda_entangle", 0.5)),
        dropout=cfg.get("dropout", 0.1),
    )
    model = NeuroQuantum(model_config)
    model.load_state_dict(ckpt.get("model_state_dict") or ckpt["model_state"])
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params on {device}")

    tokenizer = NeuroQuantumTokenizer(model_file=TOKENIZER_PATH)
    del ckpt
    gc.collect()

    # Load data
    print("\n[2/3] Loading data (streaming)...")
    all_texts = load_qa_data_streaming(MAX_SAMPLES_PER_DS)

    random.shuffle(all_texts)
    chunk_size = max(len(all_texts) // NUM_CHUNKS, 1)
    chunks = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = start + chunk_size if i < NUM_CHUNKS - 1 else len(all_texts)
        chunks.append(all_texts[start:end])

    # Free all_texts
    del all_texts
    gc.collect()

    print(f"  Chunks: {[len(c) for c in chunks]}")

    # Train chunks
    print("\n[3/3] Training...")
    results = []
    for i in range(NUM_CHUNKS):
        best_loss = train_chunk(model, cfg, tokenizer, chunks[i], i, NUM_CHUNKS, device)
        results.append({"chunk": i + 1, "best_loss": best_loss})
        save_checkpoint(model, cfg, {"chunks_completed": i + 1, "results": results})
        # Free chunk data
        chunks[i] = None
        gc.collect()
        print(f"  Memory after chunk {i+1}: {torch.cuda.memory_allocated()/1024**2:.0f}MB" if torch.cuda.is_available() else "")

    # Final summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    for r in results:
        print(f"  Chunk {r['chunk']}: Loss = {r['best_loss']:.4f}" if r['best_loss'] else f"  Chunk {r['chunk']}: skipped")
    print(f"\n  Checkpoint: {CKPT_PATH}")

    # Quick inference test
    print("\n[Test] Quick inference...")
    model.eval()
    test_prompt = "質問: 日本の首都は"
    ids = tokenizer.encode(test_prompt)
    input_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(30):
            logits = model(input_t)
            next_logits = logits[0, -1, :] / 0.7
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            input_t = torch.cat([input_t, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tokenizer.eos_id:
                break
    output = tokenizer.decode(input_t[0].tolist())
    print(f"  Prompt: {test_prompt}")
    print(f"  Output: {output}")


if __name__ == "__main__":
    main()
