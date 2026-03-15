#!/usr/bin/env python3
"""
Local training script for NeuroQuantum (32K SentencePiece) architecture.
Trains on 5 Japanese datasets, saves checkpoint for deployment.
"""
import os
import sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datetime import datetime, timezone
import json

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

# Config
CONFIG = {
    "vocab_size": 32000,
    "embed_dim": 512,
    "hidden_dim": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 512,
    "dropout": 0.1,
    "entangle_strength": 0.5,
}

DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output", "max_samples": 1540},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations", "max_samples": 3000},
    {"id": "fujiki/japanese_alpaca_data", "col": "output", "max_samples": 3000},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations", "max_samples": 3000},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations", "max_samples": 3000},
]

EPOCHS = 10
LR = 5e-4
BATCH_SIZE = 4
MAX_SEQ_LEN = CONFIG["max_seq_len"]


def extract_texts(ds, text_column, max_samples):
    """Extract text from dataset."""
    texts = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        col_data = row.get(text_column)
        if isinstance(col_data, str) and len(col_data.strip()) > 4:
            texts.append(col_data.strip())
        elif isinstance(col_data, list):
            parts = []
            for turn in col_data:
                if isinstance(turn, dict) and "value" in turn:
                    parts.append(turn["value"])
                elif isinstance(turn, dict) and "content" in turn:
                    parts.append(turn["content"])
                elif isinstance(turn, str):
                    parts.append(turn)
            combined = "\n".join(parts)
            if len(combined.strip()) > 4:
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
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    # Shuffle
    import random
    random.shuffle(sequences)

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
            avg = total_loss / n_batches
            print(f"  Epoch {epoch+1} | Batch {n_batches} | Avg Loss: {avg:.4f}")

    return total_loss / max(n_batches, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Load all datasets and extract texts
    print("\n=== Loading datasets ===")
    all_texts = []
    for ds_info in DATASETS:
        print(f"  Loading {ds_info['id']}...")
        try:
            ds = load_dataset(ds_info["id"], split="train", trust_remote_code=True)
            texts = extract_texts(ds, ds_info["col"], ds_info["max_samples"])
            print(f"    -> {len(texts)} texts extracted")
            all_texts.extend(texts)
        except Exception as e:
            print(f"    -> ERROR: {e}")

    print(f"\nTotal texts: {len(all_texts)}")

    # Step 2: Build SentencePiece tokenizer
    print("\n=== Building SentencePiece tokenizer ===")
    tokenizer = NeuroQuantumTokenizer(vocab_size=CONFIG["vocab_size"])
    tokenizer.build_vocab(
        all_texts,
        model_prefix=os.path.join(os.path.dirname(__file__), "neuroq_tokenizer"),
        character_coverage=0.9995,
    )
    actual_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    CONFIG["vocab_size"] = actual_vocab
    print(f"Actual vocab size: {actual_vocab}")

    # Step 3: Build model
    print("\n=== Building NeuroQuantum model ===")
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Step 4: Tokenize
    print("\n=== Tokenizing ===")
    sequences = tokenize_texts(all_texts, tokenizer, MAX_SEQ_LEN)
    print(f"Training sequences: {len(sequences)}")

    # Step 5: Train
    print("\n=== Training ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    training_log = []
    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, sequences, tokenizer, optimizer, epoch, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")
        training_log.append({"epoch": epoch + 1, "loss": avg_loss})

    # Step 6: Save checkpoint
    print("\n=== Saving checkpoint ===")
    ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
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
        "datasets": [d["id"] for d in DATASETS],
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved: {ckpt_path} ({os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB)")

    # Step 7: Quick inference test
    print("\n=== Inference test ===")
    model.eval()
    test_prompts = ["こんにちは", "量子コンピュータとは", "AIの未来について", "日本の首都は", "プログラミングを学ぶ"]
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(60):
                seq = input_tensor[:, -MAX_SEQ_LEN:]
                logits = model(seq)[:, -1, :] / 0.7
                # Top-K
                topk_vals = torch.topk(logits, 40)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')
                # Repetition penalty
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

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        print(f'  "{prompt}" -> "{generated_text}"')

    # Save training history
    history_path = os.path.join(os.path.dirname(__file__), "training_history.json")
    history = {
        "architecture": "neuroquantum",
        "config": CONFIG,
        "datasets": [d["id"] for d in DATASETS],
        "total_texts": len(all_texts),
        "total_sequences": len(sequences),
        "parameters": n_params,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\nTraining history saved: {history_path}")
    print("Done!")


if __name__ == "__main__":
    main()
