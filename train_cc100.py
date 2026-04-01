#!/usr/bin/env python3
"""
Additional training on range3/cc100-ja dataset.
Loads existing checkpoint and continues training.
"""
import os
import sys
import torch
import torch.nn.functional as F
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from datetime import datetime, timezone
import json
import random

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, get_gpu_adaptive_config, migrate_legacy_state_dict

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
MAX_SAMPLES = 5000
EPOCHS = 3
LR = 3e-4

# GPUの性能に基づいてバッチサイズを自動決定
_GPU_CONFIG = get_gpu_adaptive_config(vocab_size=32000)
BATCH_SIZE = _GPU_CONFIG["batch_size"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print("=== Loading checkpoint ===")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    print(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}, vocab={config['vocab_size']}, seq_len={config['max_seq_len']}")

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

    # Build model
    nq_config = NeuroQuantumConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.1),
        lambda_entangle=config.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
    model.load_state_dict(migrated)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load cc100-ja
    print("\n=== Loading range3/cc100-ja ===")
    ds = safe_load_dataset("range3/cc100-ja", split="train", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= MAX_SAMPLES:
            break
        text = row.get("text", "").strip()
        if len(text) > 10:
            texts.append(text)
    print(f"Loaded {len(texts)} texts")

    # Tokenize
    max_seq_len = config["max_seq_len"]
    sequences = []
    for t in texts:
        content_ids = tokenizer.encode(t, add_special=False)
        max_content = max_seq_len - 2  # Reserve slots for BOS and EOS
        if max_content <= 0:
            continue
        if len(content_ids) <= max_content:
            if len(content_ids) >= 2:
                seq = [tokenizer.bof_id, tokenizer.bos_id] + content_ids + [tokenizer.eos_id, tokenizer.eof_id]
                sequences.append(seq)
        else:
            stride = max(max_content // 2, 1)
            chunks = list(range(0, len(content_ids) - max_content + 1, stride))
            for idx, start in enumerate(chunks):
                chunk = content_ids[start:start + max_content]
                prefix = [tokenizer.bof_id, tokenizer.bos_id] if idx == 0 else [tokenizer.bos_id]
                suffix = [tokenizer.eos_id, tokenizer.eof_id] if idx == len(chunks) - 1 else [tokenizer.eos_id]
                seq = prefix + chunk + suffix
                sequences.append(seq)
            remaining = content_ids[-max_content:]
            tail_seq = [tokenizer.bos_id] + remaining + [tokenizer.eos_id, tokenizer.eof_id]
            if tail_seq != sequences[-1]:
                sequences.append(tail_seq)
    print(f"Training sequences: {len(sequences)}")

    # Train
    print("\n=== Training on cc100-ja ===")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    training_log = []
    for epoch in range(EPOCHS):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0

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

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, nq_config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {n_batches} | Avg Loss: {total_loss/n_batches:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")
        training_log.append({"epoch": epoch + 1, "loss": avg_loss})

    # Save checkpoint
    print("\n=== Saving checkpoint ===")
    config["datasets"] = checkpoint.get("datasets", []) + ["range3/cc100-ja"]
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": checkpoint.get("training_log", []) + training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": config["datasets"],
    }
    torch.save(new_checkpoint, CKPT_PATH)
    print(f"Saved: {CKPT_PATH} ({os.path.getsize(CKPT_PATH) / 1024 / 1024:.1f} MB)")
    sync_checkpoint_to_network_volume(CKPT_PATH)

    # Inference test
    print("\n=== Inference test ===")
    model.eval()
    test_prompts = ["こんにちは", "量子コンピュータとは", "日本の首都は", "プログラミングを学ぶ", "天気が良い日は"]
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(60):
                seq = input_tensor[:, -max_seq_len:]
                logits = model(seq)[:, -1, :] / 0.7
                topk_vals = torch.topk(logits, 40)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= 1.3
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()
                if nxt_id in (tokenizer.eos_id, tokenizer.eof_id):
                    break
                if nxt_id in (tokenizer.pad_id, tokenizer.bof_id):
                    continue
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        print(f'  "{prompt}" -> "{generated_text}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
