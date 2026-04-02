#!/usr/bin/env python3
"""
Extended training script to improve model quality.
Continues from existing checkpoint with:
- More epochs (15)
- Gradient accumulation (effective batch size 32)
- Learning rate warmup
- Lower LR for continued training
- More training data
"""
import os
import sys
import torch
import torch.nn.functional as F
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from datetime import datetime, timezone
import json
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, get_gpu_adaptive_config, migrate_legacy_state_dict

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# GPUの性能に基づいてバッチサイズを自動決定
_GPU_CONFIG = get_gpu_adaptive_config(vocab_size=32000)

# Training hyperparameters
EPOCHS = 15
LR = 1e-4  # Lower LR for continued training
BATCH_SIZE = _GPU_CONFIG["batch_size"]
GRAD_ACCUM_STEPS = max(1, 32 // BATCH_SIZE)  # 有効バッチサイズ ≈ 32を維持
WARMUP_STEPS = 100
GRAD_CLIP = 1.0

# Datasets with increased sample sizes
DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output", "max_samples": 3000},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations", "max_samples": 8000},
    {"id": "fujiki/japanese_alpaca_data", "col": "output", "max_samples": 8000},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations", "max_samples": 8000},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations", "max_samples": 8000},
]
CC100_SAMPLES = 10000


def extract_texts(ds, text_column, max_samples):
    """Extract text from dataset."""
    texts = []
    n = min(max_samples, len(ds))
    for row in ds.select(range(n)):
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
    """Tokenize texts into training sequences.

    Each chunk always starts with BOS and ends with EOS so the model
    learns proper sentence boundaries and avoids mid-sentence generation.
    """
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
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print("=== Loading checkpoint ===")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])
    print(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}, "
          f"vocab={config['vocab_size']}")
    print(f"Previous training: {len(prev_log)} epochs, last loss: {prev_log[-1]['loss']:.4f}")

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

    # Build model
    max_seq_len = config["max_seq_len"]
    nq_config = NeuroQuantumConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=max_seq_len,
        dropout=config.get("dropout", 0.1),
        lambda_entangle=config.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
    model.load_state_dict(migrated)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Load all datasets
    print("\n=== Loading datasets ===")
    all_texts = []
    for ds_info in DATASETS:
        print(f"  Loading {ds_info['id']}...")
        try:
            ds = safe_load_dataset(ds_info["id"], split="train")
            texts = extract_texts(ds, ds_info["col"], ds_info["max_samples"])
            print(f"    -> {len(texts)} texts")
            all_texts.extend(texts)
        except Exception as e:
            print(f"    -> ERROR: {e}")

    # Load CC100-ja
    print(f"  Loading range3/cc100-ja (streaming, {CC100_SAMPLES} samples)...")
    try:
        ds_cc = safe_load_dataset("range3/cc100-ja", split="train", streaming=True)
        cc_texts = []
        for i, row in enumerate(ds_cc):
            if i >= CC100_SAMPLES:
                break
            text = row.get("text", "").strip()
            if len(text) > 10:
                cc_texts.append(text)
        print(f"    -> {len(cc_texts)} texts")
        all_texts.extend(cc_texts)
    except Exception as e:
        print(f"    -> ERROR: {e}")

    print(f"\nTotal texts: {len(all_texts)}")

    # Tokenize
    print("\n=== Tokenizing ===")
    sequences = tokenize_texts(all_texts, tokenizer, max_seq_len)
    print(f"Training sequences: {len(sequences)}")

    # Calculate total steps
    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM_STEPS
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # Setup optimizer (no scheduler - we do manual LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    print(f"\n=== Training for {EPOCHS} epochs ===")
    model.train()
    training_log = []
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS):
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
            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches += 1

            # Gradient accumulation step
            if n_batches % GRAD_ACCUM_STEPS == 0:
                # Update LR
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if n_batches % 200 == 0:
                avg = total_loss / n_batches
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {n_batches} | "
                      f"Loss: {avg:.4f} | LR: {lr:.2e} | Step: {global_step}")

        # Handle remaining gradients
        if n_batches % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")
        training_log.append({"epoch": len(prev_log) + epoch + 1, "loss": avg_loss})

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  New best loss: {best_loss:.4f}, saving checkpoint...")
            save_checkpoint(model, config, prev_log + training_log, checkpoint)

    # Final save
    save_checkpoint(model, config, prev_log + training_log, checkpoint)

    # Inference test
    print("\n=== Inference test ===")
    model.eval()
    test_prompts = [
        "こんにちは",
        "量子コンピュータとは",
        "AIの未来について",
        "日本の首都は",
        "プログラミングを学ぶ",
        "天気が良い日は",
        "今日は何をしましょうか",
    ]
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(80):
                seq = input_tensor[:, -max_seq_len:]
                logits = model(seq)[:, -1, :] / 0.7
                # Top-K filtering
                topk_vals = torch.topk(logits, 40)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')
                # Repetition penalty
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= 1.3
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()
                if nxt_id in (tokenizer.eos_id, tokenizer.eof_id):
                    break
                if nxt_id in (tokenizer.pad_id, tokenizer.bof_id):
                    input_tensor = torch.cat([input_tensor, nxt], dim=1)
                    continue
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        print(f'  "{prompt}" -> "{generated_text}"')

    print("\nDone!")


def save_checkpoint(model, config, training_log, original_ckpt):
    """Save checkpoint."""
    all_datasets = list(set(
        original_ckpt.get("datasets", []) +
        [d["id"] for d in DATASETS] +
        ["range3/cc100-ja"]
    ))
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": all_datasets,
    }
    torch.save(new_checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    print(f"  Saved: {CKPT_PATH} ({size_mb:.1f} MB)")
    sync_checkpoint_to_network_volume(CKPT_PATH)


if __name__ == "__main__":
    main()
