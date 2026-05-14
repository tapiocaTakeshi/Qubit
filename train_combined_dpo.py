#!/usr/bin/env python3
"""
Combined training: standard QA training followed by DPO fine-tuning.
This approach first trains on QA pairs, then aligns with preferences using DPO.
"""
import os
import sys
import torch
import torch.nn.functional as F
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from dpo_utils import (
    load_preference_data_from_hf,
    create_preference_examples,
    tokenize_preference_pair,
    pad_sequence_pair,
    compute_dpo_loss,
    compute_dpo_metrics,
)
from datetime import datetime, timezone
import json
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, migrate_legacy_state_dict

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# QA Training hyperparameters
QA_EPOCHS = 2
QA_LR = 5e-5
QA_BATCH_SIZE = 4
QA_GRAD_ACCUM = 8

# DPO Training hyperparameters
DPO_EPOCHS = 3
DPO_LR = 1e-5
DPO_BATCH_SIZE = 2
DPO_GRAD_ACCUM = 4
DPO_BETA = 0.5

WARMUP_STEPS = 20
GRAD_CLIP = 1.0

# QA datasets (same as train_qa.py)
QA_DATASETS = [
    {
        "id": "fujiki/japanese_alpaca_data",
        "max_samples": 5000,
        "format": "alpaca",
    },
    {
        "id": "FreedomIntelligence/alpaca-gpt4-japanese",
        "max_samples": 5000,
        "format": "conversations",
    },
]


def format_qa_alpaca(row):
    """Format alpaca-style data as QA."""
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    if inp:
        q = f"{instruction}\n{inp}"
    else:
        q = instruction
    return f"質問: {q}\n回答: {output}"


def format_qa_conversations(row):
    """Format conversation-style data as QA pairs."""
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn = convs[i]
        next_turn = convs[i + 1]
        q_text = ""
        a_text = ""
        if isinstance(turn, dict):
            q_text = turn.get("value", turn.get("content", "")).strip()
        elif isinstance(turn, str):
            q_text = turn.strip()
        if isinstance(next_turn, dict):
            a_text = next_turn.get("value", next_turn.get("content", "")).strip()
        elif isinstance(next_turn, str):
            a_text = next_turn.strip()
        if q_text and a_text:
            pairs.append(f"質問: {q_text}\n回答: {a_text}")
        i += 2
    if pairs:
        return "\n\n".join(pairs)
    return None


def load_qa_data():
    """Load QA datasets."""
    all_qa = []

    for ds_info in QA_DATASETS:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        max_samples = ds_info["max_samples"]
        print(f"  Loading {ds_id}...")

        try:
            ds = safe_load_dataset(ds_id, split="train")
            n = min(max_samples, len(ds))
            count = 0

            for row in ds.select(range(n)):
                text = None
                if fmt == "alpaca":
                    text = format_qa_alpaca(row)
                elif fmt == "conversations":
                    text = format_qa_conversations(row)

                if text and len(text) > 10:
                    all_qa.append(text)
                    count += 1

            print(f"    -> {count} QA samples")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    # Add hand-crafted examples
    crafted_qa = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: プログラミングとは？\n回答: プログラミングはコンピュータに命令を書くことです。",
        "質問: 機械学習とは？\n回答: 機械学習はデータからパターンを学習するAI技術です。",
    ]
    for _ in range(10):
        all_qa.extend(crafted_qa)
    print(f"  + {len(crafted_qa) * 10} crafted QA samples")

    return all_qa


def tokenize_texts(texts, tokenizer, max_seq_len):
    """Tokenize QA texts."""
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t, add_special=True, add_boundary=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            chunks = list(range(0, len(ids) - max_seq_len + 1, stride))
            for idx, start in enumerate(chunks):
                chunk = ids[start:start + max_seq_len]
                if idx > 0 and chunk[0] == tokenizer.bof_id:
                    chunk = chunk[1:]
                if idx < len(chunks) - 1 and chunk[-1] == tokenizer.eof_id:
                    chunk = chunk[:-1]
                sequences.append(chunk)
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_qa_phase(model, tokenizer, nq_config, device, training_log):
    """QA training phase."""
    print("\n=== QA Training Phase ===")

    # Load QA data
    qa_texts = load_qa_data()
    print(f"Total QA texts: {len(qa_texts)}")

    # Tokenize
    sequences = tokenize_texts(qa_texts, tokenizer, nq_config.max_seq_len)
    print(f"Training sequences: {len(sequences)}")

    steps_per_epoch = len(sequences) // QA_BATCH_SIZE
    total_steps = (steps_per_epoch * QA_EPOCHS) // QA_GRAD_ACCUM
    print(f"Steps per epoch: {steps_per_epoch}, Total: {total_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=QA_LR, weight_decay=0.01)
    model.train()
    global_step = 0

    for epoch in range(QA_EPOCHS):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), QA_BATCH_SIZE):
            batch_seqs = sequences[i:i + QA_BATCH_SIZE]
            if not batch_seqs:
                continue

            max_len = min(max(len(s) for s in batch_seqs), nq_config.max_seq_len)
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
            loss = loss / QA_GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * QA_GRAD_ACCUM
            n_batches += 1

            if n_batches % QA_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, QA_LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        epoch_loss = total_loss / max(n_batches, 1)
        training_log.append({
            "phase": "QA",
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print(f"Epoch {epoch+1}/{QA_EPOCHS} - Loss: {epoch_loss:.4f}")


def train_dpo_phase(model, tokenizer, nq_config, device, training_log):
    """DPO training phase."""
    print("\n=== DPO Training Phase ===")

    # Load preference data
    preference_pairs = []
    preference_pairs.extend(create_preference_examples() * 15)

    try:
        prefs = load_preference_data_from_hf("argilla/ultrafeedback-binarized", split="train", max_samples=1000)
        preference_pairs.extend(prefs)
    except Exception as e:
        print(f"Could not load HF preferences: {e}")

    print(f"Total preference pairs: {len(preference_pairs)}")

    if not preference_pairs:
        print("No preference data available, skipping DPO phase")
        return

    # Prepare batches
    batches = []
    for i in range(0, len(preference_pairs), DPO_BATCH_SIZE):
        batch_pairs = preference_pairs[i:i + DPO_BATCH_SIZE]
        batch_data = {
            "input_ids_chosen": [],
            "input_ids_rejected": [],
            "labels_chosen": [],
            "labels_rejected": [],
        }

        for pair in batch_pairs:
            chosen_pair = tokenize_preference_pair(
                pair["prompt"], pair["chosen"], pair["rejected"],
                tokenizer, nq_config.max_seq_len
            )
            chosen_ids, rejected_ids = pad_sequence_pair(
                chosen_pair["chosen_ids"], chosen_pair["rejected_ids"],
                tokenizer.pad_id, max_len=nq_config.max_seq_len
            )

            batch_data["input_ids_chosen"].append(chosen_ids)
            batch_data["input_ids_rejected"].append(rejected_ids)
            batch_data["labels_chosen"].append(chosen_ids)
            batch_data["labels_rejected"].append([-100 if id == tokenizer.pad_id else id for id in rejected_ids])

        if batch_data["input_ids_chosen"]:
            batches.append(batch_data)

    print(f"DPO batches: {len(batches)}")

    steps_per_epoch = len(batches)
    total_steps = (steps_per_epoch * DPO_EPOCHS) // DPO_GRAD_ACCUM
    print(f"Total DPO steps: {total_steps}, Beta: {DPO_BETA}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=DPO_LR, weight_decay=0.01)
    model.train()
    global_step = 0

    for epoch in range(DPO_EPOCHS):
        random.shuffle(batches)
        total_loss = 0
        n_batches = 0
        all_accuracy = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(batches):
            input_ids_chosen = torch.tensor(batch["input_ids_chosen"], dtype=torch.long, device=device)
            input_ids_rejected = torch.tensor(batch["input_ids_rejected"], dtype=torch.long, device=device)
            labels_chosen = torch.tensor(batch["labels_chosen"], dtype=torch.long, device=device)
            labels_rejected = torch.tensor(batch["labels_rejected"], dtype=torch.long, device=device)

            logits_chosen = model(input_ids_chosen)
            logits_rejected = model(input_ids_rejected)

            dpo_loss, log_probs_chosen, log_probs_rejected = compute_dpo_loss(
                logits_chosen, logits_rejected,
                labels_chosen, labels_rejected,
                beta=DPO_BETA
            )

            metrics = compute_dpo_metrics(log_probs_chosen, log_probs_rejected)
            all_accuracy.append(metrics["accuracy"])

            loss = dpo_loss / DPO_GRAD_ACCUM
            loss.backward()

            total_loss += dpo_loss.item()
            n_batches += 1

            if n_batches % DPO_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, DPO_LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        epoch_loss = total_loss / max(n_batches, 1)
        avg_acc = sum(all_accuracy) / len(all_accuracy) if all_accuracy else 0
        training_log.append({
            "phase": "DPO",
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": avg_acc,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print(f"Epoch {epoch+1}/{DPO_EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {avg_acc:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print("=== Loading checkpoint ===")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])
    print(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}")

    # Load tokenizer and model
    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

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
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Combined training
    training_log = []

    train_qa_phase(model, tokenizer, nq_config, device, training_log)
    train_dpo_phase(model, tokenizer, nq_config, device, training_log)

    # Save final checkpoint
    print("\n=== Training Complete ===")
    checkpoint["model_state"] = model.state_dict()
    checkpoint["training_log"] = prev_log + training_log
    torch.save(checkpoint, CKPT_PATH)

    try:
        sync_checkpoint_to_network_volume(CKPT_PATH)
    except Exception as e:
        print(f"Could not sync: {e}")

    print("Training complete and checkpoint saved")


if __name__ == "__main__":
    main()
