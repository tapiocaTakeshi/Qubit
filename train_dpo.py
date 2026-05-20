#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training for NeuroQuantum model.
Trains the model to prefer chosen responses over rejected responses.
"""
import os
import sys
import torch
import torch.nn.functional as F
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from dpo_utils import (
    load_preference_data_from_hf,
    load_preference_data_from_dict,
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

# DPO Training hyperparameters
EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
WARMUP_STEPS = 20
GRAD_CLIP = 1.0
DPO_BETA = 0.5


def load_preference_datasets() -> list:
    """Load preference datasets from multiple sources."""
    all_preferences = []

    print("  Loading preference datasets...")

    hf_datasets = [
        ("argilla/ultrafeedback-binarized", "train", 2000),
    ]

    for ds_id, split, max_samples in hf_datasets:
        print(f"    Loading {ds_id}...")
        try:
            prefs = load_preference_data_from_hf(ds_id, split=split, max_samples=max_samples)
            all_preferences.extend(prefs)
            print(f"      -> {len(prefs)} preference pairs")
        except Exception as e:
            print(f"      -> ERROR: {e}")

    crafted_prefs = create_preference_examples()
    for _ in range(10):
        all_preferences.extend(crafted_prefs)
    print(f"  + {len(crafted_prefs) * 10} crafted preference examples")

    return all_preferences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
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
    if prev_log:
        print(f"Previous training: {len(prev_log)} sessions, last loss: {prev_log[-1]['loss']:.4f}")

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

    # Load preference data
    print("\n=== Loading preference datasets ===")
    preference_pairs = load_preference_datasets()
    print(f"\nTotal preference pairs: {len(preference_pairs)}")

    if not preference_pairs:
        print("ERROR: No preference data loaded. Aborting.")
        return

    # Prepare training batches
    print("\n=== Preparing DPO training batches ===")
    batches = []

    for i in range(0, len(preference_pairs), BATCH_SIZE):
        batch_pairs = preference_pairs[i:i + BATCH_SIZE]

        batch_data = {
            "input_ids_chosen": [],
            "input_ids_rejected": [],
            "labels_chosen": [],
            "labels_rejected": [],
        }

        for pair in batch_pairs:
            prompt = pair["prompt"]
            chosen = pair["chosen"]
            rejected = pair["rejected"]

            chosen_pair = tokenize_preference_pair(prompt, chosen, rejected, tokenizer, max_seq_len)

            chosen_ids, rejected_ids = pad_sequence_pair(
                chosen_pair["chosen_ids"],
                chosen_pair["rejected_ids"],
                tokenizer.pad_id,
                max_len=max_seq_len
            )

            batch_data["input_ids_chosen"].append(chosen_ids)
            batch_data["input_ids_rejected"].append(rejected_ids)
            batch_data["labels_chosen"].append(chosen_ids)
            batch_data["labels_rejected"].append([-100 if id == tokenizer.pad_id else id for id in rejected_ids])

        if batch_data["input_ids_chosen"]:
            batches.append(batch_data)

    print(f"Training batches: {len(batches)}")

    # Calculate steps
    steps_per_epoch = len(batches)
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM_STEPS
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total optimization steps: {total_steps}")
    print(f"DPO Beta (temperature): {DPO_BETA}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    print(f"\n=== DPO Training for {EPOCHS} epochs ===")
    model.train()
    training_log = []
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        random.shuffle(batches)
        total_loss = 0
        total_dpo_loss = 0
        n_batches = 0
        all_metrics = {}
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(batches):
            input_ids_chosen = torch.tensor(batch["input_ids_chosen"], dtype=torch.long, device=device)
            input_ids_rejected = torch.tensor(batch["input_ids_rejected"], dtype=torch.long, device=device)
            labels_chosen = torch.tensor(batch["labels_chosen"], dtype=torch.long, device=device)
            labels_rejected = torch.tensor(batch["labels_rejected"], dtype=torch.long, device=device)

            # Forward pass for chosen responses
            logits_chosen = model(input_ids_chosen)

            # Forward pass for rejected responses
            logits_rejected = model(input_ids_rejected)

            # Compute DPO loss
            dpo_loss, log_probs_chosen, log_probs_rejected = compute_dpo_loss(
                logits_chosen,
                logits_rejected,
                labels_chosen,
                labels_rejected,
                beta=DPO_BETA,
                ignore_index=-100
            )

            # Compute metrics
            metrics = compute_dpo_metrics(log_probs_chosen, log_probs_rejected)
            for key, val in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(val)

            loss = dpo_loss / GRAD_ACCUM_STEPS
            loss.backward()

            total_loss += dpo_loss.item()
            total_dpo_loss += dpo_loss.item()
            n_batches += 1

            if n_batches % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if batch_idx % 10 == 0 and batch_idx > 0:
                avg_loss = total_dpo_loss / (n_batches % GRAD_ACCUM_STEPS or GRAD_ACCUM_STEPS)
                avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
                print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(batches)}, "
                      f"Loss: {avg_loss:.4f}, "
                      f"Accuracy: {avg_metrics.get('accuracy', 0):.4f}, "
                      f"Ratio: {avg_metrics.get('avg_log_ratio', 0):.4f}")

        epoch_loss = total_loss / max(n_batches, 1)
        epoch_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        log_entry = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": epoch_metrics,
        }
        training_log.append(log_entry)

        print(f"\nEpoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_metrics.get('accuracy', 0):.4f}")
        print(f"  Avg Log Ratio: {epoch_metrics.get('avg_log_ratio', 0):.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"  Best loss improved to {best_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            print(f"\n  Saving checkpoint...")
            checkpoint["model_state"] = model.state_dict()
            checkpoint["training_log"] = prev_log + training_log
            torch.save(checkpoint, CKPT_PATH)
            print(f"  Checkpoint saved")

    # Final save
    print("\n=== Training Complete ===")
    checkpoint["model_state"] = model.state_dict()
    checkpoint["training_log"] = prev_log + training_log
    torch.save(checkpoint, CKPT_PATH)

    # Try to sync to network volume if available
    try:
        sync_checkpoint_to_network_volume(CKPT_PATH)
    except Exception as e:
        print(f"Could not sync to network volume: {e}")

    print(f"Final loss: {best_loss:.4f}")
    print(f"Training log saved with checkpoint")


if __name__ == "__main__":
    main()
