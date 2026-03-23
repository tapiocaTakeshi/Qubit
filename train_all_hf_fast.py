#!/usr/bin/env python3
"""全HuggingFaceデータセットから全データをサンプリングして高速学習。
全6データセット × 各5000サンプル = 30,000テキスト、5エポックで集中学習。
"""
import os, sys, gc, math, random, time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")

BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 8e-5
WARMUP_STEPS = 40
MIN_LR_RATIO = 0.05
GRAD_CLIP = 1.0
EPOCHS = 5
SAMPLES_PER_DS = 5000

DATASETS = [
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
]
CC100_SAMPLES = 5000


def extract_texts(ds, text_column, max_samples):
    texts = []
    indices = list(range(len(ds)))
    random.shuffle(indices)
    if max_samples and max_samples < len(ds):
        indices = indices[:max_samples]
    for idx in indices:
        row = ds[idx]
        col_data = row.get(text_column)
        if isinstance(col_data, str) and len(col_data.strip()) > 4:
            texts.append(col_data.strip())
        elif isinstance(col_data, list):
            parts = []
            for turn in col_data:
                if isinstance(turn, dict):
                    val = turn.get("value") or turn.get("content") or ""
                    if val:
                        parts.append(val)
                elif isinstance(turn, str):
                    parts.append(turn)
            combined = "\n".join(parts)
            if len(combined.strip()) > 4:
                texts.append(combined.strip())
    return texts


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
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return max_lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine_decay)


def train(model, tokenizer, cfg, device, sequences):
    max_seq_len = cfg["max_seq_len"]
    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    global_step = 0
    best_loss = float('inf')
    log = []

    for epoch in range(EPOCHS):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()
        epoch_start = time.time()

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
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if n_batches % 100 == 0:
                avg = total_loss / n_batches
                elapsed = time.time() - epoch_start
                print(f"    Batch {n_batches}/{steps_per_epoch} | Loss: {avg:.4f} | {elapsed:.0f}s")

        if n_batches % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Steps: {global_step} | {elapsed:.0f}s")
        log.append({"epoch": epoch + 1, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss

    del optimizer
    gc.collect()
    return best_loss, log


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    cfg = checkpoint["config"]
    tokenizer = NeuroQuantumTokenizer(vocab_size=cfg["vocab_size"], model_file=TOKENIZER_PATH)

    nq_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_dim=cfg.get("hidden_dim", cfg["embed_dim"] * 2),
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"], dropout=cfg.get("dropout", 0.1),
        lambda_entangle=cfg.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Phase 1: 全データセットからサンプル取得
    print(f"\n{'='*60}")
    print(f"Phase 1: 全HFデータセット ({SAMPLES_PER_DS}サンプル/DS)")
    print(f"{'='*60}")

    all_texts = []
    loaded_datasets = []

    for ds_info in DATASETS:
        ds_id = ds_info["id"]
        print(f"\n  Loading {ds_id}...")
        try:
            ds = load_dataset(ds_id, split="train")
            texts = extract_texts(ds, ds_info["col"], SAMPLES_PER_DS)
            print(f"    → {len(texts)} texts (from {len(ds)} total)")
            all_texts.extend(texts)
            loaded_datasets.append(ds_id)
            del ds
            gc.collect()
        except Exception as e:
            print(f"    → ERROR: {e}")

    print(f"\n  Loading range3/cc100-ja (streaming, {CC100_SAMPLES})...")
    try:
        ds_cc = load_dataset("range3/cc100-ja", split="train", streaming=True)
        cc_texts = []
        for i, row in enumerate(ds_cc):
            if i >= CC100_SAMPLES:
                break
            text = row.get("text", "").strip()
            if len(text) > 10:
                cc_texts.append(text)
        print(f"    → {len(cc_texts)} texts")
        all_texts.extend(cc_texts)
        loaded_datasets.append("range3/cc100-ja")
        del cc_texts
        gc.collect()
    except Exception as e:
        print(f"    → ERROR: {e}")

    print(f"\n  合計テキスト数: {len(all_texts)}")

    # Phase 2: トークナイズ
    print(f"\n{'='*60}")
    print(f"Phase 2: トークナイズ")
    print(f"{'='*60}")
    sequences = tokenize_texts(all_texts, tokenizer, cfg["max_seq_len"])
    print(f"  シーケンス数: {len(sequences)}")
    del all_texts
    gc.collect()

    # Phase 3: 学習
    print(f"\n{'='*60}")
    print(f"Phase 3: 学習 ({EPOCHS} epochs)")
    print(f"{'='*60}")

    best_loss, train_log = train(model, tokenizer, cfg, device, sequences)
    print(f"\n  Best Loss: {best_loss:.4f}")
    del sequences
    gc.collect()

    # Save
    model.eval()
    prev_log = checkpoint.get("training_log", [])
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
        "training_log": prev_log + [{"type": "hf-all-sampled", "info": f"loss={best_loss:.4f}, samples_per_ds={SAMPLES_PER_DS}", "datasets": loaded_datasets}],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(set(checkpoint.get("datasets", []) + loaded_datasets)),
        "qa_training": checkpoint.get("qa_training", False),
    }
    torch.save(new_checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    print(f"\n  Checkpoint saved: {size_mb:.1f} MB")

    # テスト推論
    print(f"\n{'='*60}")
    print(f"テスト推論")
    print(f"{'='*60}")

    import api
    api.model = model
    api.tokenizer = tokenizer
    api.config = cfg
    api.device = device
    from api import generate_text

    test_prompts = [
        "質問: 1+1は？ 回答:",
        "質問: 三角形の面積の公式は？ 回答:",
        "質問: 円周率とは？ 回答:",
        "質問: 微分とは？ 回答:",
        "質問: 素数とは？ 回答:",
        "質問: 日本の首都は？ 回答:",
        "質問: AIとは？ 回答:",
        "こんにちは",
    ]
    for prompt in test_prompts:
        result = generate_text(prompt, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3)
        print(f"\n  {prompt}")
        print(f"  → {result.strip()[:250]}")

    print(f"\n{'='*60}")
    print(f"全データセット学習完了!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
