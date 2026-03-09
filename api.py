#!/usr/bin/env python3
"""
Model Training & Inference API for NeuroQuantum.
Provides REST endpoints for training and text generation.
"""
import os
import sys
import torch
import torch.nn.functional as F
import json
import math
import random
import threading
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

app = FastAPI(title="NeuroQuantum API", version="1.0.0")

# Global state
model = None
tokenizer = None
config = None
device = None
training_status = {"running": False, "log": [], "message": "idle"}
CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")


# ========================================
# Request/Response models
# ========================================

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.3


class InferenceResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int


class TrainRequest(BaseModel):
    dataset_ids: Optional[List[str]] = None
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 4
    grad_accum_steps: int = 8
    warmup_steps: int = 100
    max_samples_per_dataset: int = 5000
    include_knowledge: bool = False  # Include Wikipedia/knowledge datasets


class TrainResponse(BaseModel):
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    running: bool
    log: list
    message: str


# ========================================
# Model loading
# ========================================

def load_model():
    """Load model and tokenizer from checkpoint."""
    global model, tokenizer, config, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]

    tokenizer = NeuroQuantumTokenizer(
        vocab_size=config["vocab_size"], model_file=TOKENIZER_PATH
    )

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
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params on {device}")
    return checkpoint


# ========================================
# Inference
# ========================================

def generate_text(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7,
                  top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.3) -> str:
    """Generate text from prompt."""
    global model, tokenizer, config, device

    tokens = tokenizer.encode(prompt, add_special=True)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = list(tokens)
    max_seq_len = config["max_seq_len"]

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq = input_tensor[:, -max_seq_len:]
            logits = model(seq)[:, -1, :] / max(temperature, 1e-5)

            # Top-K filtering
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals = torch.topk(logits, k)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')

            # Top-P filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                to_remove = cumulative_probs > top_p
                to_remove[:, 1:] = to_remove[:, :-1].clone()
                to_remove[:, 0] = False
                indices_to_remove = sorted_indices[to_remove]
                logits[0, indices_to_remove] = float('-inf')

            # Repetition penalty
            if repetition_penalty > 1.0 and len(generated) > 1:
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= repetition_penalty

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
    return generated_text


# ========================================
# Training
# ========================================

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


def tokenize_texts(texts, tok, max_seq_len):
    """Tokenize texts into training sequences."""
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


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


# Default datasets (conversation/instruction)
DEFAULT_DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
]

# Knowledge datasets to fill factual/grammatical gaps
KNOWLEDGE_DATASETS = [
    {"id": "singletongue/wikipedia-utils", "col": "text",
     "config": "passages-c400-jawiki-20240401", "streaming": True},
    {"id": "range3/wikipedia-ja-20230101", "col": "text", "streaming": True},
]


def run_training(req: TrainRequest):
    """Run training in background thread."""
    global model, tokenizer, config, device, training_status
    from datasets import load_dataset

    training_status = {"running": True, "log": [], "message": "Loading datasets..."}

    try:
        # Load datasets
        all_texts = []
        datasets_to_use = DEFAULT_DATASETS
        if req.dataset_ids:
            datasets_to_use = [{"id": did, "col": "text"} for did in req.dataset_ids]

        for ds_info in datasets_to_use:
            try:
                training_status["message"] = f"Loading {ds_info['id']}..."
                ds = load_dataset(ds_info["id"], split="train")
                texts = extract_texts(ds, ds_info["col"], req.max_samples_per_dataset)
                all_texts.extend(texts)
                training_status["log"].append(f"Loaded {ds_info['id']}: {len(texts)} texts")
            except Exception as e:
                training_status["log"].append(f"Error loading {ds_info['id']}: {e}")

        # Also load cc100-ja
        try:
            training_status["message"] = "Loading cc100-ja..."
            ds_cc = load_dataset("range3/cc100-ja", split="train", streaming=True)
            cc_texts = []
            for i, row in enumerate(ds_cc):
                if i >= req.max_samples_per_dataset:
                    break
                text = row.get("text", "").strip()
                if len(text) > 10:
                    cc_texts.append(text)
            all_texts.extend(cc_texts)
            training_status["log"].append(f"Loaded cc100-ja: {len(cc_texts)} texts")
        except Exception as e:
            training_status["log"].append(f"Error loading cc100-ja: {e}")

        # Load knowledge datasets (Wikipedia etc.) if requested
        if req.include_knowledge:
            for ds_info in KNOWLEDGE_DATASETS:
                try:
                    ds_name = ds_info["id"]
                    training_status["message"] = f"Loading knowledge: {ds_name}..."
                    kwargs = {"split": "train", "streaming": ds_info.get("streaming", False)}
                    if "config" in ds_info:
                        kwargs["name"] = ds_info["config"]
                    ds_k = load_dataset(ds_info["id"], **kwargs)
                    k_texts = []
                    for j, row in enumerate(ds_k):
                        if j >= req.max_samples_per_dataset:
                            break
                        text = row.get(ds_info["col"], "").strip()
                        if len(text) > 20:
                            k_texts.append(text)
                    all_texts.extend(k_texts)
                    training_status["log"].append(
                        f"Loaded knowledge {ds_name}: {len(k_texts)} texts")
                except Exception as e:
                    training_status["log"].append(
                        f"Error loading knowledge {ds_name}: {e}")

        training_status["message"] = "Tokenizing..."
        max_seq_len = config["max_seq_len"]
        sequences = tokenize_texts(all_texts, tokenizer, max_seq_len)
        training_status["log"].append(f"Total: {len(all_texts)} texts -> {len(sequences)} sequences")

        # Training
        steps_per_epoch = len(sequences) // req.batch_size
        total_steps = (steps_per_epoch * req.epochs) // req.grad_accum_steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=req.lr, weight_decay=0.01)

        model.train()
        global_step = 0

        for epoch in range(req.epochs):
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()

            training_status["message"] = f"Training epoch {epoch+1}/{req.epochs}..."
            log_interval = max(steps_per_epoch // 10, 1)  # Log ~10 times per epoch

            for i in range(0, len(sequences), req.batch_size):
                batch_seqs = sequences[i:i + req.batch_size]
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
                loss = loss / req.grad_accum_steps
                loss.backward()

                total_loss += loss.item() * req.grad_accum_steps
                n_batches += 1

                if n_batches % req.grad_accum_steps == 0:
                    lr = get_lr(global_step, total_steps, req.warmup_steps, req.lr)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Progress logging
                if n_batches % log_interval == 0:
                    avg = total_loss / n_batches
                    pct = int(100 * n_batches / steps_per_epoch)
                    training_status["message"] = (
                        f"Epoch {epoch+1}/{req.epochs} | {pct}% | "
                        f"Loss: {avg:.4f} | Step: {global_step}"
                    )

            if n_batches % req.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            msg = f"Epoch {epoch+1}/{req.epochs} | Loss: {avg_loss:.4f}"
            training_status["log"].append(msg)
            training_status["message"] = msg

        # Save checkpoint
        model.eval()
        checkpoint = torch.load(CKPT_PATH, map_location="cpu")
        prev_log = checkpoint.get("training_log", [])
        new_log_entries = [{"epoch": len(prev_log) + i + 1, "loss": float(l.split("Loss: ")[1])}
                          for i, l in enumerate(training_status["log"]) if "Loss:" in l]

        new_checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "training_log": prev_log + new_log_entries,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "datasets": list(set(
                checkpoint.get("datasets", []) +
                [d["id"] for d in datasets_to_use] +
                ["range3/cc100-ja"]
            )),
        }
        torch.save(new_checkpoint, CKPT_PATH)
        training_status["log"].append(f"Checkpoint saved: {CKPT_PATH}")
        training_status["message"] = "Training complete!"
        training_status["running"] = False

    except Exception as e:
        import traceback
        training_status["running"] = False
        training_status["message"] = f"Error: {e}"
        training_status["log"].append(traceback.format_exc())
        if model is not None:
            model.eval()


# ========================================
# API Endpoints
# ========================================

@app.on_event("startup")
async def startup():
    load_model()


@app.get("/")
async def root():
    return {
        "name": "NeuroQuantum API",
        "version": "1.0.0",
        "model": "neuroquantum",
        "config": config,
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
    }


@app.post("/inference", response_model=InferenceResponse)
async def inference(req: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if training_status["running"]:
        raise HTTPException(status_code=503, detail="Model is currently training")

    text = generate_text(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    tokens_count = len(tokenizer.encode(text, add_special=False)) if text else 0
    return InferenceResponse(
        prompt=req.prompt,
        generated_text=text,
        tokens_generated=tokens_count,
    )


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if training_status["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(run_training, req)
    return TrainResponse(
        status="started",
        message=f"Training started: {req.epochs} epochs, lr={req.lr}, "
                f"effective_batch={req.batch_size * req.grad_accum_steps}",
    )


@app.get("/train/status", response_model=TrainStatusResponse)
async def train_status():
    return TrainStatusResponse(**training_status)


@app.post("/reload")
async def reload_model():
    """Reload model from latest checkpoint."""
    try:
        load_model()
        return {"status": "reloaded", "message": "Model reloaded from checkpoint"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
