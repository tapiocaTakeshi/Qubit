"""
Custom Inference Handler for HuggingFace Inference Endpoints
neuroQ - QBNN-Transformer (Improved Architecture)

Implements the EndpointHandler class for deploying the QBNN model
as a HuggingFace Inference Endpoint with custom pre/post-processing.

改善点:
- SentencePieceトークナイザー（語彙32K）による意味単位の学習
- max_seq_len=512 による文脈保持
- embed_dim=256, num_layers=6 による表現力の向上
- neuroquantum_layered.py のNeuroQuantumアーキテクチャを使用

Reference: https://huggingface.co/docs/inference-endpoints/main/en/engines/toolkit#create-a-custom-inference-handler
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
from datetime import datetime, timezone

# ============================================================
# Import NeuroQuantum architecture from neuroquantum_layered.py
# ============================================================
try:
    from neuroquantum_layered import (
        NeuroQuantum,
        NeuroQuantumConfig,
        NeuroQuantumTokenizer,
    )
    NEUROQUANTUM_AVAILABLE = True
except ImportError:
    NEUROQUANTUM_AVAILABLE = False


# ============================================================
# Legacy QBNN Layer (kept for backward compatibility with old checkpoints)
# ============================================================

class QBNNLayer(nn.Module):
    def __init__(self, dim, lam=0.12):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.J = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.lam = lam
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.linear(x)
        delta = torch.einsum('bsd,od->bso', torch.tanh(x), self.J) * torch.tanh(h)
        return self.norm(F.gelu(h + self.lam * delta))


class QBNNTransformer(nn.Module):
    """Legacy small model for backward compatibility with old checkpoints."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        self.embed = nn.Embedding(cfg["vocab_size"], d)
        self.pos_embed = nn.Embedding(cfg["max_seq_len"], d)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d, cfg["num_heads"], batch_first=True),
                "qbnn": QBNNLayer(d, cfg["entangle_strength"]),
                "norm1": nn.LayerNorm(d),
                "norm2": nn.LayerNorm(d),
            }) for _ in range(cfg["num_layers"])
        ])
        self.head = nn.Linear(d, cfg["vocab_size"])
        self.cfg = cfg

    def forward(self, input_ids=None, labels=None, x=None, **kwargs):
        if input_ids is None:
            input_ids = x
        if input_ids is None and "inputs" in kwargs:
            input_ids = kwargs["inputs"]

        B, S = input_ids.shape
        h = self.embed(input_ids) + self.pos_embed(torch.arange(S, device=input_ids.device).unsqueeze(0))
        for layer in self.layers:
            a, _ = layer["attn"](h, h, h)
            h = layer["norm1"](h + a)
            h = layer["norm2"](h + layer["qbnn"](h))
        logits = self.head(h)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            from transformers.modeling_outputs import CausalLMOutput
            return CausalLMOutput(loss=loss, logits=logits)

        return logits

    def generate(self, tokens, max_new=30, temperature=0.8, top_k=0, top_p=1.0,
                 repetition_penalty=1.0):
        self.eval()
        generated = []
        with torch.no_grad():
            for _ in range(max_new):
                seq = tokens[:, -self.cfg["max_seq_len"]:]
                logits = self(seq)[:, -1, :] / max(temperature, 1e-5)

                if generated and repetition_penalty > 1.0:
                    for prev_token in set(generated[-20:]):
                        logits[0, prev_token] /= repetition_penalty

                if top_k > 0:
                    topk_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0]
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    to_remove = cumulative_probs > top_p
                    to_remove[:, 1:] = to_remove[:, :-1].clone()
                    to_remove[:, 0] = False
                    indices_to_remove = sorted_indices[to_remove]
                    logits[0, indices_to_remove] = float('-inf')

                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                tokens = torch.cat([tokens, nxt], dim=1)
                generated.append(nxt.item())
                if nxt.item() == 1:  # <EOS>
                    break
        return tokens


# ============================================================
# Legacy Character-level Tokenizer (for old checkpoints)
# ============================================================

class CharTokenizer:
    def __init__(self):
        chars = list(
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
            "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "、。？！「」（）・ー\n "
        )
        self.stoi = {"<PAD>": 0, "<EOS>": 1, "<UNK>": 2}
        for i, c in enumerate(chars):
            self.stoi[c] = i + 3
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text, max_len=32):
        return [self.stoi.get(c, 2) for c in str(text)[:max_len]]

    def decode(self, ids):
        return "".join(self.itos.get(i, "?") for i in ids if i not in (0, 1, 2))


# ============================================================
# Default model config (matches app.py)
# ============================================================

DEFAULT_CONFIG = {
    "vocab_size": 200,
    "embed_dim": 64,
    "num_heads": 2,
    "num_layers": 2,
    "max_seq_len": 32,
    "entangle_strength": 0.12,
}

DEFAULT_CONFIG = {
    "vocab_size": 32000,
    "embed_dim": 512,
    "hidden_dim": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 512,
    "entangle_strength": 0.5,
    "dropout": 0.1,
    "architecture": "neuroquantum",  # "neuroquantum" or "legacy"
}


# ============================================================
# RunPod Serverless Handler
# ============================================================

import runpod

def find_checkpoint(path: str):
    """Search for a checkpoint file in the model directory."""
    candidates = [
        os.path.join(path, "qbnn_checkpoint.pt"),
        os.path.join(path, "neuroq_checkpoint.pt"),
        os.path.join(path, "checkpoint.pt"),
        os.path.join(path, "model.pt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    # Also check for any .pt file
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith(".pt"):
                return os.path.join(path, fname)
    return None

# Use global variables for model, tokenizer, and config to persist across requests
tokenizer = CharTokenizer()

# Determine config - try loading from checkpoint first
config = dict(DEFAULT_CONFIG)
config["vocab_size"] = tokenizer.vocab_size

# Look for checkpoint file
model_path = os.environ.get("MODEL_PATH", ".")
ckpt_path = find_checkpoint(model_path)

if ckpt_path is not None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Use saved config if available
    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        config.update(saved_config)
        # Ensure vocab_size matches tokenizer
        config["vocab_size"] = tokenizer.vocab_size

    model = QBNNTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
else:
    # No checkpoint found - initialize with default weights
    model = QBNNTransformer(config)

model.eval()

def train(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    from transformers import Trainer, TrainingArguments
    from datasets import load_dataset
    import os
    import traceback
    from huggingface_hub import HfApi

    dataset_id = data.get("dataset_id")
    if not dataset_id:
        return [{"error": "dataset_id is required for training"}]
        
    text_column = data.get("text_column", "text")
    split = data.get("split", "train")
    max_samples = int(data.get("max_samples", 500))
    epochs = float(data.get("epochs", 3.0))
    lr = float(data.get("lr", 1e-3))
    
    try:
        ds = load_dataset(dataset_id, split=split, trust_remote_code=False)
    except Exception as e:
        return [{"error": f"Failed to load dataset: {str(e)}", "traceback": traceback.format_exc()}]
        
    # Extract texts
    texts = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        col_data = row.get(text_column)
        if isinstance(col_data, str) and len(col_data.strip()) > 4:
            texts.append(col_data.strip())
            
    if not texts:
        return [{"error": "No valid text found in dataset"}]
        
    # Prepare Dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_seq_len):
            self.data = []
            for t in texts:
                ids = tokenizer.encode(t, max_len=max_seq_len)
                if len(ids) >= 2:
                    self.data.append(ids)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            ids = self.data[idx]
            return {"input_ids": ids, "labels": ids.copy()}

    train_dataset = CustomDataset(texts, tokenizer, config["max_seq_len"])
    
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        labels = []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            ids = x["input_ids"] + [0] * pad_len # 0 is PAD
            lbl = x["labels"] + [-100] * pad_len
            input_ids.append(ids)
            labels.append(lbl)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    training_args = TrainingArguments(
        output_dir="/tmp/qbnn_training",
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=10,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
    
    try:
        trainer.train()
    except Exception as e:
        return [{"error": f"Training failed: {e}", "traceback": traceback.format_exc()}]
        
    # Save model
    ckpt_path_out = "qbnn_checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(), 
        "config": config
    }, ckpt_path_out)
    
    # Upload if HF_TOKEN is available
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            repo_id = os.environ.get("REPOSITORY_ID", "tapiocaTakeshi/Qubit")
            HfApi(token=token).upload_file(
                path_or_fileobj=ckpt_path_out,
                path_in_repo=ckpt_path_out,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update model checkpoint via API training"
            )
        except Exception as e:
            return [{"status": "success", "message": f"Trained successfully, but upload failed: {e}"}]
            
    return [{"status": "success", "message": f"Trained on {len(texts)} samples for {epochs} epochs."}]


def handler(job):
    """
    Handle an inference or training request for RunPod serverless.
    
    Job format:
        {
            "input": {
                "inputs": "your prompt text",
                "parameters": { ... }
            }
        }
    """
    job_input = job.get("input", {})
    
    # Check if this is a training request
    if job_input.get("action") == "train":
        return train(job_input)

    inputs = job_input.get("inputs", job_input.get("prompt", ""))
    if isinstance(inputs, list):
        inputs = inputs[0] if inputs else ""
    inputs = str(inputs)

    # Extract generation parameters
    params = job_input.get("parameters", {})
    temperature = float(params.get("temperature", 0.8))
    max_new_tokens = int(params.get("max_new_tokens", 30))
    top_k = int(params.get("top_k", 0))
    top_p = float(params.get("top_p", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))

    # Encode input
    ids = tokenizer.encode(inputs, max_len=config["max_seq_len"])
    if not ids:
        return [{"generated_text": ""}]

    input_tensor = torch.tensor([ids], dtype=torch.long)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    # Decode only the newly generated tokens
    generated_ids = output[0, len(ids):].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return [{"generated_text": generated_text}]

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
