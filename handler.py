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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any


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
# Default model config
# ============================================================

DEFAULT_CONFIG = {
    "vocab_size": 32000,
    "embed_dim": 512,
    "hidden_dim": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 512,
    "entangle_strength": 0.5,
    "dropout": 0.1,
    "architecture": "neuroquantum",
}


# ============================================================
# Utility
# ============================================================

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


# ============================================================
# HuggingFace Inference Endpoints Handler
# ============================================================

class EndpointHandler:
    """
    Custom handler for HuggingFace Inference Endpoints.

    HF Inference Endpoints will:
      1. Download the model repo to a local directory
      2. Instantiate this class with __init__(path=<model_dir>)
      3. Call __call__(data) for each inference request
    """

    def __init__(self, path: str = ""):
        """Load the model, tokenizer, and config from the model directory."""

        # Initialize tokenizer
        self.tokenizer = CharTokenizer()

        # Build config
        self.config = dict(DEFAULT_CONFIG)
        self.config["vocab_size"] = self.tokenizer.vocab_size

        # Look for checkpoint
        ckpt_path = find_checkpoint(path) if path else None

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # Use saved config if available
            if "config" in checkpoint:
                self.config.update(checkpoint["config"])
                self.config["vocab_size"] = self.tokenizer.vocab_size

            self.model = QBNNTransformer(self.config)
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            # No checkpoint found - initialize with default weights
            self.model = QBNNTransformer(self.config)

        self.model.eval()

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle an inference request.

        Args:
            data: Dictionary with:
                - "inputs" (str): The input prompt text
                - "parameters" (dict, optional): Generation parameters
                    - temperature (float): default 0.8
                    - max_new_tokens (int): default 30
                    - top_k (int): default 0
                    - top_p (float): default 1.0
                    - repetition_penalty (float): default 1.0

        Returns:
            List of dicts with "generated_text" key.
        """
        # Extract input text
        inputs = data.get("inputs", data.get("prompt", ""))
        if isinstance(inputs, list):
            inputs = inputs[0] if inputs else ""
        inputs = str(inputs)

        # Extract generation parameters
        params = data.get("parameters", {})
        temperature = float(params.get("temperature", 0.8))
        max_new_tokens = int(params.get("max_new_tokens", 30))
        top_k = int(params.get("top_k", 0))
        top_p = float(params.get("top_p", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))

        # Encode input
        ids = self.tokenizer.encode(inputs, max_len=self.config["max_seq_len"])
        if not ids:
            return [{"generated_text": ""}]

        input_tensor = torch.tensor([ids], dtype=torch.long)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_new=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # Decode only the newly generated tokens
        generated_ids = output[0, len(ids):].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return [{"generated_text": generated_text}]
