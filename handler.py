"""
Custom Inference Handler for HuggingFace Inference Endpoints
neuroQ - QBNN-Transformer

Implements the EndpointHandler class for deploying the QBNN model
as a HuggingFace Inference Endpoint with custom pre/post-processing.

Reference: https://huggingface.co/docs/inference-endpoints/main/en/engines/toolkit#create-a-custom-inference-handler
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any


# ============================================================
# QBNN Layer (same architecture as app.py)
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

    def forward(self, x):
        B, S = x.shape
        h = self.embed(x) + self.pos_embed(torch.arange(S, device=x.device).unsqueeze(0))
        for layer in self.layers:
            a, _ = layer["attn"](h, h, h)
            h = layer["norm1"](h + a)
            h = layer["norm2"](h + layer["qbnn"](h))
        return self.head(h)

    def generate(self, tokens, max_new=30, temperature=0.8, top_k=0, top_p=1.0,
                 repetition_penalty=1.0):
        self.eval()
        generated = []
        with torch.no_grad():
            for _ in range(max_new):
                seq = tokens[:, -self.cfg["max_seq_len"]:]
                logits = self(seq)[:, -1, :] / max(temperature, 1e-5)

                # Repetition penalty
                if generated and repetition_penalty > 1.0:
                    for prev_token in set(generated[-20:]):
                        logits[0, prev_token] /= repetition_penalty

                # Top-K filtering
                if top_k > 0:
                    topk_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0]
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                # Top-P (nucleus) filtering
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
# Character-level Tokenizer (same as app.py)
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


# ============================================================
# EndpointHandler
# ============================================================

class EndpointHandler:
    """
    Custom handler for HuggingFace Inference Endpoints.

    Loads the QBNN-Transformer checkpoint and serves text generation requests.

    Request format:
        {
            "inputs": "your prompt text",
            "parameters": {              # all optional
                "temperature": 0.8,
                "max_new_tokens": 30,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }
        }

    Response format:
        [{"generated_text": "..."}]
    """

    def __init__(self, path: str = ""):
        """
        Initialize the handler by loading the model from the given path.

        Args:
            path: Path to model weights directory (provided by Inference Endpoints).
        """
        self.tokenizer = CharTokenizer()

        # Determine config - try loading from checkpoint first
        self.config = dict(DEFAULT_CONFIG)
        self.config["vocab_size"] = self.tokenizer.vocab_size

        # Look for checkpoint file
        ckpt_path = self._find_checkpoint(path)

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # Use saved config if available
            if "config" in checkpoint:
                saved_config = checkpoint["config"]
                self.config.update(saved_config)
                # Ensure vocab_size matches tokenizer
                self.config["vocab_size"] = self.tokenizer.vocab_size

            self.model = QBNNTransformer(self.config)
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            # No checkpoint found - initialize with default weights
            self.model = QBNNTransformer(self.config)

        self.model.eval()

    def _find_checkpoint(self, path: str):
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

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle an inference request.

        Args:
            data: Request body with 'inputs' key and optional 'parameters'.

        Returns:
            List of dicts with 'generated_text' key.
        """
        inputs = data.get("inputs", data)
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
