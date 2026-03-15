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
# NeuroQuantum wrapper for HuggingFace Trainer API compatibility
# ============================================================

class NeuroQTrainerWrapper(nn.Module):
    """Wraps NeuroQuantum model to support HuggingFace Trainer API (input_ids/labels)."""

    def __init__(self, model: 'NeuroQuantum', vocab_size: int):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            from transformers.modeling_outputs import CausalLMOutput
            return CausalLMOutput(loss=loss, logits=logits)

        return logits

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def train(self, mode=True):
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        self.model.eval()
        return super().eval()


# ============================================================
# Default model config — Improved architecture
# ============================================================

LEGACY_CONFIG = {
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
# EndpointHandler
# ============================================================

class EndpointHandler:
    """
    Custom handler for HuggingFace Inference Endpoints.

    Loads the QBNN-Transformer checkpoint and serves text generation requests.
    Uses the improved NeuroQuantum architecture with SentencePiece tokenizer.

    Request format:
        {
            "inputs": "your prompt text",
            "parameters": {              # all optional
                "temperature": 0.8,
                "max_new_tokens": 100,
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
        self._init_log = []
        self._init_log.append(f"path={path}")
        # Look for checkpoint file first to determine architecture
        ckpt_path = self._find_checkpoint(path)
        self._init_log.append(f"ckpt_path={ckpt_path}")
        checkpoint = None
        saved_config = None

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            saved_config = checkpoint.get("config", {})

        # Determine if this is a legacy checkpoint
        is_legacy = False
        if saved_config:
            is_legacy = saved_config.get("architecture") != "neuroquantum"
            # Also detect legacy by small parameter values
            if saved_config.get("embed_dim", 256) <= 64 and saved_config.get("max_seq_len", 512) <= 32:
                is_legacy = True

        self._init_log.append(f"is_legacy={is_legacy}, NEUROQUANTUM_AVAILABLE={NEUROQUANTUM_AVAILABLE}")
        self._init_log.append(f"saved_config={saved_config}")

        if is_legacy or not NEUROQUANTUM_AVAILABLE:
            # Legacy mode: use old CharTokenizer + QBNNTransformer
            self._init_legacy(checkpoint, saved_config)
            self._init_log.append("mode=legacy")
        else:
            # New mode: use NeuroQuantum + SentencePiece
            self._init_neuroquantum(path, checkpoint, saved_config)
            self._init_log.append("mode=neuroquantum")

    def _init_legacy(self, checkpoint, saved_config):
        """Initialize with legacy architecture for old checkpoints."""
        self.architecture = "legacy"
        self.tokenizer = CharTokenizer()
        self.config = dict(LEGACY_CONFIG)
        self.config["vocab_size"] = self.tokenizer.vocab_size

        if saved_config:
            self.config.update(saved_config)
            self.config["vocab_size"] = self.tokenizer.vocab_size

        self.model = QBNNTransformer(self.config)
        if checkpoint and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.neuroq_model = None

    def _init_neuroquantum(self, path, checkpoint, saved_config):
        """Initialize with improved NeuroQuantum architecture."""
        self.architecture = "neuroquantum"
        self.config = dict(DEFAULT_CONFIG)

        # Load or create SentencePiece tokenizer
        tokenizer_path = self._find_tokenizer(path)
        if tokenizer_path:
            self.tokenizer = NeuroQuantumTokenizer(
                vocab_size=self.config["vocab_size"],
                model_file=tokenizer_path
            )
        else:
            # No trained tokenizer found — create one with fallback vocab
            self.tokenizer = NeuroQuantumTokenizer(vocab_size=self.config["vocab_size"])

        # Update vocab_size from tokenizer
        actual_vocab = self.tokenizer.actual_vocab_size or self.tokenizer.vocab_size
        self.config["vocab_size"] = actual_vocab

        if saved_config and saved_config.get("architecture") == "neuroquantum":
            # Use saved config but keep tokenizer's vocab_size
            saved_vocab = actual_vocab
            self.config.update(saved_config)
            self.config["vocab_size"] = saved_vocab

        # Build NeuroQuantum model
        nq_config = NeuroQuantumConfig(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config.get("hidden_dim", self.config["embed_dim"] * 2),
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            max_seq_len=self.config["max_seq_len"],
            dropout=self.config.get("dropout", 0.1),
            lambda_entangle=self.config.get("entangle_strength", 0.5),
        )
        self.neuroq_model = NeuroQuantum(config=nq_config)

        if checkpoint and "model_state" in checkpoint:
            try:
                self.neuroq_model.load_state_dict(checkpoint["model_state"])
                self._init_log.append("state_dict loaded OK")
            except RuntimeError as e:
                self._init_log.append(f"state_dict FAILED: {e}")
                # State dict mismatch — reinitialize
                self.neuroq_model = NeuroQuantum(config=nq_config)
        else:
            self._init_log.append(f"no checkpoint to load: checkpoint={checkpoint is not None}, has_model_state={'model_state' in checkpoint if checkpoint else False}")

        self.neuroq_model.eval()
        # For backward-compatible interface
        self.model = self.neuroq_model

    def _find_tokenizer(self, path: str):
        """Search for a SentencePiece tokenizer model file."""
        candidates = [
            os.path.join(path, "neuroq_tokenizer.model"),
            os.path.join(path, "neuroq_tokenizer_16k.model"),
            os.path.join(path, "neuroq_tokenizer_8k.model"),
            os.path.join(path, "neuroq_tokenizer.json"),
            # Also check script directory
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model"),
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer_16k.model"),
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer_8k.model"),
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.json"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _find_checkpoint(self, path: str):
        """Search for a checkpoint file in the model directory."""
        candidates = [
            os.path.join(path, "neuroq_checkpoint.pt"),
            os.path.join(path, "qbnn_checkpoint.pt"),
            os.path.join(path, "checkpoint.pt"),
            os.path.join(path, "model.pt"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.endswith(".pt"):
                    return os.path.join(path, fname)
        return None

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Handle an inference or training request.

        Args:
            data: Request body with 'inputs' key and optional 'parameters'.
                  Alternatively, for training: {"action": "train", "dataset_id": "...", ...}

        Returns:
            List of dicts with 'generated_text' or training status.
        """
        # Debug: return init log
        if data.get("action") == "debug" or data.get("inputs") == "__debug__":
            return [{"init_log": self._init_log, "architecture": self.architecture,
                     "config": self.config, "model_params": sum(p.numel() for p in self.model.parameters()),
                     "handler_version": "v3_2026_03_12"}]

        # Support training via both top-level and parameters.action
        if data.get("action") == "train":
            return self.train(data)
        params = data.get("parameters", {})
        if params.get("action") == "train":
            # Merge parameters into top-level for train()
            train_data = dict(params)
            return self.train(train_data)

        # Batch training actions
        action = data.get("action") or params.get("action", "")
        if action == "batch_train":
            bt_data = dict(data)
            bt_data.update(params)
            return self.batch_train(bt_data)
        if action == "batch_train_status":
            bt_data = dict(data)
            bt_data.update(params)
            return self.batch_train_status(bt_data)

        # Federated learning actions
        if action.startswith("federated_"):
            fed_data = dict(data)
            if params.get("action"):
                fed_data.update(params)
            if action == "federated_init":
                return self.federated_init(fed_data)
            elif action == "federated_submit":
                return self.federated_submit(fed_data)
            elif action == "federated_aggregate":
                return self.federated_aggregate(fed_data)
            elif action == "federated_status":
                return self.federated_status(fed_data)
            else:
                return [{"error": f"Unknown federated action: {action}"}]

        inputs = data.get("inputs", data)
        if isinstance(inputs, list):
            inputs = inputs[0] if inputs else ""
        inputs = str(inputs)
        temperature = float(params.get("temperature", 0.8))
        max_new_tokens = int(params.get("max_new_tokens", 100))
        top_k = int(params.get("top_k", 40))
        top_p = float(params.get("top_p", 0.9))
        repetition_penalty = float(params.get("repetition_penalty", 1.2))

        if self.architecture == "neuroquantum":
            return self._generate_neuroquantum(
                inputs, temperature, max_new_tokens, top_k, top_p, repetition_penalty
            )
        else:
            return self._generate_legacy(
                inputs, temperature, max_new_tokens, top_k, top_p, repetition_penalty
            )

    def _generate_neuroquantum(self, text, temperature, max_new_tokens,
                                top_k, top_p, repetition_penalty):
        """Generate text using the NeuroQuantum architecture."""
        tokens = self.tokenizer.encode(text, add_special=True)
        if not tokens:
            return [{"generated_text": ""}]

        device = next(self.neuroq_model.parameters()).device
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        self.neuroq_model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                seq = input_tensor[:, -self.config["max_seq_len"]:]
                logits = self.neuroq_model(seq)[:, -1, :] / max(temperature, 1e-5)

                # Repetition penalty
                if len(generated) > 1 and repetition_penalty > 1.0:
                    window = generated[-min(50, len(generated)):]
                    for prev_token in set(window):
                        if prev_token < logits.size(-1):
                            logits[0, prev_token] /= repetition_penalty

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

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()

                # EOS check
                if nxt_id == self.tokenizer.eos_id:
                    break
                # Skip PAD
                if nxt_id == self.tokenizer.pad_id:
                    continue

                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)

        # Decode only generated tokens (skip input)
        generated_ids = generated[len(tokens):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special=True)

        return [{"generated_text": generated_text}]

    def _generate_legacy(self, text, temperature, max_new_tokens,
                          top_k, top_p, repetition_penalty):
        """Generate text using the legacy QBNNTransformer architecture."""
        ids = self.tokenizer.encode(text, max_len=self.config["max_seq_len"])
        if not ids:
            return [{"generated_text": ""}]

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_new=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        generated_ids = output[0, len(ids):].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return [{"generated_text": generated_text}]

    def train(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from transformers import Trainer, TrainingArguments, TrainerCallback
        from datasets import load_dataset
        import json
        import traceback
        from datetime import datetime, timezone
        from huggingface_hub import HfApi

        dataset_id = data.get("dataset_id")
        if not dataset_id:
            return [{"error": "dataset_id is required for training"}]

        text_column = data.get("text_column", "text")
        split = data.get("split", "train")
        max_samples = int(data.get("max_samples", 3000))
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
            elif isinstance(col_data, list):
                parts = []
                for turn in col_data:
                    if isinstance(turn, dict) and "value" in turn:
                        parts.append(turn["value"])
                    elif isinstance(turn, str):
                        parts.append(turn)
                combined = "\n".join(parts)
                if len(combined.strip()) > 4:
                    texts.append(combined.strip())

        if not texts:
            return [{"error": "No valid text found in dataset"}]

        # Always use neuroquantum architecture for training if available
        # This handles upgrade from legacy to neuroquantum
        if NEUROQUANTUM_AVAILABLE:
            if self.architecture != "neuroquantum":
                # Upgrade from legacy to neuroquantum
                self.config = dict(DEFAULT_CONFIG)
                self.architecture = "neuroquantum"
            return self._train_neuroquantum(data, texts, dataset_id, text_column,
                                            epochs, lr, max_samples)
        else:
            return self._train_legacy(data, texts, dataset_id, text_column,
                                      epochs, lr)

    def _train_neuroquantum(self, data, texts, dataset_id, text_column,
                             epochs, lr, max_samples):
        """Train using the improved NeuroQuantum architecture."""
        from transformers import Trainer, TrainingArguments, TrainerCallback
        import json
        import traceback
        from datetime import datetime, timezone
        from huggingface_hub import HfApi

        max_seq_len = self.config["max_seq_len"]

        # Build SentencePiece tokenizer if not already trained
        if self.tokenizer.sp is None:
            self.tokenizer = NeuroQuantumTokenizer(vocab_size=self.config["vocab_size"])
            self.tokenizer.build_vocab(
                texts,
                model_prefix=os.path.join(os.path.dirname(__file__), "neuroq_tokenizer"),
                character_coverage=0.9995,
            )
            # Rebuild model with actual vocab size
            actual_vocab = self.tokenizer.actual_vocab_size or self.tokenizer.vocab_size
            self.config["vocab_size"] = actual_vocab
            nq_config = NeuroQuantumConfig(
                vocab_size=actual_vocab,
                embed_dim=self.config["embed_dim"],
                hidden_dim=self.config.get("hidden_dim", self.config["embed_dim"] * 2),
                num_heads=self.config["num_heads"],
                num_layers=self.config["num_layers"],
                max_seq_len=self.config["max_seq_len"],
                dropout=self.config.get("dropout", 0.1),
                lambda_entangle=self.config.get("entangle_strength", 0.5),
            )
            self.neuroq_model = NeuroQuantum(config=nq_config)
            self.model = self.neuroq_model

        # Tokenize texts into sequences
        class SeqDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, max_seq_len):
                self.data = []
                for t in texts:
                    # Encode without special tokens, wrap each chunk with BOS/EOS
                    content_ids = tokenizer.encode(t, add_special=False)
                    max_content = max_seq_len - 2  # Reserve for BOS and EOS
                    if max_content <= 0:
                        continue
                    if len(content_ids) <= max_content:
                        if len(content_ids) >= 2:
                            seq = [tokenizer.bos_id] + content_ids + [tokenizer.eos_id]
                            self.data.append(seq)
                    else:
                        stride = max(max_content // 2, 1)
                        for start in range(0, len(content_ids) - max_content + 1, stride):
                            chunk = content_ids[start:start + max_content]
                            seq = [tokenizer.bos_id] + chunk + [tokenizer.eos_id]
                            self.data.append(seq)
                        remaining = content_ids[-max_content:]
                        tail_seq = [tokenizer.bos_id] + remaining + [tokenizer.eos_id]
                        if tail_seq != self.data[-1]:
                            self.data.append(tail_seq)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ids = self.data[idx]
                return {"input_ids": ids, "labels": ids.copy()}

        train_dataset = SeqDataset(texts, self.tokenizer, max_seq_len)

        def collate_fn(batch):
            max_len = min(max(len(x["input_ids"]) for x in batch), max_seq_len)
            input_ids = []
            labels = []
            for x in batch:
                ids = x["input_ids"][:max_len]
                lbl = x["labels"][:max_len]
                pad_len = max_len - len(ids)
                ids = ids + [self.tokenizer.pad_id] * pad_len
                lbl = lbl + [-100] * pad_len
                input_ids.append(ids)
                labels.append(lbl)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        class LossCallback(TrainerCallback):
            def __init__(self):
                self.logs = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    self.logs.append({
                        "step": state.global_step,
                        "loss": round(logs["loss"], 6),
                        "epoch": round(logs.get("epoch", 0), 2)
                    })

        loss_callback = LossCallback()

        # Wrap model for Trainer compatibility
        wrapper = NeuroQTrainerWrapper(self.neuroq_model, self.config["vocab_size"])

        training_args = TrainingArguments(
            output_dir="/tmp/neuroq_training",
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_strategy="no",
            logging_steps=10,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=wrapper,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            callbacks=[loss_callback]
        )

        try:
            train_result = trainer.train()
        except Exception as e:
            import traceback
            return [{"error": f"Training failed: {e}", "traceback": traceback.format_exc()}]
        finally:
            self.neuroq_model.eval()

        final_loss = round(train_result.training_loss, 6) if train_result.training_loss else None

        # Build training history entry
        history_entry = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "epochs": int(epochs) if epochs == int(epochs) else epochs,
            "samples": len(train_dataset),
            "final_loss": final_loss,
            "learning_rate": lr,
            "text_column": text_column,
            "architecture": "neuroquantum",
            "embed_dim": self.config["embed_dim"],
            "num_layers": self.config["num_layers"],
            "max_seq_len": self.config["max_seq_len"],
            "vocab_size": self.config["vocab_size"],
        }

        # Update training_history.json
        history_path = os.path.join(os.path.dirname(__file__), "training_history.json")
        try:
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(history_entry)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception:
            history = [history_entry]

        # Save model checkpoint with architecture tag
        ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
        torch.save({
            "model_state": self.neuroq_model.state_dict(),
            "config": self.config
        }, ckpt_path)

        # Save tokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer")
        self.tokenizer.save(tokenizer_path)

        # Upload if HF_TOKEN available
        token = os.environ.get("HF_TOKEN")
        upload_status = "skipped (no HF_TOKEN)"
        if token:
            try:
                repo_id = os.environ.get("REPOSITORY_ID", "tapiocaTakeshi/Qubit")
                api = HfApi(token=token)
                api.upload_file(
                    path_or_fileobj=ckpt_path,
                    path_in_repo="neuroq_checkpoint.pt",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update neuroQ checkpoint after training on {dataset_id}"
                )
                api.upload_file(
                    path_or_fileobj=history_path,
                    path_in_repo="training_history.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update training history after {dataset_id}"
                )
                upload_status = "uploaded"
            except Exception as e:
                upload_status = f"upload failed: {e}"

        return [{
            "status": "success",
            "architecture": "neuroquantum",
            "dataset_id": dataset_id,
            "samples_trained": len(train_dataset),
            "epochs": epochs,
            "final_loss": final_loss,
            "config": {
                "vocab_size": self.config["vocab_size"],
                "embed_dim": self.config["embed_dim"],
                "num_layers": self.config["num_layers"],
                "max_seq_len": self.config["max_seq_len"],
            },
            "loss_log": loss_callback.logs,
            "upload": upload_status,
            "message": f"Trained on {len(train_dataset)} samples for {epochs} epochs. Final loss: {final_loss}"
        }]

    def _train_legacy(self, data, texts, dataset_id, text_column, epochs, lr):
        """Train using the legacy QBNNTransformer architecture."""
        from transformers import Trainer, TrainingArguments, TrainerCallback
        import json
        import traceback
        from datetime import datetime, timezone
        from huggingface_hub import HfApi

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

        train_dataset = CustomDataset(texts, self.tokenizer, self.config["max_seq_len"])

        def collate_fn(batch):
            max_len = max(len(x["input_ids"]) for x in batch)
            input_ids = []
            labels = []
            for x in batch:
                pad_len = max_len - len(x["input_ids"])
                ids = x["input_ids"] + [0] * pad_len
                lbl = x["labels"] + [-100] * pad_len
                input_ids.append(ids)
                labels.append(lbl)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        class LossCallback(TrainerCallback):
            def __init__(self):
                self.logs = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    self.logs.append({
                        "step": state.global_step,
                        "loss": round(logs["loss"], 6),
                        "epoch": round(logs.get("epoch", 0), 2)
                    })

        loss_callback = LossCallback()

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
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            callbacks=[loss_callback]
        )

        try:
            train_result = trainer.train()
        except Exception as e:
            return [{"error": f"Training failed: {e}", "traceback": traceback.format_exc()}]
        finally:
            self.model.eval()

        final_loss = round(train_result.training_loss, 6) if train_result.training_loss else None

        history_entry = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "epochs": int(epochs) if epochs == int(epochs) else epochs,
            "samples": len(train_dataset),
            "final_loss": final_loss,
            "learning_rate": lr,
            "text_column": text_column
        }

        history_path = os.path.join(os.path.dirname(__file__), "training_history.json")
        try:
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(history_entry)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception:
            history = [history_entry]

        ckpt_path = os.path.join(os.path.dirname(__file__), "qbnn_checkpoint.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config
        }, ckpt_path)

        token = os.environ.get("HF_TOKEN")
        upload_status = "skipped (no HF_TOKEN)"
        if token:
            try:
                repo_id = os.environ.get("REPOSITORY_ID", "tapiocaTakeshi/Qubit")
                api = HfApi(token=token)
                api.upload_file(
                    path_or_fileobj=ckpt_path,
                    path_in_repo="qbnn_checkpoint.pt",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update checkpoint after training on {dataset_id}"
                )
                api.upload_file(
                    path_or_fileobj=history_path,
                    path_in_repo="training_history.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update training history after {dataset_id}"
                )
                upload_status = "uploaded"
            except Exception as e:
                upload_status = f"upload failed: {e}"

        return [{
            "status": "success",
            "dataset_id": dataset_id,
            "samples_trained": len(train_dataset),
            "epochs": epochs,
            "final_loss": final_loss,
            "loss_log": loss_callback.logs,
            "upload": upload_status,
            "message": f"Trained on {len(train_dataset)} samples for {epochs} epochs. Final loss: {final_loss}"
        }]

    # ============================================================
    # Batch Training Endpoint
    # ============================================================
    #
    # データセットを分割（バッチ）して学習するエンドポイント。
    # 大規模データセットをメモリに収まるサイズに分割し、
    # 各バッチごとに学習・チェックポイント保存を行う。
    #
    # Request example:
    #   {"action": "batch_train", "dataset_id": "izumi-lab/wikipedia-ja-20230720",
    #    "batch_size": 500, "epochs_per_batch": 1, "max_samples": 5000,
    #    "text_column": "text", "lr": 1e-3}
    #
    # Status check:
    #   {"action": "batch_train_status", "job_id": "..."}
    # ============================================================

    _batch_jobs: Dict[str, Dict[str, Any]] = {}

    def batch_train(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Train on a dataset split into sequential batches.
        Each batch loads a subset of the dataset, trains, saves checkpoint,
        then moves to the next batch. This keeps memory usage bounded.
        """
        from transformers import Trainer, TrainingArguments, TrainerCallback
        from datasets import load_dataset
        import json
        import traceback
        from datetime import datetime, timezone
        from huggingface_hub import HfApi
        import uuid

        dataset_id = data.get("dataset_id")
        if not dataset_id:
            return [{"error": "dataset_id is required for batch training"}]

        text_column = data.get("text_column", "text")
        split = data.get("split", "train")
        batch_size = int(data.get("batch_size", 500))
        max_samples = int(data.get("max_samples", 5000))
        epochs_per_batch = float(data.get("epochs_per_batch", 1.0))
        lr = float(data.get("lr", 1e-3))
        start_batch = int(data.get("start_batch", 0))

        # Generate job_id
        job_id = data.get("job_id") or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Load dataset
        try:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=False)
        except Exception as e:
            return [{"error": f"Failed to load dataset: {str(e)}", "traceback": traceback.format_exc()}]

        total_available = min(max_samples, len(ds))
        num_batches = (total_available + batch_size - 1) // batch_size

        # Initialize job tracking
        job_info = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": total_available,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "epochs_per_batch": epochs_per_batch,
            "lr": lr,
            "current_batch": start_batch,
            "batch_results": [],
        }
        EndpointHandler._batch_jobs[job_id] = job_info

        # Ensure neuroquantum architecture
        if NEUROQUANTUM_AVAILABLE and self.architecture != "neuroquantum":
            self.config = dict(DEFAULT_CONFIG)
            self.architecture = "neuroquantum"

        if not NEUROQUANTUM_AVAILABLE:
            return [{"error": "NeuroQuantum architecture not available for batch training"}]

        max_seq_len = self.config["max_seq_len"]

        # --- Inner classes (same as _train_neuroquantum) ---

        class SeqDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, max_seq_len):
                self.data = []
                for t in texts:
                    content_ids = tokenizer.encode(t, add_special=False)
                    max_content = max_seq_len - 2
                    if max_content <= 0:
                        continue
                    if len(content_ids) <= max_content:
                        if len(content_ids) >= 2:
                            seq = [tokenizer.bos_id] + content_ids + [tokenizer.eos_id]
                            self.data.append(seq)
                    else:
                        stride = max(max_content // 2, 1)
                        for start in range(0, len(content_ids) - max_content + 1, stride):
                            chunk = content_ids[start:start + max_content]
                            seq = [tokenizer.bos_id] + chunk + [tokenizer.eos_id]
                            self.data.append(seq)
                        remaining = content_ids[-max_content:]
                        tail_seq = [tokenizer.bos_id] + remaining + [tokenizer.eos_id]
                        if tail_seq != self.data[-1]:
                            self.data.append(tail_seq)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ids = self.data[idx]
                return {"input_ids": ids, "labels": ids.copy()}

        def collate_fn(batch):
            max_len = min(max(len(x["input_ids"]) for x in batch), max_seq_len)
            input_ids = []
            labels = []
            for x in batch:
                ids = x["input_ids"][:max_len]
                lbl = x["labels"][:max_len]
                pad_len = max_len - len(ids)
                ids = ids + [self.tokenizer.pad_id] * pad_len
                lbl = lbl + [-100] * pad_len
                input_ids.append(ids)
                labels.append(lbl)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        class LossCallback(TrainerCallback):
            def __init__(self):
                self.logs = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    self.logs.append({
                        "step": state.global_step,
                        "loss": round(logs["loss"], 6),
                        "epoch": round(logs.get("epoch", 0), 2)
                    })

        # Build tokenizer if needed
        if self.tokenizer.sp is None:
            # Gather sample texts for tokenizer training
            sample_texts = []
            for row in ds.select(range(min(2000, len(ds)))):
                col_data = row.get(text_column)
                if isinstance(col_data, str) and len(col_data.strip()) > 4:
                    sample_texts.append(col_data.strip())
            if not sample_texts:
                return [{"error": "No valid text found for tokenizer training"}]

            self.tokenizer = NeuroQuantumTokenizer(vocab_size=self.config["vocab_size"])
            self.tokenizer.build_vocab(
                sample_texts,
                model_prefix=os.path.join(os.path.dirname(__file__), "neuroq_tokenizer"),
                character_coverage=0.9995,
            )
            actual_vocab = self.tokenizer.actual_vocab_size or self.tokenizer.vocab_size
            self.config["vocab_size"] = actual_vocab
            nq_config = NeuroQuantumConfig(
                vocab_size=actual_vocab,
                embed_dim=self.config["embed_dim"],
                hidden_dim=self.config.get("hidden_dim", self.config["embed_dim"] * 2),
                num_heads=self.config["num_heads"],
                num_layers=self.config["num_layers"],
                max_seq_len=self.config["max_seq_len"],
                dropout=self.config.get("dropout", 0.1),
                lambda_entangle=self.config.get("entangle_strength", 0.5),
            )
            self.neuroq_model = NeuroQuantum(config=nq_config)
            self.model = self.neuroq_model

        # --- Batch training loop ---
        all_loss_logs = []
        overall_first_loss = None
        overall_last_loss = None

        for batch_idx in range(start_batch, num_batches):
            job_info["current_batch"] = batch_idx
            offset = batch_idx * batch_size
            end = min(offset + batch_size, total_available)

            # Extract texts for this batch
            texts = []
            for row in ds.select(range(offset, end)):
                col_data = row.get(text_column)
                if isinstance(col_data, str) and len(col_data.strip()) > 4:
                    texts.append(col_data.strip())
                elif isinstance(col_data, list):
                    parts = []
                    for turn in col_data:
                        if isinstance(turn, dict) and "value" in turn:
                            parts.append(turn["value"])
                        elif isinstance(turn, str):
                            parts.append(turn)
                    combined = "\n".join(parts)
                    if len(combined.strip()) > 4:
                        texts.append(combined.strip())

            if not texts:
                job_info["batch_results"].append({
                    "batch": batch_idx,
                    "status": "skipped",
                    "reason": "no valid texts in batch"
                })
                continue

            train_dataset = SeqDataset(texts, self.tokenizer, max_seq_len)
            if len(train_dataset) == 0:
                job_info["batch_results"].append({
                    "batch": batch_idx,
                    "status": "skipped",
                    "reason": "no sequences after tokenization"
                })
                continue

            loss_callback = LossCallback()
            wrapper = NeuroQTrainerWrapper(self.neuroq_model, self.config["vocab_size"])

            training_args = TrainingArguments(
                output_dir="/tmp/neuroq_batch_training",
                num_train_epochs=epochs_per_batch,
                learning_rate=lr,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                warmup_ratio=0.1,
                weight_decay=0.01,
                save_strategy="no",
                logging_steps=10,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )

            trainer = Trainer(
                model=wrapper,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=collate_fn,
                callbacks=[loss_callback]
            )

            try:
                train_result = trainer.train()
                batch_loss = round(train_result.training_loss, 6) if train_result.training_loss else None
            except Exception as e:
                job_info["batch_results"].append({
                    "batch": batch_idx,
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                job_info["status"] = "error"
                self.neuroq_model.eval()
                break
            finally:
                self.neuroq_model.eval()

            # Track loss
            if loss_callback.logs:
                all_loss_logs.extend([
                    {**log, "batch": batch_idx} for log in loss_callback.logs
                ])
                if overall_first_loss is None:
                    overall_first_loss = loss_callback.logs[0]["loss"]
                overall_last_loss = loss_callback.logs[-1]["loss"]

            batch_result = {
                "batch": batch_idx,
                "status": "completed",
                "samples_raw": len(texts),
                "sequences": len(train_dataset),
                "final_loss": batch_loss,
                "loss_log": loss_callback.logs,
            }
            job_info["batch_results"].append(batch_result)

            # Save checkpoint after each batch
            ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
            torch.save({
                "model_state": self.neuroq_model.state_dict(),
                "config": self.config
            }, ckpt_path)

        # --- All batches done ---
        job_info["status"] = "completed" if job_info["status"] == "running" else job_info["status"]
        job_info["completed_at"] = datetime.now(timezone.utc).isoformat()

        completed_batches = [b for b in job_info["batch_results"] if b.get("status") == "completed"]
        total_sequences = sum(b.get("sequences", 0) for b in completed_batches)
        final_loss = completed_batches[-1]["final_loss"] if completed_batches else None

        # Save training history
        history_entry = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "mode": "batch_train",
            "total_batches": num_batches,
            "completed_batches": len(completed_batches),
            "batch_size": batch_size,
            "epochs_per_batch": epochs_per_batch,
            "total_sequences": total_sequences,
            "first_loss": overall_first_loss,
            "final_loss": final_loss,
            "learning_rate": lr,
            "text_column": text_column,
            "architecture": "neuroquantum",
            "embed_dim": self.config["embed_dim"],
            "num_layers": self.config["num_layers"],
            "max_seq_len": self.config["max_seq_len"],
            "vocab_size": self.config["vocab_size"],
        }

        history_path = os.path.join(os.path.dirname(__file__), "training_history.json")
        try:
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(history_entry)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Save tokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer")
        self.tokenizer.save(tokenizer_path)

        # Upload if HF_TOKEN available
        token = os.environ.get("HF_TOKEN")
        upload_status = "skipped (no HF_TOKEN)"
        if token:
            try:
                repo_id = os.environ.get("REPOSITORY_ID", "tapiocaTakeshi/Qubit")
                api = HfApi(token=token)
                ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
                api.upload_file(
                    path_or_fileobj=ckpt_path,
                    path_in_repo="neuroq_checkpoint.pt",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Batch training on {dataset_id}: {len(completed_batches)}/{num_batches} batches"
                )
                api.upload_file(
                    path_or_fileobj=history_path,
                    path_in_repo="training_history.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update training history after batch training on {dataset_id}"
                )
                upload_status = "uploaded"
            except Exception as e:
                upload_status = f"upload failed: {e}"

        return [{
            "status": job_info["status"],
            "job_id": job_id,
            "architecture": "neuroquantum",
            "dataset_id": dataset_id,
            "total_samples": total_available,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "completed_batches": len(completed_batches),
            "epochs_per_batch": epochs_per_batch,
            "total_sequences_trained": total_sequences,
            "first_loss": overall_first_loss,
            "final_loss": final_loss,
            "batch_results": job_info["batch_results"],
            "config": {
                "vocab_size": self.config["vocab_size"],
                "embed_dim": self.config["embed_dim"],
                "num_layers": self.config["num_layers"],
                "max_seq_len": self.config["max_seq_len"],
            },
            "upload": upload_status,
            "message": f"Batch training complete: {len(completed_batches)}/{num_batches} batches, "
                       f"{total_sequences} sequences. Loss: {overall_first_loss} -> {final_loss}"
        }]

    def batch_train_status(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the status of a batch training job."""
        job_id = data.get("job_id")

        # If no job_id, return all jobs
        if not job_id:
            jobs_summary = []
            for jid, info in EndpointHandler._batch_jobs.items():
                completed = [b for b in info["batch_results"] if b.get("status") == "completed"]
                jobs_summary.append({
                    "job_id": jid,
                    "dataset_id": info["dataset_id"],
                    "status": info["status"],
                    "current_batch": info["current_batch"],
                    "num_batches": info["num_batches"],
                    "completed_batches": len(completed),
                    "started_at": info["started_at"],
                    "completed_at": info.get("completed_at"),
                })
            return [{"jobs": jobs_summary, "total_jobs": len(jobs_summary)}]

        if job_id not in EndpointHandler._batch_jobs:
            return [{"error": f"Job '{job_id}' not found"}]

        info = EndpointHandler._batch_jobs[job_id]
        completed = [b for b in info["batch_results"] if b.get("status") == "completed"]

        return [{
            "job_id": job_id,
            "dataset_id": info["dataset_id"],
            "status": info["status"],
            "total_samples": info["total_samples"],
            "batch_size": info["batch_size"],
            "num_batches": info["num_batches"],
            "current_batch": info["current_batch"],
            "completed_batches": len(completed),
            "epochs_per_batch": info["epochs_per_batch"],
            "lr": info["lr"],
            "batch_results": info["batch_results"],
            "started_at": info["started_at"],
            "completed_at": info.get("completed_at"),
        }]

    # ============================================================
    # Federated Learning (Split Learning) Endpoint
    # ============================================================
    #
    # Flow:
    #   1. federated_init   — サーバーが現在のモデル重みとラウンドIDを返す
    #   2. federated_submit — クライアントがローカル学習後の勾配差分を送信
    #   3. federated_aggregate — 集約（FedAvg）を実行しグローバルモデルを更新
    #   4. federated_status  — 現在のラウンド状態を確認
    #
    # Request examples:
    #   {"action": "federated_init", "round_id": "round_001", "min_clients": 2}
    #   {"action": "federated_submit", "round_id": "round_001", "client_id": "client_A",
    #    "model_delta": {<param_name>: <list>}, "num_samples": 500}
    #   {"action": "federated_aggregate", "round_id": "round_001"}
    #   {"action": "federated_status", "round_id": "round_001"}
    # ============================================================

    # Class-level storage for federated rounds (shared across requests)
    _federated_rounds: Dict[str, Dict[str, Any]] = {}

    def federated_init(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Initialize a new federated learning round.
        Returns the current global model weights so clients can start local training.
        """
        round_id = data.get("round_id")
        if not round_id:
            return [{"error": "round_id is required"}]

        min_clients = int(data.get("min_clients", 2))
        epochs_per_client = int(data.get("epochs_per_client", 3))
        lr = float(data.get("lr", 1e-3))

        # Serialize current model state for distribution
        model_state = {}
        for name, param in self.model.state_dict().items():
            model_state[name] = param.cpu().tolist()

        # Register this round
        EndpointHandler._federated_rounds[round_id] = {
            "round_id": round_id,
            "status": "waiting_for_clients",
            "min_clients": min_clients,
            "epochs_per_client": epochs_per_client,
            "lr": lr,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "client_updates": {},
            "aggregated": False,
            "global_model_snapshot": copy.deepcopy(self.model.state_dict()),
        }

        return [{
            "status": "initialized",
            "round_id": round_id,
            "min_clients": min_clients,
            "epochs_per_client": epochs_per_client,
            "lr": lr,
            "model_weights": model_state,
            "architecture": self.architecture,
            "config": {
                "vocab_size": self.config["vocab_size"],
                "embed_dim": self.config["embed_dim"],
                "num_layers": self.config["num_layers"],
                "max_seq_len": self.config["max_seq_len"],
            },
            "message": f"Federated round '{round_id}' initialized. Distribute weights to clients."
        }]

    def federated_submit(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Receive a client's local training result (model delta or full weights).
        Clients send their updated model parameters after local training.
        """
        round_id = data.get("round_id")
        client_id = data.get("client_id")

        if not round_id or not client_id:
            return [{"error": "round_id and client_id are required"}]

        if round_id not in EndpointHandler._federated_rounds:
            return [{"error": f"Round '{round_id}' not found. Call federated_init first."}]

        round_info = EndpointHandler._federated_rounds[round_id]

        if round_info["aggregated"]:
            return [{"error": f"Round '{round_id}' already aggregated. Start a new round."}]

        model_delta = data.get("model_delta")
        model_weights = data.get("model_weights")
        num_samples = int(data.get("num_samples", 1))
        client_loss = data.get("final_loss")

        if not model_delta and not model_weights:
            return [{"error": "Either model_delta or model_weights is required"}]

        # Convert submitted data to tensors
        update_tensors = {}
        global_snapshot = round_info["global_model_snapshot"]

        if model_delta:
            # Client sent parameter deltas (difference from global model)
            for name, values in model_delta.items():
                if name in global_snapshot:
                    update_tensors[name] = torch.tensor(values, dtype=global_snapshot[name].dtype)
        elif model_weights:
            # Client sent full weights — compute delta from global snapshot
            for name, values in model_weights.items():
                if name in global_snapshot:
                    client_param = torch.tensor(values, dtype=global_snapshot[name].dtype)
                    update_tensors[name] = client_param - global_snapshot[name].cpu()

        if not update_tensors:
            return [{"error": "No valid parameter updates found"}]

        round_info["client_updates"][client_id] = {
            "delta": update_tensors,
            "num_samples": num_samples,
            "final_loss": client_loss,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

        num_submitted = len(round_info["client_updates"])
        min_clients = round_info["min_clients"]

        return [{
            "status": "submitted",
            "round_id": round_id,
            "client_id": client_id,
            "num_samples": num_samples,
            "final_loss": client_loss,
            "clients_submitted": num_submitted,
            "min_clients": min_clients,
            "ready_to_aggregate": num_submitted >= min_clients,
            "message": f"Client '{client_id}' update received ({num_submitted}/{min_clients} clients)."
        }]

    def federated_aggregate(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Aggregate client updates using Federated Averaging (FedAvg).
        Weighted average of deltas based on each client's num_samples.
        """
        round_id = data.get("round_id")
        if not round_id:
            return [{"error": "round_id is required"}]

        if round_id not in EndpointHandler._federated_rounds:
            return [{"error": f"Round '{round_id}' not found"}]

        round_info = EndpointHandler._federated_rounds[round_id]

        if round_info["aggregated"]:
            return [{"error": f"Round '{round_id}' already aggregated"}]

        client_updates = round_info["client_updates"]
        if not client_updates:
            return [{"error": "No client updates to aggregate"}]

        force = data.get("force", False)
        if len(client_updates) < round_info["min_clients"] and not force:
            return [{
                "error": f"Not enough clients: {len(client_updates)}/{round_info['min_clients']}. "
                         f"Use force=true to aggregate anyway.",
                "clients_submitted": len(client_updates),
                "min_clients": round_info["min_clients"]
            }]

        # FedAvg: weighted average of deltas by number of samples
        total_samples = sum(u["num_samples"] for u in client_updates.values())
        global_state = round_info["global_model_snapshot"]
        aggregated_state = {}

        for name, param in global_state.items():
            aggregated_delta = torch.zeros_like(param, dtype=torch.float32)
            has_update = False
            for client_id, update in client_updates.items():
                if name in update["delta"]:
                    weight = update["num_samples"] / total_samples
                    aggregated_delta += weight * update["delta"][name].float()
                    has_update = True

            if has_update:
                aggregated_state[name] = param.float() + aggregated_delta
                aggregated_state[name] = aggregated_state[name].to(param.dtype)
            else:
                aggregated_state[name] = param

        # Apply aggregated weights to the global model
        self.model.load_state_dict(aggregated_state)
        self.model.eval()

        round_info["aggregated"] = True
        round_info["status"] = "completed"
        round_info["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Save checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config
        }, ckpt_path)

        # Upload if HF_TOKEN available
        upload_status = "skipped (no HF_TOKEN)"
        token = os.environ.get("HF_TOKEN")
        if token:
            try:
                from huggingface_hub import HfApi
                repo_id = os.environ.get("REPOSITORY_ID", "tapiocaTakeshi/Qubit")
                api = HfApi(token=token)
                api.upload_file(
                    path_or_fileobj=ckpt_path,
                    path_in_repo="neuroq_checkpoint.pt",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Federated learning round {round_id}: aggregated {len(client_updates)} clients"
                )
                upload_status = "uploaded"
            except Exception as e:
                upload_status = f"upload failed: {e}"

        # Build client summary
        client_summary = {}
        for cid, u in client_updates.items():
            client_summary[cid] = {
                "num_samples": u["num_samples"],
                "final_loss": u["final_loss"],
                "weight": round(u["num_samples"] / total_samples, 4),
            }

        # Clean up large data from memory
        for u in client_updates.values():
            u["delta"] = None
        round_info.pop("global_model_snapshot", None)

        return [{
            "status": "aggregated",
            "round_id": round_id,
            "num_clients": len(client_updates),
            "total_samples": total_samples,
            "clients": client_summary,
            "upload": upload_status,
            "message": f"FedAvg aggregation complete. {len(client_updates)} clients, {total_samples} total samples."
        }]

    def federated_status(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the current status of a federated learning round."""
        round_id = data.get("round_id")

        # If no round_id, return all active rounds
        if not round_id:
            rounds_summary = []
            for rid, info in EndpointHandler._federated_rounds.items():
                rounds_summary.append({
                    "round_id": rid,
                    "status": info["status"],
                    "clients_submitted": len(info["client_updates"]),
                    "min_clients": info["min_clients"],
                    "aggregated": info["aggregated"],
                    "created_at": info["created_at"],
                })
            return [{"active_rounds": rounds_summary, "total_rounds": len(rounds_summary)}]

        if round_id not in EndpointHandler._federated_rounds:
            return [{"error": f"Round '{round_id}' not found"}]

        info = EndpointHandler._federated_rounds[round_id]
        client_info = {}
        for cid, u in info["client_updates"].items():
            client_info[cid] = {
                "num_samples": u["num_samples"],
                "final_loss": u["final_loss"],
                "submitted_at": u["submitted_at"],
            }

        return [{
            "round_id": round_id,
            "status": info["status"],
            "min_clients": info["min_clients"],
            "epochs_per_client": info["epochs_per_client"],
            "lr": info["lr"],
            "clients_submitted": len(info["client_updates"]),
            "clients": client_info,
            "aggregated": info["aggregated"],
            "created_at": info["created_at"],
            "completed_at": info.get("completed_at"),
        }]
