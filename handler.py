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

    Split training request format:
        {
            "action": "split_next",
            "parameters": {
                "mode": "qa",          # "qa" or "wikipedia"
                "num_chunks": 4,
                "epochs_per_chunk": 6,
                "lr": 3e-5,
                "batch_size": 4,
                "grad_accum_steps": 4,
                "warmup_steps": 30,
                "max_samples_per_dataset": 0
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
        # Split training state
        self._split_state = {
            "chunks": [],
            "current_index": 0,
            "num_chunks": 0,
            "mode": "qa",
        }

        # Device selection with CUDA health check
        self.device = self._select_device()
        self._init_log.append(f"device={self.device}")

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

        if NEUROQUANTUM_AVAILABLE:
            if is_legacy:
                # Legacy checkpoint detected — initialize fresh NeuroQuantum model
                # (ignore old small weights, start fresh with proper dimensions)
                self._init_log.append("legacy_checkpoint_detected=upgrading_to_neuroquantum")
                self._init_neuroquantum(path, checkpoint=None, saved_config=None)
                self._init_log.append("mode=neuroquantum (fresh, legacy checkpoint ignored)")
            else:
                # New mode: use NeuroQuantum + SentencePiece
                self._init_neuroquantum(path, checkpoint, saved_config)
                self._init_log.append("mode=neuroquantum")
        else:
            # Fallback: Legacy mode when NeuroQuantum not available
            self._init_legacy(checkpoint, saved_config)
            self._init_log.append("mode=legacy")

    def _select_device(self):
        """Select device with CUDA health check and CPU fallback."""
        if torch.cuda.is_available():
            try:
                # Quick CUDA health check
                test = torch.tensor([1.0], device="cuda")
                _ = test + test
                del test
                return "cuda"
            except RuntimeError:
                pass
        return "cpu"

    def _reset_cuda(self):
        """Attempt to recover from CUDA errors by resetting and falling back to CPU."""
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            test = torch.tensor([1.0], device="cuda")
            _ = test + test
            del test
            self.device = "cuda"
            return True
        except Exception:
            self.device = "cpu"
            # Move model to CPU
            if hasattr(self, 'neuroq_model') and self.neuroq_model is not None:
                self.neuroq_model = self.neuroq_model.cpu()
                self.model = self.neuroq_model
            elif hasattr(self, 'model') and self.model is not None:
                self.model = self.model.cpu()
            return False

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

        # Move to selected device
        self.neuroq_model = self.neuroq_model.to(self.device)
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
        Handle inference, training, or status requests.

        Action routing:
            "inference" (default) — テキスト生成（従来通り）
            "train"              — 一般データセットによる学習
            "train_qa"           — QA形式データでの学習
            "train_texts"        — テキストデータを直接送信して学習
            "status"             — 学習状況・モデル情報の確認
            "debug"              — 初期化ログ等の詳細情報
            "split_next"         — 分割学習の次チャンク実行
            "split_reset"        — 分割学習状態のリセット

        Request format:
            {
                "inputs": "<prompt or __action__>",
                "parameters": {
                    "action": "<action>",
                    ...action-specific params...
                }
            }
        """
        # ── Resolve action ──────────────────────────────────────
        # Priority: data["action"] > parameters["action"] > inputs command string
        params = data.get("parameters", {})
        action = data.get("action") or params.get("action", "")
        inputs_raw = data.get("inputs", "")
        if not action and isinstance(inputs_raw, str) and inputs_raw.startswith("__") and inputs_raw.endswith("__"):
            action = inputs_raw.strip("_")

        # Default to inference when no action specified
        if not action:
            action = "inference"

        # ── Action routing ──────────────────────────────────────
        # Status: return model info, training state, health
        if action == "status":
            return self._handle_status()

        # Debug: return detailed init log
        if action == "debug":
            return self._handle_debug()

        # Reinit: reset CUDA state and reinitialize model
        if action == "reinit":
            return self._handle_reinit()

        # Split training
        if action == "split_next":
            split_data = data if data.get("action") else params
            return self.split_train_next(split_data)
        if action == "split_reset":
            return self.split_train_reset()

        # Training: general dataset
        if action == "train":
            train_data = params if params.get("action") == "train" else data
            return self.train(train_data)

        # Training: QA format data
        if action == "train_qa":
            return self._train_qa(data)

        # Training: direct text data
        if action == "train_texts":
            return self._train_from_texts(data)

        # ── Inference (default) ─────────────────────────────────
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

    # ── Status & Debug handlers ─────────────────────────────

    def _handle_status(self) -> List[Dict[str, Any]]:
        """Return model status, training state, and health info."""
        import json
        from datetime import datetime, timezone

        n_params = sum(p.numel() for p in self.model.parameters())

        # Load training history if available
        history_path = os.path.join(os.path.dirname(__file__), "training_history.json")
        training_history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    training_history = json.load(f)
            except Exception:
                pass

        # Check checkpoint info
        ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
        ckpt_info = {}
        if os.path.exists(ckpt_path):
            ckpt_info["size_mb"] = round(os.path.getsize(ckpt_path) / (1024 * 1024), 1)
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_info["trained_at"] = ckpt.get("trained_at", "unknown")
                ckpt_info["datasets"] = ckpt.get("datasets", [])
            except Exception:
                pass

        # Split training state
        split_state = {
            "current_index": self._split_state.get("current_index", 0),
            "num_chunks": self._split_state.get("num_chunks", 0),
            "mode": self._split_state.get("mode", "idle"),
        }

        return [{
            "status": "ok",
            "architecture": self.architecture,
            "device": self.device,
            "model_params": n_params,
            "config": {k: v for k, v in self.config.items() if k != "architecture"},
            "checkpoint": ckpt_info,
            "training_history_count": len(training_history),
            "last_training": training_history[-1] if training_history else None,
            "split_training": split_state,
            "handler_version": "v6_2026_03_21",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]

    def _handle_debug(self) -> List[Dict[str, Any]]:
        """Return detailed debug/init information."""
        return [{
            "init_log": self._init_log,
            "architecture": self.architecture,
            "config": self.config,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "handler_version": "v6_2026_03_21",
        }]

    def _handle_reinit(self) -> List[Dict[str, Any]]:
        """Reset CUDA state and reinitialize model fresh."""
        from datetime import datetime, timezone
        old_device = self.device
        cuda_recovered = self._reset_cuda()
        self.device = self._select_device()
        return [{
            "status": "reinit_complete",
            "old_device": old_device,
            "new_device": self.device,
            "cuda_recovered": cuda_recovered,
            "architecture": self.architecture,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]

    def _generate_neuroquantum(self, text, temperature, max_new_tokens,
                                top_k, top_p, repetition_penalty):
        """Generate text using the NeuroQuantum architecture."""
        tokens = self.tokenizer.encode(text, add_special=True)
        if not tokens:
            return [{"generated_text": ""}]

        # Clamp token IDs to valid range
        vocab_size = self.config["vocab_size"]
        tokens = [min(max(tid, 0), vocab_size - 1) for tid in tokens]

        device = self.device
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

    # ── Training handlers ───────────────────────────────────

    def _train_qa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Train with QA-format data. Accepts question-answer pairs and
        formats them for optimal learning.

        Request format:
            {
                "inputs": "__train_qa__",
                "parameters": {
                    "action": "train_qa",
                    "qa_pairs": [
                        {"question": "日本の首都は？", "answer": "東京です。"},
                        ...
                    ],
                    // OR plain texts in QA format:
                    "texts": ["質問: ...\n回答: ...", ...],
                    "epochs": 6,
                    "lr": 3e-5,
                    "batch_size": 4,
                    "grad_accum_steps": 4,
                    "repeat": 3
                }
            }
        """
        params = data.get("parameters", {})
        qa_pairs = params.get("qa_pairs", [])
        texts = params.get("texts", [])
        repeat = int(params.get("repeat", 3))

        # Convert qa_pairs to formatted texts
        formatted_texts = list(texts)  # Start with any raw texts
        for qa in qa_pairs:
            q = qa.get("question", qa.get("q", ""))
            a = qa.get("answer", qa.get("a", ""))
            if q and a:
                formatted_texts.append(f"質問: {q}\n回答: {a}")

        if not formatted_texts:
            return [{"error": "No QA data provided. Send qa_pairs or texts in parameters."}]

        # Repeat QA data for better learning (QA pairs are typically small)
        expanded_texts = formatted_texts * repeat

        # Delegate to _train_from_texts with the expanded data
        train_data = {
            "parameters": {
                "texts": expanded_texts,
                "epochs": params.get("epochs", 6),
                "lr": params.get("lr", 3e-5),
                "batch_size": params.get("batch_size", 4),
                "grad_accum_steps": params.get("grad_accum_steps", 4),
                "warmup_steps": params.get("warmup_steps", 30),
            }
        }
        result = self._train_from_texts(train_data)

        # Enrich response with QA-specific info
        if result and isinstance(result, list) and result[0].get("status") == "success":
            result[0]["qa_pairs_count"] = len(formatted_texts)
            result[0]["repeat"] = repeat
            result[0]["total_texts"] = len(expanded_texts)

        return result

    def _train_from_texts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Train directly from text data sent in the request body.
        No need to load from HF datasets - client sends the texts directly.

        Request format:
            {
                "inputs": "__train_texts__",
                "parameters": {
                    "action": "train_texts",
                    "texts": ["text1", "text2", ...],
                    "epochs": 4,
                    "lr": 3e-5,
                    "batch_size": 4,
                    "grad_accum_steps": 4
                }
            }
        """
        import traceback
        import math
        import random
        import gc
        from datetime import datetime, timezone

        params = data.get("parameters", {})
        texts = params.get("texts", [])
        if not texts:
            return [{"error": "No texts provided. Send texts in parameters.texts"}]

        epochs = int(params.get("epochs", 4))
        lr = float(params.get("lr", 3e-5))
        batch_size = int(params.get("batch_size", 4))
        grad_accum = int(params.get("grad_accum_steps", 4))
        warmup_steps = int(params.get("warmup_steps", 30))

        if not NEUROQUANTUM_AVAILABLE or self.architecture != "neuroquantum":
            return [{"error": "NeuroQuantum architecture required for text training"}]

        try:
            max_seq_len = self.config["max_seq_len"]
            vocab_size = self.config["vocab_size"]

            # Tokenize with vocab bounds check
            sequences = []
            for t in texts:
                ids = self.tokenizer.encode(t, add_special=True)
                # Clamp token IDs to valid range to prevent CUDA assert
                ids = [min(max(tid, 0), vocab_size - 1) for tid in ids]
                if len(ids) <= max_seq_len:
                    if len(ids) >= 4:
                        sequences.append(ids)
                else:
                    stride = max(max_seq_len // 2, 1)
                    for start in range(0, len(ids) - max_seq_len + 1, stride):
                        sequences.append(ids[start:start + max_seq_len])

            if not sequences:
                return [{"error": "No valid sequences after tokenization"}]

            # Train (use self.device with fallback)
            device = self.device
            steps_per_epoch = len(sequences) // batch_size
            total_steps = (steps_per_epoch * epochs) // grad_accum
            optimizer = torch.optim.AdamW(self.neuroq_model.parameters(), lr=lr, weight_decay=0.01)

            self.neuroq_model.train()
            global_step = 0
            epoch_losses = []

            for epoch in range(epochs):
                random.shuffle(sequences)
                total_loss = 0
                n_batches = 0
                optimizer.zero_grad()

                for i in range(0, len(sequences), batch_size):
                    batch_seqs = sequences[i:i + batch_size]
                    if not batch_seqs:
                        continue

                    max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
                    input_ids = []
                    labels = []
                    for s in batch_seqs:
                        ids = s[:max_len]
                        pad_len = max_len - len(ids)
                        input_ids.append(ids + [self.tokenizer.pad_id] * pad_len)
                        labels.append(ids + [-100] * pad_len)

                    input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
                    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

                    logits = self.neuroq_model(input_ids_t)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels_t[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, self.config["vocab_size"]),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    loss = loss / grad_accum
                    loss.backward()

                    total_loss += loss.item() * grad_accum
                    n_batches += 1

                    if n_batches % grad_accum == 0:
                        # Learning rate schedule
                        if global_step < warmup_steps:
                            cur_lr = lr * global_step / max(warmup_steps, 1)
                        else:
                            progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                            cur_lr = lr * (0.1 + 0.9 * cosine_decay)
                        for pg in optimizer.param_groups:
                            pg['lr'] = cur_lr
                        torch.nn.utils.clip_grad_norm_(self.neuroq_model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                if n_batches % grad_accum != 0:
                    torch.nn.utils.clip_grad_norm_(self.neuroq_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                avg_loss = total_loss / max(n_batches, 1)
                epoch_losses.append(round(avg_loss, 4))

            self.neuroq_model.eval()
            del optimizer
            gc.collect()

            # Save checkpoint
            ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
            torch.save({
                "model_state": self.neuroq_model.state_dict(),
                "config": self.config,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }, ckpt_path)

            # Upload if HF_TOKEN available
            upload_status = "skipped"
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
                        commit_message=f"Update checkpoint after train_texts ({len(texts)} texts)"
                    )
                    upload_status = "uploaded"
                except Exception as e:
                    upload_status = f"failed: {e}"

            return [{"generated_text": f"Training complete: {len(texts)} texts, {len(sequences)} sequences, {epochs} epochs, final_loss={epoch_losses[-1]}", "status": "success", "epoch_losses": epoch_losses, "sequences": len(sequences), "upload": upload_status}]

        except RuntimeError as e:
            if "CUDA" in str(e):
                # CUDA error — attempt recovery by falling back to CPU
                self._reset_cuda()
                self._init_log.append(f"CUDA error during training, fell back to {self.device}")
            self.neuroq_model.eval()
            return [{"error": f"Training failed: {e}", "traceback": traceback.format_exc(),
                     "device": self.device, "hint": "CUDA error detected, device switched to CPU. Retry training."}]
        except Exception as e:
            self.neuroq_model.eval()
            return [{"error": f"Training failed: {e}", "traceback": traceback.format_exc()}]

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
        config_name = data.get("config", data.get("name", None))
        max_samples = int(data.get("max_samples", 0))  # 0 = use all samples
        epochs = float(data.get("epochs", 10.0))
        lr = float(data.get("lr", 1e-3))

        try:
            ds = load_dataset(dataset_id, name=config_name, split=split, trust_remote_code=False)
        except Exception as e:
            return [{"error": f"Failed to load dataset: {str(e)}", "traceback": traceback.format_exc()}]

        # Extract texts
        texts = []
        n = min(max_samples, len(ds)) if max_samples > 0 else len(ds)
        for row in ds.select(range(n)):
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
                    ids = tokenizer.encode(t, add_special=True)
                    # Truncate or create sliding windows for long texts
                    if len(ids) <= max_seq_len:
                        if len(ids) >= 4:
                            self.data.append(ids)
                    else:
                        # Sliding window with stride = max_seq_len // 2
                        stride = max(max_seq_len // 2, 1)
                        for start in range(0, len(ids) - max_seq_len + 1, stride):
                            chunk = ids[start:start + max_seq_len]
                            self.data.append(chunk)

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
    # Split (Chunked) Training
    # ============================================================

    # QA dataset definitions for split training
    QA_DATASETS_INFO = [
        {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
        {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
        {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "format": "alpaca"},
        {"id": "izumi-lab/llm-japanese-dataset", "format": "izumi"},
    ]

    CRAFTED_QA = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 水の化学式は何ですか？\n回答: 水の化学式はH2Oです。",
        "質問: 太陽系で一番大きい惑星は？\n回答: 太陽系で一番大きい惑星は木星です。",
        "質問: 光の速さはどのくらいですか？\n回答: 光の速さは秒速約30万キロメートル（299,792,458 m/s）です。",
        "質問: 日本で一番高い山は？\n回答: 日本で一番高い山は富士山で、標高3,776メートルです。",
        "質問: 1年は何日ですか？\n回答: 1年は通常365日です。うるう年は366日です。",
        "質問: 地球の衛星は何ですか？\n回答: 地球の衛星は月です。",
        "質問: 人間の体温は通常何度ですか？\n回答: 人間の体温は通常約36.5度から37度です。",
    ]

    def _format_qa_alpaca(self, row):
        inst = row.get("instruction", "")
        out = row.get("output", "")
        if inst and out:
            return f"質問: {inst}\n回答: {out}"
        return ""

    def _format_qa_conversations(self, row):
        convs = row.get("conversations", [])
        parts = []
        for turn in convs:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human":
                parts.append(f"質問: {value}")
            elif role == "gpt":
                parts.append(f"回答: {value}")
        return "\n".join(parts) if parts else ""

    def _format_qa_izumi(self, row):
        text = row.get("text", "")
        if text and len(text) > 10:
            return text
        return ""

    def _prepare_split_data(self, mode, num_chunks, max_samples=0):
        """Prepare and split training data into chunks."""
        from datasets import load_dataset
        import random

        all_texts = []

        if mode == "qa":
            for ds_info in self.QA_DATASETS_INFO:
                ds_id = ds_info["id"]
                fmt = ds_info["format"]
                try:
                    ds = load_dataset(ds_id, split="train")
                    n = min(max_samples, len(ds)) if max_samples > 0 else len(ds)
                    for row in ds.select(range(n)):
                        if fmt == "alpaca":
                            text = self._format_qa_alpaca(row)
                        elif fmt == "conversations":
                            text = self._format_qa_conversations(row)
                        elif fmt == "izumi":
                            text = self._format_qa_izumi(row)
                        else:
                            continue
                        if text and len(text) > 10:
                            all_texts.append(text)
                except Exception:
                    pass
            # Add crafted QA
            for _ in range(40):
                all_texts.extend(self.CRAFTED_QA)
        elif mode == "wikipedia":
            try:
                ds = load_dataset("izumi-lab/wikipedia-ja-20230720", split="train", streaming=True)
                count = 0
                for row in ds:
                    text = row.get("text", "").strip()
                    if text and len(text) > 50:
                        all_texts.append(text)
                        count += 1
                        if max_samples > 0 and count >= max_samples:
                            break
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'qa' or 'wikipedia'.")

        random.shuffle(all_texts)

        chunk_size = max(len(all_texts) // num_chunks, 1)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(all_texts)
            chunks.append(all_texts[start:end])

        return chunks

    def _run_split_chunk_training(self, chunk_texts, chunk_index, num_chunks,
                                   epochs_per_chunk, lr, batch_size,
                                   grad_accum_steps, warmup_steps):
        """Train one chunk. Returns loss info."""
        import math
        import random

        if not NEUROQUANTUM_AVAILABLE or self.architecture != "neuroquantum":
            return {"error": "Split training requires neuroquantum architecture"}

        max_seq_len = self.config["max_seq_len"]
        min_lr_ratio = 0.1

        # Tokenize chunk texts
        sequences = []
        for text in chunk_texts:
            ids = self.tokenizer.encode(text, add_special=True)
            if len(ids) >= 4:
                if len(ids) <= max_seq_len:
                    sequences.append(ids)
                else:
                    stride = max(max_seq_len // 2, 1)
                    for start in range(0, len(ids) - max_seq_len + 1, stride):
                        sequences.append(ids[start:start + max_seq_len])

        if not sequences:
            return {"chunk_index": chunk_index, "loss": None, "message": "No sequences"}

        steps_per_epoch = len(sequences) // batch_size
        total_steps = (steps_per_epoch * epochs_per_chunk) // grad_accum_steps
        optimizer = torch.optim.AdamW(self.neuroq_model.parameters(), lr=lr, weight_decay=0.01)

        self.neuroq_model.train()
        global_step = 0
        best_loss = float('inf')
        log = []

        for epoch in range(epochs_per_chunk):
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()

            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i + batch_size]
                if not batch_seqs:
                    continue

                max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
                input_ids = []
                labels = []
                for s in batch_seqs:
                    ids = s[:max_len]
                    pad_len = max_len - len(ids)
                    input_ids.append(ids + [self.tokenizer.pad_id] * pad_len)
                    labels.append(ids + [-100] * pad_len)

                device = next(self.neuroq_model.parameters()).device
                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)

                logits = self.neuroq_model(input_ids_t)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels_t[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config["vocab_size"]),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss = loss / grad_accum_steps
                loss.backward()

                total_loss += loss.item() * grad_accum_steps
                n_batches += 1

                if n_batches % grad_accum_steps == 0:
                    if global_step < warmup_steps:
                        current_lr = lr * global_step / max(warmup_steps, 1)
                    else:
                        progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                        current_lr = lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr
                    torch.nn.utils.clip_grad_norm_(self.neuroq_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if n_batches % grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.neuroq_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            msg = f"Chunk {chunk_index+1} Epoch {epoch+1}/{epochs_per_chunk} | Loss: {avg_loss:.4f}"
            log.append(msg)

            if avg_loss < best_loss:
                best_loss = avg_loss

        # Save checkpoint after chunk
        ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
        torch.save({
            "model_state": self.neuroq_model.state_dict(),
            "config": self.config
        }, ckpt_path)
        self.neuroq_model.eval()

        # Upload checkpoint if HF_TOKEN available
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
                    commit_message=f"Split training chunk {chunk_index+1}/{num_chunks} complete"
                )
                upload_status = "uploaded"
            except Exception as e:
                upload_status = f"upload failed: {e}"

        return {
            "chunk_index": chunk_index,
            "best_loss": best_loss,
            "sequences": len(sequences),
            "log": log,
            "upload": upload_status,
        }

    def split_train_next(self, data):
        """Train next chunk of split data."""
        mode = data.get("mode", "qa")
        num_chunks = int(data.get("num_chunks", 4))
        epochs_per_chunk = int(data.get("epochs_per_chunk", 6))
        lr = float(data.get("lr", 3e-5))
        batch_size = int(data.get("batch_size", 4))
        grad_accum_steps = int(data.get("grad_accum_steps", 4))
        warmup_steps = int(data.get("warmup_steps", 30))
        max_samples = int(data.get("max_samples_per_dataset", 0))

        # Prepare chunks if needed
        if (not self._split_state["chunks"]
                or self._split_state["num_chunks"] != num_chunks
                or self._split_state["mode"] != mode):
            try:
                chunks = self._prepare_split_data(mode, num_chunks, max_samples)
                self._split_state = {
                    "chunks": chunks,
                    "current_index": 0,
                    "num_chunks": num_chunks,
                    "mode": mode,
                }
            except Exception as e:
                import traceback
                return [{"error": f"Failed to prepare data: {e}",
                         "traceback": traceback.format_exc()}]

        idx = self._split_state["current_index"]
        if idx >= self._split_state["num_chunks"]:
            return [{
                "status": "all_chunks_completed",
                "chunk_index": idx,
                "chunks_remaining": 0,
                "message": "All chunks already trained. Use split_reset to start over.",
            }]

        chunk_texts = self._split_state["chunks"][idx]
        try:
            result = self._run_split_chunk_training(
                chunk_texts, idx, num_chunks,
                epochs_per_chunk, lr, batch_size,
                grad_accum_steps, warmup_steps
            )
        except Exception as e:
            import traceback
            return [{"error": f"Training error: {e}",
                     "traceback": traceback.format_exc()}]

        self._split_state["current_index"] = idx + 1
        remaining = self._split_state["num_chunks"] - self._split_state["current_index"]

        return [{
            "status": "chunk_completed",
            "chunk_index": idx,
            "chunks_remaining": remaining,
            "best_loss": result.get("best_loss"),
            "sequences": result.get("sequences"),
            "log": result.get("log", []),
            "upload": result.get("upload"),
            "message": f"Chunk {idx+1}/{num_chunks} done. {remaining} remaining.",
        }]

    def split_train_reset(self):
        """Reset split training state."""
        self._split_state = {
            "chunks": [],
            "current_index": 0,
            "num_chunks": 0,
            "mode": "qa",
        }
        return [{"status": "reset", "message": "Split training state has been reset."}]
