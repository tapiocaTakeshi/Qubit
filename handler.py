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
    "embed_dim": 256,
    "hidden_dim": 512,
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
        # Look for checkpoint file first to determine architecture
        ckpt_path = self._find_checkpoint(path)
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

        if is_legacy or not NEUROQUANTUM_AVAILABLE:
            # Legacy mode: use old CharTokenizer + QBNNTransformer
            self._init_legacy(checkpoint, saved_config)
        else:
            # New mode: use NeuroQuantum + SentencePiece
            self._init_neuroquantum(path, checkpoint, saved_config)

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
            except RuntimeError:
                # State dict mismatch — reinitialize
                self.neuroq_model = NeuroQuantum(config=nq_config)

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
        if data.get("action") == "train":
            return self.train(data)

        inputs = data.get("inputs", data)
        if isinstance(inputs, list):
            inputs = inputs[0] if inputs else ""
        inputs = str(inputs)

        params = data.get("parameters", {})
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

        # Build or rebuild tokenizer from training data if needed
        if self.architecture == "neuroquantum" and NEUROQUANTUM_AVAILABLE:
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
