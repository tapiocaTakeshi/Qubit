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
            # Support Trainer API by returning an object with 'loss' and 'logits'
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
        Handle an inference or training request.

        Args:
            data: Request body with 'inputs' key and optional 'parameters'.
                  Alternatively, for training: {"action": "train", "dataset_id": "...", ...}

        Returns:
            List of dicts with 'generated_text' or training status.
        """
        # Check if this is a training request
        if data.get("action") == "train":
            return self.train(data)

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

    def train(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from transformers import Trainer, TrainingArguments, TrainerCallback
        from datasets import load_dataset
        import os
        import json
        import traceback
        from datetime import datetime, timezone
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

        train_dataset = CustomDataset(texts, self.tokenizer, self.config["max_seq_len"])

        def collate_fn(batch):
            max_len = max(len(x["input_ids"]) for x in batch)
            input_ids = []
            labels = []
            for x in batch:
                pad_len = max_len - len(x["input_ids"])
                ids = x["input_ids"] + [0] * pad_len  # 0 is PAD
                lbl = x["labels"] + [-100] * pad_len
                input_ids.append(ids)
                labels.append(lbl)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        # Callback to capture training loss
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

        final_loss = round(train_result.training_loss, 6) if train_result.training_loss else None

        # Build training history entry
        history_entry = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "epochs": int(epochs) if epochs == int(epochs) else epochs,
            "samples": len(train_dataset),
            "final_loss": final_loss,
            "learning_rate": lr,
            "text_column": text_column
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

        # Save model checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__), "qbnn_checkpoint.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config
        }, ckpt_path)

        # Upload checkpoint and training history if HF_TOKEN is available
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
