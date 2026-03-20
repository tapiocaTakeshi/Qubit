"""
Custom Handler for HuggingFace Inference Endpoints
neuroQ - NeuroQuantum Transformer

Supports both inference and training via the "action" field:
  - action: "inference" (default) — text generation
  - action: "train"               — fine-tuning on HF datasets
  - action: "train_qa"            — QA-format fine-tuning
  - action: "status"              — check training status

Reference: https://huggingface.co/docs/inference-endpoints/main/en/engines/toolkit#create-a-custom-inference-handler
"""

import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# Import NeuroQuantum architecture
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
# Default datasets for training
# ============================================================

DEFAULT_DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
]

QA_DATASETS_INFO = [
    {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "format": "conversations"},
    {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
    {"id": "izumi-lab/llm-japanese-dataset", "format": "izumi"},
]

CRAFTED_QA = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さはどのくらいですか？\n回答: 富士山の高さは3,776メートルです。日本で最も高い山です。",
    "質問: プログラミングとは何ですか？\n回答: プログラミングとは、コンピュータに実行させる命令を書くことです。Python、Java、C++などのプログラミング言語を使って、ソフトウェアやアプリケーションを作成します。",
    "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。機械学習、深層学習、自然言語処理などの技術を含みます。",
    "質問: 量子コンピュータとは何ですか？\n回答: 量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。量子ビット（キュービット）を使い、従来のコンピュータでは困難な問題を解くことができます。",
    "質問: 機械学習とは何ですか？\n回答: 機械学習は、データからパターンを学習し、予測や判断を行う人工知能の一分野です。教師あり学習、教師なし学習、強化学習などの手法があります。",
    "質問: Pythonとは何ですか？\n回答: Pythonは、読みやすく書きやすい汎用プログラミング言語です。データ分析、AI開発、Web開発など幅広い分野で使われています。",
    "質問: インターネットとは何ですか？\n回答: インターネットは、世界中のコンピュータネットワークを相互に接続した通信網です。Webサイトの閲覧、メール、動画配信などのサービスを支えています。",
    "質問: 太陽系にはいくつの惑星がありますか？\n回答: 太陽系には8つの惑星があります。水星、金星、地球、火星、木星、土星、天王星、海王星です。",
    "質問: 光の速さはどのくらいですか？\n回答: 光の速さは秒速約30万キロメートル（299,792,458 m/s）です。これは宇宙で最も速い速度です。",
]


# ============================================================
# Utility functions
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
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith(".pt"):
                return os.path.join(path, fname)
    return None


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
    """Tokenize texts into training sequences with BOS/EOS."""
    sequences = []
    for t in texts:
        content_ids = tok.encode(t, add_special=False)
        max_content = max_seq_len - 2
        if max_content <= 0:
            continue
        if len(content_ids) <= max_content:
            if len(content_ids) >= 2:
                seq = [tok.bos_id] + content_ids + [tok.eos_id]
                sequences.append(seq)
        else:
            stride = max(max_content // 2, 1)
            for start in range(0, len(content_ids) - max_content + 1, stride):
                chunk = content_ids[start:start + max_content]
                seq = [tok.bos_id] + chunk + [tok.eos_id]
                sequences.append(seq)
            remaining = content_ids[-max_content:]
            tail_seq = [tok.bos_id] + remaining + [tok.eos_id]
            if tail_seq != sequences[-1]:
                sequences.append(tail_seq)
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def format_qa_alpaca(row):
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    q = f"{instruction}\n{inp}" if inp else instruction
    return f"質問: {q}\n回答: {output}"


def format_qa_conversations(row):
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn, next_turn = convs[i], convs[i + 1]
        q_text = turn.get("value", turn.get("content", "")).strip() if isinstance(turn, dict) else (turn.strip() if isinstance(turn, str) else "")
        a_text = next_turn.get("value", next_turn.get("content", "")).strip() if isinstance(next_turn, dict) else (next_turn.strip() if isinstance(next_turn, str) else "")
        if q_text and a_text:
            pairs.append(f"質問: {q_text}\n回答: {a_text}")
        i += 2
    return "\n\n".join(pairs) if pairs else None


def format_qa_izumi(row):
    output = row.get("output", "").strip()
    instruction = row.get("input", row.get("instruction", "")).strip() if isinstance(row.get("input", ""), str) else ""
    if not output:
        return None
    return f"質問: {instruction}\n回答: {output}" if instruction else f"回答: {output}"


# ============================================================
# HuggingFace Inference Endpoints Handler
# ============================================================

class EndpointHandler:
    """
    Custom handler for HuggingFace Inference Endpoints.

    Supports:
      - action: "inference" (default) — text generation
      - action: "train"               — general dataset training
      - action: "train_qa"            — QA-format training
      - action: "status"              — check training status
    """

    def __init__(self, path: str = ""):
        """Load the model, tokenizer, and config from the model directory."""
        self.model_path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_status = {"running": False, "log": [], "message": "idle"}

        # Find checkpoint
        self.ckpt_path = find_checkpoint(path) if path else None

        if self.ckpt_path and NEUROQUANTUM_AVAILABLE:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            self.config = checkpoint.get("config", dict(DEFAULT_CONFIG))

            # Load tokenizer
            tokenizer_path = os.path.join(path, "neuroq_tokenizer.model")
            self.tokenizer = NeuroQuantumTokenizer(
                vocab_size=self.config["vocab_size"],
                model_file=tokenizer_path if os.path.isfile(tokenizer_path) else None,
            )

            # Build model
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
            self.model = NeuroQuantum(config=nq_config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[handler] NeuroQuantum model loaded: {n_params:,} params on {self.device}")
        else:
            # Fallback: initialize fresh model
            self.config = dict(DEFAULT_CONFIG)
            if NEUROQUANTUM_AVAILABLE:
                self.tokenizer = NeuroQuantumTokenizer(vocab_size=self.config["vocab_size"])
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
                self.model = NeuroQuantum(config=nq_config).to(self.device)
                self.model.eval()
                print("[handler] NeuroQuantum model initialized (no checkpoint)")
            else:
                raise RuntimeError("neuroquantum_layered.py not available")

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Route request based on "action" field.

        Args:
            data: Dictionary with:
                - "action" (str): "inference" | "train" | "train_qa" | "status"
                - For inference:
                    - "inputs" (str): The input prompt text
                    - "parameters" (dict): Generation parameters
                - For train:
                    - "parameters" (dict): Training hyperparameters
                - For train_qa:
                    - "parameters" (dict): QA training hyperparameters

        Returns:
            List of dicts with results.
        """
        action = data.get("action", "inference")

        if action == "train":
            return self._handle_train(data)
        elif action == "train_qa":
            return self._handle_train_qa(data)
        elif action == "status":
            return self._handle_status()
        else:
            return self._handle_inference(data)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    def _handle_inference(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text from prompt."""
        inputs = data.get("inputs", data.get("prompt", ""))
        if isinstance(inputs, list):
            inputs = inputs[0] if inputs else ""
        inputs = str(inputs)

        params = data.get("parameters", {})
        temperature = float(params.get("temperature", 0.7))
        max_new_tokens = int(params.get("max_new_tokens", 100))
        top_k = int(params.get("top_k", 40))
        top_p = float(params.get("top_p", 0.9))
        repetition_penalty = float(params.get("repetition_penalty", 1.3))

        tokens = self.tokenizer.encode(inputs, add_special=True)
        if not tokens:
            return [{"generated_text": ""}]

        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated = list(tokens)
        max_seq_len = self.config["max_seq_len"]

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                seq = input_tensor[:, -max_seq_len:]
                logits = self.model(seq)[:, -1, :] / max(temperature, 1e-5)

                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    topk_vals = torch.topk(logits, k)[0]
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    to_remove = cumulative_probs > top_p
                    to_remove[:, 1:] = to_remove[:, :-1].clone()
                    to_remove[:, 0] = False
                    indices_to_remove = sorted_indices[to_remove]
                    logits[0, indices_to_remove] = float('-inf')

                if repetition_penalty > 1.0 and len(generated) > 1:
                    for prev in set(generated[-50:]):
                        if prev < logits.size(-1):
                            logits[0, prev] /= repetition_penalty

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()

                if nxt_id == self.tokenizer.eos_id:
                    break
                if nxt_id == self.tokenizer.pad_id:
                    continue

                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)

        generated_text = self.tokenizer.decode(generated[len(tokens):], skip_special=True)
        return [{"generated_text": generated_text}]

    # --------------------------------------------------------
    # Training (general)
    # --------------------------------------------------------

    def _handle_train(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run general dataset training."""
        if self.training_status["running"]:
            return [{"status": "error", "message": "Training already in progress"}]

        from datasets import load_dataset

        params = data.get("parameters", {})
        epochs = int(params.get("epochs", 10))
        lr = float(params.get("lr", 1e-4))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 8))
        warmup_steps = int(params.get("warmup_steps", 100))
        max_samples = int(params.get("max_samples_per_dataset", 5000))
        dataset_ids = params.get("dataset_ids", None)

        self.training_status = {"running": True, "log": [], "message": "Loading datasets..."}

        try:
            all_texts = []
            datasets_to_use = DEFAULT_DATASETS
            if dataset_ids:
                datasets_to_use = [{"id": did, "col": "text"} for did in dataset_ids]

            for ds_info in datasets_to_use:
                try:
                    self.training_status["message"] = f"Loading {ds_info['id']}..."
                    ds = load_dataset(ds_info["id"], split="train")
                    texts = extract_texts(ds, ds_info["col"], max_samples)
                    all_texts.extend(texts)
                    self.training_status["log"].append(f"Loaded {ds_info['id']}: {len(texts)} texts")
                except Exception as e:
                    self.training_status["log"].append(f"Error loading {ds_info['id']}: {e}")

            # Also load cc100-ja
            try:
                self.training_status["message"] = "Loading cc100-ja..."
                ds_cc = load_dataset("range3/cc100-ja", split="train", streaming=True)
                cc_texts = []
                for i, row in enumerate(ds_cc):
                    if i >= max_samples:
                        break
                    text = row.get("text", "").strip()
                    if len(text) > 10:
                        cc_texts.append(text)
                all_texts.extend(cc_texts)
                self.training_status["log"].append(f"Loaded cc100-ja: {len(cc_texts)} texts")
            except Exception as e:
                self.training_status["log"].append(f"Error loading cc100-ja: {e}")

            if not all_texts:
                self.training_status["running"] = False
                self.training_status["message"] = "No training data loaded"
                return [{"status": "error", "message": "No training data loaded",
                         "log": self.training_status["log"]}]

            self.training_status["message"] = "Tokenizing..."
            max_seq_len = self.config["max_seq_len"]
            sequences = tokenize_texts(all_texts, self.tokenizer, max_seq_len)
            self.training_status["log"].append(
                f"Total: {len(all_texts)} texts -> {len(sequences)} sequences"
            )

            self._run_training_loop(
                sequences, epochs, lr, batch_size, grad_accum_steps,
                warmup_steps, max_seq_len,
            )

            # Save checkpoint
            self._save_checkpoint(
                datasets=[d["id"] for d in datasets_to_use] + ["range3/cc100-ja"]
            )

            return [{"status": "success", "message": self.training_status["message"],
                     "log": self.training_status["log"]}]

        except Exception as e:
            import traceback
            self.training_status["running"] = False
            self.training_status["message"] = f"Error: {e}"
            self.training_status["log"].append(traceback.format_exc())
            self.model.eval()
            return [{"status": "error", "message": str(e),
                     "log": self.training_status["log"]}]

    # --------------------------------------------------------
    # Training (QA format)
    # --------------------------------------------------------

    def _handle_train_qa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run QA-format training."""
        if self.training_status["running"]:
            return [{"status": "error", "message": "Training already in progress"}]

        from datasets import load_dataset

        params = data.get("parameters", {})
        epochs = int(params.get("epochs", 20))
        lr = float(params.get("lr", 3e-5))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 4))
        warmup_steps = int(params.get("warmup_steps", 30))
        max_samples = int(params.get("max_samples_per_dataset", 1500))
        dataset_id = params.get("dataset_id", None)
        crafted_repeat = int(params.get("crafted_repeat", 40))

        self.training_status = {"running": True, "log": [], "message": "Loading QA datasets..."}

        try:
            all_qa = []

            if dataset_id:
                # Custom single dataset
                try:
                    self.training_status["message"] = f"Loading {dataset_id}..."
                    ds = load_dataset(dataset_id, split="train", streaming=True)
                    count = 0
                    for row in ds:
                        if count >= max_samples:
                            break
                        q = row.get("question", row.get("instruction", "")).strip()
                        a = row.get("answer", row.get("output", row.get("response", ""))).strip()
                        if q and a and len(q) > 2 and len(a) > 2:
                            all_qa.append(f"質問: {q}\n回答: {a}")
                            count += 1
                    self.training_status["log"].append(f"Loaded {dataset_id}: {count} QA samples")
                except Exception as e:
                    self.training_status["log"].append(f"Error loading {dataset_id}: {e}")
            else:
                # Default QA datasets
                for ds_info in QA_DATASETS_INFO:
                    ds_id = ds_info["id"]
                    fmt = ds_info["format"]
                    ms = min(1000, max_samples) if fmt == "izumi" else max_samples
                    try:
                        self.training_status["message"] = f"Loading {ds_id}..."
                        ds = load_dataset(ds_id, split="train")
                        n = min(ms, len(ds))
                        count = 0
                        for row in ds.select(range(n)):
                            if fmt == "alpaca":
                                text = format_qa_alpaca(row)
                            elif fmt == "conversations":
                                text = format_qa_conversations(row)
                            elif fmt == "izumi":
                                text = format_qa_izumi(row)
                            else:
                                continue
                            if text and len(text) > 10:
                                all_qa.append(text)
                                count += 1
                        self.training_status["log"].append(f"Loaded {ds_id}: {count} QA samples")
                    except Exception as e:
                        self.training_status["log"].append(f"Error loading {ds_id}: {e}")

            # Add crafted QA
            for _ in range(crafted_repeat):
                all_qa.extend(CRAFTED_QA)
            self.training_status["log"].append(
                f"Added {len(CRAFTED_QA) * crafted_repeat} crafted QA samples"
            )
            self.training_status["log"].append(f"Total QA texts: {len(all_qa)}")

            if not all_qa:
                self.training_status["running"] = False
                self.training_status["message"] = "No QA data loaded"
                return [{"status": "error", "message": "No QA data loaded",
                         "log": self.training_status["log"]}]

            self.training_status["message"] = "Tokenizing..."
            max_seq_len = self.config["max_seq_len"]
            sequences = tokenize_texts(all_qa, self.tokenizer, max_seq_len)
            self.training_status["log"].append(f"Training sequences: {len(sequences)}")

            self._run_training_loop(
                sequences, epochs, lr, batch_size, grad_accum_steps,
                warmup_steps, max_seq_len,
            )

            # Save checkpoint
            extra_ds = [dataset_id] if dataset_id else []
            self._save_checkpoint(
                datasets=[d["id"] for d in QA_DATASETS_INFO] + extra_ds,
                extra_meta={"qa_training": True},
            )

            return [{"status": "success", "message": self.training_status["message"],
                     "log": self.training_status["log"]}]

        except Exception as e:
            import traceback
            self.training_status["running"] = False
            self.training_status["message"] = f"Error: {e}"
            self.training_status["log"].append(traceback.format_exc())
            self.model.eval()
            return [{"status": "error", "message": str(e),
                     "log": self.training_status["log"]}]

    # --------------------------------------------------------
    # Training loop (shared)
    # --------------------------------------------------------

    def _run_training_loop(self, sequences, epochs, lr, batch_size,
                           grad_accum_steps, warmup_steps, max_seq_len):
        """Core training loop used by both general and QA training."""
        steps_per_epoch = len(sequences) // batch_size
        total_steps = (steps_per_epoch * epochs) // grad_accum_steps
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()

            self.training_status["message"] = f"Training epoch {epoch+1}/{epochs}..."

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

                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

                logits = self.model(input_ids_t)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels_t[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config["vocab_size"]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / grad_accum_steps
                loss.backward()

                total_loss += loss.item() * grad_accum_steps
                n_batches += 1

                if n_batches % grad_accum_steps == 0:
                    cur_lr = get_lr(global_step, total_steps, warmup_steps, lr)
                    for pg in optimizer.param_groups:
                        pg['lr'] = cur_lr
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if n_batches % grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            msg = f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
            self.training_status["log"].append(msg)
            self.training_status["message"] = msg

        self.model.eval()
        self.training_status["message"] = f"Training complete! Final loss: {avg_loss:.4f}"
        self.training_status["running"] = False

    # --------------------------------------------------------
    # Checkpoint save
    # --------------------------------------------------------

    def _save_checkpoint(self, datasets=None, extra_meta=None):
        """Save model checkpoint."""
        if not self.ckpt_path:
            self.ckpt_path = os.path.join(self.model_path or ".", "neuroq_checkpoint.pt")

        # Load existing checkpoint for training log history
        prev_log = []
        prev_datasets = []
        if os.path.isfile(self.ckpt_path):
            try:
                old_ckpt = torch.load(self.ckpt_path, map_location="cpu")
                prev_log = old_ckpt.get("training_log", [])
                prev_datasets = old_ckpt.get("datasets", [])
            except Exception:
                pass

        new_log_entries = [
            {"epoch": len(prev_log) + i + 1, "loss": float(l.split("Loss: ")[1])}
            for i, l in enumerate(self.training_status["log"]) if "Loss:" in l
        ]

        ds_list = list(set(prev_datasets + (datasets or [])))

        checkpoint = {
            "model_state": self.model.state_dict(),
            "config": self.config,
            "training_log": prev_log + new_log_entries,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "datasets": ds_list,
        }
        if extra_meta:
            checkpoint.update(extra_meta)

        torch.save(checkpoint, self.ckpt_path)
        self.training_status["log"].append(f"Checkpoint saved: {self.ckpt_path}")

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    def _handle_status(self) -> List[Dict[str, Any]]:
        """Return current training status."""
        return [{
            "status": "running" if self.training_status["running"] else "idle",
            "message": self.training_status["message"],
            "log": self.training_status["log"],
        }]
