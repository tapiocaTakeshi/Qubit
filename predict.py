"""Cog Predictor for NeuroQuantum / QBNN-Transformer model."""

import os
import torch
import torch.nn.functional as F
from cog import BasePredictor, Input

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, migrate_legacy_state_dict

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isfile(CKPT_PATH):
            raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

        checkpoint = torch.load(CKPT_PATH, map_location="cpu")
        self.config = checkpoint["config"]

        self.tokenizer = NeuroQuantumTokenizer(
            vocab_size=self.config["vocab_size"], model_file=TOKENIZER_PATH
        )

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

        self.model = NeuroQuantum(nq_config)
        migrated = migrate_legacy_state_dict(checkpoint["model_state"], self.model)
        self.model.load_state_dict(migrated)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=100, ge=1, le=512),
        temperature: float = Input(description="Sampling temperature", default=0.7, ge=0.01, le=2.0),
        top_k: int = Input(description="Top-k sampling (0 to disable)", default=40, ge=0, le=500),
        top_p: float = Input(description="Nucleus sampling threshold", default=0.9, ge=0.0, le=1.0),
        repetition_penalty: float = Input(description="Repetition penalty", default=1.3, ge=1.0, le=3.0),
    ) -> str:
        """Generate text from a prompt."""
        prompt = f"<s>{prompt}</s>"
        tokens = self.tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated = list(tokens)
        max_seq_len = self.config["max_seq_len"]

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

                if nxt_id in (self.tokenizer.eos_id, self.tokenizer.eof_id):
                    break
                if nxt_id in (self.tokenizer.pad_id, self.tokenizer.bof_id):
                    continue

                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)

        return self.tokenizer.decode(generated[len(tokens):], skip_special=True)
