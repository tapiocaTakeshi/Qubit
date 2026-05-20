#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) utilities for training preference-aligned models.
Supports loading preference datasets and computing DPO loss.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


def load_preference_data_from_dict(data: List[Dict]) -> List[Dict]:
    """Load preference data from list of dicts with 'prompt', 'chosen', 'rejected' fields."""
    preference_samples = []
    for item in data:
        if "prompt" in item and "chosen" in item and "rejected" in item:
            preference_samples.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
    return preference_samples


def load_preference_data_from_hf(dataset_id: str, split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
    """Load preference data from HuggingFace dataset (e.g., argilla/ultrafeedback-binarized)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split=split)

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        preference_samples = []
        for item in ds:
            prompt = item.get("prompt", "") or item.get("instruction", "") or item.get("question", "")
            chosen = item.get("chosen", "") or item.get("best", "")
            rejected = item.get("rejected", "") or item.get("worst", "")

            if prompt and chosen and rejected:
                preference_samples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        return preference_samples
    except Exception as e:
        print(f"Error loading HF dataset {dataset_id}: {e}")
        return []


def format_qa_preference(prompt: str, chosen: str, rejected: str) -> Tuple[str, str]:
    """Format preference pair as QA-style prompts with responses."""
    chosen_full = f"質問: {prompt}\n回答: {chosen}"
    rejected_full = f"質問: {prompt}\n回答: {rejected}"
    return chosen_full, rejected_full


def tokenize_preference_pair(prompt: str, chosen: str, rejected: str, tokenizer, max_seq_len: int) -> Dict:
    """Tokenize a preference pair (prompt + chosen response) and (prompt + rejected response)."""
    chosen_text = f"質問: {prompt}\n回答: {chosen}"
    rejected_text = f"質問: {prompt}\n回答: {rejected}"

    chosen_ids = tokenizer.encode(chosen_text, add_special=True, add_boundary=True)
    rejected_ids = tokenizer.encode(rejected_text, add_special=True, add_boundary=True)

    chosen_ids = chosen_ids[:max_seq_len] if len(chosen_ids) > max_seq_len else chosen_ids
    rejected_ids = rejected_ids[:max_seq_len] if len(rejected_ids) > max_seq_len else rejected_ids

    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_len": len(chosen_ids),
        "rejected_len": len(rejected_ids),
    }


def compute_dpo_loss(
    model_logits_chosen: torch.Tensor,
    model_logits_rejected: torch.Tensor,
    labels_chosen: torch.Tensor,
    labels_rejected: torch.Tensor,
    beta: float = 0.5,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DPO (Direct Preference Optimization) loss.

    Args:
        model_logits_chosen: Logits from model for chosen response [batch_size, seq_len, vocab_size]
        model_logits_rejected: Logits from model for rejected response [batch_size, seq_len, vocab_size]
        labels_chosen: Labels for chosen response [batch_size, seq_len]
        labels_rejected: Labels for rejected response [batch_size, seq_len]
        beta: Temperature parameter for DPO (higher = stronger preference learning)
        ignore_index: Token index to ignore in loss computation

    Returns:
        dpo_loss: DPO loss scalar
        log_probs_chosen: Log probabilities of chosen response
        log_probs_rejected: Log probabilities of rejected response
    """

    # Shift for next token prediction
    shift_logits_chosen = model_logits_chosen[..., :-1, :].contiguous()
    shift_logits_rejected = model_logits_rejected[..., :-1, :].contiguous()
    shift_labels_chosen = labels_chosen[..., 1:].contiguous()
    shift_labels_rejected = labels_rejected[..., 1:].contiguous()

    # Compute log probabilities
    log_probs_chosen = -F.cross_entropy(
        shift_logits_chosen.view(-1, shift_logits_chosen.size(-1)),
        shift_labels_chosen.view(-1),
        reduction='none',
        ignore_index=ignore_index
    ).view(shift_labels_chosen.shape)

    log_probs_rejected = -F.cross_entropy(
        shift_logits_rejected.view(-1, shift_logits_rejected.size(-1)),
        shift_labels_rejected.view(-1),
        reduction='none',
        ignore_index=ignore_index
    ).view(shift_labels_rejected.shape)

    # Sum log probs for each sequence (ignoring padding)
    mask_chosen = (shift_labels_chosen != ignore_index).float()
    mask_rejected = (shift_labels_rejected != ignore_index).float()

    log_probs_chosen_sum = (log_probs_chosen * mask_chosen).sum(dim=1) / mask_chosen.sum(dim=1).clamp(min=1e-8)
    log_probs_rejected_sum = (log_probs_rejected * mask_rejected).sum(dim=1) / mask_rejected.sum(dim=1).clamp(min=1e-8)

    # DPO loss: -log(sigmoid(beta * (log_p_chosen - log_p_rejected)))
    # This encourages log_p_chosen > log_p_rejected
    log_odds = beta * (log_probs_chosen_sum - log_probs_rejected_sum)
    dpo_loss = -F.logsigmoid(log_odds).mean()

    return dpo_loss, log_probs_chosen_sum, log_probs_rejected_sum


def compute_dpo_metrics(
    log_probs_chosen: torch.Tensor,
    log_probs_rejected: torch.Tensor,
) -> Dict[str, float]:
    """Compute metrics for DPO training monitoring."""
    accuracy = (log_probs_chosen > log_probs_rejected).float().mean().item()
    avg_log_ratio = (log_probs_chosen - log_probs_rejected).mean().item()

    return {
        "accuracy": accuracy,
        "avg_log_ratio": avg_log_ratio,
        "log_probs_chosen": log_probs_chosen.mean().item(),
        "log_probs_rejected": log_probs_rejected.mean().item(),
    }


def pad_sequence_pair(
    chosen_ids: List[int],
    rejected_ids: List[int],
    pad_id: int,
    max_len: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """Pad a pair of sequences to the same length."""
    if max_len is None:
        max_len = max(len(chosen_ids), len(rejected_ids))

    chosen_ids = chosen_ids[:max_len]
    rejected_ids = rejected_ids[:max_len]

    chosen_ids += [pad_id] * (max_len - len(chosen_ids))
    rejected_ids += [pad_id] * (max_len - len(rejected_ids))

    return chosen_ids, rejected_ids


def batch_preference_samples(
    samples: List[Dict],
    batch_size: int,
    tokenizer,
    max_seq_len: int,
    device: torch.device,
) -> List[Dict]:
    """Prepare batches of preference samples for training."""
    batches = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]

        batch_data = {
            "input_ids_chosen": [],
            "input_ids_rejected": [],
            "labels_chosen": [],
            "labels_rejected": [],
        }

        for sample in batch_samples:
            pair = tokenize_preference_pair(
                sample["prompt"],
                sample["chosen"],
                sample["rejected"],
                tokenizer,
                max_seq_len
            )

            chosen_ids, rejected_ids = pad_sequence_pair(
                pair["chosen_ids"],
                pair["rejected_ids"],
                tokenizer.pad_id,
                max_len=max_seq_len
            )

            batch_data["input_ids_chosen"].append(chosen_ids)
            batch_data["input_ids_rejected"].append(rejected_ids)
            batch_data["labels_chosen"].append(chosen_ids)
            batch_data["labels_rejected"].append([-100 if id == tokenizer.pad_id else id for id in rejected_ids])

        # Convert to tensors
        batch_data["input_ids_chosen"] = torch.tensor(batch_data["input_ids_chosen"], dtype=torch.long, device=device)
        batch_data["input_ids_rejected"] = torch.tensor(batch_data["input_ids_rejected"], dtype=torch.long, device=device)
        batch_data["labels_chosen"] = torch.tensor(batch_data["labels_chosen"], dtype=torch.long, device=device)
        batch_data["labels_rejected"] = torch.tensor(batch_data["labels_rejected"], dtype=torch.long, device=device)

        batches.append(batch_data)

    return batches


def create_preference_examples() -> List[Dict]:
    """Create hand-crafted preference examples for DPO training."""
    examples = [
        {
            "prompt": "日本の首都は？",
            "chosen": "日本の首都は東京です。東京都にあります。",
            "rejected": "日本の首都は京都です。",
        },
        {
            "prompt": "富士山の高さは？",
            "chosen": "富士山の高さは3,776メートルで、日本で最も高い山です。",
            "rejected": "富士山は約3,000メートルくらいです。",
        },
        {
            "prompt": "プログラミングとは？",
            "chosen": "プログラミングはコンピュータに実行させるコマンドを書くプロセスです。Python、Java、C++などの言語があります。",
            "rejected": "プログラミングはコンピュータを使うことです。",
        },
        {
            "prompt": "機械学習とは？",
            "chosen": "機械学習はデータからパターンを自動的に学習し、予測や判断を行うAI技術です。教師あり学習と教師なし学習があります。",
            "rejected": "機械学習は人工知能です。",
        },
        {
            "prompt": "DNAの役割は？",
            "chosen": "DNAは生物の遺伝情報を保持する分子で、二重らせん構造を持ち、4つの塩基（A, T, G, C）から構成されています。",
            "rejected": "DNAはタンパク質です。",
        },
    ]

    return examples
