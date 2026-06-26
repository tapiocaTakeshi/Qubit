#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use proven approach: load from OASST script and modify dataset
from train_small_oasst_ja import *

DATASET_ID = "llm-jp/databricks-dolly-15k-ja"
DEFAULT_CHECKPOINT_NAME = "neuroq_dolly_15k_ja_checkpoint.pt"

def extract_dolly_texts(ds, max_samples: int):
    """databricks-dolly-15k-ja から QA テキストを抽出。"""
    texts = []
    n = len(ds) if max_samples is None or max_samples <= 0 else min(max_samples, len(ds))
    for row in ds.select(range(n)):
        inst = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        resp = (row.get("response") or "").strip()
        if not resp:
            continue
        parts = []
        if inst:
            parts.append(f"質問: {inst}")
        if ctx:
            parts.append(f"背景: {ctx}")
        parts.append(f"回答: {resp}")
        text = "\n".join(parts).strip()
        if len(text) > 4:
            texts.append(text)
    return texts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--ckpt-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--tokenizer-prefix", default="neuroq_dolly_15k_ja_tokenizer")
    args = parser.parse_args()
    
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    progress.info(f"Device: {device}")
    
    progress.info(f"=== Loading {DATASET_ID} ===")
    ds = safe_load_dataset(DATASET_ID, split="train")
    
    CONFIG = get_model_config_by_size("small", vocab_size=args.vocab_size)
    batch_size = args.batch_size or CONFIG["batch_size"]
    max_seq_len = args.max_seq_len or CONFIG["max_seq_len"]
    
    effective_max_samples = len(ds) if (args.max_samples is None or args.max_samples <= 0) else min(args.max_samples, len(ds))
    progress.info(f"=== Extracting {effective_max_samples} texts ===")
    texts = extract_dolly_texts(ds, args.max_samples)
    progress.info(f"Extracted: {len(texts)} texts")
    
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ckpt_name)
    resume_ckpt = None
    
    progress.info("=== Building SentencePiece tokenizer ===")
    tokenizer_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tokenizer_prefix)
    tokenizer_model_path = tokenizer_prefix + ".model"
    tokenizer = NeuroQuantumTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(texts, model_prefix=tokenizer_prefix, character_coverage=0.9995)
    actual_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    CONFIG["vocab_size"] = actual_vocab
    progress.info(f"Vocab size: {actual_vocab}")
    
    progress.info("=== Building NeuroQuantum (small) ===")
    nq_config = NeuroQuantumConfig(
        vocab_size=actual_vocab,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        max_seq_len=max_seq_len,
        dropout=CONFIG["dropout"],
        lambda_entangle=CONFIG["entangle_strength"],
    )
    model = NeuroQuantum(config=nq_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    progress.info(f"Parameters: {n_params:,}")
    
    progress.info("=== Tokenizing ===")
    sequences = tokenize_texts(texts, tokenizer, max_seq_len)
    progress.info(f"Training sequences: {len(sequences)}")
    
    progress.info("=== Training ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    def save_checkpoint(epoch=None, batch=None, loss=None, final=False):
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "vocab_size": actual_vocab,
                "embed_dim": CONFIG["embed_dim"],
                "hidden_dim": CONFIG["hidden_dim"],
                "num_heads": CONFIG["num_heads"],
                "num_layers": CONFIG["num_layers"],
                "max_seq_len": max_seq_len,
            },
            "dataset": DATASET_ID,
            "trained_at": str(__import__('datetime').datetime.now()),
        }
        torch.save(checkpoint, ckpt_path)
        size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        progress.info(f"Saved: {ckpt_path} ({size_mb:.1f} MB)")
    
    training_log = []
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, sequences, tokenizer, optimizer, batch_size, max_seq_len, epoch, device, save_every=0, on_save=save_checkpoint)
        progress.info(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.6f}")
        training_log.append({"epoch": epoch+1, "loss": avg_loss})
    
    save_checkpoint(final=True)
    progress.info(f"Training complete! Checkpoint: {ckpt_path}")
