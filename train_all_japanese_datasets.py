#!/usr/bin/env python3
"""
複数の日本語データセットで統合学習するスクリプト。

各データセットをエポック数1で順番に学習し、1つのモデルに統合。
チェックポイントは各データセット学習後に保存される。

使い方:
    python train_all_japanese_datasets.py --model-size small
    python train_all_japanese_datasets.py --model-size medium --max-samples 10000
    python train_all_japanese_datasets.py --model-size large --upload --repo-id tapiocatakeshi/Qubit
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_hf_dataset import (
    extract_texts,
    tokenize_texts,
    train_epoch,
    upload_checkpoint_to_hf,
    merge_checkpoint_to_main,
    get_default_names,
    auto_detect_split,
    is_japanese_dataset,
    optimize_for_japanese_dataset,
)
from dataset_utils import safe_load_dataset
from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)
from progress_logger import ProgressLogger

DEFAULT_REPO_ID = "tapiocatakeshi/Qubit"

progress = ProgressLogger("train_all_japanese_datasets")

JAPANESE_DATASETS = [
    {
        "id": "llm-jp/databricks-dolly-15k-ja",
        "split": None,
        "max_samples": None,
        "description": "Databricks Dolly 15K 日本語版",
    },
    {
        "id": "wikimedia/wikipedia",
        "split": "ja",
        "max_samples": None,
        "description": "Wikimedia Wikipedia 日本語版",
    },
    {
        "id": "mc4",
        "split": "ja",
        "max_samples": None,
        "description": "MC4 日本語コーパス",
    },
    {
        "id": "oscar-corpus/OSCAR-2301",
        "split": "ja",
        "max_samples": None,
        "description": "OSCAR 2301 日本語版",
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-size",
        default="small",
        help="モデルサイズ: small, medium, large, xlarge, ... (デフォルト: small)",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="各データセットの最大サンプル数（Noneで全件）",
    )
    p.add_argument(
        "--gradient-accumulation-steps", type=int, default=8
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--use-bf16", action="store_true", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--enable-warmup", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument(
        "--save-every",
        type=int,
        default=100,
    )
    p.add_argument("--ckpt-name", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reset-epochs", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    p.add_argument("--tokenizer-prefix", default=None)
    p.add_argument(
        "--dataset-filter",
        default=None,
        help="学習するデータセットをフィルタ（例: dolly, wikipedia）",
    )
    return p.parse_args()


def get_filtered_datasets(filter_str: str = None):
    """フィルタ文字列に基づいてデータセットをフィルタリング。"""
    if not filter_str:
        return JAPANESE_DATASETS

    filtered = []
    for ds in JAPANESE_DATASETS:
        if filter_str.lower() in ds["id"].lower() or filter_str.lower() in ds["description"].lower():
            filtered.append(ds)

    if not filtered:
        progress.warning(f"フィルタ '{filter_str}' に一致するデータセットがありません。")
        return JAPANESE_DATASETS

    return filtered


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress.info(f"Device: {device}")

    ckpt_name = args.ckpt_name or get_default_names(args.model_size)[0]
    tokenizer_prefix_arg = args.tokenizer_prefix or get_default_names(args.model_size)[1]

    CONFIG = get_model_config_by_size(args.model_size, vocab_size=args.vocab_size)
    CONFIG = optimize_for_japanese_dataset(CONFIG, "japanese_unified")

    batch_size = args.batch_size or CONFIG["batch_size"]
    max_seq_len = args.max_seq_len or CONFIG["max_seq_len"]
    gradient_accumulation_steps = args.gradient_accumulation_steps
    use_bf16 = args.use_bf16
    gradient_checkpointing = args.gradient_checkpointing

    progress.info(f"Model size: {args.model_size}")
    progress.info(f"Training config: batch_size={batch_size}, max_seq_len={max_seq_len}")

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_name)
    tokenizer_prefix = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), tokenizer_prefix_arg
    )
    tokenizer_model_path = tokenizer_prefix + ".model"

    resume_ckpt = None
    if args.resume and os.path.isfile(ckpt_path):
        progress.info(f"=== Resuming from checkpoint: {ckpt_path} ===")
        resume_ckpt = torch.load(ckpt_path, map_location="cpu")

    datasets = get_filtered_datasets(args.dataset_filter)
    progress.info(f"\n🇯🇵 {len(datasets)} 個の日本語データセットで学習します")
    for i, ds_info in enumerate(datasets, 1):
        progress.info(f"  {i}. {ds_info['description']} ({ds_info['id']})")

    all_texts = []
    for ds_info in datasets:
        try:
            progress.info(f"\n=== Loading {ds_info['description']} ===")
            split = ds_info["split"] or auto_detect_split(ds_info["id"])
            ds = safe_load_dataset(ds_info["id"], split=split)

            max_samples = args.max_samples or ds_info.get("max_samples")
            texts = extract_texts(ds, max_samples)
            progress.log_dataset_loaded(ds_info["id"], len(texts))

            if texts:
                all_texts.extend(texts)
                progress.info(f"✅ {len(texts)} テキストを追加（総計: {len(all_texts)}）")
            else:
                progress.warning(f"⚠️ {ds_info['id']} からテキストを取得できません")

        except Exception as e:
            progress.warning(f"⚠️ {ds_info['id']} の読み込みエラー: {e}")
            continue

    if not all_texts:
        raise RuntimeError("どのデータセットからもテキストを取得できませんでした")

    progress.info(f"\n=== トークナイザー構築 ===")
    if resume_ckpt is not None and os.path.isfile(tokenizer_model_path):
        progress.info(f"既存トークナイザーをロード: {tokenizer_model_path}")
        ckpt_vocab = resume_ckpt.get("config", {}).get("vocab_size", args.vocab_size)
        tokenizer = NeuroQuantumTokenizer(vocab_size=ckpt_vocab, model_file=tokenizer_model_path)
    else:
        progress.info("新規トークナイザーを構築")
        tokenizer = NeuroQuantumTokenizer(vocab_size=args.vocab_size)
        tokenizer.build_vocab(all_texts, model_prefix=tokenizer_prefix, character_coverage=0.9995)

    actual_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    CONFIG["vocab_size"] = actual_vocab
    progress.info(f"Vocabulary size: {actual_vocab}")

    progress.info(f"\n=== NeuroQuantum モデル構築 ===")
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

    if resume_ckpt is not None and "model_state" in resume_ckpt:
        model.load_state_dict(resume_ckpt["model_state"], strict=False)
        progress.info("チェックポイントから重みをロード")

    progress.info(f"\n=== トークナイズ ===")
    sequences = tokenize_texts(all_texts, tokenizer, max_seq_len)
    progress.info(f"学習シーケンス数: {len(sequences)}")

    progress.info(f"\n=== 学習開始 ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = (len(sequences) // (batch_size * gradient_accumulation_steps)) * 1
    if args.enable_warmup:
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = min(args.warmup_steps, total_steps // 10)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

        scheduler = LambdaLR(optimizer, lr_lambda)
        progress.info(f"ウォームアップスケジューラーを有効化（warmup_steps={warmup_steps}）")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        progress.info("CosineAnnealing スケジューラーを使用")

    training_log = []
    if resume_ckpt is not None:
        training_log = list(resume_ckpt.get("training_log", []))

    def save_checkpoint_func(epoch, batch=None, loss=None, final=False):
        """チェックポイントを保存。

        毎回、中間チェックポイントをメインチェックポイントにマージします。
        """
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": nq_config.__dict__,
            "training_log": training_log,
            "batch": batch,
            "epoch": epoch,
            "loss": loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if batch is not None and not final:
            ckpt_dir = os.path.join(os.path.dirname(ckpt_path), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_basename = os.path.basename(ckpt_path)
            ckpt_name_no_ext = os.path.splitext(ckpt_basename)[0]
            intermediate_ckpt_path = os.path.join(
                ckpt_dir, f"{ckpt_name_no_ext}_epoch{epoch:03d}_batch{batch:06d}.pt"
            )
            torch.save(checkpoint, intermediate_ckpt_path)
            progress.info(f"Checkpoint saved: {intermediate_ckpt_path}")

            merge_checkpoint_to_main(intermediate_ckpt_path, ckpt_path)
        else:
            torch.save(checkpoint, ckpt_path)
            progress.info(f"Checkpoint saved: {ckpt_path}")

    progress.start_training(
        epochs=1,
        total_sequences=len(sequences),
        batch_size=batch_size,
        lr=args.lr,
    )

    avg_loss = train_epoch(
        model,
        sequences,
        tokenizer,
        optimizer,
        batch_size,
        max_seq_len,
        epoch=0,
        device=device,
        save_every=args.save_every,
        on_save=save_checkpoint_func,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_bf16=use_bf16,
        scheduler=scheduler,
        max_grad_norm=args.max_grad_norm,
    )

    if not args.enable_warmup:
        scheduler.step()

    training_log.append({"epoch": 0, "avg_loss": avg_loss, "datasets": len(datasets)})
    save_checkpoint_func(epoch=0, loss=avg_loss, final=True)

    progress.end_training()

    if args.upload and args.hf_token:
        upload_checkpoint_to_hf(ckpt_path, args.repo_id, args.hf_token, tokenizer_model_path)

    progress.info("\n=== 学習完了 ===")
    progress.info(f"✅ {len(datasets)} 個のデータセット（エポック数1）で統合学習しました")
    progress.info(f"📊 最終Loss: {avg_loss:.4f}")
    progress.info(f"💾 Checkpoint: {ckpt_path}")
    progress.info(f"🔤 Tokenizer: {tokenizer_model_path}")


if __name__ == "__main__":
    main()
