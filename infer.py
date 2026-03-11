#!/usr/bin/env python3
"""
neuroQ 推論CLI

チェックポイントからモデルを読み込み、テキスト生成を行う。

使用例:
    # 単発推論
    python infer.py "量子コンピュータとは"

    # パラメータ指定
    python infer.py "AIの未来" --max-length 200 --temperature 0.7

    # 対話モード
    python infer.py --interactive
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_gpu_adaptive_config,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(SCRIPT_DIR, "neuroq_checkpoint.pt")
DEFAULT_TOK = os.path.join(SCRIPT_DIR, "neuroq_tokenizer.model")


def load_model(ckpt_path: str, tokenizer_path: str, device: str = "auto"):
    """チェックポイントからモデルとトークナイザーを読み込む。"""
    if not os.path.exists(ckpt_path):
        print(f"エラー: チェックポイントが見つかりません: {ckpt_path}")
        sys.exit(1)

    # デバイス選択
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    print(f"デバイス: {dev}")

    # チェックポイント読み込み
    print(f"チェックポイント読み込み中: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)
    config = checkpoint["config"]
    print(f"  アーキテクチャ: {config.get('architecture', 'unknown')}")
    print(f"  embed_dim={config['embed_dim']}, num_layers={config['num_layers']}, "
          f"num_heads={config['num_heads']}")

    # トークナイザー
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"])
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"  トークナイザー: {tokenizer_path} (語彙数={tokenizer.vocab_size})")
    else:
        print(f"  警告: トークナイザーが見つかりません: {tokenizer_path}")

    # モデル構築
    nq_config = NeuroQuantumConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.1),
        lambda_entangle=config.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(nq_config).to(dev)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {n_params:,}")

    return model, tokenizer, dev, config


def generate(
    model,
    tokenizer,
    device,
    config,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    no_repeat_ngram: int = 3,
) -> str:
    """テキスト生成を実行する。"""
    max_seq_len = config["max_seq_len"]

    tokens = tokenizer.encode(prompt, add_special=True)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = list(tokens)

    with torch.no_grad():
        for step in range(max_length):
            seq = input_tensor[:, -max_seq_len:]
            logits = model(seq)[:, -1, :].clone()
            vocab_size = logits.size(-1)

            # 繰り返しペナルティ（recency-weighted）
            window_size = min(100, len(generated))
            recent = generated[-window_size:]
            token_positions = {}
            for pos, tid in enumerate(recent):
                token_positions.setdefault(tid, []).append(pos)

            for tid, positions in token_positions.items():
                if tid >= vocab_size:
                    continue
                count = len(positions)
                most_recent = max(positions)
                recency = 0.5 + 0.5 * (most_recent / max(window_size - 1, 1))
                penalty = repetition_penalty ** (1 + count * 0.3 * recency)
                logits[0, tid] /= penalty

            # N-gram重複防止
            if no_repeat_ngram > 0 and len(generated) >= no_repeat_ngram:
                prefix = tuple(generated[-(no_repeat_ngram - 1):])
                banned = set()
                for i in range(len(generated) - no_repeat_ngram + 1):
                    if tuple(generated[i:i + no_repeat_ngram - 1]) == prefix:
                        banned.add(generated[i + no_repeat_ngram - 1])
                for tid in banned:
                    if tid < vocab_size:
                        logits[0, tid] = float('-inf')

            # 温度
            logits = logits / max(temperature, 0.05)

            # Top-K
            if top_k > 0:
                k = min(top_k, (logits[0] != float('-inf')).sum().item())
                if k > 0:
                    topk_vals, _ = torch.topk(logits[0], k)
                    logits[logits < topk_vals[-1]] = float('-inf')

            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                logits[0, sorted_idx[remove_mask]] = float('-inf')

            # サンプリング
            probs = F.softmax(logits[0], dim=-1)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / vocab_size
            next_token = torch.multinomial(probs, 1)
            nxt_id = next_token.item()

            if nxt_id == tokenizer.eos_id:
                break
            if nxt_id == tokenizer.pad_id:
                continue

            generated.append(nxt_id)
            input_tensor = torch.cat(
                [input_tensor, next_token.unsqueeze(0)], dim=1
            )

    output_ids = generated[len(tokens):]
    return tokenizer.decode(output_ids, skip_special=True).strip()


def interactive_mode(model, tokenizer, device, config, args):
    """対話モードで推論を実行する。"""
    print("\n=== neuroQ 対話モード ===")
    print("プロンプトを入力してください（'quit' または 'q' で終了）\n")

    while True:
        try:
            prompt = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not prompt or prompt.lower() in ("quit", "q", "exit"):
            print("終了します。")
            break

        result = generate(
            model, tokenizer, device, config,
            prompt=prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
        )
        print(f"neuroQ> {result}\n")


def main():
    parser = argparse.ArgumentParser(
        description="neuroQ 推論CLI - QBNN量子ニューラルネットワーク言語モデル"
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="生成のためのプロンプト文字列",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="対話モードで起動",
    )
    parser.add_argument(
        "--checkpoint", "-c", default=DEFAULT_CKPT,
        help=f"チェックポイントのパス (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--tokenizer", "-t", default=DEFAULT_TOK,
        help=f"トークナイザーモデルのパス (default: {DEFAULT_TOK})",
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="使用するデバイス (default: auto)",
    )
    parser.add_argument(
        "--max-length", "-n", type=int, default=100,
        help="最大生成トークン数 (default: 100)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="サンプリング温度 (default: 0.7)",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-Kサンプリング (default: 40)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-Pサンプリング (default: 0.9)",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.3,
        help="繰り返しペナルティ (default: 1.3)",
    )
    parser.add_argument(
        "--no-repeat-ngram", type=int, default=3,
        help="N-gram重複防止サイズ (default: 3)",
    )

    args = parser.parse_args()

    if args.prompt is None and not args.interactive:
        parser.print_help()
        print("\nエラー: プロンプトを指定するか、--interactive フラグを使用してください。")
        sys.exit(1)

    # モデル読み込み
    model, tokenizer, device, config = load_model(
        args.checkpoint, args.tokenizer, args.device
    )

    if args.interactive:
        interactive_mode(model, tokenizer, device, config, args)
    else:
        print(f"\nプロンプト: {args.prompt}")
        print(f"設定: temperature={args.temperature}, top_k={args.top_k}, "
              f"top_p={args.top_p}, max_length={args.max_length}")
        print("-" * 40)
        result = generate(
            model, tokenizer, device, config,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
        )
        print(f"生成結果: {result}")


if __name__ == "__main__":
    main()
