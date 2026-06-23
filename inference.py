#!/usr/bin/env python3
"""
neuroQ ローカル推論スクリプト

使用方法:
    python inference.py "量子コンピュータとは何ですか？"
    python inference.py --prompt "AIの未来について" --max_new_tokens 200 --temperature 0.8
    python inference.py --interactive
    python inference.py --checkpoint path/to/checkpoint.pt "質問テキスト"
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    migrate_legacy_state_dict,
)

# デフォルト設定（チェックポイントがない場合に使用）
DEFAULT_CONFIG = {
    "vocab_size": 8000,
    "embed_dim": 512,
    "hidden_dim": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 10000,
    "entangle_strength": 0.5,
    "dropout": 0.1,
}

TOKENIZER_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model"),
    os.path.join(os.path.dirname(__file__), "neuroq_tokenizer_8k.model"),
    "neuroq_tokenizer.model",
    "neuroq_tokenizer_8k.model",
]

CHECKPOINT_CANDIDATES = [
    os.environ.get("CHECKPOINT_PATH", ""),
    os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt"),
    os.path.join(os.path.dirname(__file__), "qbnn_checkpoint.pt"),
    "neuroq_checkpoint.pt",
    "qbnn_checkpoint.pt",
]


def find_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def load_model(checkpoint_path=None):
    """モデルとトークナイザーをロード。チェックポイントがなければ新規初期化。"""
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ckpt_path = checkpoint_path or find_file(CHECKPOINT_CANDIDATES)
    tok_path = find_file(TOKENIZER_CANDIDATES)

    if ckpt_path:
        print(f"[neuroQ] チェックポイントを読み込み中: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        config = checkpoint.get("config", dict(DEFAULT_CONFIG))
    else:
        print("[neuroQ] チェックポイントが見つかりません。新規モデルで初期化します。")
        checkpoint = None
        config = dict(DEFAULT_CONFIG)

    if tok_path:
        print(f"[neuroQ] トークナイザー: {tok_path}")
    else:
        print("[neuroQ] トークナイザーモデルが見つかりません。フォールバックを使用します。")

    tokenizer = NeuroQuantumTokenizer(
        vocab_size=config["vocab_size"],
        model_file=tok_path,
    )

    # トークナイザーの実際の語彙サイズで設定を上書き
    tok_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    if tok_vocab and tok_vocab != config["vocab_size"]:
        print(f"[neuroQ] vocab_size 調整: {config['vocab_size']} → {tok_vocab}")
        config["vocab_size"] = tok_vocab

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

    model = NeuroQuantum(config=nq_config).to(device)

    if checkpoint:
        migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
        model_state = model.state_dict()
        for key in list(migrated.keys()):
            if key in model_state and migrated[key].shape != model_state[key].shape:
                new_tensor = model_state[key].clone()
                slices = tuple(
                    slice(0, min(o, n))
                    for o, n in zip(migrated[key].shape, model_state[key].shape)
                )
                new_tensor[slices] = migrated[key][slices]
                migrated[key] = new_tensor
        model.load_state_dict(migrated, strict=False)

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    device_label = str(device).upper()
    print(f"[neuroQ] モデル準備完了: {n_params:,} パラメータ / デバイス: {device_label}")
    print(f"[neuroQ] アーキテクチャ: embed={nq_config.embed_dim}, "
          f"hidden={nq_config.hidden_dim}, heads={nq_config.num_heads}, "
          f"layers={nq_config.num_layers}, λ={nq_config.lambda_entangle}")

    return model, tokenizer, config, device


def generate(
    model,
    tokenizer,
    config,
    device,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> str:
    """NeuroQuantum モデルでテキストを生成する。"""
    # QA 形式にフォーマット（学習データと一致させる）
    raw = prompt.strip()
    if not raw.startswith("質問:") and not raw.startswith("回答:"):
        raw = f"質問: {raw}\n回答:"

    content_ids = tokenizer.encode(raw, add_special=False)
    tokens = [tokenizer.bof_id, tokenizer.bos_id] + content_ids
    if not content_ids:
        return ""

    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = list(tokens)
    max_seq_len = config["max_seq_len"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq = input_tensor[:, -max_seq_len:]
            logits = model(seq)[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals = torch.topk(logits, k)[0]
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits[0, sorted_indices[remove]] = float("-inf")

            if repetition_penalty > 1.0 and len(generated) > 1:
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        if logits[0, prev] > 0:
                            logits[0, prev] /= repetition_penalty
                        else:
                            logits[0, prev] *= repetition_penalty

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            nxt_id = nxt.item()

            if nxt_id in (tokenizer.eos_id, tokenizer.eof_id):
                break
            if nxt_id in (tokenizer.pad_id, tokenizer.bof_id):
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                continue

            generated.append(nxt_id)
            input_tensor = torch.cat([input_tensor, nxt], dim=1)

    return tokenizer.decode(generated[len(tokens):], skip_special=True)


def interactive_mode(model, tokenizer, config, device, args):
    """対話型推論モード。"""
    print("\nneuroQ 対話モード（終了: quit / exit / q）")
    print("-" * 50)
    while True:
        try:
            prompt = input("\nあなた: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q", "終了"):
            print("終了します。")
            break
        result = generate(
            model, tokenizer, config, device, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\nneuroQ: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="neuroQ ローカル推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("prompt", nargs="?", default=None, help="推論プロンプト")
    parser.add_argument("--prompt", dest="prompt_flag", default=None, help="推論プロンプト（オプション形式）")
    parser.add_argument("--checkpoint", default=None, help="チェックポイントファイルのパス (.pt)")
    parser.add_argument("--interactive", "-i", action="store_true", help="対話モードで起動")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成トークン数 (デフォルト: 100)")
    parser.add_argument("--temperature", type=float, default=0.7, help="サンプリング温度 (デフォルト: 0.7)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K サンプリング (デフォルト: 40)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus サンプリング閾値 (デフォルト: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="繰り返しペナルティ (デフォルト: 1.3)")

    args = parser.parse_args()

    # プロンプト解決（位置引数 > --prompt オプション）
    prompt = args.prompt or args.prompt_flag

    if not args.interactive and not prompt:
        parser.print_help()
        print("\n例:")
        print('  python inference.py "量子コンピュータとは何ですか？"')
        print('  python inference.py --interactive')
        sys.exit(0)

    model, tokenizer, config, device = load_model(args.checkpoint)

    if args.interactive:
        interactive_mode(model, tokenizer, config, device, args)
    else:
        print(f"\nプロンプト: {prompt}")
        print("-" * 50)
        result = generate(
            model, tokenizer, config, device, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"生成結果:\n{result}")


if __name__ == "__main__":
    main()
