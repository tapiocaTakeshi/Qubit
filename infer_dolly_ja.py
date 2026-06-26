#!/usr/bin/env python3
"""
databricks-dolly-15k-ja で学習した NeuroQuantum small モデルの推論スクリプト。

学習時と同じ会話フォーマット

    <USER> {指示}
    <ASSISTANT>

を入力としてモデルに与え、<ASSISTANT> 以降の応答を生成する。

使い方:
    python infer_dolly_ja.py "日本の首都はどこですか？"
    python infer_dolly_ja.py --interactive
    python infer_dolly_ja.py --checkpoint neuroq_small_dolly_ja_checkpoint.pt \
        --tokenizer neuroq_small_dolly_ja_tokenizer.model "富士山の高さは？"
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import load_model  # チェックポイント/トークナイザーのロードを再利用

DEFAULT_CHECKPOINT = "neuroq_small_dolly_ja_checkpoint.pt"
DEFAULT_TOKENIZER = "neuroq_small_dolly_ja_tokenizer.model"


def generate_dolly(
    model,
    tokenizer,
    config,
    device,
    instruction: str,
    context: str = "",
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> str:
    """dolly 学習フォーマット (<USER>/<ASSISTANT>) で応答を生成する。"""
    parts = [f"<USER> {instruction.strip()}"]
    if context and context.strip():
        parts.append(context.strip())
    parts.append("<ASSISTANT>")
    prompt = "\n".join(parts)

    content_ids = tokenizer.encode(prompt, add_special=False)
    if not content_ids:
        return ""
    tokens = [tokenizer.bof_id, tokenizer.bos_id] + content_ids
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = list(tokens)
    max_seq_len = config["max_seq_len"]

    model.eval()
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

    return tokenizer.decode(generated[len(tokens):], skip_special=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", default=None, help="指示文 (instruction)")
    parser.add_argument("--context", default="", help="補助的な文脈 (任意)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    args = parser.parse_args()

    model, tokenizer, config, device = load_model(args.checkpoint, args.tokenizer)

    def run(instruction, context=""):
        return generate_dolly(
            model, tokenizer, config, device, instruction, context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    if args.interactive:
        print("\ndolly-ja 対話モード（終了: quit / exit / q）")
        print("-" * 50)
        while True:
            try:
                q = input("\n指示: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n終了します。")
                break
            if not q or q.lower() in ("quit", "exit", "q", "終了"):
                break
            print(f"\n応答: {run(q)}")
    elif args.prompt:
        print(f"\n指示: {args.prompt}")
        if args.context:
            print(f"文脈: {args.context}")
        print("-" * 50)
        print(f"応答:\n{run(args.prompt, args.context)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
