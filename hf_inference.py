#!/usr/bin/env python3
"""
HuggingFace Inference Endpoint client for neuroQ.

Usage:
    python hf_inference.py "あなたの質問"
    python hf_inference.py --interactive
    python hf_inference.py --prompt "質問内容" --max_new_tokens 200 --temperature 0.8
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))


def send_request(payload: dict, timeout: int = 600) -> dict:
    """Send a request to the HuggingFace Endpoint with retry logic."""
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(ENDPOINT_URL, data=data, headers=headers, method="POST")

    max_retries = 12
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            if e.code == 503 and attempt < max_retries - 1:
                wait = min(10 + attempt * 5, 30)
                print(f"Endpoint unavailable (503), retrying in {wait}s... ({attempt + 1}/{max_retries})", file=sys.stderr)
                import time as _time
                _time.sleep(wait)
                req = urllib.request.Request(ENDPOINT_URL, data=data, headers=headers, method="POST")
                continue
            print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
            print(f"Response: {error_body}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                wait = min(10 + attempt * 5, 30)
                print(f"Connection error, retrying in {wait}s... ({attempt + 1}/{max_retries})", file=sys.stderr)
                import time as _time
                _time.sleep(wait)
                req = urllib.request.Request(ENDPOINT_URL, data=data, headers=headers, method="POST")
                continue
            print(f"Connection error: {e.reason}", file=sys.stderr)
            sys.exit(1)


def infer(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> dict:
    """Send inference request to the HuggingFace Endpoint."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        },
    }
    return send_request(payload)


def train(action: str = "train_qa_dataset", **params) -> dict:
    """Send training request to the HuggingFace Endpoint."""
    payload = {
        "inputs": f"__{action}__",
        "parameters": {"action": action, **params},
    }
    return send_request(payload, timeout=600)


def print_result(result):
    """Pretty-print inference result."""
    if isinstance(result, list):
        for item in result:
            text = item.get("generated_text", "")
            print(f"\n--- Generated Text ---\n{text}")
            if "debug" in item:
                print(f"\n--- Debug Info ---")
                debug = item["debug"]
                print(f"  Input length: {debug.get('input_len')}")
                print(f"  Generated tokens: {debug.get('generated_token_count')}")
    elif isinstance(result, dict):
        text = result.get("generated_text", "")
        print(f"\n--- Generated Text ---\n{text}")
    else:
        print(result)


def interactive_mode(args):
    """Run in interactive chat mode."""
    print("neuroQ Interactive Mode (type 'quit' to exit)")
    print(f"Endpoint: {ENDPOINT_URL}")
    print("-" * 50)
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        result = infer(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print_result(result)


def main():
    parser = argparse.ArgumentParser(description="neuroQ HuggingFace Inference Client")
    sub = parser.add_subparsers(dest="command")

    # --- infer (default) ---
    p_infer = sub.add_parser("infer", help="Run inference")
    p_infer.add_argument("prompt", help="Prompt text")
    p_infer.add_argument("--max_new_tokens", type=int, default=100)
    p_infer.add_argument("--temperature", type=float, default=0.7)
    p_infer.add_argument("--top_k", type=int, default=40)
    p_infer.add_argument("--top_p", type=float, default=0.9)
    p_infer.add_argument("--repetition_penalty", type=float, default=1.3)
    p_infer.add_argument("--raw", action="store_true")

    # --- interactive ---
    p_int = sub.add_parser("chat", help="Interactive chat mode")
    p_int.add_argument("--max_new_tokens", type=int, default=100)
    p_int.add_argument("--temperature", type=float, default=0.7)
    p_int.add_argument("--top_k", type=int, default=40)
    p_int.add_argument("--top_p", type=float, default=0.9)
    p_int.add_argument("--repetition_penalty", type=float, default=1.3)

    # --- train ---
    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("--action", default="train_qa_dataset",
                         choices=["train", "train_qa", "train_qa_dataset", "train_split", "train_split_next"],
                         help="Training action (default: train_qa_dataset)")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--batch_size", type=int, default=None)
    p_train.add_argument("--grad_accum_steps", type=int, default=None)
    p_train.add_argument("--warmup_steps", type=int, default=None)
    p_train.add_argument("--max_samples_per_dataset", type=int, default=None)
    p_train.add_argument("--dataset_id", type=str, default=None)

    # --- status ---
    sub.add_parser("status", help="Check endpoint/model status")

    args = parser.parse_args()

    # Default: if no subcommand but has positional arg, treat as infer
    if args.command is None:
        # Check if first positional arg looks like a prompt
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Legacy: python hf_inference.py "prompt"
            result = infer(sys.argv[1])
            print_result(result)
        else:
            parser.print_help()
        return

    if args.command == "status":
        result = send_request({"inputs": "__status__", "parameters": {"action": "status"}}, timeout=60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "chat":
        interactive_mode(args)

    elif args.command == "infer":
        result = infer(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        if args.raw:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)

    elif args.command == "train":
        params = {}
        for key in ("epochs", "lr", "batch_size", "grad_accum_steps", "warmup_steps",
                     "max_samples_per_dataset", "dataset_id"):
            val = getattr(args, key, None)
            if val is not None:
                params[key] = val
        print(f"Starting training (action={args.action})...")
        result = train(action=args.action, **params)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
