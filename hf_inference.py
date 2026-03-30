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

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(ENDPOINT_URL, data=data, headers=headers, method="POST")

    max_retries = 12  # up to ~6 minutes for cold start
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            if e.code == 503 and attempt < max_retries - 1:
                wait = min(10 + attempt * 5, 30)
                print(f"Endpoint unavailable (503), retrying in {wait}s... ({attempt + 1}/{max_retries})", file=sys.stderr)
                import time as _time
                _time.sleep(wait)
                # Rebuild request (urlopen consumes the data)
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
    parser.add_argument("prompt", nargs="?", help="Prompt text for inference")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--status", action="store_true", help="Check endpoint status")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON response")
    args = parser.parse_args()

    if args.status:
        payload = {"inputs": "__status__", "parameters": {"action": "status"}}
        headers = {"Content-Type": "application/json"}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(ENDPOINT_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        return

    if args.interactive:
        interactive_mode(args)
        return

    if not args.prompt:
        parser.print_help()
        sys.exit(1)

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


if __name__ == "__main__":
    main()
