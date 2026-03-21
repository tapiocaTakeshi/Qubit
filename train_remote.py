#!/usr/bin/env python3
"""Remote training client for HuggingFace Inference Endpoints.

Sends training / status requests to the handler API.

Usage:
  # Check model status
  python train_remote.py --mode status

  # Train with QA pairs from a JSON file
  python train_remote.py --mode qa --qa_file qa_data.json

  # Train with QA pairs inline
  python train_remote.py --mode qa --qa_pairs '[{"question":"日本の首都は？","answer":"東京です。"}]'

  # General dataset training (existing)
  python train_remote.py --mode train

  # QA dataset training (existing, from HF datasets)
  python train_remote.py --mode train_qa_dataset
"""

import argparse
import json
import os
import sys
import time

import requests


def get_endpoint_url():
    """Get endpoint URL from environment or default."""
    return os.environ.get(
        "HF_ENDPOINT_URL",
        os.environ.get("ENDPOINT_URL", "http://localhost:8080"),
    )


def send_request(url, payload, token=None, timeout=600):
    """Send a POST request to the endpoint."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"[remote] POST {url}")
    print(f"[remote] action={payload.get('action', 'inference')}")

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def cmd_status(args):
    """Check model status."""
    url = get_endpoint_url()
    payload = {"action": "status"}
    result = send_request(url, payload, token=args.token)

    if isinstance(result, list):
        result = result[0]

    print("\n=== Model Status ===")
    print(f"Status:       {result.get('status', '?')}")
    print(f"Architecture: {result.get('architecture', '?')}")
    print(f"Parameters:   {result.get('model_params', '?'):,}")
    print(f"Message:      {result.get('message', '')}")

    ckpt = result.get("checkpoint")
    if ckpt:
        print(f"\nCheckpoint:   {ckpt.get('path', '?')}")
        print(f"  Size:       {ckpt.get('size_mb', '?')} MB")
        print(f"  Trained at: {ckpt.get('trained_at', '?')}")

    history = result.get("training_history", {})
    print(f"\nTraining history: {history.get('count', 0)} entries")
    if history.get("latest"):
        print(f"  Latest: {history['latest']}")

    split = result.get("split_training")
    if split:
        print(f"\nSplit training: chunk {split.get('last_completed_chunk', '?')}/{split.get('num_chunks', '?')}")

    print(f"\nTimestamp: {result.get('timestamp', '?')}")
    return result


def cmd_qa(args):
    """Train with QA pairs."""
    url = get_endpoint_url()

    # Load QA pairs
    qa_pairs = None
    if args.qa_file:
        with open(args.qa_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        print(f"[remote] Loaded {len(qa_pairs)} QA pairs from {args.qa_file}")
    elif args.qa_pairs:
        qa_pairs = json.loads(args.qa_pairs)
        print(f"[remote] Using {len(qa_pairs)} inline QA pairs")
    else:
        print("[remote] Error: --qa_file or --qa_pairs required for qa mode")
        sys.exit(1)

    payload = {
        "action": "train_qa",
        "parameters": {
            "qa_pairs": qa_pairs,
            "repeat": args.repeat,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_steps": args.warmup_steps,
        },
    }

    result = send_request(url, payload, token=args.token, timeout=args.timeout)
    _print_training_result(result)
    return result


def cmd_train(args):
    """General dataset training."""
    url = get_endpoint_url()
    payload = {
        "action": "train",
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_steps": args.warmup_steps,
        },
    }
    if args.dataset_ids:
        payload["parameters"]["dataset_ids"] = args.dataset_ids

    result = send_request(url, payload, token=args.token, timeout=args.timeout)
    _print_training_result(result)
    return result


def cmd_train_qa_dataset(args):
    """QA dataset training (from HF datasets)."""
    url = get_endpoint_url()
    payload = {
        "action": "train_qa_dataset",
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_steps": args.warmup_steps,
        },
    }
    if args.dataset_id:
        payload["parameters"]["dataset_id"] = args.dataset_id

    result = send_request(url, payload, token=args.token, timeout=args.timeout)
    _print_training_result(result)
    return result


def _print_training_result(result):
    """Pretty-print training result."""
    if isinstance(result, list):
        result = result[0]

    status = result.get("status", "?")
    print(f"\n=== Training Result: {status} ===")
    print(f"Message: {result.get('message', '')}")

    log = result.get("log", [])
    if log:
        print("\nLog:")
        for entry in log:
            print(f"  {entry}")


def main():
    parser = argparse.ArgumentParser(description="Remote training client for neuroQ endpoint")
    parser.add_argument(
        "--mode",
        choices=["status", "qa", "train", "train_qa_dataset"],
        default="status",
        help="Operation mode",
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="API token")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout (seconds)")

    # QA mode args
    parser.add_argument("--qa_file", help="JSON file with QA pairs [{question, answer}, ...]")
    parser.add_argument("--qa_pairs", help="Inline JSON string of QA pairs")
    parser.add_argument("--repeat", type=int, default=3, help="QA data repeat count")

    # Training hyperparameters (shared)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=10)

    # Dataset selection
    parser.add_argument("--dataset_ids", nargs="+", help="Dataset IDs for train mode")
    parser.add_argument("--dataset_id", help="Single dataset ID for train_qa_dataset mode")

    args = parser.parse_args()

    dispatch = {
        "status": cmd_status,
        "qa": cmd_qa,
        "train": cmd_train,
        "train_qa_dataset": cmd_train_qa_dataset,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
