#!/usr/bin/env python3
"""RunPod API Training Client for NeuroQ QBNN.

Sends training requests to the RunPod serverless endpoint and polls for results.

Usage:
  # QA dataset training (recommended - uses 4 curated Japanese datasets)
  python train_runpod.py --mode train_qa

  # General dataset training (5 datasets + cc100-ja)
  python train_runpod.py --mode train

  # Check model status
  python train_runpod.py --mode status

  # Split training (timeout-safe, chunked)
  python train_runpod.py --mode train_split

  # Resume split training
  python train_runpod.py --mode train_split_next

  # Check split training progress
  python train_runpod.py --mode split_status
"""

import argparse
import json
import os
import sys
import time

import requests

# ==============================================================
# RunPod API Configuration
# ==============================================================

RUNPOD_API_BASE = os.environ.get(
    "RUNPOD_API_BASE",
    "https://api.runpod.ai/v2/54wu6q7dj4hole",
)
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
if not RUNPOD_API_KEY:
    print("[runpod] Warning: RUNPOD_API_KEY not set. Set via environment variable or --api_key flag.")
    print("  export RUNPOD_API_KEY='your-api-key-here'")


# ==============================================================
# Recommended datasets (built into handler.py defaults)
# ==============================================================
# QA datasets (action: train_qa):
#   - fujiki/japanese_alpaca_data          (alpaca format)
#   - FreedomIntelligence/alpaca-gpt4-japanese (conversations)
#   - kunishou/oasst1-chat-44k-ja          (conversations)
#   - izumi-lab/llm-japanese-dataset        (izumi format)
#
# General datasets (action: train):
#   - izumi-lab/llm-japanese-dataset
#   - kunishou/oasst1-chat-44k-ja
#   - fujiki/japanese_alpaca_data
#   - shi3z/Japanese_wikipedia_conversation_100K
#   - FreedomIntelligence/alpaca-gpt4-japanese
#   - range3/cc100-ja (supplementary)


def runpod_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }


def runpod_run(payload, timeout=30):
    """Submit an async job to RunPod. Returns job ID."""
    url = f"{RUNPOD_API_BASE}/run"
    print(f"[runpod] POST {url}")
    resp = requests.post(url, json=payload, headers=runpod_headers(), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("id")
    status = data.get("status")
    print(f"[runpod] Job submitted: id={job_id}, status={status}")
    return data


def runpod_status(job_id, timeout=30):
    """Check the status of a RunPod job."""
    url = f"{RUNPOD_API_BASE}/status/{job_id}"
    resp = requests.get(url, headers=runpod_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def poll_job(job_id, poll_interval=10, max_wait=3600):
    """Poll a RunPod job until completion."""
    print(f"\n[runpod] Polling job {job_id} (interval={poll_interval}s, max_wait={max_wait}s)")
    elapsed = 0
    while elapsed < max_wait:
        result = runpod_status(job_id)
        status = result.get("status", "UNKNOWN")

        if status == "COMPLETED":
            print(f"\n[runpod] Job COMPLETED after ~{elapsed}s")
            return result
        elif status == "FAILED":
            print(f"\n[runpod] Job FAILED after ~{elapsed}s")
            return result
        elif status == "CANCELLED":
            print(f"\n[runpod] Job CANCELLED after ~{elapsed}s")
            return result
        else:
            mins = elapsed // 60
            secs = elapsed % 60
            print(f"  [{mins:02d}:{secs:02d}] status={status}", end="\r")
            time.sleep(poll_interval)
            elapsed += poll_interval

    print(f"\n[runpod] Timeout after {max_wait}s. Job may still be running.")
    return runpod_status(job_id)


def print_result(result):
    """Pretty-print a RunPod job result."""
    status = result.get("status", "?")
    output = result.get("output", result)

    if isinstance(output, list) and len(output) == 1:
        output = output[0]

    print(f"\n{'='*60}")
    print(f"  Job Status: {status}")
    print(f"{'='*60}")

    if isinstance(output, dict):
        print(f"  Result:  {output.get('status', '?')}")
        print(f"  Message: {output.get('message', '')}")

        log = output.get("log", [])
        if log:
            print(f"\n  Training Log:")
            for entry in log:
                print(f"    {entry}")

        # Status-specific fields
        if "architecture" in output:
            print(f"\n  Architecture: {output.get('architecture')}")
            print(f"  Parameters:   {output.get('model_params', '?'):,}")

        checkpoint = output.get("checkpoint")
        if checkpoint:
            print(f"\n  Checkpoint: {checkpoint.get('path', '?')}")
            print(f"    Size:     {checkpoint.get('size_mb', '?')} MB")
            print(f"    Trained:  {checkpoint.get('trained_at', '?')}")

        split = output.get("split_training")
        if split:
            print(f"\n  Split Training: chunk {split.get('last_completed_chunk', '?')}/{split.get('num_chunks', '?')}")
    else:
        print(f"  Output: {json.dumps(output, ensure_ascii=False, indent=2)}")

    print(f"{'='*60}\n")


# ==============================================================
# Commands
# ==============================================================

def cmd_status(args):
    """Check model status."""
    payload = {"input": {"action": "status", "prompt": ""}}
    result = runpod_run(payload)
    job_id = result.get("id")
    if not job_id:
        print("[runpod] Error: No job ID returned")
        return
    final = poll_job(job_id, poll_interval=5, max_wait=120)
    print_result(final)


def cmd_train(args):
    """General dataset training (5 Japanese datasets + cc100-ja)."""
    payload = {
        "input": {
            "action": "train",
            "prompt": "",
            "parameters": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "warmup_steps": args.warmup_steps,
                "max_samples_per_dataset": args.max_samples,
            },
        }
    }
    if args.dataset_ids:
        payload["input"]["parameters"]["dataset_ids"] = args.dataset_ids

    print(f"\n[runpod] Starting general training")
    print(f"  epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"  grad_accum={args.grad_accum_steps}, warmup={args.warmup_steps}")
    print(f"  max_samples_per_dataset={args.max_samples}")

    result = runpod_run(payload)
    job_id = result.get("id")
    if not job_id:
        print("[runpod] Error: No job ID returned")
        return
    final = poll_job(job_id, poll_interval=args.poll_interval, max_wait=args.max_wait)
    print_result(final)


def cmd_train_qa(args):
    """QA dataset training (recommended - 4 curated Japanese datasets).

    Uses action 'train_qa_dataset' which loads from HuggingFace datasets.
    """
    payload = {
        "input": {
            "action": "train_qa_dataset",
            "prompt": "",
            "parameters": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "warmup_steps": args.warmup_steps,
                "max_samples_per_dataset": args.max_samples,
                "crafted_repeat": args.crafted_repeat,
            },
        }
    }
    if args.dataset_id:
        payload["input"]["parameters"]["dataset_id"] = args.dataset_id

    print(f"\n[runpod] Starting QA dataset training (recommended)")
    print(f"  epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"  grad_accum={args.grad_accum_steps}, warmup={args.warmup_steps}")
    print(f"  max_samples={args.max_samples}, crafted_repeat={args.crafted_repeat}")
    if args.dataset_id:
        print(f"  custom dataset: {args.dataset_id}")
    else:
        print(f"  datasets: fujiki/japanese_alpaca_data,")
        print(f"            FreedomIntelligence/alpaca-gpt4-japanese,")
        print(f"            kunishou/oasst1-chat-44k-ja,")
        print(f"            izumi-lab/llm-japanese-dataset")

    result = runpod_run(payload)
    job_id = result.get("id")
    if not job_id:
        print("[runpod] Error: No job ID returned")
        return
    final = poll_job(job_id, poll_interval=args.poll_interval, max_wait=args.max_wait)
    print_result(final)


def cmd_train_split(args):
    """Split training (timeout-safe, chunked)."""
    payload = {
        "input": {
            "action": "train_split",
            "prompt": "",
            "parameters": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_chunks": args.num_chunks,
                "mode": args.split_mode,
            },
        }
    }

    print(f"\n[runpod] Starting split training")
    print(f"  mode={args.split_mode}, chunks={args.num_chunks}")
    print(f"  epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")

    result = runpod_run(payload)
    job_id = result.get("id")
    if not job_id:
        print("[runpod] Error: No job ID returned")
        return
    final = poll_job(job_id, poll_interval=args.poll_interval, max_wait=args.max_wait)
    print_result(final)


def cmd_train_split_next(args):
    """Resume split training (next chunk)."""
    payload = {
        "input": {
            "action": "train_split_next",
            "prompt": "",
        }
    }

    print(f"\n[runpod] Resuming split training (next chunk)")
    result = runpod_run(payload)
    job_id = result.get("id")
    if not job_id:
        print("[runpod] Error: No job ID returned")
        return
    final = poll_job(job_id, poll_interval=args.poll_interval, max_wait=args.max_wait)
    print_result(final)


def cmd_split_status(args):
    """Check split training progress."""
    payload = {"input": {"action": "split_status", "prompt": ""}}
    result = runpod_run(payload)
    job_id = result.get("id")
    if not job_id:
        print("[runpod] Error: No job ID returned")
        return
    final = poll_job(job_id, poll_interval=5, max_wait=120)
    print_result(final)


def main():
    parser = argparse.ArgumentParser(
        description="RunPod API Training Client for NeuroQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended usage:
  python train_runpod.py --mode train_qa              # QA training with defaults
  python train_runpod.py --mode train_qa --epochs 30  # More epochs
  python train_runpod.py --mode train                 # General training
  python train_runpod.py --mode status                # Check model status
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["status", "train", "train_qa", "train_split", "train_split_next", "split_status"],
        default="train_qa",
        help="Operation mode (default: train_qa)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--warmup_steps", type=int, default=30, help="Warmup steps (default: 30)")
    parser.add_argument("--max_samples", type=int, default=1500, help="Max samples per dataset (default: 1500)")

    # QA-specific
    parser.add_argument("--dataset_id", help="Custom dataset ID for train_qa mode")
    parser.add_argument("--crafted_repeat", type=int, default=40, help="Crafted QA repeat count (default: 40)")

    # General training
    parser.add_argument("--dataset_ids", nargs="+", help="Custom dataset IDs for train mode")

    # Split training
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunks for split training (default: 5)")
    parser.add_argument("--split_mode", choices=["qa", "general", "both"], default="both",
                        help="Split training mode (default: both)")

    # Polling
    parser.add_argument("--poll_interval", type=int, default=15, help="Poll interval in seconds (default: 15)")
    parser.add_argument("--max_wait", type=int, default=3600, help="Max wait time in seconds (default: 3600)")

    # API config override
    parser.add_argument("--api_base", help="RunPod API base URL override")
    parser.add_argument("--api_key", help="RunPod API key override")

    args = parser.parse_args()

    # Apply overrides
    global RUNPOD_API_BASE, RUNPOD_API_KEY
    if args.api_base:
        RUNPOD_API_BASE = args.api_base
    if args.api_key:
        RUNPOD_API_KEY = args.api_key

    print(f"[runpod] Endpoint: {RUNPOD_API_BASE}")
    print(f"[runpod] Mode: {args.mode}")

    dispatch = {
        "status": cmd_status,
        "train": cmd_train,
        "train_qa": cmd_train_qa,
        "train_split": cmd_train_split,
        "train_split_next": cmd_train_split_next,
        "split_status": cmd_split_status,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
