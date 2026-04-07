#!/usr/bin/env python3
"""RunPod training client for NeuroQ with HuggingFace datasets.

Sends training requests to a RunPod serverless endpoint, supporting
all training modes available in handler.py:
  - train:             General HF dataset training
  - train_qa_dataset:  QA-format training from HF datasets
  - train_qa:          QA pairs training (inline or from file)
  - train_split:       Chunked dataset training (timeout-safe)
  - train_split_next:  Single chunk training (iterative)
  - status:            Check model status
  - split_status:      Check split training progress
  - split_reset:       Reset split training state
  - inference:         Run inference

Usage:
  # Set your RunPod endpoint
  export RUNPOD_ENDPOINT_ID="your_endpoint_id"
  export RUNPOD_API_KEY="your_api_key"

  # Check status
  python train_runpod.py status

  # General training with default HF datasets
  python train_runpod.py train

  # Train with specific HF datasets
  python train_runpod.py train --dataset-ids fujiki/japanese_alpaca_data kunishou/oasst1-chat-44k-ja

  # QA dataset training
  python train_runpod.py train-qa-dataset --dataset-id fujiki/japanese_alpaca_data

  # QA pairs from file
  python train_runpod.py train-qa --qa-file qa_data.json

  # Split training (all chunks)
  python train_runpod.py train-split --mode qa --num-chunks 4

  # Iterative split training (one chunk at a time)
  python train_runpod.py train-split-next --mode qa --num-chunks 4

  # Auto-loop split training (calls train-split-next until done)
  python train_runpod.py train-split-auto --mode qa --num-chunks 4

  # Inference
  python train_runpod.py inference --prompt "日本の首都は"
"""

import argparse
import json
import os
import sys
import time
import logging

import runpod

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_runpod")


def get_endpoint():
    """Get RunPod endpoint object."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")

    if not api_key:
        logger.error("RUNPOD_API_KEY not set")
        sys.exit(1)
    if not endpoint_id:
        logger.error("RUNPOD_ENDPOINT_ID not set")
        sys.exit(1)

    runpod.api_key = api_key
    return runpod.Endpoint(endpoint_id)


def run_job(payload, timeout=86400, poll_interval=5):
    """Submit a job to RunPod and wait for results with polling.

    Uses async run + polling for long-running training jobs,
    since run_sync may timeout for training tasks.
    """
    endpoint = get_endpoint()

    logger.info("Submitting job: action=%s", payload.get("action", "inference"))
    run_request = endpoint.run(payload)
    job_id = run_request.job_id
    logger.info("Job submitted: %s", job_id)

    # Poll for completion
    start_time = time.time()
    while True:
        status = run_request.status()
        elapsed = time.time() - start_time

        if status == "COMPLETED":
            result = run_request.output()
            logger.info("Job completed in %.1fs", elapsed)
            return result

        if status == "FAILED":
            logger.error("Job failed after %.1fs", elapsed)
            try:
                output = run_request.output()
                return {"status": "error", "message": str(output)}
            except Exception:
                return {"status": "error", "message": "Job failed with no output"}

        if status == "CANCELLED":
            return {"status": "error", "message": "Job was cancelled"}

        if elapsed > timeout:
            logger.warning("Timeout after %.1fs, job %s still %s", elapsed, job_id, status)
            return {
                "status": "timeout",
                "message": f"Job {job_id} still {status} after {timeout}s",
                "job_id": job_id,
            }

        logger.info("  [%3.0fs] %s ...", elapsed, status)
        time.sleep(poll_interval)


def run_sync_job(payload, timeout=120):
    """Submit a job and wait synchronously (for quick operations)."""
    endpoint = get_endpoint()
    logger.info("Submitting sync job: action=%s", payload.get("action", "inference"))
    result = endpoint.run_sync(payload, timeout=timeout)
    return result


# ============================================================
# Commands
# ============================================================

def cmd_status(args):
    """Check model status."""
    result = run_sync_job({"action": "status"}, timeout=60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def cmd_train(args):
    """General HF dataset training."""
    payload = {
        "action": "train",
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_steps": args.warmup_steps,
            "max_samples_per_dataset": args.max_samples,
        },
    }
    if args.dataset_ids:
        payload["parameters"]["dataset_ids"] = args.dataset_ids

    result = run_job(payload, timeout=args.timeout)
    _print_result(result)


def cmd_train_qa_dataset(args):
    """QA-format training from HF datasets."""
    payload = {
        "action": "train_qa_dataset",
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_steps": args.warmup_steps,
            "max_samples_per_dataset": args.max_samples,
        },
    }
    if args.dataset_id:
        payload["parameters"]["dataset_id"] = args.dataset_id

    result = run_job(payload, timeout=args.timeout)
    _print_result(result)


def cmd_train_qa(args):
    """Train with QA pairs (inline or from file)."""
    qa_pairs = None
    if args.qa_file:
        with open(args.qa_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        logger.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_file)
    elif args.qa_pairs:
        qa_pairs = json.loads(args.qa_pairs)
        logger.info("Using %d inline QA pairs", len(qa_pairs))
    else:
        logger.error("--qa-file or --qa-pairs required")
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

    result = run_job(payload, timeout=args.timeout)
    _print_result(result)


def cmd_train_split(args):
    """Split dataset training (all chunks, background)."""
    payload = {
        "action": "train_split",
        "parameters": {
            "mode": args.mode,
            "num_chunks": args.num_chunks,
            "epochs_per_chunk": args.epochs_per_chunk,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_steps": args.warmup_steps,
            "max_samples_per_dataset": args.max_samples,
            "crafted_repeat": args.crafted_repeat,
        },
    }
    if args.dataset_ids:
        payload["parameters"]["dataset_ids"] = args.dataset_ids
    if args.samples_per_batch:
        payload["parameters"]["samples_per_batch"] = args.samples_per_batch
    if args.max_minutes:
        payload["parameters"]["max_minutes_per_chunk"] = args.max_minutes

    result = run_job(payload, timeout=args.timeout)
    _print_result(result)


def cmd_train_split_next(args):
    """Train the next single chunk (timeout-safe)."""
    payload = {
        "action": "train_split_next",
        "parameters": {
            "mode": args.mode,
            "num_chunks": args.num_chunks,
            "epochs_per_chunk": args.epochs_per_chunk,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_steps": args.warmup_steps,
            "max_samples_per_dataset": args.max_samples,
            "crafted_repeat": args.crafted_repeat,
        },
    }
    if args.dataset_ids:
        payload["parameters"]["dataset_ids"] = args.dataset_ids
    if args.samples_per_batch:
        payload["parameters"]["samples_per_batch"] = args.samples_per_batch

    result = run_job(payload, timeout=args.timeout)
    _print_result(result)
    return result


def cmd_train_split_auto(args):
    """Automatically loop train-split-next until all chunks are done."""
    chunk_num = 0
    while True:
        chunk_num += 1
        logger.info("=== Auto split training: iteration %d ===", chunk_num)

        result = cmd_train_split_next(args)

        if isinstance(result, dict):
            status = result.get("status", "")
            chunks_remaining = result.get("chunks_remaining", 0)

            if status == "error":
                logger.error("Training failed, stopping auto-loop")
                break

            if status == "completed" or chunks_remaining == 0:
                logger.info("All chunks completed!")
                break

            if status == "timeout":
                logger.warning("Job timed out, retrying...")
                continue

        # Brief pause between chunks
        logger.info("Waiting %ds before next chunk...", args.chunk_interval)
        time.sleep(args.chunk_interval)


def cmd_split_status(args):
    """Check split training progress."""
    result = run_sync_job({"action": "split_status"}, timeout=60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def cmd_split_reset(args):
    """Reset split training state."""
    result = run_sync_job({"action": "split_reset"}, timeout=60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def cmd_inference(args):
    """Run inference."""
    payload = {
        "action": "inference",
        "prompt": args.prompt,
        "parameters": {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
    }
    result = run_sync_job(payload, timeout=120)
    if isinstance(result, dict):
        print(result.get("generated_text", json.dumps(result, ensure_ascii=False)))
    elif isinstance(result, list) and result:
        r = result[0]
        print(r.get("generated_text", json.dumps(r, ensure_ascii=False)))
    else:
        print(result)


def cmd_health(args):
    """Check endpoint health."""
    endpoint = get_endpoint()
    result = endpoint.health()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def _print_result(result):
    """Pretty-print training result."""
    if isinstance(result, list):
        result = result[0] if result else {}

    if not isinstance(result, dict):
        print(result)
        return

    status = result.get("status", "?")
    print(f"\n{'='*50}")
    print(f" Result: {status}")
    print(f"{'='*50}")

    if result.get("message"):
        print(f" Message: {result['message']}")

    log = result.get("log", [])
    if log:
        print(f"\n Log ({len(log)} entries):")
        for entry in log:
            print(f"   {entry}")

    chunks_remaining = result.get("chunks_remaining")
    if chunks_remaining is not None:
        print(f"\n Chunks remaining: {chunks_remaining}")

    job_id = result.get("job_id")
    if job_id:
        print(f" Job ID: {job_id}")

    print()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RunPod training client for NeuroQ with HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  RUNPOD_API_KEY       RunPod API key (required)
  RUNPOD_ENDPOINT_ID   RunPod endpoint ID (required)

Examples:
  %(prog)s status
  %(prog)s train --dataset-ids fujiki/japanese_alpaca_data
  %(prog)s train-qa-dataset --epochs 20
  %(prog)s train-split --mode qa --num-chunks 4
  %(prog)s train-split-auto --mode qa --num-chunks 4
  %(prog)s inference --prompt "日本の首都は"
""",
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # --- status ---
    sub.add_parser("status", help="Check model status")

    # --- health ---
    sub.add_parser("health", help="Check endpoint health")

    # --- train ---
    p_train = sub.add_parser("train", help="General HF dataset training")
    p_train.add_argument("--dataset-ids", nargs="+", help="HF dataset IDs (default: built-in Japanese datasets)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--grad-accum-steps", type=int, default=8)
    p_train.add_argument("--warmup-steps", type=int, default=100)
    p_train.add_argument("--max-samples", type=int, default=5000)
    p_train.add_argument("--timeout", type=int, default=86400, help="Max wait time in seconds")

    # --- train-qa-dataset ---
    p_tqd = sub.add_parser("train-qa-dataset", help="QA-format training from HF datasets")
    p_tqd.add_argument("--dataset-id", help="Specific HF dataset ID")
    p_tqd.add_argument("--epochs", type=int, default=20)
    p_tqd.add_argument("--lr", type=float, default=3e-5)
    p_tqd.add_argument("--batch-size", type=int, default=4)
    p_tqd.add_argument("--grad-accum-steps", type=int, default=4)
    p_tqd.add_argument("--warmup-steps", type=int, default=30)
    p_tqd.add_argument("--max-samples", type=int, default=1500)
    p_tqd.add_argument("--timeout", type=int, default=86400)

    # --- train-qa ---
    p_qa = sub.add_parser("train-qa", help="Train with QA pairs")
    p_qa.add_argument("--qa-file", help="JSON file with QA pairs")
    p_qa.add_argument("--qa-pairs", help="Inline JSON string of QA pairs")
    p_qa.add_argument("--repeat", type=int, default=3)
    p_qa.add_argument("--epochs", type=int, default=20)
    p_qa.add_argument("--lr", type=float, default=3e-5)
    p_qa.add_argument("--batch-size", type=int, default=4)
    p_qa.add_argument("--grad-accum-steps", type=int, default=4)
    p_qa.add_argument("--warmup-steps", type=int, default=10)
    p_qa.add_argument("--timeout", type=int, default=86400)

    # --- train-split ---
    p_split = sub.add_parser("train-split", help="Split dataset training (all chunks)")
    _add_split_args(p_split)

    # --- train-split-next ---
    p_next = sub.add_parser("train-split-next", help="Train next single chunk")
    _add_split_args(p_next)

    # --- train-split-auto ---
    p_auto = sub.add_parser("train-split-auto", help="Auto-loop split training until done")
    _add_split_args(p_auto)
    p_auto.add_argument("--chunk-interval", type=int, default=10,
                        help="Seconds to wait between chunks")

    # --- split-status ---
    sub.add_parser("split-status", help="Check split training progress")

    # --- split-reset ---
    sub.add_parser("split-reset", help="Reset split training state")

    # --- inference ---
    p_inf = sub.add_parser("inference", help="Run text generation")
    p_inf.add_argument("--prompt", required=True, help="Input prompt")
    p_inf.add_argument("--max-tokens", type=int, default=100)
    p_inf.add_argument("--temperature", type=float, default=0.7)
    p_inf.add_argument("--top-k", type=int, default=40)
    p_inf.add_argument("--top-p", type=float, default=0.9)
    p_inf.add_argument("--repetition-penalty", type=float, default=1.3)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "status": cmd_status,
        "health": cmd_health,
        "train": cmd_train,
        "train-qa-dataset": cmd_train_qa_dataset,
        "train-qa": cmd_train_qa,
        "train-split": cmd_train_split,
        "train-split-next": cmd_train_split_next,
        "train-split-auto": cmd_train_split_auto,
        "split-status": cmd_split_status,
        "split-reset": cmd_split_reset,
        "inference": cmd_inference,
    }

    dispatch[args.command](args)


def _add_split_args(parser):
    """Add common split training arguments."""
    parser.add_argument("--mode", choices=["qa", "general"], default="qa")
    parser.add_argument("--dataset-ids", nargs="+", help="HF dataset IDs")
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--samples-per-batch", type=int, default=None)
    parser.add_argument("--epochs-per-chunk", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--crafted-repeat", type=int, default=20)
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Max minutes per chunk")
    parser.add_argument("--timeout", type=int, default=86400)


if __name__ == "__main__":
    main()
