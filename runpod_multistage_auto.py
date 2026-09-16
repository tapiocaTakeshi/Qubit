#!/usr/bin/env python3
"""
Automated multi-stage training orchestrator for RunPod.

Monitors training job status and automatically submits the next stage
when the current one completes. Manages the full 7-stage pipeline:
1. FineWeb-2 Japanese
2. ABEJA-CC-JA-edu
3. Wikipedia
4. Instruction
5. Conversation
6. Mathematics
7. Code (swallow-code-v2)
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Configuration
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "aic6yigpthbck5")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_API_URL = "https://api.runpod.ai/v2"

# Training stages configuration
TRAINING_STAGES = [
    {
        "id": 1,
        "name": "fineweb_japanese",
        "description": "FineWeb-2 Japanese",
        "datasets": [{"id": "HuggingFaceFW/fineweb-2-edu-japanese", "split": None, "max_samples": 50000}],
        "learning_rate": 1e-4,
        "epochs": 1,
    },
    {
        "id": 2,
        "name": "abeja_cc_ja_edu",
        "description": "ABEJA-CC-JA-edu",
        "datasets": [{"id": "ABEJA/abeja-cc-ja-edu", "split": None, "max_samples": 30000}],
        "learning_rate": 8e-5,
        "epochs": 1,
    },
    {
        "id": 3,
        "name": "wikipedia",
        "description": "Wikipedia (Japanese + English)",
        "datasets": [
            {"id": "wikimedia/wikipedia", "split": "20220301.ja", "max_samples": 20000},
            {"id": "wikimedia/wikipedia", "split": "20220301.en", "max_samples": 20000},
        ],
        "learning_rate": 5e-5,
        "epochs": 1,
    },
    {
        "id": 4,
        "name": "instruction",
        "description": "Instruction-following",
        "datasets": [
            {"id": "Open-Orca/OpenOrca", "split": None, "max_samples": 10000},
            {"id": "HuggingFaceH4/ultrachat_200k", "split": None, "max_samples": 10000},
        ],
        "learning_rate": 3e-5,
        "epochs": 1,
    },
    {
        "id": 5,
        "name": "conversation",
        "description": "Conversation",
        "datasets": [
            {"id": "kunishou/hh-rlhf-ja", "split": None, "max_samples": 5000},
            {"id": "HuggingFaceH4/ultrachat_200k", "split": None, "max_samples": 5000},
        ],
        "learning_rate": 2e-5,
        "epochs": 1,
    },
    {
        "id": 6,
        "name": "mathematics",
        "description": "Mathematics",
        "datasets": [
            {"id": "meta-math/MetaMathQA", "split": None, "max_samples": 10000},
            {"id": "openai/gsm8k", "split": "main", "max_samples": 5000},
        ],
        "learning_rate": 2e-5,
        "epochs": 1,
    },
    {
        "id": 7,
        "name": "code",
        "description": "Code (swallow-code-v2)",
        "datasets": [{"id": "tokyotech-llm/swallow-code-v2", "split": None, "max_samples": 100000}],
        "learning_rate": 1e-5,
        "epochs": 1,
    },
]

# State tracking
STATE_FILE = Path("multistage_state.json")

def load_state():
    """Load training state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"current_stage": 0, "job_ids": [], "completed_stages": []}

def save_state(state):
    """Save training state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def check_job_status(job_id):
    """Check status of a RunPod job."""
    try:
        url = f"{RUNPOD_API_URL}/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
        headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to check job status: {e}")
        return None

def submit_stage(stage):
    """Submit a training stage to RunPod."""
    try:
        url = f"{RUNPOD_API_URL}/{RUNPOD_ENDPOINT_ID}/run"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
        }

        # Convert dataset dicts to string IDs (handler expects "owner/dataset" format)
        dataset_ids = []
        for ds in stage["datasets"]:
            if isinstance(ds, dict):
                ds_id = ds.get("id")
                split = ds.get("split")
                # Format: "owner/dataset" or "owner/dataset:config" for split
                if split:
                    dataset_ids.append(f"{ds_id}:{split}")
                else:
                    dataset_ids.append(ds_id)
            else:
                dataset_ids.append(str(ds))

        payload = {
            "input": {
                "action": "train",
                "data": {
                    "parameters": {
                        "dataset_ids": dataset_ids,
                        "epochs": stage["epochs"],
                        "lr": stage["learning_rate"],
                        "batch_size": 2,
                        "grad_accum_steps": 4,
                        "mode": "general",
                        "max_samples_per_dataset": stage["datasets"][0].get("max_samples", 10000),
                    }
                }
            }
        }

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to submit stage: {e}")
        return None

def print_status(state):
    """Print current training status."""
    print("\n" + "="*60)
    print("📊 Multi-Stage Training Status")
    print("="*60)
    print(f"Current Stage: {state['current_stage']}/{len(TRAINING_STAGES)}")
    print(f"Completed Stages: {len(state['completed_stages'])}")

    for i, stage in enumerate(TRAINING_STAGES, 1):
        status = "✓" if i in state["completed_stages"] else "○"
        current = "→" if i == state["current_stage"] else " "
        print(f"  {current} {status} Stage {i}: {stage['description']}")

    if state["job_ids"]:
        print(f"\nLatest Job ID: {state['job_ids'][-1]}")
    print("="*60 + "\n")

def main():
    """Main orchestration loop."""
    if not RUNPOD_API_KEY:
        print("[ERROR] RUNPOD_API_KEY not set")
        sys.exit(1)

    state = load_state()
    print(f"[INFO] Loaded state: {state}")

    # Check if all stages are complete
    if len(state["completed_stages"]) == len(TRAINING_STAGES):
        print("[SUCCESS] All training stages completed!")
        print_status(state)
        return

    print_status(state)

    # If no current job, start the next stage
    if not state["job_ids"]:
        next_stage_idx = state["current_stage"]
    else:
        # Check current job status
        current_job_id = state["job_ids"][-1]
        print(f"[INFO] Checking status of job {current_job_id}...")

        job_status = check_job_status(current_job_id)
        if not job_status:
            print("[WARN] Could not check job status, will retry later")
            return

        status = job_status.get("status", "UNKNOWN")
        print(f"[INFO] Job status: {status}")

        if status == "COMPLETED":
            # Check if job succeeded
            if job_status.get("error"):
                print(f"[WARN] Job failed: {job_status.get('error')[:200]}")
                # Mark as completed anyway to move to next stage
            else:
                print(f"[INFO] Job succeeded!")

            # Mark stage as completed
            current_stage_id = state["current_stage"]
            if current_stage_id not in state["completed_stages"]:
                state["completed_stages"].append(current_stage_id)

            # Move to next stage
            next_stage_idx = current_stage_id + 1
        elif status == "FAILED":
            print(f"[ERROR] Job failed with error")
            print(f"Error: {job_status.get('error')[:500]}")
            return
        else:
            print(f"[INFO] Job still running ({status})...")
            return

    # Submit next stage if available
    if next_stage_idx <= len(TRAINING_STAGES):
        stage = TRAINING_STAGES[next_stage_idx - 1]
        print(f"\n[INFO] Submitting Stage {next_stage_idx}: {stage['description']}")
        print(f"  Datasets: {len(stage['datasets'])}")
        print(f"  Learning Rate: {stage['learning_rate']}")

        result = submit_stage(stage)
        if result:
            job_id = result.get("id")
            print(f"[SUCCESS] Job submitted: {job_id}")

            state["current_stage"] = next_stage_idx
            state["job_ids"].append(job_id)
            save_state(state)

            print_status(state)
        else:
            print("[ERROR] Failed to submit stage")
            return
    else:
        print("[SUCCESS] All stages completed!")

if __name__ == "__main__":
    main()
