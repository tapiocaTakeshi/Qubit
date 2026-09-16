#!/usr/bin/env python3
"""
Submit multi-stage 3B training job to RunPod.

Configuration:
- Budget: $10 USD
- Model: 3B NeuroQuantum (new)
- Reset: Fresh training from random initialization
- Logging: Save to training_logs/

Usage:
  export RUNPOD_API_KEY="your-api-key"
  export RUNPOD_ENDPOINT_ID="your-endpoint-id"
  python submit_runpod_training.py --budget 10
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime

# RunPod configuration
RUNPOD_API_URL = "https://api.runpod.io/v2"
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")


def submit_training_job(endpoint_id, api_key, budget_dollars=10):
    """Submit multi-stage training job to RunPod."""

    if not endpoint_id or not api_key:
        print("Error: RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables required")
        print("Set them with:")
        print("  export RUNPOD_ENDPOINT_ID='your-endpoint-id'")
        print("  export RUNPOD_API_KEY='your-api-key'")
        sys.exit(1)

    # Job configuration
    job_payload = {
        "input": {
            "action": "train_multistage_3b",
            "budget_dollars": budget_dollars,
            "reset_checkpoint": True,
            "log_to_file": True,
            "timestamp": datetime.now().isoformat(),
            "training_stages": [
                "fineweb_japanese",
                "abeja_cc_ja_edu",
                "wikipedia",
                "instruction",
                "conversation",
                "mathematics",
                "code",
            ],
        }
    }

    # Submit job
    url = f"{RUNPOD_API_URL}/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    print(f"Submitting training job to RunPod endpoint: {endpoint_id}")
    print(f"Budget: ${budget_dollars}")
    print(f"Payload: {json.dumps(job_payload, indent=2)}")
    print()

    try:
        response = requests.post(url, json=job_payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        job_id = result.get("id")

        print(f"✓ Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Status URL: {RUNPOD_API_URL}/{endpoint_id}/status/{job_id}")
        print()
        print("You can check the status with:")
        print(f"  curl https://api.runpod.io/v2/{endpoint_id}/status/{job_id} \\")
        print(f"    -H 'Authorization: Bearer $RUNPOD_API_KEY'")
        print()
        print("Check logs from the container:")
        print(f"  training_logs/train_multistage_3b_*.log")

        return job_id

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to submit job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        sys.exit(1)


def create_docker_instructions():
    """Generate Docker setup instructions for the endpoint."""
    instructions = """
# Docker setup for RunPod Qubit AI training endpoint

# 1. Build image with Qubit AI and training scripts
docker build -t qubit-ai-3b:latest -f Dockerfile .

# 2. Push to RunPod registry (if using private endpoint)
docker tag qubit-ai-3b:latest your-registry/qubit-ai-3b:latest
docker push your-registry/qubit-ai-3b:latest

# 3. Create RunPod endpoint with:
#    - Base image: your-registry/qubit-ai-3b:latest
#    - Handler: runpod_handler.py
#    - Environment variables:
#      MODEL_DIR=/app
#      NETWORK_VOLUME_PATH=/runpod-volume
#    - Network volume: attached for persistent checkpoints

# 4. Note the endpoint ID and set environment variables:
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"

# 5. Submit training job:
python submit_runpod_training.py --budget 10
"""
    print(instructions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit multi-stage training to RunPod")
    parser.add_argument("--budget", type=float, default=10, help="Budget in dollars")
    parser.add_argument("--show-docker", action="store_true", help="Show Docker setup instructions")
    parser.add_argument("--endpoint", type=str, help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID)")
    parser.add_argument("--api-key", type=str, help="RunPod API key (or set RUNPOD_API_KEY)")

    args = parser.parse_args()

    if args.show_docker:
        create_docker_instructions()
        sys.exit(0)

    # Override environment variables if provided
    if args.endpoint:
        os.environ["RUNPOD_ENDPOINT_ID"] = args.endpoint
    if args.api_key:
        os.environ["RUNPOD_API_KEY"] = args.api_key

    # Submit job
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    job_id = submit_training_job(endpoint_id, api_key, budget_dollars=args.budget)

    if job_id:
        print(f"\n✓ Training job {job_id} submitted")
