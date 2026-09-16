#!/usr/bin/env python3
"""
Multi-stage training orchestrator for 3B Qubit AI on RunPod.

Executes the following training schedule:
1. FineWeb-2 Japanese
2. ABEJA-CC-JA-edu
3. Wikipedia
4. Instruction
5. Conversation
6. Math
7. Code (tokyotech-llm/swallow-code-v2)

Each stage trains on relevant datasets with appropriate learning rates.
Checkpoints are saved after each stage.

Usage:
  python train_multistage_3b_runpod.py --budget 10 --log-file training.log

Or submit to RunPod via API:
  curl https://api.runpod.io/v2/ENDPOINT_ID/run \
    -H "Content-Type: application/json" \
    -d '{"input": {"action": "train_multistage_3b", "budget_dollars": 10}}'
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
log_dir = Path("training_logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"train_multistage_3b_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Training stages with dataset config
TRAINING_STAGES = [
    {
        "name": "fineweb_japanese",
        "description": "FineWeb-2 Japanese",
        "datasets": [
            {"id": "HuggingFaceFW/fineweb-2-edu-japanese", "split": None, "max_samples": 1000000}
        ],
        "learning_rate": 1e-4,
        "epochs": 1,
    },
    {
        "name": "abeja_cc_ja_edu",
        "description": "ABEJA-CC-JA-edu",
        "datasets": [
            {"id": "ABEJA/abeja-cc-ja-edu", "split": None, "max_samples": 500000}
        ],
        "learning_rate": 8e-5,
        "epochs": 1,
    },
    {
        "name": "wikipedia",
        "description": "Wikipedia (Japanese + English)",
        "datasets": [
            {"id": "wikimedia/wikipedia", "split": "20220301.ja", "max_samples": 200000},
            {"id": "wikimedia/wikipedia", "split": "20220301.en", "max_samples": 200000},
        ],
        "learning_rate": 5e-5,
        "epochs": 1,
    },
    {
        "name": "instruction",
        "description": "Instruction-following datasets",
        "datasets": [
            {"id": "Open-Orca/OpenOrca", "split": None, "max_samples": 100000},
            {"id": "HuggingFaceH4/ultrachat_200k", "split": None, "max_samples": 100000},
        ],
        "learning_rate": 3e-5,
        "epochs": 2,
    },
    {
        "name": "conversation",
        "description": "Conversational datasets",
        "datasets": [
            {"id": "kunishou/hh-rlhf-ja", "split": None, "max_samples": 50000},
            {"id": "HuggingFaceH4/ultrachat_200k", "split": None, "max_samples": 50000},
        ],
        "learning_rate": 2e-5,
        "epochs": 2,
    },
    {
        "name": "mathematics",
        "description": "Mathematical reasoning datasets",
        "datasets": [
            {"id": "meta-math/MetaMathQA", "split": None, "max_samples": 100000},
            {"id": "openai/gsm8k", "split": "main", "max_samples": 50000},
        ],
        "learning_rate": 2e-5,
        "epochs": 2,
    },
    {
        "name": "code",
        "description": "Code datasets (tokyotech-llm/swallow-code-v2)",
        "datasets": [
            {
                "id": "tokyotech-llm/swallow-code-v2",
                "split": None,
                "max_samples": 5000000,
                "exclude": ["generated", "duplicate", "license_unclear"],
            }
        ],
        "learning_rate": 1e-5,
        "epochs": 1,
    },
]


def reset_checkpoint():
    """Reset checkpoint to start fresh."""
    logger.info("Resetting checkpoint...")
    try:
        from handler import EndpointHandler
        handler = EndpointHandler(path=".")
        result = handler._handle_reset_checkpoint()
        logger.info(f"Checkpoint reset result: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to reset checkpoint: {e}", exc_info=True)
        return False


def run_training_stage(stage_idx, stage_config):
    """Execute a single training stage."""
    stage_name = stage_config["name"]
    description = stage_config["description"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Stage {stage_idx + 1}/{len(TRAINING_STAGES)}: {description}")
    logger.info(f"{'='*60}")

    try:
        from handler import EndpointHandler

        handler = EndpointHandler(path=".")

        # Build training request
        request = {
            "action": "train",
            "data": {
                "datasets": stage_config["datasets"],
                "learning_rate": stage_config["learning_rate"],
                "epochs": stage_config["epochs"],
                "batch_size": 2,  # Adjust based on GPU memory
                "gradient_accumulation_steps": 4,
                "save_every_n_steps": 500,
                "log_every_n_steps": 10,
            },
        }

        logger.info(f"Starting training with config: {json.dumps(request, indent=2)}")

        # Run training
        start_time = time.time()
        results = handler(request)
        elapsed = time.time() - start_time

        logger.info(f"Stage completed in {elapsed:.1f}s")
        logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")

        # Save checkpoint after stage
        checkpoint_path = f"checkpoints/stage_{stage_name}.pt"
        logger.info(f"Saving checkpoint to {checkpoint_path}")

        return True

    except Exception as e:
        logger.error(f"Stage {stage_name} failed: {e}", exc_info=True)
        return False


def run_multistage_training(budget_dollars=10):
    """Execute all training stages."""
    logger.info(f"Starting multi-stage 3B training (Budget: ${budget_dollars})")
    logger.info(f"Total stages: {len(TRAINING_STAGES)}")

    # Log stage order
    for i, stage in enumerate(TRAINING_STAGES, 1):
        logger.info(f"  {i}. {stage['description']}")

    # Reset checkpoint
    if not reset_checkpoint():
        logger.error("Failed to reset checkpoint, aborting")
        return False

    # Run each stage
    successful_stages = 0
    for idx, stage in enumerate(TRAINING_STAGES):
        if run_training_stage(idx, stage):
            successful_stages += 1
            logger.info(f"Progress: {successful_stages}/{len(TRAINING_STAGES)} stages completed")
        else:
            logger.error(f"Failed at stage {idx + 1}: {stage['description']}")
            # Continue to next stage despite error

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete: {successful_stages}/{len(TRAINING_STAGES)} stages successful")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*60}")

    return successful_stages == len(TRAINING_STAGES)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-stage 3B training orchestrator")
    parser.add_argument("--budget", type=float, default=10, help="Budget in dollars")
    parser.add_argument("--log-file", type=str, help="Override log file path")

    args = parser.parse_args()

    if args.log_file:
        log_file = Path(args.log_file)

    logger.info(f"Log file: {log_file}")
    success = run_multistage_training(budget_dollars=args.budget)

    sys.exit(0 if success else 1)
