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

from training_stages import TRAINING_STAGES

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


def run_training_stage(handler, stage_idx, stage_config, reset_checkpoint):
    """Run one stage via the handler's train_multistage_3b action.

    handler.train_multistage_3b resolves exactly what to train from
    training_stages.py by stage_idx, so this only needs to pass the index.
    """
    description = stage_config["description"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Stage {stage_idx + 1}/{len(TRAINING_STAGES)}: {description}")
    logger.info(f"{'='*60}")

    request = {
        "action": "train_multistage_3b",
        "parameters": {
            "stage_index": stage_idx,
            "reset_checkpoint": reset_checkpoint,
        },
    }

    start_time = time.time()
    results = handler(request)
    elapsed = time.time() - start_time

    result = results[0] if results else {}
    logger.info(f"Stage completed in {elapsed:.1f}s")
    logger.info(f"Result: {json.dumps(result, indent=2, default=str, ensure_ascii=False)}")

    return result.get("status") == "success" or result.get("status") == "completed"


def run_multistage_training(budget_dollars=10, reset_checkpoint=True):
    """Execute all training stages sequentially against a single loaded model."""
    logger.info(f"Starting multi-stage 3B training (Budget: ${budget_dollars})")
    logger.info(f"Total stages: {len(TRAINING_STAGES)}")

    # Log stage order
    for i, stage in enumerate(TRAINING_STAGES, 1):
        logger.info(f"  {i}. {stage['description']}")

    from handler import EndpointHandler

    if reset_checkpoint:
        # Reset the on-disk checkpoint *before* loading the model, since
        # loading already pulls any existing checkpoint into memory and a
        # reset afterwards can't undo that (see EndpointHandler's own
        # requires_worker_restart note on _handle_reset_checkpoint).
        logger.info("Resetting checkpoint for a fresh run...")
        reset_handler = EndpointHandler(path=".")
        logger.info(f"Checkpoint reset result: {reset_handler._handle_reset_checkpoint()}")
        del reset_handler

    handler = EndpointHandler(path=".")

    # Run each stage against the same loaded model/handler instead of
    # reloading the 3B checkpoint from disk for every stage.
    successful_stages = 0
    for idx, stage in enumerate(TRAINING_STAGES):
        # reset_checkpoint was already handled above; the handler's own
        # multistage state file guards against re-resetting mid-run.
        if run_training_stage(handler, idx, stage, reset_checkpoint=False):
            successful_stages += 1
            logger.info(f"Progress: {successful_stages}/{len(TRAINING_STAGES)} stages completed")
        else:
            logger.error(f"Failed at stage {idx + 1}: {stage['description']}")
            break

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
