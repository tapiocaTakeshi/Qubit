"""
RunPod Serverless Handler for NeuroQ QBNN

Wraps the EndpointHandler from handler.py to work with RunPod's
serverless infrastructure via runpod.serverless.start().

RunPod sends requests in the format:
    {"input": {"prompt": "...", "parameters": {...}}}

This handler translates that into the EndpointHandler format and
returns the result.
"""

import os
import sys
import traceback
import runpod

sys.path.insert(0, os.path.dirname(__file__))

# ------------------------------------------------------------------
# Global model instance (loaded once at cold start)
# ------------------------------------------------------------------

NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")
MODEL_DIR = os.environ.get("MODEL_DIR", "/app")

# Prefer network volume for persistent checkpoints across pod restarts
if os.path.isdir(NETWORK_VOLUME_PATH):
    print(f"[runpod_handler] Network volume available at {NETWORK_VOLUME_PATH}")

handler = None
startup_error = None

# Keep the RunPod queue worker alive even when application imports or model
# initialization fail.  A status request can then return the full traceback
# instead of leaving every job stuck in IN_QUEUE while the container restarts.
try:
    from handler import EndpointHandler

    handler = EndpointHandler(path=MODEL_DIR)
    print(f"[runpod_handler] Model loaded (MODEL_DIR={MODEL_DIR}, checkpoint={handler.ckpt_path})")
except BaseException:
    startup_error = traceback.format_exc()
    print("[runpod_handler] Application startup failed; diagnostic mode enabled")
    print(startup_error)


# ------------------------------------------------------------------
# RunPod handler function
# ------------------------------------------------------------------

def run_handler(event):
    """
    RunPod serverless handler.

    Expected input format:
        {
            "input": {
                "prompt": "こんにちは",
                "action": "inference",       # optional
                "parameters": {              # optional
                    "temperature": 0.7,
                    "max_new_tokens": 100,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repetition_penalty": 1.3
                }
            }
        }

    Supported actions:
        inference (default), agent, train, train_qa, train_qa_dataset,
        train_split, train_split_next, train_dpo, train_combined_dpo,
        split_status, split_reset, status
    """
    if handler is None:
        return {
            "status": "error",
            "message": "Worker application startup failed",
            "startup_error": startup_error,
        }

    job_input = event.get("input", {})

    # Translate RunPod input to EndpointHandler format
    data = {}

    if "action" in job_input:
        data["action"] = job_input["action"]

    if "prompt" in job_input:
        data["inputs"] = job_input["prompt"]
    elif "inputs" in job_input:
        data["inputs"] = job_input["inputs"]

    if "parameters" in job_input:
        data["parameters"] = dict(job_input["parameters"])

    # Preserve all training controls that may be sent at input top level.
    # In particular, support both dataset_ids and the common datasets alias.
    # The previous version silently discarded datasets and max_samples_per_dataset,
    # causing training to fall back to DEFAULT_DATASETS and default hyperparameters.
    passthrough_keys = (
        "qa_pairs", "dataset_ids", "datasets", "dataset", "dataset_id",
        "training_dataset", "max_samples_per_dataset", "max_seq_len",
        "epochs", "lr", "batch_size", "mode", "num_chunks", "resume",
        "dpo_beta", "grad_accum_steps", "warmup_steps", "grad_clip",
        "max_samples_hf", "qa_epochs", "qa_lr", "qa_batch_size",
        "qa_grad_accum", "dpo_epochs", "dpo_lr", "dpo_batch_size",
        "dpo_grad_accum",
    )
    for key in passthrough_keys:
        if key in job_input:
            data.setdefault("parameters", {})[key] = job_input[key]

    from neuroquantum_agent_progress import dispatch_job
    result = dispatch_job(handler, data, event)

    if isinstance(result, list) and len(result) == 1:
        return result[0]
    return result


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

runpod.serverless.start({"handler": run_handler})
