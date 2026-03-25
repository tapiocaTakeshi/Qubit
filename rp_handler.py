"""
RunPod Serverless Handler for Qubit (NeuroQuantum) model.

This wraps the existing EndpointHandler for RunPod's serverless infrastructure.
RunPod calls handler(job) with job["input"] containing the request data.
"""

import runpod
import os
import sys

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(__file__))

from handler import EndpointHandler

# Initialize model once at module load (cold start)
print("Initializing Qubit NeuroQuantum model...")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.dirname(__file__))
endpoint_handler = EndpointHandler(MODEL_DIR)
print(f"Model loaded: {endpoint_handler.architecture}, device: {endpoint_handler.device}")


def handler(job):
    """
    RunPod serverless handler entry point.

    Input format (compatible with HF Inference Endpoints):
        {
            "prompt": "日本の首都は",           # Simple inference
            "temperature": 0.8,
            "max_new_tokens": 100,
            ...
        }
    OR HF-style:
        {
            "inputs": "日本の首都は",
            "parameters": { "temperature": 0.8, ... }
        }
    OR action-based:
        {
            "action": "train_qa",
            "qa_pairs": [...],
            ...
        }
    """
    job_input = job.get("input", {})

    # Convert RunPod-style input to HF EndpointHandler format
    if "inputs" in job_input:
        # Already in HF format
        data = job_input
    elif "action" in job_input:
        # Action-based request (train, status, etc.)
        action = job_input.pop("action")
        data = {
            "inputs": f"__{action}__",
            "parameters": {
                "action": action,
                **job_input,
            }
        }
    elif "prompt" in job_input:
        # Simple prompt-based inference
        prompt = job_input.pop("prompt")
        data = {
            "inputs": prompt,
            "parameters": job_input,
        }
    else:
        return {"error": "No 'prompt', 'inputs', or 'action' found in input"}

    try:
        result = endpoint_handler(data)

        # Unwrap single-element list for cleaner RunPod response
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result
    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
