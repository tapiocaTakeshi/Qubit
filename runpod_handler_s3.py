"""
RunPod Serverless Handler with S3 Integration for NeuroQ QBNN

Extends the base RunPod handler to support S3 operations:
- upload_model: Upload generated models to S3
- download_model: Download models from S3 before inference
- upload_checkpoint: Upload training checkpoints to S3
- download_checkpoint: Resume training from S3 checkpoints
- list_s3_models: List available models in S3
- list_s3_checkpoints: List available checkpoints in S3

Usage example:
    {
        "input": {
            "action": "upload_model",
            "model_path": "/app/outputs/model.pt",
            "model_name": "my_model_v1"
        }
    }

    {
        "input": {
            "action": "download_model",
            "model_name": "my_model_v1",
            "model_dir": "/models"
        }
    }

    {
        "input": {
            "action": "inference",
            "prompt": "こんにちは",
            "s3_model": "my_model_v1"
        }
    }
"""

import os
import sys
import json
import logging
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))

from runpod_s3_integration import RunPodS3Manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RunPodS3Handler:
    """Handler for S3 operations on RunPod."""

    def __init__(self):
        """Initialize S3 handler."""
        try:
            self.s3_manager = RunPodS3Manager()
            logger.info("S3 Manager initialized successfully")
        except Exception as e:
            logger.warning(f"S3 Manager initialization failed: {e}")
            self.s3_manager = None

    def handle_upload_model(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model upload to S3."""
        if not self.s3_manager:
            return {"error": "S3 not configured"}

        model_path = input_data.get("model_path")
        model_name = input_data.get("model_name")

        if not model_path or not model_name:
            return {"error": "model_path and model_name are required"}

        try:
            result = self.s3_manager.upload_model(model_path, model_name)
            logger.info(f"Model uploaded: {model_name}")
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"status": "error", "error": str(e)}

    def handle_download_model(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model download from S3."""
        if not self.s3_manager:
            return {"error": "S3 not configured"}

        model_name = input_data.get("model_name")
        model_dir = input_data.get("model_dir", "/tmp/models")

        if not model_name:
            return {"error": "model_name is required"}

        try:
            results = self.s3_manager.download_model(model_name, model_dir)
            logger.info(f"Model downloaded: {model_name}")
            return {"status": "success", "data": results}
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return {"status": "error", "error": str(e)}

    def handle_upload_checkpoint(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle checkpoint upload to S3."""
        if not self.s3_manager:
            return {"error": "S3 not configured"}

        checkpoint_path = input_data.get("checkpoint_path")
        checkpoint_name = input_data.get("checkpoint_name")

        if not checkpoint_path or not checkpoint_name:
            return {"error": "checkpoint_path and checkpoint_name are required"}

        try:
            result = self.s3_manager.upload_checkpoint(
                checkpoint_path, checkpoint_name
            )
            logger.info(f"Checkpoint uploaded: {checkpoint_name}")
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error(f"Checkpoint upload failed: {e}")
            return {"status": "error", "error": str(e)}

    def handle_download_checkpoint(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle checkpoint download from S3."""
        if not self.s3_manager:
            return {"error": "S3 not configured"}

        checkpoint_name = input_data.get("checkpoint_name")
        checkpoint_dir = input_data.get("checkpoint_dir", "/tmp/checkpoints")

        if not checkpoint_name:
            return {"error": "checkpoint_name is required"}

        try:
            results = self.s3_manager.download_checkpoint(
                checkpoint_name, checkpoint_dir
            )
            logger.info(f"Checkpoint downloaded: {checkpoint_name}")
            return {"status": "success", "data": results}
        except Exception as e:
            logger.error(f"Checkpoint download failed: {e}")
            return {"status": "error", "error": str(e)}

    def handle_list_models(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle listing S3 models."""
        if not self.s3_manager:
            return {"error": "S3 not configured"}

        try:
            files = self.s3_manager.list_files(prefix="models/")
            logger.info(f"Listed {len(files)} models")
            return {"status": "success", "data": files}
        except Exception as e:
            logger.error(f"List models failed: {e}")
            return {"status": "error", "error": str(e)}

    def handle_list_checkpoints(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle listing S3 checkpoints."""
        if not self.s3_manager:
            return {"error": "S3 not configured"}

        try:
            checkpoints = self.s3_manager.list_checkpoints()
            logger.info(f"Listed {len(checkpoints)} checkpoints")
            return {"status": "success", "data": checkpoints}
        except Exception as e:
            logger.error(f"List checkpoints failed: {e}")
            return {"status": "error", "error": str(e)}

    def handle_list_datasets(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle listing S3 datasets."""
        if not self.s3_manager:
            return {"error": "S3 not configured"}

        try:
            datasets = self.s3_manager.list_datasets()
            logger.info(f"Listed {len(datasets)} datasets")
            return {"status": "success", "data": datasets}
        except Exception as e:
            logger.error(f"List datasets failed: {e}")
            return {"status": "error", "error": str(e)}

    def handle_s3_action(self, action: str, input_data: Dict[str, Any]) -> Any:
        """Route S3 action to appropriate handler."""
        handlers = {
            "upload_model": self.handle_upload_model,
            "download_model": self.handle_download_model,
            "upload_checkpoint": self.handle_upload_checkpoint,
            "download_checkpoint": self.handle_download_checkpoint,
            "list_models": self.handle_list_models,
            "list_checkpoints": self.handle_list_checkpoints,
            "list_datasets": self.handle_list_datasets,
        }

        handler = handlers.get(action)
        if not handler:
            return {"error": f"Unknown S3 action: {action}"}

        return handler(input_data)


# Global S3 handler instance
s3_handler = RunPodS3Handler()


def handle_s3_request(input_data: Dict[str, Any]) -> Any:
    """
    Handle S3-related requests from RunPod.

    This can be integrated into the main RunPod handler to support
    S3 operations alongside inference and training.
    """
    action = input_data.get("action")

    # Check if this is an S3 action
    s3_actions = {
        "upload_model",
        "download_model",
        "upload_checkpoint",
        "download_checkpoint",
        "list_models",
        "list_checkpoints",
        "list_datasets",
    }

    if action in s3_actions:
        return s3_handler.handle_s3_action(action, input_data)

    return None


# ------------------------------------------------------------------
# Integration example for main RunPod handler
# ------------------------------------------------------------------

"""
To integrate S3 support into your existing RunPod handler,
add this to your run_handler function:

def run_handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action")

    # Check for S3 actions
    s3_result = handle_s3_request(job_input)
    if s3_result is not None:
        return s3_result

    # Continue with existing handler logic...
    # ... (inference, training, etc.)

"""


# ------------------------------------------------------------------
# CLI for testing
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test RunPod S3 Handler"
    )
    sub = parser.add_subparsers(dest="action")

    p_upload = sub.add_parser("upload-model")
    p_upload.add_argument("model_path")
    p_upload.add_argument("model_name")

    p_download = sub.add_parser("download-model")
    p_download.add_argument("model_name")
    p_download.add_argument("--model-dir", default="/tmp/models")

    sub.add_parser("list-models")
    sub.add_parser("list-checkpoints")

    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        raise SystemExit(1)

    input_data = vars(args)
    input_data.pop("action", None)
    input_data["action"] = args.action

    result = handle_s3_request(input_data)
    print(json.dumps(result, indent=2, default=str))
