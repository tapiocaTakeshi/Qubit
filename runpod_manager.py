"""
RunPod Endpoint Manager for NeuroQ QBNN

Provides programmatic management of RunPod serverless endpoints
using the RunPod API. Supports creating, listing, updating, and
deleting endpoints, as well as checking job status and endpoint health.

Usage:
    from runpod_manager import RunPodManager

    manager = RunPodManager()  # uses RUNPOD_API_KEY env var
    # or
    manager = RunPodManager(api_key="your_key")

    # List endpoints
    endpoints = manager.list_endpoints()

    # Run inference
    result = manager.run_sync("endpoint_id", {"inputs": "こんにちは"})
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional

import runpod

logger = logging.getLogger(__name__)


class RunPodManager:
    """Manages RunPod serverless endpoints for NeuroQ QBNN deployment."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "RunPod API key is required. Set RUNPOD_API_KEY environment "
                "variable or pass api_key argument."
            )
        runpod.api_key = self.api_key

    # ------------------------------------------------------------------
    # Endpoint CRUD
    # ------------------------------------------------------------------

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all serverless endpoints."""
        return runpod.get_endpoints()

    def get_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Get details of a specific endpoint."""
        return runpod.get_endpoint(endpoint_id)

    def create_endpoint(
        self,
        name: str,
        template_id: str,
        gpu_ids: str = "AMPERE_80",
        workers_min: int = 0,
        workers_max: int = 1,
        idle_timeout: int = 5,
        flash_boot: bool = True,
        network_volume_id: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a new serverless endpoint.

        Args:
            name: Display name for the endpoint.
            template_id: RunPod template ID (Docker image config).
            gpu_ids: GPU type filter (e.g. "AMPERE_80", "AMPERE_16", "ADA_24").
            workers_min: Minimum active workers (0 = scale to zero).
            workers_max: Maximum workers to scale up to.
            idle_timeout: Seconds before idle worker shuts down.
            flash_boot: Enable flash boot for faster cold starts.
            network_volume_id: Optional network volume to attach.
            env: Optional environment variables for the worker.
        """
        params = {
            "name": name,
            "templateId": template_id,
            "gpuIds": gpu_ids,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "idleTimeout": idle_timeout,
            "flashBoot": flash_boot,
        }
        if network_volume_id:
            params["networkVolumeId"] = network_volume_id
        if env:
            params["env"] = env

        return runpod.create_endpoint(**params)

    def update_endpoint(
        self,
        endpoint_id: str,
        workers_min: Optional[int] = None,
        workers_max: Optional[int] = None,
        idle_timeout: Optional[int] = None,
        gpu_ids: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing endpoint's scaling configuration."""
        params = {"endpointId": endpoint_id}
        if workers_min is not None:
            params["workersMin"] = workers_min
        if workers_max is not None:
            params["workersMax"] = workers_max
        if idle_timeout is not None:
            params["idleTimeout"] = idle_timeout
        if gpu_ids is not None:
            params["gpuIds"] = gpu_ids

        return runpod.update_endpoint(**params)

    def delete_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Delete a serverless endpoint."""
        return runpod.delete_endpoint(endpoint_id)

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def run_async(
        self, endpoint_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit an async job. Returns job ID immediately."""
        endpoint = runpod.Endpoint(endpoint_id)
        run_request = endpoint.run(payload)
        return {"job_id": run_request.job_id, "status": "IN_QUEUE"}

    def run_sync(
        self,
        endpoint_id: str,
        payload: Dict[str, Any],
        timeout: int = 120,
    ) -> Any:
        """Submit a job and wait for the result.

        Args:
            endpoint_id: The endpoint to call.
            payload: The input payload (e.g. {"inputs": "text"}).
            timeout: Max seconds to wait for completion.
        """
        endpoint = runpod.Endpoint(endpoint_id)
        run_request = endpoint.run_sync(payload, timeout=timeout)
        return run_request

    def get_job_status(
        self, endpoint_id: str, job_id: str
    ) -> Dict[str, Any]:
        """Check the status of an async job."""
        endpoint = runpod.Endpoint(endpoint_id)
        return endpoint.status(job_id)

    def cancel_job(self, endpoint_id: str, job_id: str) -> Dict[str, Any]:
        """Cancel a running or queued job."""
        endpoint = runpod.Endpoint(endpoint_id)
        return endpoint.cancel(job_id)

    # ------------------------------------------------------------------
    # Health & diagnostics
    # ------------------------------------------------------------------

    def health(self, endpoint_id: str) -> Dict[str, Any]:
        """Get health/status info for an endpoint."""
        endpoint = runpod.Endpoint(endpoint_id)
        return endpoint.health()

    def purge_queue(self, endpoint_id: str) -> Dict[str, Any]:
        """Purge all queued jobs for an endpoint."""
        endpoint = runpod.Endpoint(endpoint_id)
        return endpoint.purge_queue()


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="RunPod Endpoint Manager for NeuroQ")
    parser.add_argument("--api-key", help="RunPod API key (or set RUNPOD_API_KEY)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all endpoints")

    p_health = sub.add_parser("health", help="Check endpoint health")
    p_health.add_argument("endpoint_id")

    p_run = sub.add_parser("run", help="Run sync inference")
    p_run.add_argument("endpoint_id")
    p_run.add_argument("prompt", help="Input text prompt")
    p_run.add_argument("--temperature", type=float, default=0.8)
    p_run.add_argument("--max-tokens", type=int, default=30)

    p_status = sub.add_parser("status", help="Check job status")
    p_status.add_argument("endpoint_id")
    p_status.add_argument("job_id")

    p_cancel = sub.add_parser("cancel", help="Cancel a job")
    p_cancel.add_argument("endpoint_id")
    p_cancel.add_argument("job_id")

    p_purge = sub.add_parser("purge", help="Purge job queue")
    p_purge.add_argument("endpoint_id")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    mgr = RunPodManager(api_key=args.api_key)

    if args.command == "list":
        print(json.dumps(mgr.list_endpoints(), indent=2, default=str))

    elif args.command == "health":
        print(json.dumps(mgr.health(args.endpoint_id), indent=2, default=str))

    elif args.command == "run":
        payload = {
            "inputs": args.prompt,
            "parameters": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_tokens,
            },
        }
        result = mgr.run_sync(args.endpoint_id, payload)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "status":
        print(json.dumps(mgr.get_job_status(args.endpoint_id, args.job_id), indent=2, default=str))

    elif args.command == "cancel":
        print(json.dumps(mgr.cancel_job(args.endpoint_id, args.job_id), indent=2, default=str))

    elif args.command == "purge":
        print(json.dumps(mgr.purge_queue(args.endpoint_id), indent=2, default=str))
