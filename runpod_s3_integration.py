"""
RunPod S3 Integration for NeuroQ QBNN

Provides S3 storage integration for RunPod serverless endpoints.
Supports uploading/downloading models, checkpoints, and datasets
to/from RunPod's S3 bucket (y5bpqo1548).

Usage:
    from runpod_s3_integration import RunPodS3Manager

    manager = RunPodS3Manager()
    manager.list_files()
    manager.upload_file("local_path", "s3_key")
    manager.download_file("s3_key", "local_path")
"""

import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError(
        "boto3 is required for S3 integration. Install with: pip install boto3"
    )

logger = logging.getLogger(__name__)


class RunPodS3Manager:
    """Manages S3 storage for RunPod NeuroQ QBNN models and datasets."""

    # RunPod S3 Configuration
    DEFAULT_ENDPOINT = "https://s3api-us-il-1.runpod.io"
    DEFAULT_REGION = "us-il-1"
    DEFAULT_BUCKET = "y5bpqo1548"

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize RunPod S3 Manager.

        Args:
            bucket_name: S3 bucket name (default: y5bpqo1548)
            endpoint_url: S3 endpoint URL (default: RunPod S3 API)
            region: AWS region (default: us-il-1)
            access_key: AWS access key ID
            secret_key: AWS secret access key
        """
        self.bucket_name = bucket_name or os.environ.get(
            "RUNPOD_S3_BUCKET", self.DEFAULT_BUCKET
        )
        self.endpoint_url = endpoint_url or os.environ.get(
            "RUNPOD_S3_ENDPOINT", self.DEFAULT_ENDPOINT
        )
        self.region = region or os.environ.get(
            "RUNPOD_S3_REGION", self.DEFAULT_REGION
        )

        # Get credentials from environment
        self.access_key = access_key or os.environ.get("RUNPOD_S3_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("RUNPOD_S3_SECRET_KEY")

        if not self.access_key or not self.secret_key:
            logger.warning(
                "S3 credentials not found. Set RUNPOD_S3_ACCESS_KEY and "
                "RUNPOD_S3_SECRET_KEY environment variables."
            )

        # Initialize boto3 S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        logger.info(
            f"RunPod S3 Manager initialized "
            f"(bucket={self.bucket_name}, endpoint={self.endpoint_url})"
        )

    # ------------------------------------------------------------------
    # List & Browse
    # ------------------------------------------------------------------

    def list_files(
        self, prefix: str = "", delimiter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket.

        Args:
            prefix: Filter by prefix (e.g. "models/", "checkpoints/")
            delimiter: For directory listing (e.g. "/" for folders)

        Returns:
            List of file metadata dicts
        """
        try:
            params = {"Bucket": self.bucket_name, "Prefix": prefix}
            if delimiter:
                params["Delimiter"] = delimiter

            response = self.s3_client.list_objects_v2(**params)

            files = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    files.append(
                        {
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                        }
                    )

            logger.info(f"Listed {len(files)} files in {self.bucket_name}/{prefix}")
            return files
        except ClientError as e:
            logger.error(f"Error listing files: {e}")
            raise

    def get_file_size(self, s3_key: str) -> int:
        """Get size of a file in S3."""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response["ContentLength"]
        except ClientError as e:
            logger.error(f"Error getting file size for {s3_key}: {e}")
            raise

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Error checking file existence: {e}")
            raise

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 object key
            extra_args: Additional S3 put_object args (e.g. ServerSideEncryption)

        Returns:
            Upload result metadata
        """
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")

            file_size = local_file.stat().st_size
            logger.info(
                f"Uploading {local_path} ({file_size} bytes) to s3://{self.bucket_name}/{s3_key}"
            )

            # Upload with progress
            put_args = extra_args or {}
            self.s3_client.upload_file(
                str(local_file), self.bucket_name, s3_key, ExtraArgs=put_args
            )

            logger.info(f"Successfully uploaded to {s3_key}")
            return {
                "s3_key": s3_key,
                "local_path": local_path,
                "file_size": file_size,
                "status": "uploaded",
            }
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def upload_directory(
        self,
        local_dir: str,
        s3_prefix: str = "",
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Upload a directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix for uploaded files
            recursive: Recursively upload subdirectories

        Returns:
            List of upload results
        """
        try:
            local_path = Path(local_dir)
            if not local_path.is_dir():
                raise NotADirectoryError(f"Not a directory: {local_dir}")

            results = []
            pattern = "**/*" if recursive else "*"

            for file_path in local_path.glob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{relative_path}".lstrip("/")

                    result = self.upload_file(str(file_path), s3_key)
                    results.append(result)

            logger.info(f"Uploaded {len(results)} files from {local_dir}")
            return results
        except Exception as e:
            logger.error(f"Error uploading directory: {e}")
            raise

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_file(
        self,
        s3_key: str,
        local_path: str,
    ) -> Dict[str, Any]:
        """
        Download a file from S3.

        Args:
            s3_key: S3 object key
            local_path: Local destination path

        Returns:
            Download result metadata
        """
        try:
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}"
            )

            self.s3_client.download_file(self.bucket_name, s3_key, str(local_file))

            file_size = local_file.stat().st_size
            logger.info(f"Successfully downloaded {s3_key} ({file_size} bytes)")

            return {
                "s3_key": s3_key,
                "local_path": local_path,
                "file_size": file_size,
                "status": "downloaded",
            }
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

    def download_directory(
        self,
        s3_prefix: str,
        local_dir: str,
    ) -> List[Dict[str, Any]]:
        """
        Download all files under an S3 prefix.

        Args:
            s3_prefix: S3 prefix to download
            local_dir: Local destination directory

        Returns:
            List of download results
        """
        try:
            files = self.list_files(prefix=s3_prefix)
            results = []

            for file_info in files:
                s3_key = file_info["key"]
                relative_key = s3_key[len(s3_prefix) :].lstrip("/")
                local_path = os.path.join(local_dir, relative_key)

                result = self.download_file(s3_key, local_path)
                results.append(result)

            logger.info(f"Downloaded {len(results)} files to {local_dir}")
            return results
        except Exception as e:
            logger.error(f"Error downloading directory: {e}")
            raise

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_file(self, s3_key: str) -> Dict[str, Any]:
        """Delete a file from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted {s3_key}")
            return {"s3_key": s3_key, "status": "deleted"}
        except ClientError as e:
            logger.error(f"Error deleting file: {e}")
            raise

    def delete_prefix(self, s3_prefix: str) -> Dict[str, Any]:
        """Delete all files under an S3 prefix."""
        try:
            files = self.list_files(prefix=s3_prefix)
            deleted_count = 0

            for file_info in files:
                self.delete_file(file_info["key"])
                deleted_count += 1

            logger.info(f"Deleted {deleted_count} files under {s3_prefix}")
            return {"prefix": s3_prefix, "deleted_count": deleted_count}
        except Exception as e:
            logger.error(f"Error deleting prefix: {e}")
            raise

    # ------------------------------------------------------------------
    # Model Management
    # ------------------------------------------------------------------

    def upload_model(
        self,
        model_path: str,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Upload a model directory/file to S3 under models/ prefix.

        Args:
            model_path: Local model path
            model_name: Name for the model in S3

        Returns:
            Upload result
        """
        local_path = Path(model_path)
        if local_path.is_file():
            s3_key = f"models/{model_name}"
            return self.upload_file(str(local_path), s3_key)
        elif local_path.is_dir():
            return self.upload_directory(str(local_path), f"models/{model_name}")
        else:
            raise FileNotFoundError(f"Path not found: {model_path}")

    def download_model(
        self,
        model_name: str,
        local_dir: str,
    ) -> List[Dict[str, Any]]:
        """
        Download a model from S3 under models/ prefix.

        Args:
            model_name: Model name in S3
            local_dir: Local destination directory

        Returns:
            Download results
        """
        return self.download_directory(f"models/{model_name}", local_dir)

    # ------------------------------------------------------------------
    # Checkpoint Management
    # ------------------------------------------------------------------

    def upload_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> Dict[str, Any]:
        """Upload a training checkpoint to S3."""
        local_path = Path(checkpoint_path)
        if local_path.is_file():
            s3_key = f"checkpoints/{checkpoint_name}"
            return self.upload_file(str(local_path), s3_key)
        elif local_path.is_dir():
            return self.upload_directory(
                str(local_path), f"checkpoints/{checkpoint_name}"
            )
        else:
            raise FileNotFoundError(f"Path not found: {checkpoint_path}")

    def download_checkpoint(
        self,
        checkpoint_name: str,
        local_dir: str,
    ) -> List[Dict[str, Any]]:
        """Download a training checkpoint from S3."""
        return self.download_directory(f"checkpoints/{checkpoint_name}", local_dir)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints in S3."""
        return self.list_files(prefix="checkpoints/")

    # ------------------------------------------------------------------
    # Dataset Management
    # ------------------------------------------------------------------

    def upload_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Upload a dataset to S3."""
        local_path = Path(dataset_path)
        if local_path.is_file():
            s3_key = f"datasets/{dataset_name}"
            return self.upload_file(str(local_path), s3_key)
        elif local_path.is_dir():
            return self.upload_directory(
                str(local_path), f"datasets/{dataset_name}"
            )
        else:
            raise FileNotFoundError(f"Path not found: {dataset_path}")

    def download_dataset(
        self,
        dataset_name: str,
        local_dir: str,
    ) -> List[Dict[str, Any]]:
        """Download a dataset from S3."""
        return self.download_directory(f"datasets/{dataset_name}", local_dir)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in S3."""
        return self.list_files(prefix="datasets/")


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="RunPod S3 Manager for NeuroQ")
    parser.add_argument("--bucket", default=RunPodS3Manager.DEFAULT_BUCKET)
    parser.add_argument("--endpoint", default=RunPodS3Manager.DEFAULT_ENDPOINT)
    parser.add_argument("--region", default=RunPodS3Manager.DEFAULT_REGION)
    parser.add_argument("--access-key", help="AWS access key (or RUNPOD_S3_ACCESS_KEY)")
    parser.add_argument(
        "--secret-key", help="AWS secret key (or RUNPOD_S3_SECRET_KEY)"
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List files in bucket")
    p_list = sub.add_parser("list-prefix", help="List files with prefix")
    p_list.add_argument("prefix")

    p_upload = sub.add_parser("upload", help="Upload file to S3")
    p_upload.add_argument("local_path")
    p_upload.add_argument("s3_key")

    p_download = sub.add_parser("download", help="Download file from S3")
    p_download.add_argument("s3_key")
    p_download.add_argument("local_path")

    p_delete = sub.add_parser("delete", help="Delete file from S3")
    p_delete.add_argument("s3_key")

    p_model_upload = sub.add_parser("upload-model", help="Upload model to S3")
    p_model_upload.add_argument("model_path")
    p_model_upload.add_argument("model_name")

    p_model_download = sub.add_parser("download-model", help="Download model from S3")
    p_model_download.add_argument("model_name")
    p_model_download.add_argument("local_dir")

    p_checkpoint_list = sub.add_parser("list-checkpoints", help="List checkpoints")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    mgr = RunPodS3Manager(
        bucket_name=args.bucket,
        endpoint_url=args.endpoint,
        region=args.region,
        access_key=args.access_key,
        secret_key=args.secret_key,
    )

    if args.command == "list":
        files = mgr.list_files()
        print(json.dumps(files, indent=2, default=str))

    elif args.command == "list-prefix":
        files = mgr.list_files(prefix=args.prefix)
        print(json.dumps(files, indent=2, default=str))

    elif args.command == "upload":
        result = mgr.upload_file(args.local_path, args.s3_key)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "download":
        result = mgr.download_file(args.s3_key, args.local_path)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "delete":
        result = mgr.delete_file(args.s3_key)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "upload-model":
        result = mgr.upload_model(args.model_path, args.model_name)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "download-model":
        results = mgr.download_model(args.model_name, args.local_dir)
        print(json.dumps(results, indent=2, default=str))

    elif args.command == "list-checkpoints":
        checkpoints = mgr.list_checkpoints()
        print(json.dumps(checkpoints, indent=2, default=str))
