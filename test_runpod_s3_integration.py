"""
Test suite for RunPod S3 Integration

Tests S3 manager functionality with mock and real connections.
Run with: pytest test_runpod_s3_integration.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))

from runpod_s3_integration import RunPodS3Manager


class TestRunPodS3Manager:
    """Test suite for RunPodS3Manager."""

    @patch("runpod_s3_integration.boto3.client")
    def test_initialization(self, mock_boto3_client):
        """Test S3 manager initialization."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        manager = RunPodS3Manager(
            bucket_name="test-bucket",
            endpoint_url="https://s3.example.com",
            region="us-west-1",
            access_key="test-key",
            secret_key="test-secret",
        )

        assert manager.bucket_name == "test-bucket"
        assert manager.endpoint_url == "https://s3.example.com"
        assert manager.region == "us-west-1"
        assert manager.access_key == "test-key"
        assert manager.secret_key == "test-secret"
        assert manager.s3_client is not None

    @patch("runpod_s3_integration.boto3.client")
    def test_list_files(self, mock_boto3_client):
        """Test listing files."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "file1.txt", "Size": 100, "LastModified": "2023-01-01"},
                {"Key": "file2.txt", "Size": 200, "LastModified": "2023-01-02"},
            ]
        }

        manager = RunPodS3Manager()
        files = manager.list_files(prefix="test/")

        assert len(files) == 2
        assert files[0]["key"] == "file1.txt"
        assert files[0]["size"] == 100
        assert files[1]["key"] == "file2.txt"

    @patch("runpod_s3_integration.boto3.client")
    def test_file_exists(self, mock_boto3_client):
        """Test checking file existence."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        manager = RunPodS3Manager()

        # File exists
        mock_s3_client.head_object.return_value = {"ContentLength": 100}
        assert manager.file_exists("existing_file.txt") is True

        # File does not exist
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, "HeadObject")
        assert manager.file_exists("missing_file.txt") is False

    @patch("runpod_s3_integration.boto3.client")
    def test_upload_file(self, mock_boto3_client):
        """Test file upload."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        manager = RunPodS3Manager()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            result = manager.upload_file(temp_file, "test_key.txt")

            assert result["status"] == "uploaded"
            assert result["s3_key"] == "test_key.txt"
            assert result["local_path"] == temp_file
            assert result["file_size"] > 0

            # Verify upload_file was called
            mock_s3_client.upload_file.assert_called_once()
        finally:
            os.unlink(temp_file)

    @patch("runpod_s3_integration.boto3.client")
    def test_download_file(self, mock_boto3_client):
        """Test file download."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        manager = RunPodS3Manager()

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, "downloaded.txt")

            # Create a file to simulate download
            Path(local_path).write_text("test")

            result = manager.download_file("test_key.txt", local_path)

            assert result["status"] == "downloaded"
            assert result["s3_key"] == "test_key.txt"
            assert result["local_path"] == local_path

    @patch("runpod_s3_integration.boto3.client")
    def test_delete_file(self, mock_boto3_client):
        """Test file deletion."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        manager = RunPodS3Manager()
        result = manager.delete_file("test_key.txt")

        assert result["status"] == "deleted"
        assert result["s3_key"] == "test_key.txt"
        mock_s3_client.delete_object.assert_called_once()

    @patch("runpod_s3_integration.boto3.client")
    def test_upload_model(self, mock_boto3_client):
        """Test model upload."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        manager = RunPodS3Manager()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pt") as f:
            f.write("model data")
            temp_file = f.name

        try:
            result = manager.upload_model(temp_file, "my_model")

            assert result["status"] == "uploaded"
            assert "models/my_model" in result["s3_key"]
            mock_s3_client.upload_file.assert_called_once()
        finally:
            os.unlink(temp_file)

    @patch("runpod_s3_integration.boto3.client")
    def test_list_checkpoints(self, mock_boto3_client):
        """Test listing checkpoints."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "checkpoints/run1/checkpoint.pt",
                    "Size": 500,
                    "LastModified": "2023-01-01",
                },
                {
                    "Key": "checkpoints/run2/checkpoint.pt",
                    "Size": 600,
                    "LastModified": "2023-01-02",
                },
            ]
        }

        manager = RunPodS3Manager()
        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 2
        mock_s3_client.list_objects_v2.assert_called_once()

    @patch("runpod_s3_integration.boto3.client")
    def test_environment_variables(self, mock_boto3_client):
        """Test initialization from environment variables."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        # Set environment variables
        os.environ["RUNPOD_S3_BUCKET"] = "env-bucket"
        os.environ["RUNPOD_S3_ENDPOINT"] = "https://env.example.com"
        os.environ["RUNPOD_S3_REGION"] = "eu-west-1"

        try:
            manager = RunPodS3Manager()

            assert manager.bucket_name == "env-bucket"
            assert manager.endpoint_url == "https://env.example.com"
            assert manager.region == "eu-west-1"
        finally:
            # Clean up environment variables
            for var in ["RUNPOD_S3_BUCKET", "RUNPOD_S3_ENDPOINT", "RUNPOD_S3_REGION"]:
                os.environ.pop(var, None)


class TestRunPodS3Handler:
    """Test suite for RunPodS3Handler."""

    @patch("runpod_handler_s3.RunPodS3Manager")
    def test_handler_upload_model(self, mock_s3_manager_class):
        """Test S3 handler model upload."""
        from runpod_handler_s3 import RunPodS3Handler

        mock_manager = MagicMock()
        mock_s3_manager_class.return_value = mock_manager
        mock_manager.upload_model.return_value = {
            "status": "uploaded",
            "s3_key": "models/test",
        }

        handler = RunPodS3Handler()
        result = handler.handle_upload_model(
            {"model_path": "/tmp/model.pt", "model_name": "test"}
        )

        assert result["status"] == "success"
        mock_manager.upload_model.assert_called_once()

    @patch("runpod_handler_s3.RunPodS3Manager")
    def test_handler_missing_params(self, mock_s3_manager_class):
        """Test handler with missing required parameters."""
        from runpod_handler_s3 import RunPodS3Handler

        mock_manager = MagicMock()
        mock_s3_manager_class.return_value = mock_manager

        handler = RunPodS3Handler()
        result = handler.handle_upload_model({})

        assert "error" in result
        assert "model_path" in result["error"] or "model_name" in result["error"]


def test_s3_defaults():
    """Test S3 manager default values."""
    assert RunPodS3Manager.DEFAULT_BUCKET == "y5bpqo1548"
    assert RunPodS3Manager.DEFAULT_REGION == "us-il-1"
    assert RunPodS3Manager.DEFAULT_ENDPOINT == "https://s3api-us-il-1.runpod.io"


if __name__ == "__main__":
    # Run basic tests
    print("Running RunPod S3 Integration Tests...")

    test = TestRunPodS3Manager()
    try:
        test.test_s3_defaults()
        print("✓ S3 defaults test passed")
    except Exception as e:
        print(f"✗ S3 defaults test failed: {e}")

    print("\nAll tests completed. For full test suite, run: pytest test_runpod_s3_integration.py")
