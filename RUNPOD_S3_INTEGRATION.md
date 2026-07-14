# RunPod S3 Integration Guide

Complete guide for integrating S3 storage with RunPod serverless endpoints for NeuroQ QBNN models.

## Overview

The RunPod S3 Integration provides:
- Model upload/download to/from S3
- Training checkpoint management
- Dataset storage and retrieval
- Automated backup of generated models
- Persistent storage across pod restarts

## Configuration

### Environment Variables

Set these environment variables in your RunPod endpoint:

```bash
# S3 Bucket Configuration
RUNPOD_S3_BUCKET=y5bpqo1548
RUNPOD_S3_ENDPOINT=https://s3api-us-il-1.runpod.io
RUNPOD_S3_REGION=us-il-1

# AWS Credentials (from RunPod or your provider)
RUNPOD_S3_ACCESS_KEY=your_access_key
RUNPOD_S3_SECRET_KEY=your_secret_key
```

### Installation

```bash
pip install boto3
```

## Usage

### Basic S3 Manager

```python
from runpod_s3_integration import RunPodS3Manager

# Initialize
manager = RunPodS3Manager()

# List files
files = manager.list_files(prefix="models/")
print(f"Found {len(files)} models")

# Upload a model
result = manager.upload_model("/local/path/to/model.pt", "model_v1")

# Download a model
manager.download_model("model_v1", "/local/models/")

# List checkpoints
checkpoints = manager.list_checkpoints()

# Upload checkpoint
manager.upload_checkpoint("/checkpoints/run1", "run1_final")

# Download checkpoint (for resuming training)
manager.download_checkpoint("run1_final", "/checkpoints/")
```

### RunPod Handler Integration

Integrate S3 into your RunPod handler:

```python
import runpod
from runpod_handler import EndpointHandler
from runpod_handler_s3 import handle_s3_request

handler = EndpointHandler()
s3_handler = handle_s3_request

def run_handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action")
    
    # Check for S3 actions
    s3_result = s3_handler(job_input)
    if s3_result is not None:
        return s3_result
    
    # Continue with inference/training...
    return handler(job_input)

runpod.serverless.start({"handler": run_handler})
```

## Supported Actions

### Model Management

#### Upload Model
```json
{
    "input": {
        "action": "upload_model",
        "model_path": "/app/outputs/model.pt",
        "model_name": "my_model_v1"
    }
}
```

Response:
```json
{
    "status": "success",
    "data": {
        "s3_key": "models/my_model_v1",
        "local_path": "/app/outputs/model.pt",
        "file_size": 1024000,
        "status": "uploaded"
    }
}
```

#### Download Model
```json
{
    "input": {
        "action": "download_model",
        "model_name": "my_model_v1",
        "model_dir": "/models"
    }
}
```

#### List Models
```json
{
    "input": {
        "action": "list_models"
    }
}
```

Response:
```json
{
    "status": "success",
    "data": [
        {
            "key": "models/my_model_v1",
            "size": 1024000,
            "last_modified": "2023-01-15T10:30:00Z"
        }
    ]
}
```

### Checkpoint Management

#### Upload Checkpoint
```json
{
    "input": {
        "action": "upload_checkpoint",
        "checkpoint_path": "/checkpoints/epoch_10",
        "checkpoint_name": "run1_epoch10"
    }
}
```

#### Download Checkpoint
```json
{
    "input": {
        "action": "download_checkpoint",
        "checkpoint_name": "run1_epoch10",
        "checkpoint_dir": "/checkpoints"
    }
}
```

#### List Checkpoints
```json
{
    "input": {
        "action": "list_checkpoints"
    }
}
```

### Dataset Management

#### List Datasets
```json
{
    "input": {
        "action": "list_datasets"
    }
}
```

## CLI Usage

### List Files
```bash
python runpod_s3_integration.py list
python runpod_s3_integration.py list-prefix models/
```

### Upload File
```bash
python runpod_s3_integration.py upload /local/file.pt models/file.pt
```

### Download File
```bash
python runpod_s3_integration.py download models/file.pt /local/file.pt
```

### Model Operations
```bash
# Upload entire model directory
python runpod_s3_integration.py upload-model /local/model_dir my_model_v1

# Download model
python runpod_s3_integration.py download-model my_model_v1 /models/

# List checkpoints
python runpod_s3_integration.py list-checkpoints
```

## Advanced Usage

### Upload Directory
```python
manager = RunPodS3Manager()

# Upload entire checkpoint directory
results = manager.upload_directory(
    local_dir="/checkpoints/run1",
    s3_prefix="checkpoints/run1",
    recursive=True
)

print(f"Uploaded {len(results)} files")
```

### Download Directory
```python
# Download all models
manager.download_directory(
    s3_prefix="models/",
    local_dir="/models/"
)
```

### Custom S3 Configuration
```python
manager = RunPodS3Manager(
    bucket_name="custom-bucket",
    endpoint_url="https://custom-s3.example.com",
    region="us-west-2",
    access_key="your_key",
    secret_key="your_secret"
)
```

### Check File Existence
```python
if manager.file_exists("models/my_model_v1"):
    print("Model exists in S3")
else:
    print("Model not found")

size = manager.get_file_size("models/my_model_v1")
print(f"Model size: {size} bytes")
```

## S3 Bucket Structure

Recommended folder organization:

```
s3://y5bpqo1548/
├── models/
│   ├── my_model_v1/
│   ├── my_model_v2/
│   └── ...
├── checkpoints/
│   ├── run1/
│   │   ├── checkpoint_epoch1/
│   │   ├── checkpoint_epoch2/
│   │   └── ...
│   ├── run2/
│   └── ...
└── datasets/
    ├── dataset_1/
    ├── dataset_2/
    └── ...
```

## Error Handling

```python
from botocore.exceptions import ClientError

try:
    manager.upload_file("model.pt", "models/model.pt")
except FileNotFoundError as e:
    print(f"Local file not found: {e}")
except ClientError as e:
    print(f"S3 error: {e}")
    if e.response["Error"]["Code"] == "NoSuchBucket":
        print("Bucket does not exist")
    elif e.response["Error"]["Code"] == "AccessDenied":
        print("Access denied - check credentials")
```

## Performance Tips

1. **Use Multipart Upload for Large Files**: boto3 automatically uses multipart upload for files > 5GB

2. **Batch Operations**: Group multiple file operations to reduce API calls

3. **Network Volume**: Use RunPod Network Volumes for faster I/O during training

4. **S3 Lifecycle Policies**: Archive old checkpoints automatically

5. **Concurrent Downloads**: Use threading for parallel downloads

```python
from concurrent.futures import ThreadPoolExecutor

def download_models_parallel(model_names, local_dir):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(manager.download_model, name, local_dir)
            for name in model_names
        ]
        return [f.result() for f in futures]
```

## Troubleshooting

### Authentication Error: "SignatureDoesNotMatch"
- Verify `RUNPOD_S3_ACCESS_KEY` and `RUNPOD_S3_SECRET_KEY` are correct
- Check credentials haven't expired

### Error: "NoSuchBucket"
- Verify bucket name is correct (case-sensitive)
- Check bucket exists in the specified region

### Slow Upload/Download Speeds
- Check network connectivity to S3 endpoint
- Verify endpoint URL is correct for your region
- Consider using RunPod Network Volume for local caching

### "Connection Refused" Error
- Verify endpoint URL: `https://s3api-us-il-1.runpod.io`
- Check if RunPod S3 service is operational
- Try using S3 web browser: https://console.runpod.io/storage

## Integration with Training

```python
from handler import EndpointHandler
from runpod_s3_integration import RunPodS3Manager

class TrainingWithS3:
    def __init__(self):
        self.handler = EndpointHandler()
        self.s3 = RunPodS3Manager()
    
    def train_with_checkpoint_save(self, data):
        # Start training
        result = self.handler.train(data)
        
        # Upload checkpoint after training
        if os.path.exists("/checkpoints/latest"):
            self.s3.upload_checkpoint(
                "/checkpoints/latest",
                f"training_run_{data.get('run_id')}"
            )
        
        return result
    
    def resume_training_from_s3(self, checkpoint_name, data):
        # Download checkpoint
        self.s3.download_checkpoint(checkpoint_name, "/checkpoints")
        
        # Resume training with checkpoint
        data["resume"] = True
        data["checkpoint_path"] = "/checkpoints"
        
        return self.handler.train(data)
```

## Security Considerations

1. **Protect Credentials**: Never commit AWS credentials to git
   ```bash
   # Use .gitignore
   echo "RUNPOD_S3_*" >> .gitignore
   ```

2. **Use IAM Roles**: Prefer IAM roles over explicit credentials when possible

3. **Encrypt Sensitive Data**: Use S3 server-side encryption
   ```python
   manager.upload_file(
       "model.pt",
       "models/model.pt",
       extra_args={"ServerSideEncryption": "AES256"}
   )
   ```

4. **Access Control**: Restrict S3 bucket access to necessary endpoints

5. **Audit Logging**: Enable S3 access logging for compliance

## References

- [RunPod Documentation](https://docs.runpod.io/)
- [RunPod S3 Storage](https://docs.runpod.io/storage/runpod-storage/)
- [Boto3 S3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/BestPractices.html)
