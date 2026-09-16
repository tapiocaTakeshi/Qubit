# Multi-Stage 3B Training Guide

## Overview

This guide covers multi-stage training of the new 3B Qubit AI model using RunPod serverless infrastructure with a $10 budget allocation.

**Training Schedule:**
1. **FineWeb-2 Japanese** — Pre-training on high-quality Japanese text
2. **ABEJA-CC-JA-edu** — Curated Japanese educational data
3. **Wikipedia** — Japanese + English Wikipedia articles
4. **Instruction** — Instruction-following datasets (OpenOrca, UltraChat)
5. **Conversation** — Conversational datasets (HH-RLHF-JA, UltraChat)
6. **Mathematics** — Mathematical reasoning (MetaMathQA, GSM8K)
7. **Code** — Programming code (tokyotech-llm/swallow-code-v2)

Each stage:
- Decreases learning rate progressively (prevents catastrophic forgetting)
- Uses appropriate batch sizes and gradient accumulation
- Saves checkpoints after completion
- Logs metrics to `training_logs/`

## Requirements

### Software
- Python 3.10+
- PyTorch 2.0+
- Hugging Face `transformers`, `datasets`, `accelerate`
- RunPod Serverless API access

### Hardware (RunPod)
- **GPU:** A100 (40GB) or H100 recommended for 3B model
- **Budget:** $10 USD
- **Storage:** Network volume for persistent checkpoints

## Setup

### 1. Local Preparation

```bash
# Clone and setup
git clone https://github.com/tapiocaTakeshi/Qubit.git
cd Qubit
pip install -r requirements.txt

# Verify model configuration
python -c "from handler import EndpointHandler; h = EndpointHandler(); print(h.model_config)"
```

### 2. RunPod Endpoint Configuration

Create a RunPod serverless endpoint with:

**Base Configuration:**
- **Container Image:** `qubit-ai-3b:latest`
- **Handler:** `runpod_handler.py`
- **GPU Tier:** A100 40GB (minimum)
- **vCPU:** 4+
- **Memory:** 20GB RAM

**Environment Variables:**
```
MODEL_DIR=/app
NETWORK_VOLUME_PATH=/runpod-volume
HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache
CUDA_VISIBLE_DEVICES=0
```

**Network Volume:**
- Mount at `/runpod-volume`
- Recommended: 100GB+ for model checkpoints and HF cache
- Used for persistent storage across pod restarts

### 3. Authentication

Set environment variables:
```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"
```

Get these from:
- Endpoint ID: RunPod dashboard → Your endpoint → Endpoint ID
- API Key: RunPod dashboard → Settings → API Keys

## Usage

### Option A: Direct Python Script (Local)

```bash
python train_multistage_3b_runpod.py --budget 10
```

This runs training locally (requires GPU locally).

### Option B: Submit to RunPod (Recommended)

```bash
python submit_runpod_training.py --budget 10
```

The script will:
1. Reset checkpoint to start fresh
2. Submit job to RunPod endpoint
3. Return job ID and status URL
4. Print logging instructions

### Option C: Manual cURL Submission

```bash
curl https://api.runpod.io/v2/YOUR_ENDPOINT_ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "action": "train_multistage_3b",
      "budget_dollars": 10,
      "reset_checkpoint": true,
      "log_to_file": true
    }
  }'
```

## Monitoring

### Check Job Status

```bash
curl https://api.runpod.io/v2/YOUR_ENDPOINT_ID/status/YOUR_JOB_ID \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### View Logs

Logs are saved to `training_logs/train_multistage_3b_*.log` on the network volume.

To retrieve them after training:
```bash
# Mount the network volume locally or download from RunPod dashboard
cat /runpod-volume/Qubit/training_logs/train_multistage_3b_*.log
```

### Real-time Monitoring

The handler publishes progress events to stdout/stderr:
- Dataset loading progress
- Batch processing updates
- Loss/accuracy metrics
- Checkpoint save confirmations

These appear in the RunPod job logs.

## Checkpoints

### Automatic Checkpoint Saving

After each stage completes, checkpoints are saved:
```
checkpoints/
  stage_fineweb_japanese.pt
  stage_abeja_cc_ja_edu.pt
  stage_wikipedia.pt
  stage_instruction.pt
  stage_conversation.pt
  stage_mathematics.pt
  stage_code.pt
```

### Resuming from Checkpoint

To resume from a specific stage:

1. **Option A:** Use the stage checkpoint directly:
   ```python
   model.load_state_dict(torch.load("checkpoints/stage_mathematics.pt"))
   # Then continue with remaining stages
   ```

2. **Option B:** Copy checkpoint to network volume:
   ```bash
   cp checkpoints/stage_mathematics.pt /runpod-volume/neuroq_checkpoint.pt
   ```
   Then restart training from the next stage.

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `train_multistage_3b_runpod.py`:
```python
"batch_size": 1,  # Down from 2
"gradient_accumulation_steps": 8,  # Up from 4
```

### Dataset Loading Fails

Some datasets may have temporary issues. The script continues to the next stage on error. To retry a specific stage:

```bash
# Send training request for just one stage
curl https://api.runpod.io/v2/YOUR_ENDPOINT_ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "action": "train",
      "data": {
        "datasets": [{"id": "meta-math/MetaMathQA", "split": null, "max_samples": 100000}],
        "learning_rate": 2e-5,
        "epochs": 2
      }
    }
  }'
```

### Network Volume Not Persisting

Verify network volume is mounted:
```python
import os
assert os.path.isdir("/runpod-volume"), "Network volume not mounted"
```

### Cold Start Delays

First job may take 5-10 minutes to start (container initialization). Subsequent jobs are faster.

## Budget Breakdown ($10 Estimate)

**GPU Time (A100 40GB):** ~$0.44/hour
- FineWeb-2: 4-6 hours → $2-3
- ABEJA-CC-JA-edu: 2-3 hours → $1-1.50
- Wikipedia: 1-2 hours → $0.50-1
- Instruction: 2-3 hours → $1-1.50
- Conversation: 1-2 hours → $0.50-1
- Mathematics: 2-3 hours → $1-1.50
- Code: 4-6 hours → $2-3

**Total:** ~$10-12 USD (may exceed budget depending on dataset sizes)

**To stay under $10:**
- Reduce `max_samples` per dataset (proportional reduction)
- Decrease number of epochs
- Use smaller GPU tier (A10G: $0.14/hr, less VRAM)

## Configuration Reference

### Learning Rate Schedule

```
Stage              | Learning Rate | Purpose
==================|===============|==============================
FineWeb-2 Japanese | 1e-4          | Foundation learning
ABEJA-CC-JA-edu   | 8e-5          | Preserve basics, refine
Wikipedia         | 5e-5          | Mix languages, add context
Instruction       | 3e-5          | Learn following instructions
Conversation      | 2e-5          | Natural dialogue patterns
Mathematics       | 2e-5          | Reasoning capability
Code              | 1e-5          | Fine-tune for programming
```

Progressive learning rate reduction prevents catastrophic forgetting and allows specialization.

### Batch Configuration

```python
batch_size = 2  # Per-device batch size
gradient_accumulation_steps = 4  # Effective batch = 2 * 4 = 8
save_every_n_steps = 500
log_every_n_steps = 10
```

Adjust based on available GPU memory.

## Advanced: Custom Stage Configuration

To modify stages, edit `train_multistage_3b_runpod.py`:

```python
TRAINING_STAGES = [
    {
        "name": "custom_stage",
        "description": "My custom dataset",
        "datasets": [
            {"id": "your-dataset/name", "split": None, "max_samples": 50000}
        ],
        "learning_rate": 5e-5,
        "epochs": 1,
    },
    # ... more stages
]
```

Then submit with the custom script.

## References

- **Handler:** `handler.py` — Main training logic
- **RunPod Integration:** `runpod_handler.py` — Serverless wrapper
- **Model Config:** `training_config.json` — Current model parameters
- **API Docs:** https://docs.runpod.io/serverless/workers
- **Dataset Docs:**
  - FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2-edu
  - tokyotech-llm/swallow-code-v2: https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2

## Notes

- Network volume provides persistence across pod restarts
- Checkpoint backups are created automatically on stage completion
- All logs and metrics are saved locally for analysis
- Budget tracking should be monitored via RunPod dashboard
- This is a research training pipeline; hyperparameters may need tuning

---

Last Updated: 2026-09-16
