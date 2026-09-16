# Automated Multi-Stage Training on RunPod

Automatically orchestrates the full 7-stage training pipeline with automatic progression to the next stage when each completes.

## Quick Start

### 1. Set Environment Variables

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"
```

Get these from:
- **Endpoint ID:** RunPod dashboard → Your endpoint → Endpoint ID
- **API Key:** RunPod dashboard → Settings → API Keys

### 2. Run Once (Status Check)

Check current status and submit next stage if ready:

```bash
chmod +x start_multistage_auto.sh
python3 runpod_multistage_auto.py
```

### 3. Run in Daemon Mode (Recommended)

Continuously monitor and auto-progress through stages:

```bash
./start_multistage_auto.sh --daemon 300 &
```

This checks every 300 seconds (5 minutes) and:
- ✓ Monitors current job status
- ✓ Detects stage completion
- ✓ Automatically submits next stage
- ✓ Logs all activity with timestamps

## Training Stages

| Stage | Dataset | Samples | LR | Description |
|-------|---------|---------|-----|-------------|
| 1 | FineWeb-2 Japanese | 50K | 1e-4 | Foundation learning |
| 2 | ABEJA-CC-JA-edu | 30K | 8e-5 | Curated Japanese |
| 3 | Wikipedia | 40K | 5e-5 | Mixed language |
| 4 | Instruction | 20K | 3e-5 | Following instructions |
| 5 | Conversation | 10K | 2e-5 | Natural dialogue |
| 6 | Mathematics | 15K | 2e-5 | Reasoning |
| 7 | Code (swallow-code-v2) | 100K | 1e-5 | Programming |

## State Tracking

Progress is saved to `multistage_state.json`:

```json
{
  "current_stage": 2,
  "job_ids": ["job-id-1", "job-id-2"],
  "completed_stages": [1]
}
```

This allows resuming if the script is interrupted.

## Monitoring

### Check Status

```bash
python3 runpod_multistage_auto.py
```

Output shows:
- Current stage progress (1/7)
- Completed stages ✓
- Current job ID
- Next stage to submit

### View Job Details

```bash
export RUNPOD_API_KEY="..."
JOB_ID="..."

curl -s "https://api.runpod.ai/v2/aic6yigpthbck5/status/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -m json.tool
```

### Check Logs

Daemon mode logs appear in stdout. For persistent logging:

```bash
./start_multistage_auto.sh --daemon 300 > multistage.log 2>&1 &
tail -f multistage.log
```

## Troubleshooting

### Job Fails at a Stage

If a stage fails:
1. Check the error in job status
2. Fix the underlying issue (disk quota, dataset availability, etc.)
3. Manually edit `multistage_state.json` to reset to that stage
4. Re-run the orchestrator

### Disk Quota Exceeded

The script automatically sets `clear_cache: True` for each stage. If issues persist:

```bash
# Request manual cache cleanup via RunPod dashboard
# Or reduce max_samples in runpod_multistage_auto.py
```

### Network Issues

The orchestrator has retry logic. Transient network errors are logged as warnings and retried on the next check.

## Advanced Configuration

### Adjust Samples or Learning Rates

Edit `TRAINING_STAGES` in `runpod_multistage_auto.py`:

```python
{
    "id": 1,
    "name": "fineweb_japanese",
    "datasets": [{"id": "HuggingFaceFW/fineweb-2-edu-japanese", "split": None, "max_samples": 50000}],
    "learning_rate": 1e-4,
    "epochs": 1,
},
```

### Change Check Interval

Run daemon with custom interval (in seconds):

```bash
./start_multistage_auto.sh --daemon 600 &  # 10 minutes
```

### Reset Training

Delete state file and all jobs:

```bash
rm multistage_state.json
# Manually cancel running jobs in RunPod dashboard
python3 runpod_multistage_auto.py  # Starts fresh from stage 1
```

## Integration with RunPod

### Recommended Setup

1. **Terminal 1: Run daemon**
   ```bash
   export RUNPOD_API_KEY="..."
   ./start_multistage_auto.sh --daemon 300 &
   ```

2. **Terminal 2: Monitor logs**
   ```bash
   tail -f multistage.log
   ```

3. **Web: RunPod Dashboard**
   - Monitor real-time GPU utilization
   - Watch job queue
   - Check billing/budget

### Cost Tracking

Each stage runs for ~30 minutes (A100 40GB):
- Total time: ~3.5 hours for all 7 stages
- Cost at $0.44/hr: ~$1.50 (well under $10 budget)

## Files

- `runpod_multistage_auto.py` — Main orchestrator script
- `start_multistage_auto.sh` — Daemon launcher with cron-like behavior
- `multistage_state.json` — Persistent training state (auto-created)

## Tips

- Run daemon in a persistent screen/tmux session
- Monitor the log file periodically
- Check RunPod dashboard for GPU health
- Save stage checkpoints to persistent volume automatically
- Each stage can be re-run by editing state.json

---

Last Updated: 2026-09-16
