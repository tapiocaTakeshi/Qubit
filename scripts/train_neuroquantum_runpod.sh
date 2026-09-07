#!/usr/bin/env bash
# Run inside an existing Runpod GPU Pod. No Pod or Serverless job is created.
set -euo pipefail

NQ_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NQ_PYTHON="${NQ_PYTHON:-python}"
NQ_RUN_DIR="${NQ_RUN_DIR:-/workspace/neuroquantum}"
NQ_MODEL_SIZE="${NQ_MODEL_SIZE:-small}"

if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/train_neuroquantum_runpod.sh --dataset-id OWNER/DATASET [training options]' >&2
    exit 2
fi
for argument in "$@"; do
    if [[ "$argument" == '--help' || "$argument" == '-h' ]]; then
        exec "$NQ_PYTHON" "$NQ_REPO_DIR/train_hf_dataset.py" --help
    fi
done

"$NQ_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Use a Runpod GPU Pod with CUDA-enabled PyTorch.")
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name} | VRAM: {gpu.total_memory / 1024**3:.1f} GiB | PyTorch: {torch.__version__}")
print(f"Training precision: {'BF16' if torch.cuda.is_bf16_supported() else 'FP32 (BF16 unsupported)'}")
PY

mkdir -p "$NQ_RUN_DIR"
NQ_RUN_DIR="$(cd -- "$NQ_RUN_DIR" && pwd)"
export HF_HOME="${HF_HOME:-$NQ_RUN_DIR/hf-cache}"

# Later CLI arguments can override these conservative single-GPU defaults.
exec "$NQ_PYTHON" "$NQ_REPO_DIR/train_hf_dataset.py" \
    --model-size "$NQ_MODEL_SIZE" \
    --batch-size 1 \
    --max-seq-len 1024 \
    --attention-window 256 \
    --gradient-accumulation-steps 8 \
    --gradient-checkpointing \
    --use-bf16 \
    --ckpt-name "$NQ_RUN_DIR/checkpoint.pt" \
    --tokenizer-prefix "$NQ_RUN_DIR/tokenizer" \
    "$@"
