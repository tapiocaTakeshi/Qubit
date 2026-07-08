#!/bin/bash
# quick_train_megabyte_100mb.sh
# megabyte_100mb モデルの簡単学習スクリプト
# embedding-gemma-300m（日本語対応）で学習

set -e

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                  NeuroQuantum megabyte_100mb Training                  ║"
echo "║        with embedding-gemma-300m (Japanese-optimized embeddings)       ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check required packages
echo ""
echo "📦 Checking dependencies..."
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || {
    echo "❌ PyTorch not found. Install with:"
    echo "   pip install torch torchvision torchaudio"
    exit 1
}

python3 -c "from sentence_transformers import SentenceTransformer; print('✓ sentence-transformers installed')" || {
    echo "⚠️  sentence-transformers not found. Installing..."
    pip install -q sentence-transformers
}

python3 -c "import google.generativeai; print('✓ google-generativeai installed')" || {
    echo "⚠️  google-generativeai not found. Installing (optional)..."
    pip install -q google-generativeai
}

python3 -c "from datasets import load_dataset; print('✓ datasets installed')" || {
    echo "⚠️  datasets not found. Installing..."
    pip install -q datasets
}

# Google API Key check (optional - sentence-transformers is default)
echo ""
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ℹ️  GOOGLE_API_KEY not set (OK - using sentence-transformers)"
    echo "    For Google API: export GOOGLE_API_KEY='your-api-key-here'"
else
    echo "✓ GOOGLE_API_KEY is set (will use Google API option)"
fi

# Training parameters
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_SAMPLES_PER_DATASET=${MAX_SAMPLES_PER_DATASET:-5000}
SAVE_DIR=${SAVE_DIR:-./checkpoints}

echo ""
echo "📊 Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Samples per Dataset: $MAX_SAMPLES_PER_DATASET"
echo "  Checkpoint Directory: $SAVE_DIR"

# Create checkpoint directory
mkdir -p "$SAVE_DIR"

echo ""
echo "🚀 Starting training..."
echo ""

# Run training
python3 train_megabyte_100mb_multilingual.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples-per-dataset "$MAX_SAMPLES_PER_DATASET" \
    --save-dir "$SAVE_DIR"

echo ""
echo "✨ Training completed!"
echo "Checkpoints saved to: $SAVE_DIR"
