#!/bin/bash

# Kaggle Training Setup Script

echo "=========================================="
echo "NeuroQuantum Kaggle GPU Training Setup"
echo "=========================================="

# 必要なパッケージをインストール
echo "Installing required packages..."
pip install -q kaggle torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q sentencepiece transformers datasets tqdm numpy psutil

# Kaggle認証情報の設定
echo ""
echo "Setting up Kaggle authentication..."
echo "Note: You need to have .kaggle/kaggle.json in your home directory"
echo "See: https://www.kaggle.com/settings/account"

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "Please download kaggle.json from https://www.kaggle.com/settings/account"
    echo "and place it at ~/.kaggle/kaggle.json"
    echo ""
    mkdir -p ~/.kaggle
    echo "Created ~/.kaggle directory"
else
    echo "✓ Kaggle credentials found"
    chmod 600 ~/.kaggle/kaggle.json
fi

# 出力ディレクトリの作成
echo ""
echo "Creating output directories..."
mkdir -p ./kaggle_datasets
mkdir -p ./checkpoints
mkdir -p ./logs

# GPU情報の表示
echo ""
echo "GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "CUDA not found"
fi

# Python環境情報
echo ""
echo "Python environment:"
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'Current GPU: {torch.cuda.get_device_name()}')"
    python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  python train_kaggle_japanese_datasets.py [options]"
echo ""
echo "Options:"
echo "  --datasets-dir DIR         Dataset directory (default: ./kaggle_datasets)"
echo "  --batch-size NUM           Batch size (auto-detected if not specified)"
echo "  --num-epochs NUM           Number of epochs (default: 3)"
echo "  --learning-rate LR         Learning rate (default: 1e-4)"
echo "  --max-samples NUM          Max samples to load (default: 50000)"
echo "  --skip-download            Skip downloading datasets"
echo "  --model-size SIZE          Model size: auto, large, medium, small"
echo ""
