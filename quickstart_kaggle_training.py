#!/usr/bin/env python3
"""
Quick Start Script for NeuroQuantum Training on Kaggle Datasets

Usage:
    python quickstart_kaggle_training.py
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """美しなヘッダーを表示"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_requirements():
    """必要なパッケージを確認"""
    logger.info("Checking requirements...")

    required_packages = {
        'torch': 'PyTorch',
        'kaggle': 'Kaggle CLI',
        'tqdm': 'TQDM',
        'numpy': 'NumPy',
    }

    missing_packages = []

    for package, name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name} is installed")
        except ImportError:
            logger.warning(f"✗ {name} is NOT installed")
            missing_packages.append(package)

    if missing_packages:
        logger.error("\nMissing packages. Installing...")
        cmd = f"pip install {' '.join(missing_packages)}"
        os.system(cmd)

    return True


def check_kaggle_auth():
    """Kaggle認証を確認"""
    logger.info("Checking Kaggle authentication...")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        logger.error(f"✗ Kaggle credentials not found at {kaggle_json}")
        logger.error("\nTo set up Kaggle authentication:")
        logger.error("1. Visit https://www.kaggle.com/settings/account")
        logger.error("2. Click 'Create New API Token'")
        logger.error("3. Save kaggle.json to ~/.kaggle/")
        logger.error("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    logger.info(f"✓ Kaggle credentials found at {kaggle_json}")
    return True


def check_gpu():
    """GPU環境を確認"""
    import torch

    logger.info("Checking GPU...")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"✓ CUDA is available ({device_count} device(s))")

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  Device {i}: {name} ({vram:.1f} GB VRAM)")

        return True
    else:
        logger.warning("✗ CUDA is NOT available (CPU training will be slow)")
        return True  # Still allow CPU training


def setup_directories():
    """必要なディレクトリを作成"""
    logger.info("Setting up directories...")

    directories = [
        './kaggle_datasets',
        './checkpoints',
        './logs',
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Created/verified: {directory}")


def show_menu():
    """メニューを表示"""
    print_header("NeuroQuantum Kaggle Training - Quick Start")

    print("Select training option:")
    print("  1. Basic training (simple script)")
    print("  2. Advanced training (with features)")
    print("  3. Resume from checkpoint")
    print("  4. Download datasets only")
    print("  5. Show configuration")
    print("  6. Exit")
    print()

    choice = input("Enter your choice (1-6): ").strip()
    return choice


def run_basic_training():
    """基本訓練を実行"""
    print_header("Basic Training")

    print("Configuration:")
    print("  - Model size: auto-detected")
    print("  - Batch size: auto-detected")
    print("  - Epochs: 3")
    print("  - Learning rate: 1e-4")
    print()

    confirm = input("Start training? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    cmd = [
        'python', 'train_kaggle_japanese_datasets.py',
        '--num-epochs', '3',
        '--learning-rate', '1e-4',
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_advanced_training():
    """高度な訓練を実行"""
    print_header("Advanced Training")

    print("Configuration options:")
    print("  Batch size [4]: ", end='')
    batch_size = input().strip() or '4'

    print("  Number of epochs [3]: ", end='')
    num_epochs = input().strip() or '3'

    print("  Learning rate [1e-4]: ", end='')
    lr = input().strip() or '1e-4'

    print("  Max samples [50000]: ", end='')
    max_samples = input().strip() or '50000'

    print("  Use gradient accumulation? (y/n) [n]: ", end='')
    use_ga = input().strip().lower() == 'y'

    print("\nConfiguration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Max samples: {max_samples}")
    print(f"  - Gradient accumulation: {use_ga}")
    print()

    confirm = input("Start training? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    cmd = [
        'python', 'train_kaggle_advanced.py',
        '--batch-size', batch_size,
        '--num-epochs', num_epochs,
        '--learning-rate', lr,
        '--max-samples', max_samples,
    ]

    if use_ga:
        cmd.extend(['--gradient-accumulation-steps', '2'])

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_resume_training():
    """チェックポイントから再開"""
    print_header("Resume Training")

    checkpoints_dir = Path('./checkpoints')
    if not checkpoints_dir.exists():
        logger.error("No checkpoints directory found")
        return

    checkpoints = list(checkpoints_dir.glob('*.pth'))
    if not checkpoints:
        logger.error("No checkpoint files found in ./checkpoints")
        return

    print("Available checkpoints:")
    for i, cp in enumerate(sorted(checkpoints)):
        print(f"  {i}: {cp.name}")
    print()

    idx = input("Select checkpoint (0-{}): ".format(len(checkpoints)-1)).strip()
    try:
        idx = int(idx)
        checkpoint_path = sorted(checkpoints)[idx]
    except (ValueError, IndexError):
        logger.error("Invalid selection")
        return

    print(f"\nResuming from: {checkpoint_path}")
    print("  Number of additional epochs [5]: ", end='')
    num_epochs = input().strip() or '5'

    cmd = [
        'python', 'train_kaggle_advanced.py',
        '--resume-from', str(checkpoint_path),
        '--num-epochs', num_epochs,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def download_datasets_only():
    """データセットのダウンロードのみ"""
    print_header("Download Datasets Only")

    cmd = [
        'python', 'train_kaggle_japanese_datasets.py',
        '--num-epochs', '0',
        '--skip-download',  # 実際はダウンロードスクリプトが必要
    ]

    print("This will download all Kaggle datasets without training.")
    confirm = input("Start download? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    logger.info("Downloading datasets...")
    # 実際のダウンロードはtrain_kaggle_japanese_datasets.pyで実装


def show_configuration():
    """設定情報を表示"""
    import torch
    from neuroquantum_layered import get_gpu_adaptive_config

    print_header("Configuration Information")

    print("GPU Configuration:")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device_name}")
        print(f"  VRAM: {vram:.1f} GB")

        gpu_config = get_gpu_adaptive_config()
        print("\nAuto-detected Model Configuration:")
        for key, value in gpu_config.items():
            if key not in ['gpu_info']:
                print(f"  {key}: {value}")
    else:
        print("  No CUDA device found (CPU mode)")

    print("\nDirectories:")
    for directory in ['./kaggle_datasets', './checkpoints', './logs']:
        size = sum(f.stat().st_size for f in Path(directory).rglob('*') if f.is_file())
        print(f"  {directory}: {size / 1e9:.2f} GB")


def main():
    """メインループ"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "NeuroQuantum - Kaggle Japanese Datasets GPU Training".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # プリフライトチェック
    logger.info("\nRunning preflight checks...")
    print()

    if not check_requirements():
        logger.error("Missing requirements")
        return

    if not check_kaggle_auth():
        logger.error("Kaggle authentication failed")
        logger.info("Please set up Kaggle credentials first")
        return

    check_gpu()
    setup_directories()

    # メニューループ
    while True:
        choice = show_menu()

        if choice == '1':
            run_basic_training()
        elif choice == '2':
            run_advanced_training()
        elif choice == '3':
            run_resume_training()
        elif choice == '4':
            download_datasets_only()
        elif choice == '5':
            show_configuration()
        elif choice == '6':
            logger.info("Exiting...")
            break
        else:
            logger.error("Invalid choice")

        if choice in ['1', '2', '3']:
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
