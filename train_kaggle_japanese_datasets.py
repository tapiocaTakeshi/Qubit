#!/usr/bin/env python3
"""
Kaggle日本語データセットを使ってneuroquantum_layered.pyを学習するスクリプト

ダウンロード対象:
- wikimedia/wikipedia [20231101.ja]
- mc4 [ja]
- oscar-corpus/OSCAR-2301 [ja]
- globis-university/aozorabunko-clean
- shunk031/JGLUE [JNLI, JCommonsenseQA, JaQuAD]
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import gzip
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
import argparse

# Kaggle APIの使用
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️ Kaggle APIがインストールされていません。pip install kaggle を実行してください。")

# neuroquantum_layered.py をインポート
from neuroquantum_layered import NeuroQuantum, get_gpu_adaptive_config
from dataset_utils import load_jglue_dataset, create_text_dataset

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleDatasetDownloader:
    """Kaggleデータセットをダウンロード"""

    def __init__(self, datasets_dir: str = "./kaggle_datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)

        if not KAGGLE_AVAILABLE:
            raise ImportError("Kaggle APIが必要です。pip install kaggle を実行してください。")

        # Kaggle API初期化
        self.api = KaggleApi()
        self.api.authenticate()
        logger.info("Kaggle APIが認証されました")

    def download_wikipedia_ja(self) -> Path:
        """Wikipedia (日本語 2023-11-01)をダウンロード"""
        logger.info("Wikipedia (日本語) をダウンロード中...")
        dataset_name = "wikimedia/wikipedia"
        target_dir = self.datasets_dir / "wikipedia"

        if (target_dir / "20231101.ja").exists():
            logger.info("Wikipedia (日本語) は既にダウンロード済み")
            return target_dir

        self.api.dataset_download_files(
            dataset_name,
            path=target_dir,
            unzip=True
        )
        logger.info(f"Wikipedia (日本語) をダウンロードしました: {target_dir}")
        return target_dir

    def download_mc4_ja(self) -> Path:
        """MC4 (日本語)をダウンロード"""
        logger.info("MC4 (日本語) をダウンロード中...")
        dataset_name = "allenai/mc4"
        target_dir = self.datasets_dir / "mc4"

        if (target_dir / "ja").exists():
            logger.info("MC4 (日本語) は既にダウンロード済み")
            return target_dir

        self.api.dataset_download_files(
            dataset_name,
            path=target_dir,
            unzip=True
        )
        logger.info(f"MC4 (日本語) をダウンロードしました: {target_dir}")
        return target_dir

    def download_oscar_2301_ja(self) -> Path:
        """OSCAR 2301 (日本語)をダウンロード"""
        logger.info("OSCAR 2301 (日本語) をダウンロード中...")
        dataset_name = "oscar-corpus/OSCAR-2301"
        target_dir = self.datasets_dir / "oscar"

        if (target_dir / "ja").exists():
            logger.info("OSCAR 2301 (日本語) は既にダウンロード済み")
            return target_dir

        self.api.dataset_download_files(
            dataset_name,
            path=target_dir,
            unzip=True
        )
        logger.info(f"OSCAR 2301 (日本語) をダウンロードしました: {target_dir}")
        return target_dir

    def download_aozorabunko(self) -> Path:
        """青空文庫 (Clean)をダウンロード"""
        logger.info("青空文庫 (Clean) をダウンロード中...")
        dataset_name = "globis-university/aozorabunko-clean"
        target_dir = self.datasets_dir / "aozorabunko"

        if target_dir.exists() and list(target_dir.glob("**/*.txt")):
            logger.info("青空文庫 (Clean) は既にダウンロード済み")
            return target_dir

        self.api.dataset_download_files(
            dataset_name,
            path=target_dir,
            unzip=True
        )
        logger.info(f"青空文庫 (Clean) をダウンロードしました: {target_dir}")
        return target_dir

    def download_jglue(self) -> Path:
        """JGLUE (JNLI, JCommonsenseQA, JaQuAD)をダウンロード"""
        logger.info("JGLUE (JNLI, JCommonsenseQA, JaQuAD) をダウンロード中...")
        dataset_name = "shunk031/JGLUE"
        target_dir = self.datasets_dir / "jglue"

        if target_dir.exists() and list(target_dir.glob("**/*.json")):
            logger.info("JGLUE は既にダウンロード済み")
            return target_dir

        self.api.dataset_download_files(
            dataset_name,
            path=target_dir,
            unzip=True
        )
        logger.info(f"JGLUE をダウンロードしました: {target_dir}")
        return target_dir

    def download_all(self) -> Dict[str, Path]:
        """すべてのデータセットをダウンロード"""
        datasets = {}
        try:
            datasets['wikipedia'] = self.download_wikipedia_ja()
        except Exception as e:
            logger.error(f"Wikipedia ダウンロードエラー: {e}")

        try:
            datasets['mc4'] = self.download_mc4_ja()
        except Exception as e:
            logger.error(f"MC4 ダウンロードエラー: {e}")

        try:
            datasets['oscar'] = self.download_oscar_2301_ja()
        except Exception as e:
            logger.error(f"OSCAR ダウンロードエラー: {e}")

        try:
            datasets['aozorabunko'] = self.download_aozorabunko()
        except Exception as e:
            logger.error(f"青空文庫 ダウンロードエラー: {e}")

        try:
            datasets['jglue'] = self.download_jglue()
        except Exception as e:
            logger.error(f"JGLUE ダウンロードエラー: {e}")

        return datasets


class JapaneseDataset(Dataset):
    """日本語テキストデータセット"""

    def __init__(self, texts: List[str], max_length: int = 1024, vocab_size: int = 32000):
        self.texts = texts
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        # 簡易的なトークン化（実際にはSentencePieceやMeCabを使用）
        # 文字列をバイト値に変換
        tokens = [min(ord(c), self.vocab_size - 1) for c in text[:self.max_length]]

        # パディング
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'text': text[:100]  # プレビュー用
        }


def load_text_from_datasets(datasets_dir: Dict[str, Path], max_samples: int = 10000) -> List[str]:
    """複数のKaggleデータセットからテキストを読み込む"""
    texts = []

    for dataset_name, dataset_path in datasets_dir.items():
        logger.info(f"Loading texts from {dataset_name}...")

        if dataset_name == 'wikipedia':
            # JSONファイルを探す
            for json_file in dataset_path.rglob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if 'text' in item:
                                    texts.append(item['text'][:2000])
                                elif 'content' in item:
                                    texts.append(item['content'][:2000])
                                if len(texts) >= max_samples:
                                    return texts
                except Exception as e:
                    logger.warning(f"Error reading {json_file}: {e}")

        elif dataset_name == 'aozorabunko':
            # テキストファイルを読む
            for txt_file in dataset_path.rglob('*.txt'):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 段落に分割
                        for para in content.split('\n\n'):
                            if para.strip():
                                texts.append(para[:2000])
                                if len(texts) >= max_samples:
                                    return texts
                except Exception as e:
                    logger.warning(f"Error reading {txt_file}: {e}")

        elif dataset_name == 'jglue':
            # JSONLファイルを読む
            for json_file in dataset_path.rglob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                # 各データセットの形式に対応
                                if 'premise' in data:
                                    texts.append(data['premise'][:2000])
                                elif 'question' in data:
                                    texts.append(data['question'][:2000])
                                elif 'text' in data:
                                    texts.append(data['text'][:2000])
                                if len(texts) >= max_samples:
                                    return texts
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.warning(f"Error reading {json_file}: {e}")

        else:
            # MC4やOSCAR: gzipファイルを読む
            for gz_file in dataset_path.rglob('*.json.gz'):
                try:
                    with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if 'text' in data:
                                    texts.append(data['text'][:2000])
                                elif 'content' in data:
                                    texts.append(data['content'][:2000])
                                if len(texts) >= max_samples:
                                    return texts
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.warning(f"Error reading {gz_file}: {e}")

    logger.info(f"Loaded {len(texts)} texts from all datasets")
    return texts


def train_neuroquantum_on_kaggle(
    model: NeuroQuantum,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 3,
    checkpoint_dir: str = "./checkpoints"
) -> None:
    """neuroquantumモデルを訓練"""

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)

            # Forward pass
            try:
                logits = model(input_ids)

                # Loss: next token prediction
                # logits: (batch, seq_len, vocab_size)
                # labels: (batch, seq_len) - shift by 1
                batch_size, seq_len, vocab_size = logits.shape

                logits_flat = logits[:, :-1].reshape(-1, vocab_size)
                labels_flat = input_ids[:, 1:].reshape(-1)

                loss = loss_fn(logits_flat, labels_flat)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}, "
                        f"Batch {batch_idx + 1}, "
                        f"Loss: {avg_loss:.4f}"
                    )

                # Save checkpoint every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    checkpoint_file = checkpoint_path / f"checkpoint_epoch{epoch}_batch{batch_idx}.pth"
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_file)
                    logger.info(f"Checkpoint saved: {checkpoint_file}")

            except RuntimeError as e:
                logger.error(f"Error during forward pass: {e}")
                continue

        avg_epoch_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        epoch_checkpoint = checkpoint_path / f"model_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.config.__dict__,
            'loss': avg_epoch_loss,
        }, epoch_checkpoint)
        logger.info(f"Epoch checkpoint saved: {epoch_checkpoint}")


def main():
    parser = argparse.ArgumentParser(
        description="Train NeuroQuantum on Kaggle Japanese datasets"
    )
    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='./kaggle_datasets',
        help='Directory to save downloaded datasets'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (auto-detected if not specified)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='Maximum number of samples to load'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading datasets (use existing ones)'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='auto',
        help='Model size (auto, large, medium, small) or use auto-detection based on GPU'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("NeuroQuantum Layered GPU Training on Kaggle Japanese Datasets")
    logger.info("=" * 80)

    # デバイス確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Step 1: Kaggleデータセットをダウンロード
    if not args.skip_download:
        logger.info("Step 1: Downloading Kaggle datasets...")
        downloader = KaggleDatasetDownloader(datasets_dir=args.datasets_dir)
        datasets = downloader.download_all()
        logger.info(f"Downloaded {len(datasets)} datasets")
    else:
        logger.info("Skipping dataset download")
        datasets = {}
        for ds_name in ['wikipedia', 'mc4', 'oscar', 'aozorabunko', 'jglue']:
            ds_path = Path(args.datasets_dir) / ds_name
            if ds_path.exists():
                datasets[ds_name] = ds_path

    if not datasets:
        logger.error("No datasets available. Please download first.")
        return

    # Step 2: GPU適応設定
    logger.info("\nStep 2: Detecting GPU and configuring model...")
    gpu_config = get_gpu_adaptive_config()

    # batch_size 設定
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = gpu_config.get('batch_size', 4)

    logger.info(f"Batch size: {batch_size}")

    # Step 3: テキストデータ読み込み
    logger.info("\nStep 3: Loading text data...")
    texts = load_text_from_datasets(datasets, max_samples=args.max_samples)

    if not texts:
        logger.error("No texts loaded from datasets")
        return

    logger.info(f"Loaded {len(texts)} texts")

    # Step 4: データセットとDataLoaderを作成
    logger.info("\nStep 4: Creating dataset and dataloader...")
    dataset = JapaneseDataset(
        texts=texts,
        max_length=gpu_config.get('max_seq_len', 1024),
        vocab_size=gpu_config.get('vocab_size', 32000)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )

    logger.info(f"Created dataloader with {len(dataset)} samples, batch size: {batch_size}")

    # Step 5: モデルを初期化
    logger.info("\nStep 5: Initializing NeuroQuantum model...")
    try:
        model = NeuroQuantum(config_dict=gpu_config)
        logger.info(f"Model initialized with config: {gpu_config}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # モデルパラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Step 6: オプティマイザーを作成
    logger.info("\nStep 6: Setting up optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    logger.info(f"Using AdamW optimizer with LR: {args.learning_rate}")

    # Step 7: 訓練開始
    logger.info("\nStep 7: Starting training...")
    logger.info("=" * 80)

    try:
        train_neuroquantum_on_kaggle(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            checkpoint_dir="./checkpoints"
        )
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
