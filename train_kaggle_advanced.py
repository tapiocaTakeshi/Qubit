#!/usr/bin/env python3
"""
Advanced Kaggle Japanese Dataset Training for NeuroQuantum

Features:
- Multi-dataset support (Wikipedia, MC4, OSCAR, Aozorabunko, JGLUE)
- GPU adaptive configuration
- Gradient accumulation support
- Mixed precision training
- Checkpoint and resume capabilities
- Training monitoring and logging
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import json
import gzip
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import argparse
import time
from collections import defaultdict
import random

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# neuroquantum_layered をインポート
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, get_gpu_adaptive_config

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiJapaneseDataset(Dataset):
    """複数の日本語テキストソースを統合したデータセット"""

    def __init__(
        self,
        texts: List[str],
        max_length: int = 1024,
        vocab_size: int = 32000,
        augment: bool = False
    ):
        self.texts = texts
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.augment = augment

        logger.info(f"Dataset initialized with {len(texts)} texts")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx].strip()

        if not text:
            # 空のテキストの場合、別のインデックスを選ぶ
            idx = random.randint(0, len(self.texts) - 1)
            text = self.texts[idx].strip()

        # テキストを正規化
        text = text[:min(len(text), self.max_length * 4)]  # 4倍までトークン化の余裕

        # 簡易的なトークン化
        tokens = self._tokenize(text)

        # パディングまたはトリミング
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(self.max_length, dtype=torch.long),
            'text_preview': text[:100]
        }

    def _tokenize(self, text: str) -> List[int]:
        """簡易トークナイザー（実際にはSentencePieceを使用すべき）"""
        # UTF-8バイト列をトークンに変換（vocab_size内に制限）
        tokens = []
        for char in text:
            token = min(ord(char) % (self.vocab_size - 1) + 1, self.vocab_size - 1)
            tokens.append(token)
        return tokens


class TrainingMetrics:
    """訓練メトリクスの追跡"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def get_avg(self, key: str) -> float:
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])

    def get_summary(self) -> Dict:
        return {
            key: {
                'avg': self.get_avg(key),
                'last': self.metrics[key][-1] if self.metrics[key] else 0,
                'count': len(self.metrics[key])
            }
            for key in self.metrics
        }

    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class KaggleTrainer:
    """Kaggleデータセット用訓練クラス"""

    def __init__(
        self,
        model: NeuroQuantum,
        device: torch.device,
        learning_rate: float = 1e-4,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # オプティマイザー
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.95)
        )

        # Mixed Precision scaler
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using Mixed Precision Training (AMP)")
        else:
            self.scaler = None

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Metrics
        self.metrics = TrainingMetrics()

        logger.info(f"Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        num_epochs: int
    ) -> float:
        """1エポックを訓練"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulated_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not TQDM_AVAILABLE
        ) if TQDM_AVAILABLE else train_loader

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)

            try:
                if self.use_amp:
                    with autocast(dtype=torch.float16):
                        logits = self.model(input_ids)
                        loss = self._compute_loss(logits, input_ids)
                    loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(input_ids)
                    loss = self._compute_loss(logits, input_ids)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                accumulated_loss += loss.item()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    batch_loss = accumulated_loss * self.gradient_accumulation_steps
                    self.metrics.update(loss=batch_loss)
                    total_loss += batch_loss
                    num_batches += 1
                    accumulated_loss = 0

                    if TQDM_AVAILABLE:
                        pbar.set_postfix({
                            'loss': f"{batch_loss:.4f}",
                            'avg_loss': f"{total_loss / max(num_batches, 1):.4f}"
                        })

            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                self.optimizer.zero_grad()
                continue

        avg_epoch_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.6f}")

        return avg_epoch_loss

    def _compute_loss(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Next token prediction lossを計算"""
        # logits: (batch, seq_len, vocab_size)
        # input_ids: (batch, seq_len)

        batch_size, seq_len, vocab_size = logits.shape

        # シフト: 最後のトークン以外で予測
        logits_flat = logits[:, :-1].reshape(-1, vocab_size)
        labels_flat = input_ids[:, 1:].reshape(-1)

        # 無効なラベルをマスク
        valid_mask = labels_flat < self.model.config.vocab_size
        logits_flat = logits_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]

        if logits_flat.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = self.loss_fn(logits_flat, labels_flat)
        return loss

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """チェックポイント保存"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}.pth"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config.__dict__,
            'metrics': self.metrics.get_summary(),
            'timestamp': datetime.now().isoformat(),
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "model_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """チェックポイント読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")

        return epoch + 1


def load_dataset_texts(
    dataset_paths: Dict[str, Path],
    max_samples: int = 10000,
    sample_ratio: Dict[str, float] = None
) -> List[str]:
    """複数のデータセットからテキストを読み込み"""

    if sample_ratio is None:
        sample_ratio = {
            'wikipedia': 0.3,
            'mc4': 0.2,
            'oscar': 0.2,
            'aozorabunko': 0.15,
            'jglue': 0.15
        }

    all_texts = []
    texts_by_source = defaultdict(list)

    logger.info("Loading texts from datasets...")

    # データセットごとのテキスト読み込み
    for dataset_name, dataset_path in dataset_paths.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            continue

        logger.info(f"  Loading from {dataset_name}...")

        if dataset_name == 'aozorabunko':
            # テキストファイル
            for txt_file in dataset_path.rglob('*.txt'):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 段落で分割
                        for para in content.split('\n\n'):
                            if para.strip() and len(para.strip()) > 50:
                                texts_by_source[dataset_name].append(para.strip()[:2000])
                except Exception as e:
                    logger.debug(f"Error reading {txt_file}: {e}")

        elif dataset_name == 'jglue':
            # JSONファイル
            for json_file in dataset_path.rglob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if 'premise' in data and len(data['premise']) > 50:
                                    texts_by_source[dataset_name].append(data['premise'][:2000])
                                elif 'question' in data and len(data['question']) > 50:
                                    texts_by_source[dataset_name].append(data['question'][:2000])
                                elif 'text' in data and len(data['text']) > 50:
                                    texts_by_source[dataset_name].append(data['text'][:2000])
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.debug(f"Error reading {json_file}: {e}")

        else:
            # JSON/JSONL or gzip files
            for file_path in dataset_path.rglob('*'):
                if file_path.suffix == '.gz':
                    try:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    if 'text' in data and len(data['text']) > 50:
                                        texts_by_source[dataset_name].append(data['text'][:2000])
                                    elif 'content' in data and len(data['content']) > 50:
                                        texts_by_source[dataset_name].append(data['content'][:2000])
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")

        logger.info(f"    Loaded {len(texts_by_source[dataset_name])} texts from {dataset_name}")

    # サンプル数に応じてテキストをサンプリング
    logger.info("Sampling texts...")
    for dataset_name, texts in texts_by_source.items():
        ratio = sample_ratio.get(dataset_name, 0.2)
        num_samples = int(max_samples * ratio)
        sampled = random.sample(texts, min(len(texts), num_samples))
        all_texts.extend(sampled)
        logger.info(f"  Sampled {len(sampled)} texts from {dataset_name} (ratio: {ratio})")

    logger.info(f"Total texts loaded: {len(all_texts)}")
    return all_texts


def main():
    parser = argparse.ArgumentParser(
        description="Advanced NeuroQuantum training on Kaggle Japanese datasets"
    )
    parser.add_argument('--datasets-dir', type=str, default='./kaggle_datasets')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=50000)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 80)
    logger.info("NeuroQuantum Advanced Kaggle Training")
    logger.info("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name()}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Get GPU config
    logger.info("\nConfiguring model based on GPU...")
    gpu_config = get_gpu_adaptive_config()
    batch_size = args.batch_size or gpu_config.get('batch_size', 4)

    # Load data
    logger.info("\nLoading datasets...")
    datasets_dir = Path(args.datasets_dir)
    dataset_paths = {
        'wikipedia': datasets_dir / 'wikipedia',
        'mc4': datasets_dir / 'mc4',
        'oscar': datasets_dir / 'oscar',
        'aozorabunko': datasets_dir / 'aozorabunko',
        'jglue': datasets_dir / 'jglue',
    }

    texts = load_dataset_texts(dataset_paths, max_samples=args.max_samples)

    if not texts:
        logger.error("No texts loaded. Please check dataset directories.")
        return

    # Create dataset
    logger.info("\nCreating dataset...")
    dataset = MultiJapaneseDataset(
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

    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"DataLoader: batch_size={batch_size}, num_batches={len(train_loader)}")

    # Initialize model
    logger.info("\nInitializing model...")
    try:
        model = NeuroQuantum(config_dict=gpu_config)
        model.to(device)
        logger.info(f"Model config: {gpu_config}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = KaggleTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80 + "\n")

    try:
        best_loss = float('inf')

        for epoch in range(start_epoch, args.num_epochs):
            epoch_loss = trainer.train_epoch(train_loader, epoch, args.num_epochs)

            # Save checkpoint
            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss

            trainer.save_checkpoint(epoch, is_best=is_best)

            # Periodic logging
            elapsed = trainer.metrics.elapsed_time()
            logger.info(
                f"Epoch {epoch + 1}/{args.num_epochs} - "
                f"Loss: {epoch_loss:.6f} - "
                f"Elapsed: {elapsed/60:.1f}m"
            )

        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
