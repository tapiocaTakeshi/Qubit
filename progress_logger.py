#!/usr/bin/env python3
"""
統一プログレスロガー (Unified Progress Logger)

全トレーニングスクリプト・API共通の進捗ログモジュール。
ファイルログ (JSON Lines) + コンソール出力 + インメモリステータスを一元管理する。

使い方:
    from progress_logger import ProgressLogger

    logger = ProgressLogger("train_local")
    logger.start_training(epochs=3, total_sequences=1000)
    logger.log_epoch(epoch=1, total_epochs=3, loss=5.12, extra={"lr": 0.001})
    logger.log_batch(epoch=1, batch=50, loss=4.8)
    logger.end_training(final_loss=4.5)
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
_lock = threading.Lock()

# Python logging の設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ProgressLogger:
    """統一プログレスロガー"""

    def __init__(self, source: str, log_dir: Optional[str] = None):
        """
        Args:
            source: ログの発信元 (例: "train_local", "api", "handler", "split_learning")
            log_dir: ログファイルの出力先ディレクトリ (デフォルト: ./logs)
        """
        self.source = source
        self.log_dir = log_dir or LOG_DIR
        self.logger = logging.getLogger(f"qubit.{source}")

        # インメモリステータス (API互換)
        self.status: Dict[str, Any] = {
            "running": False,
            "log": [],
            "message": "idle",
        }

        # 訓練メトリクス
        self._start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._metrics: List[Dict[str, Any]] = []

        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def log_file(self) -> str:
        """当日のログファイルパス"""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"progress_{self.source}_{date_str}.jsonl")

    # ------------------------------------------------------------------
    # コアログメソッド
    # ------------------------------------------------------------------

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """JSON Lines 形式でファイルに追記"""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry["source"] = self.source
        with _lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def info(self, message: str, **extra: Any) -> None:
        """汎用情報ログ"""
        self.logger.info(message)
        self.status["log"].append(message)
        self.status["message"] = message
        self._write_entry({"level": "INFO", "message": message, **extra})

    def warning(self, message: str, **extra: Any) -> None:
        """警告ログ"""
        self.logger.warning(message)
        self.status["log"].append(f"[WARN] {message}")
        self._write_entry({"level": "WARNING", "message": message, **extra})

    def error(self, message: str, **extra: Any) -> None:
        """エラーログ"""
        self.logger.error(message)
        self.status["log"].append(f"[ERROR] {message}")
        self.status["message"] = f"Error: {message}"
        self._write_entry({"level": "ERROR", "message": message, **extra})

    # ------------------------------------------------------------------
    # トレーニングライフサイクル
    # ------------------------------------------------------------------

    def start_training(
        self,
        epochs: int,
        total_sequences: int = 0,
        batch_size: int = 0,
        lr: float = 0.0,
        **extra: Any,
    ) -> None:
        """トレーニング開始をログに記録"""
        self._start_time = time.time()
        self._metrics = []
        self.status = {"running": True, "log": [], "message": "Training started"}

        msg = (
            f"Training started: epochs={epochs}, sequences={total_sequences}, "
            f"batch_size={batch_size}, lr={lr}"
        )
        self.logger.info(msg)
        self.status["log"].append(msg)
        self._write_entry({
            "event": "training_start",
            "epochs": epochs,
            "total_sequences": total_sequences,
            "batch_size": batch_size,
            "lr": lr,
            **extra,
        })

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        lr: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """エポック完了をログに記録"""
        elapsed = ""
        if self._epoch_start_time:
            secs = time.time() - self._epoch_start_time
            elapsed = f" ({secs:.1f}s)"

        msg = f"Epoch {epoch}/{total_epochs} | Loss: {loss:.6f}"
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        msg += elapsed

        self.logger.info(msg)
        self.status["log"].append(msg)
        self.status["message"] = msg

        metric = {"epoch": epoch, "loss": round(loss, 6), **extra}
        if lr is not None:
            metric["lr"] = lr
        self._metrics.append(metric)

        self._write_entry({
            "event": "epoch_end",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": round(loss, 6),
            **({"lr": lr} if lr is not None else {}),
            **extra,
        })

        # 次のエポック計測用にリセット
        self._epoch_start_time = time.time()

    def log_batch(
        self,
        epoch: int,
        batch: int,
        loss: float,
        total_batches: Optional[int] = None,
        **extra: Any,
    ) -> None:
        """バッチ進捗をログに記録 (コンソール + ファイル)"""
        batch_str = f"{batch}/{total_batches}" if total_batches else str(batch)
        msg = f"  Epoch {epoch} | Batch {batch_str} | Loss: {loss:.4f}"
        self.logger.info(msg)
        self.status["message"] = msg

        self._write_entry({
            "event": "batch",
            "epoch": epoch,
            "batch": batch,
            "total_batches": total_batches,
            "loss": round(loss, 4),
            **extra,
        })

    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        """エポック開始を記録"""
        self._epoch_start_time = time.time()
        msg = f"Training epoch {epoch}/{total_epochs}..."
        self.status["message"] = msg
        self.logger.info(msg)
        self._write_entry({"event": "epoch_start", "epoch": epoch, "total_epochs": total_epochs})

    def end_training(
        self,
        final_loss: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """トレーニング完了をログに記録"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        minutes = elapsed / 60

        parts = ["Training complete!"]
        if final_loss is not None:
            parts.append(f"Final loss: {final_loss:.6f}")
        parts.append(f"Duration: {minutes:.1f}min")
        if checkpoint_path:
            parts.append(f"Checkpoint: {checkpoint_path}")

        msg = " | ".join(parts)
        self.logger.info(msg)
        self.status["log"].append(msg)
        self.status["message"] = "Training complete!"
        self.status["running"] = False

        self._write_entry({
            "event": "training_end",
            "final_loss": round(final_loss, 6) if final_loss is not None else None,
            "duration_sec": round(elapsed, 1),
            "checkpoint_path": checkpoint_path,
            "metrics": self._metrics,
            **extra,
        })

    def end_training_error(self, error: str, traceback_str: Optional[str] = None) -> None:
        """トレーニングエラー終了をログに記録"""
        self.logger.error(f"Training failed: {error}")
        self.status["running"] = False
        self.status["message"] = f"Error: {error}"
        self.status["log"].append(f"Error: {error}")
        if traceback_str:
            self.status["log"].append(traceback_str)

        self._write_entry({
            "event": "training_error",
            "error": error,
            "traceback": traceback_str,
            "metrics": self._metrics,
        })

    # ------------------------------------------------------------------
    # データセット読み込み
    # ------------------------------------------------------------------

    def log_dataset_loaded(self, dataset_id: str, count: int, **extra: Any) -> None:
        """データセット読み込み完了"""
        msg = f"Loaded {dataset_id}: {count} texts"
        self.info(msg, event="dataset_loaded", dataset_id=dataset_id, count=count, **extra)

    def log_dataset_error(self, dataset_id: str, error: str) -> None:
        """データセット読み込みエラー"""
        msg = f"Error loading {dataset_id}: {error}"
        self.warning(msg, event="dataset_error", dataset_id=dataset_id, error=error)

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> List[Dict[str, Any]]:
        """これまでの訓練メトリクスを返す"""
        return list(self._metrics)

    def get_status(self) -> Dict[str, Any]:
        """現在のステータスを返す (API互換)"""
        return dict(self.status)
