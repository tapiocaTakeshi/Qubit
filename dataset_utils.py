"""
Hugging Face datasets ローディングユーティリティ。
datasets v3.x で trust_remote_code が廃止されたことへの互換レイヤー。
"""
import os
import shutil
import logging
import warnings
from contextlib import contextmanager
from datasets import load_dataset as _hf_load_dataset

logger = logging.getLogger(__name__)

_TRUST_REMOTE_CODE_PATTERNS = [
    "trust_remote_code",
    "loading script",
    "standard format like parquet",
]


class _TrustRemoteCodeFilter(logging.Filter):
    """datasets ライブラリの trust_remote_code 関連ログメッセージを抑制する。"""

    def filter(self, record):
        msg = record.getMessage().lower()
        return not any(p in msg for p in _TRUST_REMOTE_CODE_PATTERNS)


@contextmanager
def _suppress_trust_remote_code_noise():
    """trust_remote_code 廃止に関する warnings と logging の両方を抑制する。"""
    log_filter = _TrustRemoteCodeFilter()
    ds_logger = logging.getLogger("datasets")
    ds_load_logger = logging.getLogger("datasets.load")
    ds_logger.addFilter(log_filter)
    ds_load_logger.addFilter(log_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
            warnings.filterwarnings("ignore", message=".*loading script.*")
            yield
    finally:
        ds_logger.removeFilter(log_filter)
        ds_load_logger.removeFilter(log_filter)


def safe_load_dataset(dataset_id, split="train", streaming=False, **kwargs):
    """load_dataset のラッパー。複数の方法を順に試行する。

    1. 通常ロード（Parquet/標準フォーマット対応データセット向け）
    2. trust_remote_code=True（datasets 2.x でカスタムスクリプト使用時）
    3. streaming モードへのフォールバック

    trust_remote_code 廃止に関する警告は自動的に抑制される。
    """
    # Attempt 1: standard load
    try:
        with _suppress_trust_remote_code_noise():
            return _hf_load_dataset(dataset_id, split=split, streaming=streaming, **kwargs)
    except Exception as e1:
        err_msg = str(e1).lower()
        if "trust_remote_code" not in err_msg and "loading script" not in err_msg:
            raise

    # Attempt 2: trust_remote_code=True (datasets <3.0)
    try:
        with _suppress_trust_remote_code_noise():
            return _hf_load_dataset(
                dataset_id, split=split, streaming=streaming,
                trust_remote_code=True, **kwargs
            )
    except TypeError:
        # datasets 3.x: trust_remote_code parameter removed entirely
        pass
    except Exception:
        pass

    # Attempt 3: streaming fallback (if not already streaming)
    if not streaming:
        try:
            logger.info(
                "%s: 通常ロード失敗。streaming モードで再試行します。", dataset_id
            )
            with _suppress_trust_remote_code_noise():
                return _hf_load_dataset(dataset_id, split=split, streaming=True, **kwargs)
        except Exception:
            pass

    raise RuntimeError(
        f"{dataset_id} のロードに失敗しました。"
        f"このデータセットはカスタムローディングスクリプトを使用しており、"
        f"datasets v3.x では非対応です。"
        f"対処法: pip install 'datasets>=2.18.0,<3'"
    )


# ============================================================
# Network Volume sync utility
# ============================================================
NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")


def sync_checkpoint_to_network_volume(ckpt_path, tokenizer_path=None):
    """チェックポイントをネットワークボリュームにコピーして永続化する。

    Args:
        ckpt_path: 保存済みチェックポイントのパス
        tokenizer_path: トークナイザーモデルのパス（任意）

    Returns:
        ネットワークボリューム上のチェックポイントパス、またはNone
    """
    if not os.path.isdir(NETWORK_VOLUME_PATH):
        return None

    nv_ckpt_path = os.path.join(NETWORK_VOLUME_PATH, os.path.basename(ckpt_path))
    try:
        shutil.copy2(ckpt_path, nv_ckpt_path)
        print(f"  Checkpoint synced to network volume: {nv_ckpt_path}")

        # Also sync tokenizer if provided
        if tokenizer_path and os.path.isfile(tokenizer_path):
            nv_tok_path = os.path.join(NETWORK_VOLUME_PATH, os.path.basename(tokenizer_path))
            shutil.copy2(tokenizer_path, nv_tok_path)
            print(f"  Tokenizer synced to network volume: {nv_tok_path}")

        return nv_ckpt_path
    except Exception as e:
        print(f"  Warning: failed to sync to network volume: {e}")
        return None
