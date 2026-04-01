"""
Hugging Face datasets ローディングユーティリティ。
datasets v3.x で trust_remote_code が廃止されたことへの互換レイヤー。
"""
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
