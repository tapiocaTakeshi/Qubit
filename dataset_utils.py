"""
Hugging Face datasets ローディングユーティリティ。
datasets v3.x で trust_remote_code が廃止されたことへの互換レイヤー。
"""
import warnings
from datasets import load_dataset as _hf_load_dataset


def safe_load_dataset(dataset_id, split="train", streaming=False, **kwargs):
    """load_dataset のラッパー。複数の方法を順に試行する。

    1. 通常ロード（Parquet/標準フォーマット対応データセット向け）
    2. trust_remote_code=True（datasets 2.x でカスタムスクリプト使用時）
    3. streaming モードへのフォールバック
    """
    # Attempt 1: standard load
    try:
        return _hf_load_dataset(dataset_id, split=split, streaming=streaming, **kwargs)
    except Exception as e1:
        err_msg = str(e1).lower()
        if "trust_remote_code" not in err_msg and "loading script" not in err_msg:
            raise

    # Attempt 2: trust_remote_code=True (datasets <3.0)
    try:
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
            warnings.warn(
                f"{dataset_id}: 通常ロード失敗。streaming モードで再試行します。"
            )
            return _hf_load_dataset(dataset_id, split=split, streaming=True, **kwargs)
        except Exception:
            pass

    raise RuntimeError(
        f"{dataset_id} のロードに失敗しました。"
        f"このデータセットはカスタムローディングスクリプトを使用しており、"
        f"datasets v3.x では非対応です。"
        f"対処法: pip install 'datasets>=2.18.0,<3'"
    )
