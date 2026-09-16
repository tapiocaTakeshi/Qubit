"""
Hugging Face datasets ローディングユーティリティ。
datasets v3.x で trust_remote_code が廃止されたことへの互換レイヤー。
"""
import os
import shutil
import logging
import warnings
import json
import urllib.parse
import urllib.request
from contextlib import contextmanager

# Keep Hugging Face metadata, locks, and streaming cache on the persistent
# network volume.  The container root filesystem is intentionally small and
# filling /root/.cache causes otherwise unrelated training failures.
NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")
HF_CACHE_ROOT = os.path.join(NETWORK_VOLUME_PATH, "huggingface-cache")
HF_DATASETS_CACHE = os.path.join(HF_CACHE_ROOT, "datasets")
HF_HUB_CACHE = os.path.join(HF_CACHE_ROOT, "hub")
for _cache_dir in (HF_CACHE_ROOT, HF_DATASETS_CACHE, HF_HUB_CACHE):
    try:
        os.makedirs(_cache_dir, exist_ok=True)
    except OSError:
        # Fall back to the container cache only if the mounted volume is absent.
        pass

# RunPod workers may inherit offline HF flags from the base image.  This worker
# trains from public Hugging Face datasets, so make the intended online mode
# explicit before importing datasets (which snapshots these flags at import).
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HOME"] = HF_CACHE_ROOT
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_ROOT, "transformers")
from datasets import load_dataset as _hf_load_dataset

logger = logging.getLogger(__name__)

_TRUST_REMOTE_CODE_PATTERNS = [
    "trust_remote_code",
    "loading script",
    "standard format like parquet",
]

_DISK_FULL_MARKERS = (
    "disk quota exceeded",
    "no space left on device",
    "[errno 28]",
    "[errno 122]",
)


def _is_disk_full_error(err):
    """Detect EDQUOT/ENOSPC-style failures from datasets/huggingface_hub."""
    msg = str(err).lower()
    return any(marker in msg for marker in _DISK_FULL_MARKERS)


def clear_hf_dataset_cache(reason=None):
    """Purge cached Hugging Face dataset/hub downloads.

    These directories only exist to speed up repeat loads; nothing in them
    is needed once a caller has pulled the rows it wants out of a dataset.
    RunPod's network volume enforces a hard disk quota, so caches left over
    from an earlier dataset (or an earlier job on a warm container) can
    starve the very next load before it writes a single byte. Clearing them
    on a disk-full error frees quota for the retry that follows.
    """
    freed_any = False
    for cache_dir in (HF_DATASETS_CACHE, HF_HUB_CACHE):
        if not os.path.isdir(cache_dir):
            continue
        for entry in os.listdir(cache_dir):
            entry_path = os.path.join(cache_dir, entry)
            try:
                if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)
                freed_any = True
            except OSError as e:
                logger.warning(
                    "Failed to remove %s during cache cleanup: %s", entry_path, e
                )
    if freed_any:
        logger.warning(
            "Cleared Hugging Face dataset cache%s",
            f" ({reason})" if reason else "",
        )
    return freed_any


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



def _load_hf_parquet_fallback(dataset_id, split="train", **kwargs):
    """Load a public HF dataset directly from its Parquet shard list.

    Some worker images fail to resolve a perfectly valid Parquet dataset through
    the high-level datasets builder.  Querying the Hub tree and opening the
    first matching shard keeps streaming startup reliable and avoids downloading
    the whole corpus.
    """
    if dataset_id != "hotchpotch/fineweb-2-edu-japanese":
        raise RuntimeError("direct Parquet fallback is not configured for this dataset")

    requested_config = kwargs.pop("name", None) or "default"
    api_url = (
        "https://huggingface.co/api/datasets/"
        f"{dataset_id}/tree/main?recursive=true&expand=false"
    )
    request = urllib.request.Request(
        api_url, headers={"User-Agent": "Qubit-RunPod/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.loads(response.read().decode("utf-8"))

    prefix = "data" if requested_config == "default" else requested_config
    candidates = [
        item.get("path", "")
        for item in entries
        if item.get("type") == "file"
        and item.get("path", "").startswith(prefix + "/")
        and item.get("path", "").endswith((".parquet", ".parquet.zst"))
        and ("/" + split + "-") in ("/" + item.get("path", ""))
    ]
    if not candidates:
        # The Hub tree endpoint may return only directory entries for very
        # large repositories.  FineWeb2 Edu Japanese publishes stable shard
        # names, so use the first known shard as a deterministic fallback.
        known_shards = {
            "small_tokens_cleaned": 283,
            "small_tokens": 283,
            "sample_10BT": 60,
            "default": 535,
        }
        shard_count = known_shards.get(requested_config)
        if shard_count and split == "train":
            candidates = [
                f"{prefix}/train-00000-of-{shard_count:05d}.parquet"
            ]
    if not candidates:
        raise RuntimeError(
            f"No Parquet shard found for {dataset_id}:{requested_config}/{split}"
        )

    # One shard is enough for the caller's bounded sample loop.
    shard = sorted(candidates)[0]
    file_url = (
        "https://huggingface.co/datasets/"
        f"{dataset_id}/resolve/main/{urllib.parse.quote(shard, safe='/')}"
    )
    logger.warning("Using direct HF Parquet fallback: %s", shard)
    # Explicitly pass the persistent cache directory: datasets may still
    # materialize builder metadata even when the data reader is streaming.
    return _hf_load_dataset(
        "parquet",
        data_files={split: file_url},
        split=split,
        streaming=True,
        cache_dir=HF_DATASETS_CACHE,
        **kwargs,
    )

def safe_load_dataset(dataset_id, split="train", streaming=False, **kwargs):
    """load_dataset のラッパー。複数の方法を順に試行する。

    1. 通常ロード（Parquet/標準フォーマット対応データセット向け）
    2. trust_remote_code=True（datasets 2.x でカスタムスクリプト使用時）
    3. streaming モードへのフォールバック

    trust_remote_code 廃止に関する警告は自動的に抑制される。
    """
    # FineWeb2 is too large for a regular materialized load on a worker.
    # Always use streaming for this bounded training pipeline.
    if dataset_id == "hotchpotch/fineweb-2-edu-japanese":
        streaming = True
        kwargs.setdefault("cache_dir", HF_DATASETS_CACHE)

    # Attempt 1: standard load
    try:
        with _suppress_trust_remote_code_noise():
            return _hf_load_dataset(dataset_id, split=split, streaming=streaming, **kwargs)
    except Exception as e1:
        last_error = e1
        err_msg = str(e1).lower()
        # FineWeb2 Edu Japanese is a public Parquet dataset, but some
        # datasets/HF Hub combinations fail before the standard builder can
        # resolve its repository.  Use the direct shard fallback in that case.
        if dataset_id == "hotchpotch/fineweb-2-edu-japanese":
            try:
                return _load_hf_parquet_fallback(
                    dataset_id, split=split, **dict(kwargs)
                )
            except Exception as fallback_error:
                logger.warning(
                    "%s: direct Parquet fallback failed: %s",
                    dataset_id, fallback_error,
                )
        disk_full = _is_disk_full_error(e1)
        if disk_full:
            # A full cache dir fails even the smallest dataset before it
            # writes anything useful. Reclaim the space and fall through to
            # the retries below instead of giving up immediately.
            clear_hf_dataset_cache(reason=f"disk quota hit while loading {dataset_id}")
        elif "trust_remote_code" not in err_msg and "loading script" not in err_msg:
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

    if disk_full:
        raise RuntimeError(
            f"{dataset_id} のロードに失敗しました: ディスククォータ超過。"
            f"キャッシュ ({HF_DATASETS_CACHE}) をクリアして再試行しましたが、"
            f"依然として空き容量が不足しています。"
        ) from last_error

    raise RuntimeError(
        f"{dataset_id} のロードに失敗しました。"
        f"このデータセットはカスタムローディングスクリプトを使用しており、"
        f"datasets v3.x では非対応です。"
        f"対処法: pip install 'datasets>=2.18.0,<3'"
    )


# ============================================================
# Network Volume sync utility
# ============================================================
# NETWORK_VOLUME_PATH is initialized above before importing datasets.


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
