"""Common Crawl WET streaming loader for Japanese training text."""
from __future__ import annotations

import gzip
import io
import os
import re
from typing import List

import requests
from warcio.archiveiterator import ArchiveIterator

_JP_RE = re.compile(r"[\\u3040-\\u30ff\\u3400-\\u4dbf\\u4e00-\\u9fff]")
_DEFAULT_COLLECTION = "CC-MAIN-2026-34"
_DEFAULT_BASE = "https://data.commoncrawl.org"


def _is_japanese(text: str) -> bool:
    compact = re.sub(r"\\s+", "", text)
    if len(compact) < 80:
        return False
    jp = len(_JP_RE.findall(compact))
    return jp >= 20 and jp / max(len(compact), 1) >= 0.12


def load_commoncrawl_japanese(
    collection: str | None = None,
    max_samples: int = 50000,
    max_wet_files: int = 2,
    timeout: int = 120,
) -> List[str]:
    """Stream a bounded number of WET files and return Japanese text samples.

    The Common Crawl bucket is public; only the selected WET files are read.
    This avoids downloading the entire multi-terabyte archive.
    """
    collection = collection or os.environ.get("COMMONCRAWL_COLLECTION", _DEFAULT_COLLECTION)
    base = os.environ.get("COMMONCRAWL_BASE_URL", _DEFAULT_BASE).rstrip("/")
    paths_url = f"{base}/crawl-data/{collection}/wet.paths.gz"
    response = requests.get(paths_url, timeout=timeout)
    response.raise_for_status()
    paths = gzip.decompress(response.content).decode("utf-8").splitlines()
    if not paths:
        raise RuntimeError(f"No WET paths found for {collection}")

    texts: List[str] = []
    for path in paths[:max(1, max_wet_files)]:
        if len(texts) >= max_samples:
            break
        url = f"{base}/{path.lstrip('/')}"
        with requests.get(url, stream=True, timeout=timeout) as wet_response:
            wet_response.raise_for_status()
            wet_response.raw.decode_content = True
            for record in ArchiveIterator(wet_response.raw, arc2warc=True):
                if len(texts) >= max_samples:
                    break
                if record.rec_type != "conversion":
                    continue
                try:
                    raw = record.content_stream().read()
                    text = raw.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if _is_japanese(text):
                    texts.append(text[:100000])
    return texts
