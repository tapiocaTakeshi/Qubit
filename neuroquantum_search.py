#!/usr/bin/env python3
"""
NeuroQuantum 検索機能 (Search / Retrieval)

NeuroQuantum に「文書検索」と「検索拡張生成 (RAG)」を追加するモジュール。

2 種類のスコアを組み合わせたハイブリッド検索を提供します。

1. **BM25 (語彙一致)** - 純 Python 実装。日本語でも動くように
   文字 bigram + 単語トークンでインデックスします。torch 不要。
2. **Dense (意味的類似)** - NeuroQuantum モデルの最終 LayerNorm 出力を
   平均プーリングした文ベクトルによるコサイン類似度。モデルとトークナイザーを
   渡した場合のみ有効になります。

使用例::

    from neuroquantum_search import NeuroQuantumSearchIndex

    index = NeuroQuantumSearchIndex()                 # BM25 のみ
    index = NeuroQuantumSearchIndex(model, tokenizer)  # ハイブリッド

    index.add_documents([
        "量子コンピュータは量子力学の原理を利用した計算機です。",
        "ニューラルネットワークは脳の神経細胞を模倣した計算モデルです。",
    ])
    for hit in index.search("量子力学", top_k=3):
        print(hit.score, hit.text)

    # 検索結果をプロンプトに埋め込む (RAG)
    prompt = build_rag_prompt("量子コンピュータとは？", index.search("量子コンピュータとは？"))
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # torch はオプション (BM25 だけなら不要)
    import torch
except ImportError:  # pragma: no cover - torch が無い環境
    torch = None  # type: ignore[assignment]


# ========================================
# トークン化
# ========================================

# ラテン文字 / 数字の連続 (単語) と、それ以外の 1 文字 (CJK など) に分割する
_WORD_RE = re.compile(r"[a-z0-9_]+|[^\sa-z0-9_]", re.IGNORECASE)
# 記号だけのトークンは捨てる
_PUNCT_CATEGORIES = ("P", "S", "Z", "C")


def _is_punct(token: str) -> bool:
    """1 文字トークンが記号・空白・制御文字なら True (単語トークンは常に False)"""
    return len(token) == 1 and unicodedata.category(token)[0] in _PUNCT_CATEGORIES


def tokenize_for_search(text: str, ngram: int = 2) -> List[str]:
    """検索用トークン列を返す。

    - Unicode NFKC 正規化 + 小文字化
    - 英数字はそのまま単語トークン
    - それ以外 (日本語など) は 1 文字トークン + 文字 n-gram (既定は bigram)

    n-gram を混ぜることで、形態素解析なしでも「量子力学」が「量子」「力学」
    のどちらの問い合わせにもマッチします。
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    words = [w for w in _WORD_RE.findall(normalized) if not _is_punct(w)]
    tokens: List[str] = list(words)

    # 連続する非ラテン文字 (CJK 等) の並びから文字 n-gram を作る
    if ngram > 1:
        run: List[str] = []

        def flush() -> None:
            if len(run) >= ngram:
                for i in range(len(run) - ngram + 1):
                    tokens.append("".join(run[i:i + ngram]))
            run.clear()

        for w in words:
            if len(w) == 1 and not w.isascii():
                run.append(w)
            else:
                flush()
        flush()
    return tokens


# ========================================
# データ構造
# ========================================


@dataclass
class SearchDocument:
    """インデックスに登録された 1 文書"""

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """検索結果 1 件"""

    doc_id: str
    text: str
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ========================================
# BM25 (純 Python)
# ========================================


class BM25Index:
    """Okapi BM25 の最小実装。追加専用 (削除は全消去のみ)。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, ngram: int = 2):
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self._doc_term_freqs: List[Counter] = []
        self._doc_lengths: List[int] = []
        self._doc_freq: Counter = Counter()

    def __len__(self) -> int:
        return len(self._doc_term_freqs)

    @property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths) / len(self._doc_lengths)

    def add(self, text: str) -> int:
        tokens = tokenize_for_search(text, self.ngram)
        tf = Counter(tokens)
        self._doc_term_freqs.append(tf)
        self._doc_lengths.append(len(tokens))
        for term in tf:
            self._doc_freq[term] += 1
        return len(self._doc_term_freqs) - 1

    def clear(self) -> None:
        self._doc_term_freqs.clear()
        self._doc_lengths.clear()
        self._doc_freq.clear()

    def idf(self, term: str) -> float:
        n = len(self._doc_term_freqs)
        df = self._doc_freq.get(term, 0)
        # 0 以下にならないよう +1 した標準的な BM25+ 形式
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> List[float]:
        """全文書に対する BM25 スコア (文書順)"""
        n = len(self._doc_term_freqs)
        if n == 0:
            return []
        query_terms = Counter(tokenize_for_search(query, self.ngram))
        if not query_terms:
            return [0.0] * n
        avgdl = self.avg_doc_length or 1.0
        result = [0.0] * n
        for term in query_terms:
            if term not in self._doc_freq:
                continue
            idf = self.idf(term)
            for idx, tf in enumerate(self._doc_term_freqs):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self._doc_lengths[idx] / avgdl)
                result[idx] += idf * (f * (self.k1 + 1)) / denom
        return result


# ========================================
# Dense embedding (NeuroQuantum モデル)
# ========================================


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class NeuroQuantumEncoder:
    """NeuroQuantum モデルを文エンコーダとして使うためのラッパー。

    最終 LayerNorm (``final_norm``) の出力を forward hook で捕まえ、
    パディングを除いた平均プーリング → L2 正規化した文ベクトルを返します。
    モデル本体は変更しません。
    """

    def __init__(self, model, tokenizer, device=None, max_len: Optional[int] = None):
        if torch is None:
            raise ImportError("Dense 検索には torch が必要です")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_len = max_len or getattr(model.config, "max_seq_len", 512)
        self._captured: Optional["torch.Tensor"] = None
        self._hook = model.final_norm.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self._captured = output

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    @property
    def pad_id(self) -> int:
        return getattr(self.tokenizer, "pad_id", 0)

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> List[List[float]]:
        """テキスト列 → 正規化済み文ベクトル (Python リスト)"""
        if torch is None:  # pragma: no cover
            raise ImportError("Dense 検索には torch が必要です")
        was_training = self.model.training
        self.model.eval()
        vectors: List[List[float]] = []
        try:
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    chunk = texts[start:start + batch_size]
                    ids = [self.tokenizer.encode(t, add_special=True)[: self.max_len] for t in chunk]
                    ids = [seq if seq else [self.pad_id] for seq in ids]
                    width = max(len(seq) for seq in ids)
                    batch = torch.full((len(ids), width), self.pad_id, dtype=torch.long)
                    attn = torch.zeros((len(ids), width), dtype=torch.float32)
                    for row, seq in enumerate(ids):
                        batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                        attn[row, : len(seq)] = 1.0
                    batch = batch.to(self.device)
                    attn = attn.to(self.device)

                    self._captured = None
                    self.model(batch)
                    hidden = self._captured
                    if hidden is None:  # pragma: no cover - hook が外れている
                        raise RuntimeError("final_norm の出力を取得できませんでした")
                    pooled = (hidden * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True).clamp(min=1.0)
                    pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
                    vectors.extend(pooled.cpu().tolist())
        finally:
            if was_training:
                self.model.train()
        return vectors


# ========================================
# 検索インデックス
# ========================================


class NeuroQuantumSearchIndex:
    """BM25 + NeuroQuantum 埋め込みのハイブリッド検索インデックス。

    Args:
        model: ``NeuroQuantum`` モデル (省略時は BM25 のみ)
        tokenizer: ``NeuroQuantumTokenizer`` (model と一緒に指定)
        device: 推論デバイス (省略時はモデルのデバイス)
        alpha: ハイブリッドスコアにおける BM25 の重み (0.0-1.0)。
               ``score = alpha * bm25_norm + (1 - alpha) * dense``
        ngram: BM25 の文字 n-gram サイズ
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        alpha: float = 0.5,
        ngram: int = 2,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha は 0.0-1.0 の範囲で指定してください")
        self.alpha = alpha
        self.documents: List[SearchDocument] = []
        self._bm25 = BM25Index(k1=k1, b=b, ngram=ngram)
        self._embeddings: List[Optional[List[float]]] = []
        self._encoder: Optional[NeuroQuantumEncoder] = None
        self._id_to_index: Dict[str, int] = {}
        if model is not None:
            if tokenizer is None:
                raise ValueError("model を指定する場合は tokenizer も必要です")
            self._encoder = NeuroQuantumEncoder(model, tokenizer, device=device)

    # -- 基本情報 -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def has_dense(self) -> bool:
        return self._encoder is not None

    @property
    def mode(self) -> str:
        return "hybrid" if self.has_dense else "bm25"

    def status(self) -> Dict[str, Any]:
        return {
            "documents": len(self.documents),
            "mode": self.mode,
            "alpha": self.alpha,
            "dense_available": self.has_dense,
            "ngram": self._bm25.ngram,
        }

    # -- 登録 -----------------------------------------------------------

    def add_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """1 文書を追加して doc_id を返す"""
        return self.add_documents([text], [doc_id] if doc_id else None, [metadata or {}])[0]

    def add_documents(
        self,
        texts: Sequence[str],
        doc_ids: Optional[Sequence[Optional[str]]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[str]:
        """複数文書をまとめて追加して doc_id の一覧を返す。

        空文字列の文書は無視されます。同じ doc_id を再登録すると古い文書を
        置き換えます (BM25 統計は再構築されます)。
        """
        if doc_ids is not None and len(doc_ids) != len(texts):
            raise ValueError("doc_ids と texts の長さが一致しません")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas と texts の長さが一致しません")

        new_docs: List[SearchDocument] = []
        replaced = False
        for i, text in enumerate(texts):
            if text is None or not str(text).strip():
                continue
            text = str(text)
            doc_id = (doc_ids[i] if doc_ids is not None else None) or f"doc-{len(self.documents) + len(new_docs) + 1}"
            meta = dict(metadatas[i] or {}) if metadatas is not None else {}
            if doc_id in self._id_to_index:
                replaced = True
            new_docs.append(SearchDocument(doc_id=doc_id, text=text, metadata=meta))

        if not new_docs:
            return []

        if replaced:
            # 置き換え: 既存文書とマージしてから全再構築
            merged = {d.doc_id: d for d in self.documents}
            for d in new_docs:
                merged[d.doc_id] = d
            self._rebuild(list(merged.values()))
            return [d.doc_id for d in new_docs]

        vectors: List[Optional[List[float]]]
        if self._encoder is not None:
            vectors = list(self._encoder.encode([d.text for d in new_docs]))
        else:
            vectors = [None] * len(new_docs)
        for doc, vec in zip(new_docs, vectors):
            self._id_to_index[doc.doc_id] = len(self.documents)
            self.documents.append(doc)
            self._bm25.add(doc.text)
            self._embeddings.append(vec)
        return [d.doc_id for d in new_docs]

    def _rebuild(self, docs: List[SearchDocument]) -> None:
        self.documents = []
        self._embeddings = []
        self._id_to_index = {}
        self._bm25.clear()
        self.add_documents([d.text for d in docs], [d.doc_id for d in docs], [d.metadata for d in docs])

    def remove_document(self, doc_id: str) -> bool:
        """doc_id の文書を削除。存在した場合 True"""
        if doc_id not in self._id_to_index:
            return False
        self._rebuild([d for d in self.documents if d.doc_id != doc_id])
        return True

    def clear(self) -> None:
        self._rebuild([])

    def get(self, doc_id: str) -> Optional[SearchDocument]:
        idx = self._id_to_index.get(doc_id)
        return self.documents[idx] if idx is not None else None

    # -- 検索 -----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """クエリに近い文書を上位 ``top_k`` 件返す。

        Args:
            query: 検索クエリ
            top_k: 返す件数
            mode: ``"hybrid"`` / ``"bm25"`` / ``"dense"``。省略時はインデックスの
                  既定 (モデルがあれば hybrid、無ければ bm25)。
            min_score: このスコア未満の結果は除外
            metadata_filter: ``{"key": value}`` が全て一致する文書のみ対象
        """
        if not query or not query.strip() or not self.documents or top_k <= 0:
            return []
        mode = (mode or self.mode).lower()
        if mode not in ("hybrid", "bm25", "dense"):
            raise ValueError(f"不明な mode: {mode} (hybrid / bm25 / dense)")
        if mode in ("hybrid", "dense") and not self.has_dense:
            if mode == "dense":
                raise ValueError("dense 検索にはモデルとトークナイザーが必要です")
            mode = "bm25"

        n = len(self.documents)
        bm25_raw = self._bm25.scores(query) if mode in ("hybrid", "bm25") else [0.0] * n
        bm25_max = max(bm25_raw) if bm25_raw else 0.0
        bm25_norm = [s / bm25_max if bm25_max > 0 else 0.0 for s in bm25_raw]

        if mode in ("hybrid", "dense"):
            qvec = self._encoder.encode([query])[0]  # type: ignore[union-attr]
            dense = [
                max(0.0, _dot(qvec, vec)) if vec is not None else 0.0
                for vec in self._embeddings
            ]
        else:
            dense = [0.0] * n

        results: List[SearchResult] = []
        for idx, doc in enumerate(self.documents):
            if metadata_filter and any(doc.metadata.get(k) != v for k, v in metadata_filter.items()):
                continue
            if mode == "bm25":
                score = bm25_norm[idx]
                if bm25_raw[idx] <= 0.0:
                    continue
            elif mode == "dense":
                score = dense[idx]
            else:
                score = self.alpha * bm25_norm[idx] + (1.0 - self.alpha) * dense[idx]
            if score < min_score or score <= 0.0:
                continue
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=round(score, 6),
                    bm25_score=round(bm25_raw[idx], 6),
                    dense_score=round(dense[idx], 6),
                    metadata=dict(doc.metadata),
                )
            )

        results.sort(key=lambda r: (-r.score, r.doc_id))
        results = results[:top_k]
        for rank, r in enumerate(results, start=1):
            r.rank = rank
        return results

    # -- 永続化 ---------------------------------------------------------

    def save(self, path: str) -> None:
        """文書と (あれば) 埋め込みを JSON に保存"""
        payload = {
            "version": 1,
            "alpha": self.alpha,
            "ngram": self._bm25.ngram,
            "documents": [asdict(d) for d in self.documents],
            "embeddings": self._embeddings if self.has_dense else None,
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str, reembed: bool = False) -> int:
        """JSON からインデックスを復元して文書数を返す。

        保存時に埋め込みがあり、かつ現在もモデルがある場合は保存済み埋め込みを
        再利用します (``reembed=True`` で再計算)。
        """
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        docs = [SearchDocument(**d) for d in payload.get("documents", [])]
        saved_embeddings = payload.get("embeddings")
        self.alpha = float(payload.get("alpha", self.alpha))

        if self.has_dense and saved_embeddings and not reembed and len(saved_embeddings) == len(docs):
            # エンコード無しで復元
            self.documents = []
            self._embeddings = []
            self._id_to_index = {}
            self._bm25.clear()
            for doc, vec in zip(docs, saved_embeddings):
                self._id_to_index[doc.doc_id] = len(self.documents)
                self.documents.append(doc)
                self._bm25.add(doc.text)
                self._embeddings.append(vec)
        else:
            self._rebuild(docs)
        return len(self.documents)

    @classmethod
    def from_file(cls, path: str, model=None, tokenizer=None, **kwargs) -> "NeuroQuantumSearchIndex":
        index = cls(model=model, tokenizer=tokenizer, **kwargs)
        index.load(path)
        return index


# ========================================
# RAG プロンプト
# ========================================

DEFAULT_RAG_TEMPLATE = (
    "以下の参考情報を踏まえて質問に答えてください。\n\n"
    "{context}\n\n"
    "質問: {query}\n"
    "回答:"
)


def build_rag_prompt(
    query: str,
    results: Iterable[SearchResult],
    template: str = DEFAULT_RAG_TEMPLATE,
    max_context_chars: int = 1500,
    numbered: bool = True,
) -> str:
    """検索結果をコンテキストとして埋め込んだプロンプトを作る。

    ``results`` が空ならクエリをそのまま返します (検索が外れてもプロンプトを
    汚さない)。
    """
    lines: List[str] = []
    used = 0
    for i, r in enumerate(results, start=1):
        text = " ".join(r.text.split())
        if not text:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[: max(0, remaining - 1)] + "…"
        lines.append(f"[{i}] {text}" if numbered else text)
        used += len(text)
    if not lines:
        return query
    return template.format(context="\n".join(lines), query=query)


def load_documents_from_file(path: str) -> Tuple[List[str], List[Optional[str]], List[Dict[str, Any]]]:
    """テキスト / JSON / JSONL ファイルから文書を読み込む。

    - ``.jsonl``: 1 行 1 JSON。``text`` (必須) / ``id`` / ``metadata`` キーを使用
    - ``.json``: 文字列のリスト、または上記オブジェクトのリスト
    - それ以外: 空行区切りの段落を 1 文書として扱う
    """
    texts: List[str] = []
    ids: List[Optional[str]] = []
    metas: List[Dict[str, Any]] = []

    def push(item: Any) -> None:
        if isinstance(item, str):
            texts.append(item)
            ids.append(None)
            metas.append({})
        elif isinstance(item, dict) and "text" in item:
            texts.append(str(item["text"]))
            ids.append(str(item["id"]) if item.get("id") is not None else None)
            metas.append(dict(item.get("metadata") or {}))

    lower = path.lower()
    with open(path, encoding="utf-8") as f:
        if lower.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    push(json.loads(line))
        elif lower.endswith(".json"):
            data = json.load(f)
            for item in (data if isinstance(data, list) else [data]):
                push(item)
        else:
            for paragraph in re.split(r"\n\s*\n", f.read()):
                if paragraph.strip():
                    push(paragraph.strip())
    return texts, ids, metas


__all__ = [
    "BM25Index",
    "DEFAULT_RAG_TEMPLATE",
    "NeuroQuantumEncoder",
    "NeuroQuantumSearchIndex",
    "SearchDocument",
    "SearchResult",
    "build_rag_prompt",
    "load_documents_from_file",
    "tokenize_for_search",
]
