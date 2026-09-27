#!/usr/bin/env python3
"""
Qubit Search Engine - Algolia を使わない自前の全文検索サーバー

riddle アプリの「謎解き (Riddles)」「チャンネル (Channels)」検索を
Algolia から置き換えるための、依存ゼロ (標準ライブラリのみ) の検索エンジンです。

特徴:

- **複数インデックス**: ``Riddles`` / ``Channels`` など任意の名前で作成
- **増分インデックス**: 1 件の追加・更新・削除で全体を再構築しない転置インデックス
- **日本語対応**: ``neuroquantum_search.tokenize_for_search`` と同じ
  文字 + 文字 bigram トークン化で、形態素解析なしに部分一致します
- **属性重み付き BM25F**: ``searchableAttributes`` の順に重みを付けて採点
- **入力途中の検索**: 最後の英数字の単語は前方一致で展開
- **カスタムランキング**: ``desc(playCount)`` のような同点時の並び順
- **フィルタ**: ``category:謎解き AND NOT contentType:game`` / ``playCount>=10``
- **類似レコメンド**: TF-IDF コサイン類似度による「似ている謎解き」
- **クリック / コンバージョン計測**: イベント数を人気度として順位に反映
- **永続化**: インデックスごとの JSON スナップショット (アトミック書き込み)
- **API キー**: 書き込み用 (admin) と検索用 (search) を分離

HTTP API (JSON)::

    GET    /health
    GET    /1/indexes
    POST   /1/indexes/{index}/query        {"query", "page", "hitsPerPage", "filters", ...}
    POST   /1/indexes/{index}/recommend    {"objectID", "maxRecommendations", "threshold"}
    GET    /1/indexes/{index}/settings
    PUT    /1/indexes/{index}/settings     (admin)
    POST   /1/indexes/{index}/batch        (admin) {"requests": [{"action", "body"}]}
    POST   /1/indexes/{index}/clear        (admin)
    DELETE /1/indexes/{index}              (admin)
    GET    /1/indexes/{index}/{objectID}
    PUT    /1/indexes/{index}/{objectID}   (admin)
    PATCH  /1/indexes/{index}/{objectID}   (admin, 部分更新)
    DELETE /1/indexes/{index}/{objectID}   (admin)
    POST   /1/events                       {"events": [{"eventType", "index", "objectIDs"}]}

起動::

    QUBIT_SEARCH_ADMIN_KEY=... QUBIT_SEARCH_API_KEY=... \\
        python qubit_search_engine.py serve --port 8080 --data-dir ./search_data

ライブラリとして::

    from qubit_search_engine import SearchEngine

    engine = SearchEngine()
    engine.save_object("Riddles", {"objectID": "r1", "title": "量子の謎"})
    engine.search("Riddles", {"query": "量子"})
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import signal
import tempfile
import threading
import time
import unicodedata
from bisect import bisect_left
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

from neuroquantum_search import tokenize_for_search

logger = logging.getLogger("qubit_search")

VERSION = "1.0.0"
MAX_BODY_BYTES = 10 * 1024 * 1024
MAX_HITS_PER_PAGE = 1000
MAX_PREFIX_EXPANSIONS = 50
PREFIX_WEIGHT = 0.7
SIMILAR_QUERY_TERMS = 64
_INDEX_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# 設定を省略したときの既定値 (riddle アプリのコレクションに合わせています)
DEFAULT_SETTINGS: Dict[str, Any] = {
    "searchableAttributes": ["title", "name", "displayName", "tags", "category", "description", "text"],
    "customRanking": [],
    "unretrievableAttributes": [],
    "minCoverage": 0.5,
    "popularityWeight": 0.1,
}

PRESET_SETTINGS: Dict[str, Dict[str, Any]] = {
    "Riddles": {
        "searchableAttributes": ["title", "tags", "category", "authorName", "description", "text", "content"],
        "customRanking": ["desc(playCount)", "desc(answerCount)", "desc(likes)", "desc(createdAt)"],
        # 公開検索キーで答えが漏れないよう、答え系のフィールドは返さない
        "unretrievableAttributes": ["answer", "answers", "correctAnswers", "hints"],
    },
    "Channels": {
        "searchableAttributes": ["displayName", "name", "channelName", "bio", "description"],
        "customRanking": ["desc(followerCount)", "desc(subscriberCount)"],
        "unretrievableAttributes": ["email", "fcmToken", "fcmTokens"],
    },
}


class SearchError(Exception):
    """API エラー (HTTP ステータス付き)"""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = int(status)


# ========================================
# ユーティリティ
# ========================================


def normalize_text(text: str) -> str:
    """NFKC + 小文字化 + 空白除去 (完全一致・フレーズ判定用)"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())


def _flatten_text(value: Any) -> str:
    """属性値 (文字列 / 数値 / リスト / 辞書) を検索用テキストにする"""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)


def _get_path(obj: Dict[str, Any], path: str) -> Any:
    """``author.name`` のようなドット区切りで値を取得"""
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _sortable(value: Any) -> Optional[float]:
    """customRanking 用に値を数値化 (ISO 日時も可)。比較できなければ None"""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return float(len(value))
    if isinstance(value, dict):
        # Firestore Timestamp を JSON 化した形 {"_seconds": ...}
        for key in ("_seconds", "seconds"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
        return None
    if isinstance(value, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _bounded_int(value: Any, name: str, default: int, lo: int, hi: int) -> int:
    try:
        result = int(default if value is None or value == "" else value)
    except (TypeError, ValueError):
        raise SearchError(f"{name} は整数で指定してください")
    if not lo <= result <= hi:
        raise SearchError(f"{name} は {lo}-{hi} で指定してください")
    return result


def _parse_ranking(rules: Sequence[str]) -> List[Tuple[str, bool]]:
    """``["desc(playCount)", "asc(title)"]`` → ``[("playCount", True), ("title", False)]``"""
    parsed = []
    for rule in rules:
        m = re.fullmatch(r"\s*(asc|desc)\(([^)]+)\)\s*", str(rule))
        if not m:
            raise SearchError(f"customRanking の形式が不正です: {rule!r} (asc(attr) / desc(attr))")
        parsed.append((m.group(2).strip(), m.group(1) == "desc"))
    return parsed


# ========================================
# フィルタ
# ========================================

_FILTER_TERM_RE = re.compile(
    r"""^\s*(?P<not>NOT\s+)?
        (?P<attr>[A-Za-z0-9_.\-]+)\s*
        (?P<op>:|<=|>=|!=|=|<|>)\s*
        (?P<value>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\S+)\s*$""",
    re.VERBOSE,
)


def _unquote_filter_value(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return re.sub(r"\\(.)", r"\1", raw[1:-1])
    return raw


def _match_value(actual: Any, op: str, expected: str) -> bool:
    if op == ":":
        if isinstance(actual, list):
            return any(_match_value(a, ":", expected) for a in actual)
        if isinstance(actual, bool):
            return str(actual).lower() == expected.lower()
        if actual is None:
            return expected.lower() == "null"
        return str(actual) == expected
    left = _sortable(actual)
    try:
        right = float(expected)
    except ValueError:
        right = _sortable(expected)
    if left is None or right is None:
        return op == "!=" and str(actual) != expected
    return {
        "=": left == right,
        "!=": left != right,
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
    }[op]


def compile_filters(filters: Optional[str], facet_filters: Any = None) -> Callable[[Dict[str, Any]], bool]:
    """Algolia 風のフィルタ文字列を判定関数にする。

    サポート: ``attr:value`` / ``attr=数値`` / ``<`` ``<=`` ``>`` ``>=`` ``!=`` /
    ``NOT`` / ``AND`` / ``OR`` (括弧なし、AND が OR より優先)。
    ``facetFilters`` は ``["a:x", ["b:y", "b:z"]]`` (外側 AND・内側 OR)。
    """
    groups: List[List[List[Tuple[bool, str, str, str]]]] = []

    if filters and str(filters).strip():
        if "(" in filters or ")" in filters:
            raise SearchError("filters の括弧はサポートしていません")
        or_group = []
        for or_part in re.split(r"\s+OR\s+", str(filters).strip()):
            and_terms = []
            for term in re.split(r"\s+AND\s+", or_part.strip()):
                m = _FILTER_TERM_RE.match(term)
                if not m:
                    raise SearchError(f"filters を解釈できません: {term!r}")
                and_terms.append(
                    (bool(m.group("not")), m.group("attr"), m.group("op"), _unquote_filter_value(m.group("value")))
                )
            or_group.append(and_terms)
        groups.append(or_group)

    if facet_filters:
        items = facet_filters if isinstance(facet_filters, list) else [facet_filters]
        for item in items:
            alternatives = item if isinstance(item, list) else [item]
            or_group = []
            for alt in alternatives:
                text = str(alt)
                negate = text.startswith("-")
                attr, sep, value = text.lstrip("-").partition(":")
                if not sep:
                    raise SearchError(f"facetFilters の形式が不正です: {alt!r}")
                or_group.append([(negate, attr.strip(), ":", value.strip())])
            groups.append(or_group)

    if not groups:
        return lambda _obj: True

    def predicate(obj: Dict[str, Any]) -> bool:
        for or_group in groups:
            ok = False
            for and_terms in or_group:
                if all(neg != _match_value(_get_path(obj, attr), op, val) for neg, attr, op, val in and_terms):
                    ok = True
                    break
            if not ok:
                return False
        return True

    return predicate


# ========================================
# インデックス
# ========================================


class SearchIndex:
    """1 つのインデックス (Algolia の index 相当)。

    スレッドセーフではないので ``SearchEngine`` のロック下で使います。
    """

    def __init__(self, name: str, settings: Optional[Dict[str, Any]] = None, ngram: int = 2,
                 k1: float = 1.2, b: float = 0.75):
        self.name = name
        self.ngram = ngram
        self.k1 = k1
        self.b = b
        self.settings: Dict[str, Any] = {}
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.popularity: Counter = Counter()
        # term -> {objectID: 重み付き tf}
        self._postings: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._doc_terms: Dict[str, Dict[str, float]] = {}
        self._doc_len: Dict[str, float] = {}
        self._total_len = 0.0
        self._phrase_text: Dict[str, str] = {}
        self._vocab_sorted: Optional[List[str]] = None
        self._norm_cache: Dict[str, Tuple[float, int]] = {}
        self.updated_at = time.time()
        self.set_settings({**DEFAULT_SETTINGS, **PRESET_SETTINGS.get(name, {}), **(settings or {})}, reindex=False)

    # -- 設定 -------------------------------------------------------------

    def set_settings(self, settings: Dict[str, Any], reindex: bool = True) -> None:
        merged = {**DEFAULT_SETTINGS, **self.settings, **settings}
        attrs = merged.get("searchableAttributes") or []
        if not isinstance(attrs, list) or not all(isinstance(a, str) and a for a in attrs):
            raise SearchError("searchableAttributes は文字列のリストで指定してください")
        _parse_ranking(merged.get("customRanking") or [])
        if not 0.0 <= float(merged.get("minCoverage", 0.5)) <= 1.0:
            raise SearchError("minCoverage は 0.0-1.0 で指定してください")
        self.settings = merged
        # 先頭の属性ほど重い: 1.0, 0.8, 0.65, ... (最低 0.3)
        self._weights = [(a, max(0.3, 1.0 * (0.82 ** i))) for i, a in enumerate(attrs)]
        self._ranking = _parse_ranking(merged.get("customRanking") or [])
        if reindex:
            objects = list(self.objects.values())
            self._reset_postings()
            for obj in objects:
                self._index(obj)
        self.updated_at = time.time()

    # -- 転置インデックス ---------------------------------------------------

    def _reset_postings(self) -> None:
        self._postings = defaultdict(dict)
        self._doc_terms = {}
        self._doc_len = {}
        self._total_len = 0.0
        self._phrase_text = {}
        self._vocab_sorted = None
        self._norm_cache: Dict[str, Tuple[float, int]] = {}

    def _index(self, obj: Dict[str, Any]) -> None:
        oid = obj["objectID"]
        terms: Dict[str, float] = defaultdict(float)
        length = 0.0
        phrase_parts = []
        for attr, weight in self._weights:
            text = _flatten_text(_get_path(obj, attr))
            if not text:
                continue
            phrase_parts.append(normalize_text(text))
            tokens = tokenize_for_search(text, self.ngram)
            length += weight * len(tokens)
            for tok in tokens:
                terms[tok] += weight
        self._doc_terms[oid] = dict(terms)
        self._doc_len[oid] = length
        self._total_len += length
        self._phrase_text[oid] = "\u0000".join(phrase_parts)
        for term, tf in terms.items():
            if term not in self._postings:
                self._vocab_sorted = None
            self._postings[term][oid] = tf

    def _unindex(self, oid: str) -> None:
        terms = self._doc_terms.pop(oid, None)
        if terms is None:
            return
        self._total_len -= self._doc_len.pop(oid, 0.0)
        self._phrase_text.pop(oid, None)
        self._norm_cache.pop(oid, None)
        for term in terms:
            posting = self._postings.get(term)
            if posting is None:
                continue
            posting.pop(oid, None)
            if not posting:
                del self._postings[term]
                self._vocab_sorted = None

    # -- オブジェクト操作 ---------------------------------------------------

    def save_object(self, obj: Dict[str, Any]) -> str:
        if not isinstance(obj, dict):
            raise SearchError("オブジェクトは JSON object で指定してください")
        oid = obj.get("objectID")
        if oid is None or str(oid) == "":
            raise SearchError("objectID が必要です")
        oid = str(oid)
        obj = {**obj, "objectID": oid}
        self._unindex(oid)
        self.objects[oid] = obj
        self._index(obj)
        self.updated_at = time.time()
        return oid

    def partial_update(self, oid: str, fields: Dict[str, Any], create_if_missing: bool = True) -> Optional[str]:
        current = self.objects.get(oid)
        if current is None and not create_if_missing:
            return None
        merged = {**(current or {}), **fields, "objectID": oid}
        return self.save_object(merged)

    def delete_object(self, oid: str) -> bool:
        if oid not in self.objects:
            return False
        self._unindex(oid)
        del self.objects[oid]
        self.popularity.pop(oid, None)
        self.updated_at = time.time()
        return True

    def clear(self) -> None:
        self.objects.clear()
        self.popularity.clear()
        self._reset_postings()
        self.updated_at = time.time()

    def get_object(self, oid: str, attributes: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
        obj = self.objects.get(oid)
        return self._retrievable(obj, attributes) if obj is not None else None

    def _retrievable(self, obj: Dict[str, Any], attributes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        hidden = set(self.settings.get("unretrievableAttributes") or [])
        if attributes and "*" not in attributes:
            wanted = set(attributes) | {"objectID"}
            return {k: copy.deepcopy(v) for k, v in obj.items() if k in wanted and k not in hidden}
        return {k: copy.deepcopy(v) for k, v in obj.items() if k not in hidden}

    # -- 採点 ---------------------------------------------------------------

    def _idf(self, term: str) -> float:
        n = len(self.objects)
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _expand_prefix(self, prefix: str) -> List[str]:
        if self._vocab_sorted is None:
            self._vocab_sorted = sorted(self._postings)
        vocab = self._vocab_sorted
        out = []
        i = bisect_left(vocab, prefix)
        while i < len(vocab) and vocab[i].startswith(prefix) and len(out) < MAX_PREFIX_EXPANSIONS:
            if vocab[i] != prefix:
                out.append(vocab[i])
            i += 1
        return out

    def _query_terms(self, query: str, prefix_last: bool) -> List[Tuple[str, float, int]]:
        """(term, 重み, 被覆グループ番号) のリスト。前方一致展開は元の語と同じグループ"""
        tokens = tokenize_for_search(query, self.ngram)
        seen: Dict[str, int] = {}
        terms: List[Tuple[str, float, int]] = []
        for tok in tokens:
            if tok in seen:
                continue
            seen[tok] = len(seen)
            terms.append((tok, 1.0, seen[tok]))
        if prefix_last and tokens:
            normalized = unicodedata.normalize("NFKC", query).lower()
            m = re.search(r"([a-z0-9_]+)$", normalized)
            if m and m.group(1) in seen:
                group = seen[m.group(1)]
                for expanded in self._expand_prefix(m.group(1)):
                    if expanded not in seen:
                        terms.append((expanded, PREFIX_WEIGHT, group))
        return terms

    def _custom_key(self, oid: str) -> Tuple:
        obj = self.objects[oid]
        key = []
        for attr, desc in self._ranking:
            val = _sortable(_get_path(obj, attr))
            if val is None:
                key.append((1, 0.0))  # 値なしは常に後ろ
            else:
                key.append((0, -val if desc else val))
        return tuple(key)

    def score_query(self, query: str, prefix_last: bool = True) -> Dict[str, Tuple[float, float]]:
        """objectID -> (関連度スコア, 被覆率)"""
        terms = self._query_terms(query, prefix_last)
        if not terms or not self.objects:
            return {}
        groups = {g for _, _, g in terms}
        n_groups = len(groups)
        avgdl = (self._total_len / len(self.objects)) or 1.0
        scores: Dict[str, float] = defaultdict(float)
        matched: Dict[str, set] = defaultdict(set)
        for term, qweight, group in terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf(term)
            for oid, tf in posting.items():
                denom = tf + self.k1 * (1 - self.b + self.b * self._doc_len[oid] / avgdl)
                scores[oid] += qweight * idf * tf * (self.k1 + 1) / denom
                matched[oid].add(group)

        min_cov = float(self.settings.get("minCoverage", 0.5)) if n_groups > 1 else 0.0
        phrase = normalize_text(query)
        result = {}
        for oid, score in scores.items():
            coverage = len(matched[oid]) / n_groups
            if coverage < min_cov:
                continue
            score *= 0.5 + 0.5 * coverage
            if phrase and phrase in self._phrase_text.get(oid, ""):
                score *= 1.5  # クエリがそのまま含まれていれば優遇
            result[oid] = (score, coverage)
        return result

    # -- 検索 ---------------------------------------------------------------

    def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        query = str(params.get("query") or "")
        page = _bounded_int(params.get("page"), "page", 0, 0, 10**6)
        hits_per_page = _bounded_int(params.get("hitsPerPage"), "hitsPerPage", 20, 1, MAX_HITS_PER_PAGE)
        predicate = compile_filters(params.get("filters"), params.get("facetFilters"))
        attributes = params.get("attributesToRetrieve")
        if attributes is not None and not isinstance(attributes, list):
            raise SearchError("attributesToRetrieve はリストで指定してください")
        prefix_last = params.get("prefixSearch", True) is not False
        popularity_weight = float(self.settings.get("popularityWeight", 0.1))

        if query.strip():
            scored = self.score_query(query, prefix_last=prefix_last)
            candidates = [
                (oid, score * (1.0 + popularity_weight * math.log1p(self.popularity.get(oid, 0))), cov)
                for oid, (score, cov) in scored.items()
                if predicate(self.objects[oid])
            ]
            candidates.sort(key=lambda c: (-round(c[1], 9), self._custom_key(c[0]), c[0]))
        else:
            # 空クエリは customRanking 順で全件 (人気度で同点を崩す)
            candidates = [(oid, 0.0, 0.0) for oid, obj in self.objects.items() if predicate(obj)]
            candidates.sort(key=lambda c: (self._custom_key(c[0]), -self.popularity.get(c[0], 0), c[0]))

        nb_hits = len(candidates)
        nb_pages = math.ceil(nb_hits / hits_per_page) if nb_hits else 0
        start = page * hits_per_page
        hits = []
        for position, (oid, score, cov) in enumerate(candidates[start:start + hits_per_page], start=start + 1):
            hit = self._retrievable(self.objects[oid], attributes)
            hit["_rankingInfo"] = {"score": round(score, 6), "coverage": round(cov, 4), "position": position}
            hits.append(hit)
        return {
            "hits": hits,
            "nbHits": nb_hits,
            "page": page,
            "nbPages": nb_pages,
            "hitsPerPage": hits_per_page,
            "query": query,
            "index": self.name,
            "processingTimeMS": int((time.perf_counter() - started) * 1000),
        }

    def _tfidf_norm(self, oid: str) -> float:
        """TF-IDF ベクトルのノルム。IDF は文書数の変化でゆっくり動くので、
        文書数が 10% 以上変わるまではキャッシュを使う"""
        n = len(self.objects)
        cached = self._norm_cache.get(oid)
        if cached is not None and abs(n - cached[1]) <= 0.1 * max(n, 1):
            return cached[0]
        norm = math.sqrt(sum((tf * self._idf(t)) ** 2 for t, tf in self._doc_terms[oid].items())) or 1.0
        self._norm_cache[oid] = (norm, n)
        return norm

    def similar(self, oid: str, max_results: int = 5, threshold: float = 0.0,
                filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """TF-IDF コサイン類似度で ``oid`` に似たオブジェクトを返す"""
        source = self._doc_terms.get(oid)
        if source is None:
            raise SearchError(f"objectID {oid!r} が見つかりません", HTTPStatus.NOT_FOUND)
        predicate = compile_filters(filters)
        weighted = {t: tf * self._idf(t) for t, tf in source.items()}
        top_terms = sorted(weighted.items(), key=lambda kv: -kv[1])[:SIMILAR_QUERY_TERMS]
        src_norm = math.sqrt(sum(w * w for _, w in top_terms)) or 1.0
        dots: Dict[str, float] = defaultdict(float)
        for term, w in top_terms:
            idf = self._idf(term)
            for other, tf in self._postings.get(term, {}).items():
                if other != oid:
                    dots[other] += w * tf * idf
        results = []
        for other, dot in dots.items():
            if not predicate(self.objects[other]):
                continue
            norm = self._tfidf_norm(other)
            sim = dot / (src_norm * norm)
            if sim > threshold:
                results.append((other, sim))
        results.sort(key=lambda r: (-r[1], self._custom_key(r[0]), r[0]))
        hits = []
        for other, sim in results[:max(0, max_results)]:
            hit = self._retrievable(self.objects[other])
            hit["_score"] = round(sim, 6)
            hits.append(hit)
        return hits

    # -- 状態 / 永続化 -------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entries": len(self.objects),
            "terms": len(self._postings),
            "updatedAt": self.updated_at,
        }

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "settings": self.settings,
            "objects": list(self.objects.values()),
            "popularity": dict(self.popularity),
        }

    @classmethod
    def from_snapshot(cls, payload: Dict[str, Any]) -> "SearchIndex":
        index = cls(payload["name"], payload.get("settings"))
        for obj in payload.get("objects", []):
            index.save_object(obj)
        index.popularity.update({k: int(v) for k, v in (payload.get("popularity") or {}).items()
                                 if k in index.objects})
        return index


# ========================================
# エンジン (複数インデックス + 永続化)
# ========================================


class SearchEngine:
    """複数の ``SearchIndex`` を管理し、ディスクへ永続化する。

    Args:
        data_dir: スナップショットの保存先 (None ならメモリのみ)
        flush_interval: 変更をまとめてディスクへ書き出す間隔 (秒)
    """

    def __init__(self, data_dir: Optional[str] = None, flush_interval: float = 2.0):
        self.data_dir = data_dir
        self.flush_interval = flush_interval
        self._indexes: Dict[str, SearchIndex] = {}
        self._dirty: set = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._flusher: Optional[threading.Thread] = None
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._load_all()

    # -- インデックス取得 ----------------------------------------------------

    @staticmethod
    def _check_name(name: str) -> str:
        if not _INDEX_NAME_RE.match(name or ""):
            raise SearchError("インデックス名は英数字・_・- の 1-64 文字で指定してください")
        return name

    def _get(self, name: str, create: bool = False) -> SearchIndex:
        self._check_name(name)
        index = self._indexes.get(name)
        if index is None:
            if not create:
                raise SearchError(f"インデックス {name!r} はありません", HTTPStatus.NOT_FOUND)
            index = SearchIndex(name)
            self._indexes[name] = index
        return index

    def _touch(self, name: str) -> None:
        self._dirty.add(name)

    def list_indexes(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [idx.info() for idx in sorted(self._indexes.values(), key=lambda i: i.name)]

    # -- 書き込み ------------------------------------------------------------

    def save_object(self, name: str, obj: Dict[str, Any]) -> str:
        with self._lock:
            oid = self._get(name, create=True).save_object(obj)
            self._touch(name)
            return oid

    def partial_update(self, name: str, oid: str, fields: Dict[str, Any], create_if_missing: bool = True) -> Optional[str]:
        with self._lock:
            result = self._get(name, create=True).partial_update(oid, fields, create_if_missing)
            self._touch(name)
            return result

    def delete_object(self, name: str, oid: str) -> bool:
        with self._lock:
            if name not in self._indexes:
                return False
            deleted = self._indexes[name].delete_object(oid)
            self._touch(name)
            return deleted

    def batch(self, name: str, requests: Sequence[Dict[str, Any]]) -> List[Optional[str]]:
        """Algolia の batch 相当。``action`` は addObject / updateObject /
        partialUpdateObject / partialUpdateObjectNoCreate / deleteObject / clear"""
        if not isinstance(requests, list):
            raise SearchError("requests はリストで指定してください")
        with self._lock:
            index = self._get(name, create=True)
            results: List[Optional[str]] = []
            for req in requests:
                if not isinstance(req, dict):
                    raise SearchError("requests の要素は object で指定してください")
                action = req.get("action")
                body = req.get("body") or {}
                if action in ("addObject", "updateObject"):
                    results.append(index.save_object(body))
                elif action in ("partialUpdateObject", "partialUpdateObjectNoCreate"):
                    oid = str(body.get("objectID") or "")
                    if not oid:
                        raise SearchError("partialUpdateObject には objectID が必要です")
                    fields = {k: v for k, v in body.items() if k != "objectID"}
                    results.append(index.partial_update(oid, fields, action == "partialUpdateObject"))
                elif action == "deleteObject":
                    oid = str(body.get("objectID") or "")
                    results.append(oid if index.delete_object(oid) else None)
                elif action == "clear":
                    index.clear()
                    results.append(None)
                else:
                    raise SearchError(f"不明な action: {action!r}")
            self._touch(name)
            return results

    def clear_index(self, name: str) -> None:
        with self._lock:
            self._get(name, create=True).clear()
            self._touch(name)

    def delete_index(self, name: str) -> bool:
        with self._lock:
            self._check_name(name)
            existed = self._indexes.pop(name, None) is not None
            self._dirty.discard(name)
            if self.data_dir:
                path = self._path(name)
                if os.path.exists(path):
                    os.remove(path)
            return existed

    def set_settings(self, name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(settings, dict):
            raise SearchError("settings は object で指定してください")
        with self._lock:
            index = self._get(name, create=True)
            index.set_settings(settings)
            self._touch(name)
            return dict(index.settings)

    def record_events(self, events: Sequence[Dict[str, Any]]) -> int:
        """click / conversion / view イベントを人気度として加算"""
        if not isinstance(events, list):
            raise SearchError("events はリストで指定してください")
        weights = {"click": 1, "conversion": 3, "view": 0}
        counted = 0
        with self._lock:
            for event in events:
                if not isinstance(event, dict):
                    continue
                name = str(event.get("index") or "")
                weight = weights.get(str(event.get("eventType") or "click"), 1)
                if name not in self._indexes or weight == 0:
                    continue
                index = self._indexes[name]
                for oid in event.get("objectIDs") or []:
                    if str(oid) in index.objects:
                        index.popularity[str(oid)] += weight
                        counted += 1
                self._touch(name)
        return counted

    # -- 読み込み ------------------------------------------------------------

    def search(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if name not in self._indexes:
                self._check_name(name)
                # 未作成のインデックスは空結果 (Algolia と同じ扱い)
                return SearchIndex(name).search(params)
            return self._indexes[name].search(params)

    def get_object(self, name: str, oid: str, attributes: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            index = self._indexes.get(name)
            return index.get_object(oid, attributes) if index else None

    def get_settings(self, name: str) -> Dict[str, Any]:
        with self._lock:
            index = self._indexes.get(self._check_name(name))
            if index is None:
                return SearchIndex(name).settings
            return dict(index.settings)

    def recommend(self, name: str, oid: str, max_results: int = 5, threshold: float = 0.0,
                  filters: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return self._get(name).similar(oid, max_results, threshold, filters)

    # -- 永続化 ---------------------------------------------------------------

    def _path(self, name: str) -> str:
        assert self.data_dir
        return os.path.join(self.data_dir, f"{name}.json")

    def _load_all(self) -> None:
        assert self.data_dir
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.data_dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    index = SearchIndex.from_snapshot(json.load(f))
                self._indexes[index.name] = index
                logger.info("loaded index %s (%d objects)", index.name, len(index.objects))
            except (OSError, ValueError, KeyError, SearchError) as e:
                logger.error("failed to load %s: %s", path, e)

    def flush(self) -> int:
        """変更のあったインデックスをディスクに書き出し、書いた数を返す"""
        if not self.data_dir:
            with self._lock:
                self._dirty.clear()
            return 0
        with self._lock:
            names = sorted(self._dirty)
            snapshots = [(n, self._indexes[n].to_snapshot()) for n in names if n in self._indexes]
            self._dirty.clear()
        for name, snapshot in snapshots:
            fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=self.data_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False)
                os.replace(tmp, self._path(name))
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                with self._lock:
                    self._dirty.add(name)
                raise
        return len(snapshots)

    def start_background_flush(self) -> None:
        if self._flusher or not self.data_dir:
            return

        def loop() -> None:
            while not self._stop.wait(self.flush_interval):
                try:
                    self.flush()
                except Exception:  # pragma: no cover - ディスク障害時もサーバーは継続
                    logger.exception("flush failed")

        self._flusher = threading.Thread(target=loop, name="qubit-search-flush", daemon=True)
        self._flusher.start()

    def close(self) -> None:
        self._stop.set()
        if self._flusher:
            self._flusher.join(timeout=5)
            self._flusher = None
        self.flush()


# ========================================
# HTTP サーバー
# ========================================


class SearchRequestHandler(BaseHTTPRequestHandler):
    """``SearchEngine`` を JSON API として公開するハンドラ"""

    server_version = f"QubitSearch/{VERSION}"
    engine: SearchEngine
    admin_key: Optional[str] = None
    search_key: Optional[str] = None
    cors_origin: str = "*"

    # -- 共通 ---------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - 標準ロガーへ
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", self.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Qubit-Search-Key")
        self.send_header("Access-Control-Max-Age", "86400")

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise SearchError("リクエストが大きすぎます", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if length == 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SearchError("JSON を解釈できません")
        if not isinstance(payload, dict):
            raise SearchError("リクエストボディは JSON object で指定してください")
        return payload

    def _presented_key(self) -> Optional[str]:
        key = self.headers.get("X-Qubit-Search-Key")
        if key:
            return key.strip()
        auth = self.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def _authorize(self, admin: bool) -> None:
        import hmac

        key = self._presented_key() or ""
        if self.admin_key and hmac.compare_digest(key, self.admin_key):
            return
        if admin:
            if self.admin_key:
                raise SearchError("admin キーが必要です", HTTPStatus.FORBIDDEN)
            return  # キー未設定 (開発用) は誰でも書き込み可
        if self.search_key and not hmac.compare_digest(key, self.search_key):
            raise SearchError("検索キーが不正です", HTTPStatus.FORBIDDEN)

    def _route(self) -> Tuple[List[str], Dict[str, str]]:
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        query = {}
        for pair in parsed.query.split("&") if parsed.query else []:
            k, _, v = pair.partition("=")
            query[unquote(k)] = unquote(v.replace("+", " "))
        return parts, query

    def _handle(self) -> None:
        try:
            parts, query = self._route()
            status, payload = self._dispatch(self.command, parts, query)
            self._send(status, payload)
        except SearchError as e:
            self._send(e.status, {"message": str(e), "status": e.status})
        except (ValueError, TypeError) as e:
            self._send(HTTPStatus.BAD_REQUEST, {"message": str(e), "status": 400})
        except Exception:  # pragma: no cover - 予期しない例外
            logger.exception("internal error")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"message": "internal error", "status": 500})

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- ルーティング -----------------------------------------------------------

    def _dispatch(self, method: str, parts: List[str], query: Dict[str, str]) -> Tuple[int, Any]:
        engine = self.engine
        if parts in ([], ["health"]) and method == "GET":
            return 200, {"status": "ok", "version": VERSION, "indexes": len(engine.list_indexes())}

        if parts == ["1", "events"] and method == "POST":
            self._authorize(admin=False)
            counted = engine.record_events(self._body().get("events") or [])
            return 200, {"status": "OK", "counted": counted}

        if parts[:2] != ["1", "indexes"]:
            raise SearchError("not found", HTTPStatus.NOT_FOUND)
        rest = parts[2:]

        if not rest and method == "GET":
            self._authorize(admin=True)
            return 200, {"items": engine.list_indexes()}
        if not rest:
            raise SearchError("method not allowed", HTTPStatus.METHOD_NOT_ALLOWED)

        name = rest[0]
        if len(rest) == 1:
            if method == "DELETE":
                self._authorize(admin=True)
                return 200, {"deleted": engine.delete_index(name)}
            if method == "GET":  # GET /1/indexes/{index}?query=...
                self._authorize(admin=False)
                return 200, engine.search(name, {"query": query.get("query", ""),
                                                 "page": query.get("page"),
                                                 "hitsPerPage": query.get("hitsPerPage")})
            raise SearchError("method not allowed", HTTPStatus.METHOD_NOT_ALLOWED)

        action = rest[1]
        if len(rest) == 2 and action == "query" and method == "POST":
            self._authorize(admin=False)
            return 200, engine.search(name, self._body())
        if len(rest) == 2 and action == "recommend" and method == "POST":
            self._authorize(admin=False)
            body = self._body()
            oid = str(body.get("objectID") or "")
            if not oid:
                raise SearchError("objectID が必要です")
            hits = engine.recommend(
                name, oid,
                max_results=_bounded_int(body.get("maxRecommendations"), "maxRecommendations", 5, 1, 100),
                threshold=float(body.get("threshold") or 0.0),
                filters=body.get("filters"),
            )
            return 200, {"hits": hits, "objectID": oid, "index": name}
        if len(rest) == 2 and action == "settings":
            if method == "GET":
                self._authorize(admin=False)
                return 200, engine.get_settings(name)
            if method == "PUT":
                self._authorize(admin=True)
                return 200, engine.set_settings(name, self._body())
        if len(rest) == 2 and action == "batch" and method == "POST":
            self._authorize(admin=True)
            ids = engine.batch(name, self._body().get("requests"))
            return 200, {"objectIDs": ids}
        if len(rest) == 2 and action == "clear" and method == "POST":
            self._authorize(admin=True)
            engine.clear_index(name)
            return 200, {"status": "OK"}

        if len(rest) == 2:
            oid = action
            if method == "GET":
                self._authorize(admin=False)
                attrs = query.get("attributesToRetrieve")
                obj = engine.get_object(name, oid, attrs.split(",") if attrs else None)
                if obj is None:
                    raise SearchError("object not found", HTTPStatus.NOT_FOUND)
                return 200, obj
            if method == "PUT":
                self._authorize(admin=True)
                return 200, {"objectID": engine.save_object(name, {**self._body(), "objectID": oid})}
            if method == "PATCH":
                self._authorize(admin=True)
                create = query.get("createIfNotExists", "true") != "false"
                result = engine.partial_update(name, oid, self._body(), create)
                return 200, {"objectID": result}
            if method == "DELETE":
                self._authorize(admin=True)
                return 200, {"deleted": engine.delete_object(name, oid), "objectID": oid}

        raise SearchError("not found", HTTPStatus.NOT_FOUND)


def create_server(engine: SearchEngine, host: str = "0.0.0.0", port: int = 8080,
                  admin_key: Optional[str] = None, search_key: Optional[str] = None,
                  cors_origin: str = "*") -> ThreadingHTTPServer:
    handler = type("BoundSearchRequestHandler", (SearchRequestHandler,), {
        "engine": engine,
        "admin_key": admin_key or None,
        "search_key": search_key or None,
        "cors_origin": cors_origin,
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


# ========================================
# CLI
# ========================================


def _load_objects(path: str) -> List[Dict[str, Any]]:
    """JSON 配列 / {"objects": [...]} / JSON Lines を読む"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        payload = json.loads(text)
        items = payload.get("objects", payload.get("hits", [])) if isinstance(payload, dict) else payload
    except json.JSONDecodeError:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    objects = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise SearchError(f"{path}: {i} 番目がオブジェクトではありません")
        if "objectID" not in item:
            item = {**item, "objectID": str(item.get("id") or item.get("uid") or i)}
        objects.append(item)
    return objects


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Qubit Search Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="HTTP サーバーを起動")
    serve.add_argument("--host", default=os.environ.get("QUBIT_SEARCH_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", os.environ.get("QUBIT_SEARCH_PORT", "8080"))))
    serve.add_argument("--data-dir", default=os.environ.get("QUBIT_SEARCH_DATA_DIR", "./search_data"))
    serve.add_argument("--cors-origin", default=os.environ.get("QUBIT_SEARCH_CORS_ORIGIN", "*"))

    imp = sub.add_parser("import", help="JSON / JSONL ファイルをインデックスに取り込む")
    imp.add_argument("file")
    imp.add_argument("--index", required=True)
    imp.add_argument("--data-dir", default=os.environ.get("QUBIT_SEARCH_DATA_DIR", "./search_data"))
    imp.add_argument("--replace", action="store_true", help="取り込み前にインデックスを空にする")

    query = sub.add_parser("search", help="ローカルのインデックスを検索")
    query.add_argument("query")
    query.add_argument("--index", required=True)
    query.add_argument("--data-dir", default=os.environ.get("QUBIT_SEARCH_DATA_DIR", "./search_data"))
    query.add_argument("--filters")
    query.add_argument("--hits", type=int, default=10)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "serve":
        admin_key = os.environ.get("QUBIT_SEARCH_ADMIN_KEY")
        search_key = os.environ.get("QUBIT_SEARCH_API_KEY")
        if not admin_key:
            logger.warning("QUBIT_SEARCH_ADMIN_KEY が未設定です: 誰でも書き込みできます (開発用)")
        engine = SearchEngine(args.data_dir)
        engine.start_background_flush()
        server = create_server(engine, args.host, args.port, admin_key, search_key, args.cors_origin)
        def _on_sigterm(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt  # docker stop でも最後に flush する

        signal.signal(signal.SIGTERM, _on_sigterm)
        logger.info("Qubit Search Engine %s listening on %s:%d (data: %s)", VERSION, args.host, args.port, args.data_dir)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            engine.close()
        return 0

    engine = SearchEngine(args.data_dir)
    if args.command == "import":
        objects = _load_objects(args.file)
        requests = ([{"action": "clear"}] if args.replace else []) + [
            {"action": "addObject", "body": obj} for obj in objects
        ]
        engine.batch(args.index, requests)
        engine.close()
        print(f"{len(objects)} 件を {args.index} に取り込みました")
        return 0

    result = engine.search(args.index, {"query": args.query, "filters": args.filters, "hitsPerPage": args.hits})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
