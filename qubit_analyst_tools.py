"""Deterministic, stdlib-only data tools for Qubit Analyst.

Cells and column names are untrusted data: they are parsed, never executed and never used
as format strings. Statistics are pure Python (checked against scipy in tests) and every
tool returns JSON-safe output: no NaN/Infinity, floats rounded to 6 significant digits (integer
digits are never rounded away). Requires Python 3.11+ (possessive regex quantifiers).
"""
import calendar
import csv
import datetime
import io
import itertools
import math
import operator
import re
import unicodedata
from collections import Counter

LIMITS = {"max_chars": 2_000_000, "max_rows": 20_000, "max_columns": 60,
          "max_cell_chars": 500, "max_column_name": 80, "max_groups": 50,
          "max_corr_columns": 20, "max_outliers": 20, "max_top_n": 50,
          "max_forecast_periods": 12, "max_crosstab_levels": 12}
MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "-", "—", "欠損", "不明", "#n/a"})
TRUE_TOKENS = frozenset({"true", "yes", "はい", "○", "◯"})
FALSE_TOKENS = frozenset({"false", "no", "いいえ", "×"})
KIND_LABELS = {"numeric": "数値", "datetime": "日付", "boolean": "真偽値",
               "categorical": "カテゴリ", "text": "テキスト"}
DIRECTION_LABELS = {"increasing": "増加傾向", "decreasing": "減少傾向",
                    "flat": "明確な傾向なし", "unknown": "判定不能"}
MAX_ABS = 1e50          # larger magnitudes are treated as unparseable cells
DESCRIBE_LIMIT = 20
OTHER = "その他"
FORECAST_CAVEAT = "過去の線形トレンドを延長した参考値です。季節性・構造変化・外部要因は考慮していません。"


class DataError(ValueError):
    """User-facing error; the message is Japanese and never contains internals."""


def _nfkc(text):
    return unicodedata.normalize("NFKC", text)


def name_key(name):
    """Loose column/label key: NFKC, casefold, whitespace removed."""
    return re.sub(r"\s+", "", _nfkc(str(name)).casefold())


def _clip(text, limit=80):
    text = "".join(" " if unicodedata.category(ch)[0] == "C" else ch for ch in str(text)[:limit * 4])
    return " ".join(text.split())[:limit]


class Column:
    """Parsed column: values hold float / datetime.date / bool / str, or None when missing."""

    def __init__(self, name, kind, values, raw_missing=0, coerced=0, unit=None, bool_labels=None):
        self.name, self.kind, self.values = name, kind, list(values)
        self.raw_missing, self.coerced, self.unit = raw_missing, coerced, unit
        self.bool_labels = bool_labels or {}      # {True: "はい", False: "いいえ"}: the data's own tokens
        self._year = None
        self.cache = {}                           # per-column derived facts (values never change)

    @property
    def missing(self):
        return sum(v is None for v in self.values)

    def present(self):
        return [v for v in self.values if v is not None]

    @property
    def is_year(self):
        """Numeric column whose whole name says year (年, 年度, 会計年度, 入社年, year, fiscal_year, FY; not
        年収, 年商, 年齢, yearly_sales) holding integer calendar years (cached: values never change)."""
        if self._year is None:
            values = self.present() if self.kind == "numeric" and _YEAR_NAME.search(name_key(self.name)) else []
            self._year = bool(values) and all(v.is_integer() and 1800 <= v <= 2200 for v in values)
        return self._year

    def __repr__(self):
        return f"Column({self.name!r}, {self.kind!r}, rows={len(self.values)})"


_YEAR_NAME = re.compile(r"(?:年|年度|西暦)$|^(?:fy|西暦)$|(?<![a-z])year(?![a-z])|year$")


class Table:
    def __init__(self, columns, name="data", dropped_rows=()):
        self.columns, self.name = list(columns), name
        self.dropped_rows = list(dropped_rows)      # [(row number, label)] of excluded 合計/Total rows
        self.n_rows = len(self.columns[0].values) if self.columns else 0
        self._exact = {c.name: c for c in self.columns}
        self._loose = {}
        for c in self.columns:
            key = name_key(c.name)
            self._loose[key] = None if key in self._loose else c   # None marks an ambiguous key

    def column(self, name):
        if isinstance(name, str):
            if name in self._exact:
                return self._exact[name]
            found = self._loose.get(name_key(name))
            if found is not None:
                return found
        raise DataError(f"列「{_clip(name, 40)}」が見つかりません")

    def kinds(self):
        return {c.name: c.kind for c in self.columns}

    def names(self, *kinds):
        return [c.name for c in self.columns if not kinds or c.kind in kinds]

    @property
    def coerced(self):
        return sum(c.coerced for c in self.columns)

    def __repr__(self):
        return f"Table({self.name!r}, rows={self.n_rows}, columns={len(self.columns)})"


# ---------------------------------------------------------------- cell parsing

# Possessive quantifiers keep matching linear: nothing else in the pattern matches ASCII
# whitespace, and digit runs cannot be re-split (a failing 64-char cell used to take seconds).
_NUMBER = re.compile(
    r"(\()?\s*+([+\-−▲△])?\s*+([¥$€£])?\s*+([+\-−])?\s*+"
    r"(\d{1,3}(?:,\d{3})++(?:\.\d++)?|(?:\d++(?:\.\d*+)?|\.\d++)(?:[eE][+-]?\d++)?)"
    r"\s*+([千万億兆])?\s*+(%|円|ドル|[¥$€£])?\s*+(\))?", re.ASCII)
_SCALE = {"千": 1e3, "万": 1e4, "億": 1e8, "兆": 1e12}
_CURRENCY = {"¥": "¥", "円": "¥", "$": "$", "ドル": "$", "€": "€", "£": "£"}
_DATE = re.compile(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})"
                   r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}:?\d{2})?)?", re.ASCII)
_MONTH = re.compile(r"(\d{4})[-/](\d{1,2})", re.ASCII)
_JDATE = re.compile(r"(\d{4})\s*年(?:\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?)?", re.ASCII)
# Fiscal years: "2016年度" / "FY2016" start in April; "2016年3月期" is dated by its closing month.
_FY = re.compile(r"(?:fy\s*)?(\d{4})\s*(?:年度|年\s*(\d{1,2})\s*月期)|fy\s*(\d{4})", re.ASCII | re.I)
_DOT_MONTH = re.compile(r"(?:19|20)\d{2}\.(?:0?[1-9]|1[0-2])", re.ASCII)


def scrub(text):
    """Replace lone UTF-16 surrogates (legal in JSON escapes, illegal in UTF-8) with '?'."""
    return text.encode("utf-8", "replace").decode("utf-8")


def _number(value):
    """-> (float, unit or None) or None. Bools, NaN and Infinity are never numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            x = float(value)
        except OverflowError:
            return None
        return (x + 0.0, None) if math.isfinite(x) and abs(x) <= MAX_ABS else None
    if not isinstance(value, str) or len(value) > 64:
        return None
    m = _NUMBER.fullmatch(_nfkc(value).strip())
    if not m:
        return None
    opened, sign, pre, sign2, digits, scale, suffix, closed = m.groups()
    if bool(opened) != bool(closed) or (sign and sign2) or (opened and (sign or sign2)):
        return None
    if pre and suffix and suffix != "%":
        return None
    x = float(digits.replace(",", "")) * _SCALE.get(scale, 1.0)
    if opened or (sign or sign2 or "+") in "-−▲△":
        x = -x
    if not math.isfinite(x) or abs(x) > MAX_ABS:
        return None
    unit = "%" if suffix == "%" else _CURRENCY.get(pre or suffix)
    return x + 0.0, unit


def parse_number(value):
    result = _number(value)
    return None if result is None else result[0]


def parse_date(value):
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if not isinstance(value, str) or len(value) > 40:
        return None
    text = _nfkc(value).strip()
    m = _DATE.fullmatch(text)
    if m:
        parts = m.group(1), m.group(3), m.group(4)
    elif (m := _MONTH.fullmatch(text)):
        parts = m.group(1), m.group(2), 1
    elif (m := _JDATE.fullmatch(text)):
        parts = m.group(1), m.group(2) or 1, m.group(3) or 1
    elif (m := _FY.fullmatch(text)):
        parts = m.group(1) or m.group(3), m.group(2) or 4, 1
    else:
        return None
    try:
        return datetime.date(*map(int, parts))
    except ValueError:
        return None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    text = _nfkc(value).strip().casefold()
    return True if text in TRUE_TOKENS else False if text in FALSE_TOKENS else None


def _label(value):
    """Stable display string for a cell / group key."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 2 ** 53 else repr(value)
    if isinstance(value, datetime.date):
        return value.isoformat()
    return str(value)


def _key(value):
    """JSON scalar for a group key (integral floats become ints, dates ISO strings)."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 2 ** 53:
        return int(value)
    if isinstance(value, datetime.date):
        return value.isoformat()
    return value


# ---------------------------------------------------------------- loading

def _too_large():
    return DataError(f"入力データが大きすぎます（最大{LIMITS['max_chars']:,}文字）")


def _too_many_rows():
    return DataError(f"行数が上限（{LIMITS['max_rows']:,}行）を超えています")


def _too_many_columns():
    return DataError(f"列数が上限（{LIMITS['max_columns']}列）を超えています")


def _cell(value, row):
    if value is None or isinstance(value, (bool, int, datetime.date)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, str):
        if len(value) > LIMITS["max_cell_chars"]:
            raise DataError(f"{row}行目のセルが長すぎます（最大{LIMITS['max_cell_chars']}文字）")
        text = _nfkc(scrub(value)).strip()[:LIMITS["max_cell_chars"]]
        return None if text.casefold() in MISSING_TOKENS else text
    raise DataError("セルの値は文字列・数値・真偽値・null のいずれかで指定してください")


def _decode(blob):
    if len(blob) > LIMITS["max_chars"] * 4:
        raise _too_large()
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return bytes(blob).decode(encoding)
        except UnicodeDecodeError:
            pass
    raise DataError("文字コードを判別できません（UTF-8 または Shift_JIS を使用してください）")


def _sniff(text):
    sample = text[:65536]
    cut = max(sample.rfind("\n"), sample.rfind("\r"))      # CR-only files (classic Mac exports) too
    if len(text) > len(sample) and cut > 0:
        sample = sample[:cut]
    best, best_score = ",", None
    for delimiter in (",", "\t", ";", "|"):
        try:
            rows = [r for r in itertools.islice(csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter), 60)
                    if any(cell.strip() for cell in r)]
        except csv.Error:
            continue
        if not rows or len(rows[0]) < 2:
            continue
        width = len(rows[0])
        same = sum(len(r) == width for r in rows[1:]) / (len(rows) - 1) if len(rows) > 1 else 1.0
        score = (same >= 0.9, width if same >= 0.9 else same)
        if best_score is None or score > best_score:
            best, best_score = delimiter, score
    return best


def _read_text(text):
    if len(text) > LIMITS["max_chars"]:
        raise _too_large()
    text = text.lstrip("﻿")
    if not text.strip():
        raise DataError("データが空です")
    names, raw, rows = None, None, 0
    try:
        for record in csv.reader(io.StringIO(text, newline=""), delimiter=_sniff(text)):
            if not any(cell.strip() for cell in record):
                continue
            if names is None:
                if len(record) > LIMITS["max_columns"]:
                    raise _too_many_columns()
                names, raw = record, [[] for _ in record]
                continue
            rows += 1
            if rows > LIMITS["max_rows"]:
                raise _too_many_rows()
            if len(record) > len(names) and any(c.strip() for c in record[len(names):]):
                raise DataError(f"{rows}行目の列数がヘッダーより多いです")
            for j, cells in enumerate(raw):
                cells.append(_cell(record[j], rows) if j < len(record) else None)
    except csv.Error:
        raise DataError("CSVを解析できません。区切り文字や引用符を確認してください") from None
    if names is None:
        raise DataError("ヘッダー行が必要です")
    return names, raw


def _read_records(records):
    if not records:
        raise DataError("データ行がありません")
    if len(records) > LIMITS["max_rows"]:
        raise _too_many_rows()
    keys = {}
    for record in records:
        if not isinstance(record, dict):
            raise DataError("レコードはオブジェクト（列名→値）で指定してください")
        for key in record:
            if key not in keys:
                if len(keys) >= LIMITS["max_columns"]:
                    raise _too_many_columns()
                keys[key] = None
    budget, raw = LIMITS["max_chars"], []
    for key in keys:
        cells = []
        for row, record in enumerate(records, 1):
            value = record.get(key)
            budget -= (len(value) + 1) if isinstance(value, str) else 1
            if budget < 0:
                raise _too_large()
            cells.append(_cell(value, row))
        raw.append(cells)
    return [str(k) for k in keys], raw


def _read_columns(mapping):
    if not mapping:
        raise DataError("データが空です")
    if len(mapping) > LIMITS["max_columns"]:
        raise _too_many_columns()
    lists = list(mapping.values())
    if any(not isinstance(v, (list, tuple)) for v in lists):
        raise DataError("列ごとの値は配列で指定してください")
    if len({len(v) for v in lists}) != 1:
        raise DataError("列ごとの配列の長さが揃っていません")
    if len(lists[0]) > LIMITS["max_rows"]:
        raise _too_many_rows()
    budget, raw = LIMITS["max_chars"], []
    for values in lists:
        budget -= sum((len(v) + 1) if isinstance(v, str) else 1 for v in values)
        if budget < 0:
            raise _too_large()
        raw.append([_cell(v, row) for row, v in enumerate(values, 1)])
    return [str(k) for k in mapping], raw


def _unique_names(names):
    result, seen = [], set()
    for index, raw in enumerate(names, 1):
        base = _clip(raw, LIMITS["max_column_name"]) or f"列{index}"
        name, suffix = base, 2
        while name_key(name) in seen:
            name, suffix = f"{base}_{suffix}", suffix + 1
        seen.add(name_key(name))
        result.append(name)
    return result


def _dotted_months(name, present):
    """"2024.1" … "2024.12" are months, not decimals, when "2024.1"/"2024.10" collide or the name says so."""
    if not present or not all(isinstance(v, str) and _DOT_MONTH.fullmatch(v) for _, v in present):
        return False
    texts = {v for _, v in present}
    # names of amounts per month (月間売上, 月額, 月商, monthly_sales) are not period names
    return len({float(v) for v in texts}) < len(texts) or bool(re.search(
        r"月(?!間|額|商|収|謝|給|販)|month(?!ly)|date|period|日付|時期", name_key(name)))


def _infer(name, cells):
    present = [(i, v) for i, v in enumerate(cells) if v is not None]
    total = len(present)
    allowed = total - (9 * total + 9) // 10      # kind needs >= 90% conforming values
    parsers = [("numeric", _number), ("datetime", parse_date), ("boolean", parse_bool)]
    if _dotted_months(name, present):
        parsers.insert(0, ("datetime", lambda v: datetime.date(int(v[:4]), int(v[5:]), 1)))
    for kind, parser in parsers:
        parsed, failures = [], 0
        for i, value in present:
            result = parser(value)
            if result is None:
                failures += 1
                if failures > allowed:
                    break
            else:
                parsed.append((i, result))
        else:
            if not total:
                continue
            values, unit, labels = [None] * len(cells), None, None
            if kind == "numeric":
                # The majority unit wins, counting unitless cells: one "5%" among plain amounts leaves the
                # column unitless and is coerced; a unit applies only when at least half the cells carry it.
                tagged = [(i, u) for i, (_, u) in parsed if u]
                units = Counter(u for _, u in tagged)
                unit = units.most_common(1)[0][0] if units and 2 * len(tagged) >= len(parsed) else None
                odd = {i for i, u in tagged if u != unit and (unit is not None or u == "%")}
                failures += len(odd)
                if failures > max(allowed, 1):     # one stray-unit cell never turns a short column into text
                    continue
                parsed = [(i, x) for i, (x, _) in parsed if i not in odd]
            elif kind == "boolean":     # keep the data's own words (はい/いいえ) for display
                raw = dict(present)
                tokens = {}
                for i, value in parsed:
                    if isinstance(raw[i], str):
                        tokens.setdefault(value, Counter())[raw[i]] += 1
                labels = {b: c.most_common(1)[0][0] for b, c in tokens.items()}
            for i, value in parsed:
                values[i] = value
            return Column(name, kind, values, len(cells) - total, failures, unit, labels)
    values = [None if v is None else _label(v) for v in cells]
    unique = len({v for v in values if v is not None})
    kind = "categorical" if unique <= max(20, 0.05 * len(cells)) else "text"
    return Column(name, kind, values, len(cells) - total)


def load_table(data, *, name="data"):
    """CSV/TSV text (header row required), list of record dicts, or dict of column -> list."""
    if isinstance(data, Table):
        return data
    title = (_clip(name, LIMITS["max_column_name"]) if isinstance(name, str) else "") or "data"
    if isinstance(data, (bytes, bytearray)):
        data = _decode(data)
    if isinstance(data, str):
        names, raw = _read_text(data)
    elif isinstance(data, list):
        names, raw = _read_records(data)
    elif isinstance(data, dict):
        names, raw = _read_columns(data)
    else:
        raise DataError("データはCSV文字列、レコードの配列、または列ごとの配列で指定してください")
    if not raw or not raw[0]:
        raise DataError("データ行がありません")
    names = _unique_names(names)
    columns = [_infer(n, cells) for n, cells in zip(names, raw)]
    dropped = _total_rows(columns)
    if dropped:          # re-type without the 合計 row so that its label and amount never count as data
        keep = len(raw[0]) - len(dropped)
        columns = [_infer(n, cells[:keep]) for n, cells in zip(names, raw)]
    return Table(columns, title, dropped)


TOTAL_LABELS = frozenset({"合計", "総計", "小計", "計", "total", "grand total", "sum", "合計値", "総合計"})


def _total_rows(columns):
    """Trailing 合計/Total rows (at most 2) of a spreadsheet export -> [(row number, label)]: the row's first
    label cell is a total word and, for some numeric column, its value is the sum of the rows above it."""
    labels = [c for c in columns if c.kind in ("categorical", "text")]
    nums = [c for c in columns if c.kind == "numeric"]
    n = len(columns[0].values) if columns else 0
    if not labels or not nums or n < 3:
        return []

    def label(r):
        return next((c.values[r] for c in labels if c.values[r] is not None), None)

    trailing = []
    for r in (n - 1, n - 2):
        text = label(r)
        if text is None or re.sub(r"\s+", " ", _nfkc(text).casefold()).strip() not in TOTAL_LABELS:
            break
        trailing.append(r)
    found = []
    for r in trailing:
        for c in nums:
            v = c.values[r]
            sums = {_fsum(x for x in c.values[:end] if x is not None) for end in (r, min(trailing))}
            if v is not None and any(s and abs(v - s) <= 0.005 * abs(s) for s in sums):
                found.append((r + 1, label(r)))
                break
        else:
            break
    return sorted(found)          # contiguous from the last row: a failed check stops the scan


# ---------------------------------------------------------------- statistics

_EPS, _FPMIN, _MAXIT = 1e-15, 1e-300, 10_000
_SMALL = "データ数が不足しています"
_FLAT = "値が一定のため計算できません"


def _as_float(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        x = float(x)
    except OverflowError:
        return None
    return x if math.isfinite(x) else None


def _clean(xs):
    return [v for v in map(_as_float, xs) if v is not None]


def _out(x):
    return x if isinstance(x, float) and math.isfinite(x) else None


def _fsum(xs):
    try:
        return math.fsum(xs)
    except OverflowError:
        return math.inf
    except ValueError:
        return math.nan


def _sq(xs, m):
    return _fsum(d * d for d in (x - m for x in xs))


def _flat(xs, m, ss):
    """Exactly constant, or variance at rounding-noise level (as numpy/scipy treat it)."""
    tiny = 1e-15 * abs(m)
    return max(xs) == min(xs) or ss / len(xs) <= tiny * tiny


def mean(xs):
    xs = _clean(xs)
    return _out(_fsum(xs) / len(xs)) if xs else None


def _variance(xs):
    m = _fsum(xs) / len(xs)
    return 0.0 if max(xs) == min(xs) else _sq(xs, m) / (len(xs) - 1)


def sample_std(xs):
    xs = _clean(xs)
    return _out(math.sqrt(_variance(xs))) if len(xs) >= 2 else None


def _quantile_sorted(xs, q):
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    a, b, t = xs[lo], xs[min(lo + 1, len(xs) - 1)], h - lo
    value = a + (b - a) * t if t < 0.5 else b - (b - a) * (1 - t)   # numpy's _lerp
    return _out(value if math.isfinite(value) else a * (1 - t) + b * t)


def quantile(xs, q):
    """Linear interpolation (numpy default / R type 7)."""
    xs, q = sorted(_clean(xs)), _as_float(q)
    return _quantile_sorted(xs, q) if xs and q is not None and 0 <= q <= 1 else None


def median(xs):
    return quantile(xs, 0.5)


def skewness(xs):
    """Adjusted Fisher-Pearson sample skewness (scipy.stats.skew(bias=False))."""
    xs = _clean(xs)
    n = len(xs)
    if n < 3:
        return None
    m = _fsum(xs) / n
    m2 = _sq(xs, m)
    if _flat(xs, m, m2):
        return None
    m2 /= n
    m3 = _fsum(d * d * d for d in (x - m for x in xs)) / n
    denom = m2 * math.sqrt(m2)
    if not denom > 0:          # m2 underflowed for |x| below ~1e-100 (scipy gives nan)
        return None
    return _out(m3 / denom * math.sqrt(n * (n - 1.0)) / (n - 2.0))


def _paired(xs, ys):
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys):
        return None
    pairs = [(a, b) for a, b in zip(map(_as_float, xs), map(_as_float, ys))
             if a is not None and b is not None]
    return [a for a, _ in pairs], [b for _, b in pairs]


def _corr(sxx, syy, sxy):
    product = sxx * syy
    denominator = math.sqrt(product) if 0 < product < math.inf else math.sqrt(sxx) * math.sqrt(syy)
    return _out(sxy / denominator)


def _moments(x, y):
    n = len(x)
    mx, my = _fsum(x) / n, _fsum(y) / n
    dx, dy = [v - mx for v in x], [v - my for v in y]
    return (mx, my, _fsum(map(operator.mul, dx, dx)), _fsum(map(operator.mul, dy, dy)),
            _fsum(map(operator.mul, dx, dy)))


def _r_pvalue(r, n):
    """Two-sided p for H0: rho=0 (exact under normality, as scipy.stats.pearsonr)."""
    if n < 3:
        return None
    if abs(r) >= 1:
        return 0.0
    a = n / 2.0 - 1.0
    return _out(min(1.0, 2.0 * _betainc(a, a, (1.0 - abs(r)) / 2.0, (1.0 + abs(r)) / 2.0)))


def pearson(xs, ys):
    """-> {r, p_value, n} (+ reason when undefined). Pairs with a missing side are dropped."""
    pairs = _paired(xs, ys)
    if pairs is None:
        return {"r": None, "p_value": None, "n": 0, "reason": "長さが一致しません"}
    x, y = pairs
    result = {"r": None, "p_value": None, "n": len(x)}
    if len(x) < 3:
        return {**result, "reason": _SMALL}
    mx, my, sxx, syy, sxy = _moments(x, y)
    if _flat(x, mx, sxx) or _flat(y, my, syy):
        return {**result, "reason": _FLAT}
    r = _corr(sxx, syy, sxy)
    if r is None:
        return {**result, "reason": "数値が大きすぎます"}
    r = max(-1.0, min(1.0, r))
    return {**result, "r": r, "p_value": _r_pvalue(r, len(x))}


def rankdata(xs):
    """1-based ranks; ties get the average rank."""
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = _paired(xs, ys)
    if pairs is None:
        return {"r": None, "p_value": None, "n": 0, "reason": "長さが一致しません"}
    x, y = pairs
    if len(x) < 3:
        return {"r": None, "p_value": None, "n": len(x), "reason": _SMALL}
    if max(x) == min(x) or max(y) == min(y):
        return {"r": None, "p_value": None, "n": len(x), "reason": _FLAT}
    return pearson(rankdata(x), rankdata(y))


def linregress(xs, ys):
    """OLS y = intercept + slope*x -> slope, intercept, r, r2, stderr, intercept_stderr, p_value."""
    pairs = _paired(xs, ys) or ([], [])
    x, y = pairs
    n = len(x)
    result = dict.fromkeys(("slope", "intercept", "r", "r2", "stderr", "intercept_stderr",
                            "p_value", "residual_std"))
    result.update(n=n, df=max(n - 2, 0))
    if n < 2:
        return {**result, "reason": _SMALL}
    mx, my, sxx, syy, sxy = _moments(x, y)
    if _flat(x, mx, sxx):
        return {**result, "reason": "説明変数が一定のため計算できません"}
    slope = sxy / sxx
    intercept = my - slope * mx
    flat_y = _flat(y, my, syy)
    r = 0.0 if flat_y else max(-1.0, min(1.0, _corr(sxx, syy, sxy) or 0.0))
    result.update(slope=_out(slope), intercept=_out(intercept), r=_out(r), r2=_out(r * r))
    if n < 3:
        return {**result, "reason": "n<3 のため標準誤差と p 値は計算できません"}
    sse = 0.0 if flat_y else _fsum(e * e for e in (b - intercept - slope * a for a, b in zip(x, y)))
    s = math.sqrt(sse / (n - 2))
    stderr = s / math.sqrt(sxx)
    result.update(stderr=_out(stderr), intercept_stderr=_out(stderr * math.sqrt(sxx / n + mx * mx)),
                  residual_std=_out(s), p_value=1.0 if flat_y else _r_pvalue(r, n))
    return result


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes betacf, modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= _FPMIN else _FPMIN)
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) >= _FPMIN else _FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) >= _FPMIN else _FPMIN
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betainc(a, b, x, xc=None):
    """I_x(a, b); xc is 1-x computed by the caller without cancellation when available."""
    xc = 1.0 - x if xc is None else xc
    if x <= 0.0:
        return 0.0
    if xc <= 0.0:
        return 1.0
    log_x = math.log1p(-xc) if xc < 0.5 else math.log(x)
    log_xc = math.log1p(-x) if x < 0.5 else math.log(xc)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * log_x + b * log_xc)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, xc) / b


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    a, b, x = _as_float(a), _as_float(b), _as_float(x)
    if None in (a, b, x) or a <= 0 or b <= 0 or not 0 <= x <= 1:
        return None
    return _out(_betainc(a, b, x))


def gammaincc(a, x):
    """Regularized upper incomplete gamma Q(a, x): series for P when x < a+1, else Lentz CF."""
    a, x = _as_float(a), _as_float(x)
    if a is None or x is None or a <= 0 or x < 0:
        return None
    if x == 0:
        return 1.0
    front = math.exp(-x + a * math.log(x) - math.lgamma(a))
    if x < a + 1.0:
        ap, total = a, 1.0 / a
        term = total
        for _ in range(_MAXIT):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return _out(max(0.0, 1.0 - total * front))
    b = x + 1.0 - a
    c, d = 1.0 / _FPMIN, 1.0 / b
    h = d
    for i in range(1, _MAXIT + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1.0 / (d if abs(d) >= _FPMIN else _FPMIN)
        c = b + an / c
        c = c if abs(c) >= _FPMIN else _FPMIN
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return _out(min(1.0, front * h))


def _valid_df(df):
    df = _as_float(df)
    return df if df is not None and df > 0 else None


def student_t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom (df may be fractional)."""
    df = _valid_df(df)
    if df is None or isinstance(t, bool) or not isinstance(t, (int, float)) or math.isnan(t):
        return None
    t = float(t)
    if t == 0:
        return 0.5
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    if df > 1e7:
        tail = 0.5 * math.erfc(abs(t) / math.sqrt(2.0))
    else:
        tt = t * t
        tail = 0.0 if math.isinf(tt) else 0.5 * _betainc(df / 2.0, 0.5, df / (df + tt), tt / (df + tt))
    return _out(tail if t > 0 else 1.0 - tail)


def student_t_cdf(t, df):
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        return None
    return student_t_sf(-t, df)


def _t_center(t, df):
    """P(0 < T < t) for t >= 0, accurate for tiny t."""
    if df > 1e7:
        return 0.5 * math.erf(t / math.sqrt(2.0))
    tt = t * t
    return 0.5 if math.isinf(tt) else 0.5 * _betainc(0.5, df / 2.0, tt / (df + tt), df / (df + tt))


def t_ppf(p, df):
    """Inverse CDF of Student's t by bisection (on the centre mass near 0.5, else the tail)."""
    p, df = _as_float(p), _valid_df(df)
    if p is None or df is None or not 0 < p < 1:
        return None
    if p == 0.5:
        return 0.0
    if 0.25 <= p <= 0.75:
        target, below = abs(p - 0.5), lambda t: _t_center(t, df) < target    # p - 0.5 is exact
    else:
        target = 1.0 - p if p > 0.5 else p
        below = lambda t: student_t_sf(t, df) > target
    lo, hi = 0.0, 1.0
    while below(hi):
        lo, hi = hi, hi * 2.0
        if hi > 1e300:
            return None
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi or hi - lo <= 1e-15 * hi:
            break
        if below(mid):
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return t if p > 0.5 else -t


def chi2_sf(x, k):
    """P(X > x) for chi-square with k degrees of freedom."""
    x, k = _as_float(x), _valid_df(k)
    if k is None or x is None:
        return None
    return 1.0 if x <= 0 else gammaincc(k / 2.0, x / 2.0)


def f_sf(f, d1, d2):
    """P(F > f) for Snedecor's F with (d1, d2) degrees of freedom (fractional allowed):
    I_{d2/(d2+d1 f)}(d2/2, d1/2), with the complement passed exactly."""
    f, d1, d2 = _as_float(f), _valid_df(d1), _valid_df(d2)
    if None in (f, d1, d2):
        return None
    if f <= 0:
        return 1.0
    scaled = d1 * f
    if math.isinf(scaled):
        return 0.0
    denominator = d2 + scaled
    return _out(_betainc(d2 / 2.0, d1 / 2.0, d2 / denominator, scaled / denominator))


def _rescaled(groups):
    """Groups shifted by the grand mean and divided by the largest deviation (None when every value
    is equal). F, Welch's F and eta-squared are affine-invariant, so this only keeps squares of
    1e50 or 1e-200 magnitudes from overflowing or underflowing."""
    values = [v for g in groups for v in g]
    if max(values) == min(values):
        return None
    m = _fsum(values) / len(values)
    scale = max(abs(v - m) for v in values)
    if not 0 < scale < math.inf:
        return None
    return [[(v - m) / scale for v in g] for g in groups]


def _group_ss(g, m):
    return 0.0 if max(g) == min(g) else _sq(g, m)


def anova_oneway(groups):
    """Classic one-way ANOVA (scipy.stats.f_oneway) -> f, df1, df2, p_value, eta_squared, k, n."""
    groups = [g for g in (_clean(g) for g in groups) if g]
    k, n = len(groups), sum(map(len, groups))
    result = {"f": None, "df1": k - 1 if k >= 2 else None, "df2": n - k if n > k >= 2 else None,
              "p_value": None, "eta_squared": None, "k": k, "n": n}
    if k < 2 or n <= k:
        return {**result, "reason": "2グループ以上と、グループ数を超えるデータ数が必要です"}
    z = _rescaled(groups)
    if z is None:
        return {**result, "reason": _FLAT}
    means = [_fsum(g) / len(g) for g in z]
    grand = _fsum(v for g in z for v in g) / n
    ssb = _fsum(len(g) * (m - grand) * (m - grand) for g, m in zip(z, means))
    ssw = _fsum(_group_ss(g, m) for g, m in zip(z, means))
    result["eta_squared"] = _out(ssb / (ssb + ssw))
    if not ssw > 0:
        return {**result, "reason": "グループ内のばらつきが0のため検定できません"}
    f = (ssb / (k - 1)) / (ssw / (n - k))
    return {**result, "f": _out(f), "p_value": f_sf(f, k - 1, n - k)}


def welch_anova(groups):
    """Welch's heteroscedastic one-way ANOVA (Welch 1951) -> f, df1, df2 (fractional), p_value.

    w_i = n_i / s_i², m_w = Σ w_i m_i / W, A = Σ w_i (m_i − m_w)² / (k − 1),
    Λ = Σ (1 − w_i / W)² / (n_i − 1), F = A / (1 + 2(k − 2)Λ / (k² − 1)), df2 = (k² − 1) / (3Λ).
    """
    groups = [g for g in (_clean(g) for g in groups) if g]
    k = len(groups)
    result = {"f": None, "df1": k - 1 if k >= 2 else None, "df2": None, "p_value": None, "k": k,
              "n": sum(map(len, groups))}
    if k < 2 or any(len(g) < 2 for g in groups):
        return {**result, "reason": "2グループ以上で、各グループに2件以上の値が必要です"}
    z = _rescaled(groups)
    if z is None:
        return {**result, "reason": _FLAT}
    variances = [_variance(g) for g in z]
    if not all(v > 0 for v in variances):
        return {**result, "reason": "値が一定のグループがあるため Welch の分散分析は計算できません"}
    means = [_fsum(g) / len(g) for g in z]
    weights = [len(g) / v for g, v in zip(z, variances)]
    total = _fsum(weights)
    centre = _fsum(w * m for w, m in zip(weights, means)) / total
    a = _fsum(w * (m - centre) * (m - centre) for w, m in zip(weights, means)) / (k - 1)
    lam = _fsum((1.0 - w / total) ** 2 / (len(g) - 1) for w, g in zip(weights, z))
    if not lam > 0 or not math.isfinite(a):
        return {**result, "reason": "数値が極端なため計算できません"}
    f = a / (1.0 + 2.0 * (k - 2) * lam / (k * k - 1.0))
    df2 = (k * k - 1.0) / (3.0 * lam)
    return {**result, "f": _out(f), "df2": _out(df2), "p_value": f_sf(f, k - 1, df2)}


def welch_ttest(a, b):
    """Welch's unequal-variance t-test (scipy ttest_ind(equal_var=False))."""
    a, b = _clean(a), _clean(b)
    result = {"t": None, "df": None, "p_value": None, "n_a": len(a), "n_b": len(b),
              "mean_a": mean(a), "mean_b": mean(b)}
    if len(a) < 2 or len(b) < 2:
        return {**result, "reason": "各グループに2件以上の値が必要です"}
    va, vb = _variance(a) / len(a), _variance(b) / len(b)
    if va + vb <= 0 or not math.isfinite(va + vb):
        return {**result, "reason": "両グループとも値が一定のため検定できません"}
    t = (result["mean_a"] - result["mean_b"]) / math.sqrt(va + vb)
    ra, rb = va / (va + vb), vb / (va + vb)          # scale-free Welch df: no underflow for tiny values
    df = 1.0 / (ra * ra / (len(a) - 1) + rb * rb / (len(b) - 1))
    p = student_t_sf(abs(t), df)
    return {**result, "t": _out(t), "df": _out(df),
            "p_value": None if p is None else min(1.0, 2.0 * p)}


def cohen_d(a, b):
    """Standardised mean difference with the pooled sample standard deviation."""
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = ((len(a) - 1) * _variance(a) + (len(b) - 1) * _variance(b)) / (len(a) + len(b) - 2)
    if not pooled > 0:
        return None
    return _out((_fsum(a) / len(a) - _fsum(b) / len(b)) / math.sqrt(pooled))


def chi2_independence(observed):
    """Pearson chi-square test of independence (no Yates correction); empty rows/cols dropped."""
    rows = [[_as_float(v) or 0.0 for v in row] for row in observed]
    rows = [row for row in rows if sum(row) > 0]
    keep = [j for j in range(max((len(r) for r in rows), default=0))
            if sum(r[j] for r in rows if j < len(r)) > 0]
    rows = [[r[j] if j < len(r) else 0.0 for j in keep] for r in rows]
    n = _fsum(v for r in rows for v in r)
    result = {"chi2": None, "dof": None, "p_value": None, "cramers_v": None, "n": n,
              "low_expected_share": None}
    if len(rows) < 2 or len(keep) < 2:
        return {**result, "reason": "2水準以上の行と列が必要です"}
    row_totals = [_fsum(r) for r in rows]
    col_totals = [_fsum(r[j] for r in rows) for j in range(len(keep))]
    expected = [[rt * ct / n for ct in col_totals] for rt in row_totals]
    chi2 = _fsum((o - e) * (o - e) / e for orow, erow in zip(rows, expected) for o, e in zip(orow, erow))
    dof = (len(rows) - 1) * (len(keep) - 1)
    low = sum(e < 5 for erow in expected for e in erow) / (len(rows) * len(keep))
    v = math.sqrt(min(1.0, chi2 / (n * (min(len(rows), len(keep)) - 1))))
    return {**result, "chi2": _out(chi2), "dof": dof, "p_value": chi2_sf(chi2, dof),
            "cramers_v": _out(v), "low_expected_share": low}


def cramers_v(observed):
    return chi2_independence(observed)["cramers_v"]


def apqb(r):
    """APQB view of a certainty r in [-1, 1]: r = cos(2θ), T = |sin(2θ)|, r² + T² = 1."""
    r = max(-1.0, min(1.0, float(r)))
    return {"theta": math.acos(r) / 2.0, "r": r, "T": math.sqrt(max(0.0, (1.0 - r) * (1.0 + r)))}


def _strength(r):
    a = abs(r)
    if a >= 0.7:
        return "strong", "強い相関"
    if a >= 0.4:
        return "moderate", "中程度の相関"
    if a >= 0.2:
        return "weak", "弱い相関"
    return "negligible", "ほぼ相関なし"


def _effect(d):
    a = abs(d)
    if a >= 0.8:
        return "large", "大きい"
    if a >= 0.5:
        return "medium", "中程度"
    if a >= 0.2:
        return "small", "小さい"
    return "negligible", "ほぼなし"


# ---------------------------------------------------------------- JSON helpers

def round_sig(x, digits=6):
    """Round to `digits` significant digits, but never round away integer digits (12345678 stays exact)."""
    x = _as_float(x)
    if x is None:
        return None
    if x == 0:
        return 0.0
    digits = min(17, max(digits, math.floor(math.log10(abs(x))) + 1))
    return float(format(x, f".{digits}g")) + 0.0


def json_safe(obj, _depth=0):
    """Non-finite floats -> None, floats -> 6 significant digits (integer part kept), dates -> ISO."""
    if _depth > 64:
        return None
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return round_sig(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k if isinstance(k, str) else str(json_safe(k, _depth + 1)): json_safe(v, _depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, _depth + 1) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [json_safe(v, _depth + 1) for v in sorted(obj, key=repr)]
    return str(obj)[:200]


_TOKEN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+", re.ASCII)


def number_tokens(obj):
    """Every finite number in a JSON-like object (values and digits inside strings), deduped."""
    found, seen = [], set()

    def add(x):
        x = _as_float(x)
        if x is not None and x not in seen:
            seen.add(x)
            found.append(x)

    def walk(o, depth):
        if depth > 64 or o is None or isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            add(o)
        elif isinstance(o, str):
            for m in _TOKEN.finditer(_nfkc(o)):
                add(float(m.group().replace(",", "")))
        elif isinstance(o, datetime.date):
            walk(o.isoformat(), depth)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for v in o:
                walk(v, depth + 1)

    walk(obj, 0)
    return found


# ---------------------------------------------------------------- tools

TOOLS = {}


def _tool(name):
    def register(fn):
        def run(table, args=None):
            return json_safe(fn(table, dict(args or {})))
        run.__name__, run.__doc__ = fn.__name__, fn.__doc__
        TOOLS[name] = run
        return run
    return register


def _rows(column):
    return [(i + 1, v) for i, v in enumerate(column.values) if v is not None]


def is_id_like(col):
    """ID / 番号 / コード columns and 1, 2, 3… row counters: skipped by default analyses."""
    key = name_key(col.name)
    if re.search(r"(?<![a-z])id$|^id(?![a-z])|番号|コード|^no\.?$|^#$|\bcode$", key):
        return True
    values = col.present()
    return (col.kind == "numeric" and len(values) > 2
            and values in (list(map(float, range(1, len(values) + 1))), list(map(float, range(len(values))))))


@_tool("profile")
def profile(table, args):
    """行数・列の型・欠損・変換できなかったセル。"""
    columns = []
    for col in table.columns:
        present = col.present()
        sample, seen = [], set()
        for value in present:
            label = _label(value)
            if label not in seen:
                seen.add(label)
                sample.append(value[:40] if isinstance(value, str) else value)
                if len(sample) == 3:
                    break
        entry = {"name": col.name, "kind": col.kind, "missing": len(col.values) - len(present),
                 "missing_share": (len(col.values) - len(present)) / max(1, len(col.values)),
                 "unique": len(set(present)), "coerced": col.coerced, "sample": sample}
        if col.unit:
            entry["unit"] = col.unit
        columns.append(entry)
    return {"rows": table.n_rows, "n_columns": len(table.columns), "columns": columns,
            "kinds": dict(Counter(c.kind for c in table.columns)),
            "missing_cells": sum(c["missing"] for c in columns), "coerced": table.coerced,
            "coerced_columns": {c.name: c.coerced for c in table.columns if c.coerced}}


def _describe_column(col):
    present = col.present()
    out = {"name": col.name, "kind": col.kind, "count": len(present),
           "missing": len(col.values) - len(present)}
    if col.kind == "numeric":
        xs = sorted(present)
        out.update(mean=mean(xs), std=sample_std(xs), min=xs[0] if xs else None,
                   q1=_quantile_sorted(xs, 0.25) if xs else None,
                   median=_quantile_sorted(xs, 0.5) if xs else None,
                   q3=_quantile_sorted(xs, 0.75) if xs else None,
                   max=xs[-1] if xs else None, skew=skewness(xs), sum=_out(_fsum(xs)) if xs else None)
        if col.unit:
            out["unit"] = col.unit
    elif col.kind == "datetime":
        out.update(min=min(present, default=None), max=max(present, default=None),
                   span_days=(max(present) - min(present)).days if present else None,
                   unique=len(set(present)))
    else:
        counts = Counter(present)
        top = sorted(counts.items(), key=lambda kv: (-kv[1], _label(kv[0])))[:10]
        out.update(unique=len(counts), top=[{"value": v, "count": c, "share": c / len(present)}
                                            for v, c in top])
    if not present:
        out["reason"] = "値がありません"
    return out


@_tool("describe")
def describe(table, args):
    """列の要約統計。column 省略時は ID 的な列を除いて先頭から最大20列。"""
    names = ([args["column"]] if args.get("column") else
             [n for n in table.names() if not is_id_like(table.column(n))] or table.names())
    return {"columns": [_describe_column(table.column(n)) for n in names[:DESCRIBE_LIMIT]],
            "truncated": len(names) > DESCRIBE_LIMIT}


def _centred(col, method):
    """Centred values of a column without missing cells (reused across matrix pairs)."""
    if len(col.values) < 3 or None in col.values:
        return None
    xs = rankdata(col.values) if method == "spearman" else col.values
    m = _fsum(xs) / len(xs)
    d = [v - m for v in xs]
    ss = _fsum(map(operator.mul, d, d))
    return d, ss, _flat(xs, m, ss)


def _pair(cx, cy, method, cache=None):
    cx_c, cy_c = (cache.get(cx.name), cache.get(cy.name)) if cache else (None, None)
    if cx_c and cy_c:
        n = len(cx_c[0])
        res = {"r": None, "p_value": None, "n": n}
        if cx_c[2] or cy_c[2]:
            res["reason"] = _FLAT
        else:
            r = _corr(cx_c[1], cy_c[1], _fsum(map(operator.mul, cx_c[0], cy_c[0])))
            r = None if r is None else max(-1.0, min(1.0, r))
            res.update(r=r, p_value=None if r is None else _r_pvalue(r, n))
    else:
        res = (spearman if method == "spearman" else pearson)(cx.values, cy.values)
    out = {"x": cx.name, "y": cy.name, "r": res["r"], "n": res["n"], "p_value": res["p_value"]}
    if res["r"] is None:
        out["reason"] = res.get("reason")
        return out
    strength, label = _strength(res["r"])
    out.update(strength=strength, strength_ja=label,
               direction="positive" if res["r"] > 0 else "negative" if res["r"] < 0 else "none",
               apqb=apqb(res["r"]))
    return out


@_tool("correlate")
def correlate(table, args):
    """2列の相関、または数値列どうしの相関上位10組。"""
    method = args.get("method") or "pearson"
    x, y = args.get("x"), args.get("y")
    if x and y:
        return {"mode": "pair", "method": method, **_pair(table.column(x), table.column(y), method)}
    limit = LIMITS["max_corr_columns"]
    numeric = [c for c in table.columns if c.kind == "numeric"]
    numeric = [c for c in numeric if not (is_id_like(c) or c.is_year)] or numeric   # unless named
    anchor = table.column(x or y) if (x or y) else None
    if anchor is not None:
        others = [c for c in numeric if c is not anchor]
        cols, pairs = [anchor, *others[:limit - 1]], [(anchor, c) for c in others[:limit - 1]]
        truncated = len(others) > limit - 1
    else:
        cols, truncated = numeric[:limit], len(numeric) > limit
        pairs = list(itertools.combinations(cols, 2))
    cache = {c.name: _centred(c, method) for c in cols}
    results = [_pair(a, b, method, cache) for a, b in pairs]
    valid = sorted((r for r in results if r["r"] is not None),
                   key=lambda r: (-abs(r["r"]), r["x"], r["y"]))
    return {"mode": "column" if anchor is not None else "matrix", "method": method,
            "columns": [c.name for c in cols], "pairs": valid[:10], "n_pairs": len(valid),
            "skipped": len(results) - len(valid), "truncated": truncated}


def _aggregate(values, agg):
    if agg == "count":
        return len(values)
    if not values:
        return None
    if agg == "sum":
        return _out(_fsum(values))
    if agg == "mean":
        return mean(values)
    if agg == "median":
        return median(values)
    return min(values) if agg == "min" else max(values)


@_tool("group_by")
def group_by(table, args):
    """グループ別集計（合計・平均・中央値・件数・最小・最大）と構成比。"""
    bcol = table.column(args["by"])
    vcol = table.column(args["value"]) if args.get("value") else None
    agg = args.get("agg") or ("sum" if vcol else "count")
    groups, missing_keys = {}, 0
    for i, key in enumerate(bcol.values):
        if key is None:
            missing_keys += 1
            continue
        bucket = groups.setdefault(key, [])
        value = 1.0 if vcol is None else vcol.values[i]
        if value is not None:
            bucket.append(value)
    rows = [(key, _aggregate(vals, agg), len(vals)) for key, vals in groups.items()]
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0), _label(r[0])))
    everything = [v for vals in groups.values() for v in vals]
    total = _aggregate(everything, agg)
    shares = (agg in ("sum", "count") and (vcol is None or vcol.unit != "%") and total and total > 0
              and all((r[1] or 0) >= 0 for r in rows))     # a share of summed rates is meaningless
    limit = LIMITS["max_groups"]
    out = {"by": bcol.name, "value": vcol.name if vcol else None, "agg": agg,
           "groups": [{"key": _key(k), "value": v, "count": c,
                       "share": v / total if shares and v is not None else None}
                      for k, v, c in rows[:limit]],
           "n_groups": len(rows), "total": total, "truncated": len(rows) > limit,
           "missing_keys": missing_keys}
    if len(rows) > limit:
        rest = rows[limit:]
        other = {"groups": len(rest), "count": sum(r[2] for r in rest)}
        if agg in ("sum", "count"):
            other["value"] = _out(_fsum(r[1] for r in rest if r[1] is not None))
        out["other"] = other
    if vcol is not None and vcol.unit and agg != "count":
        out["unit"] = vcol.unit
    if not rows:
        out["reason"] = "集計できるグループがありません"
    return out


def _bucket_end(key, period):
    """Last day of the month / (fiscal) year bucket that starts on key."""
    return _add_months(key, 1 if period == "month" else 12) - datetime.timedelta(days=1)


def _partial_edges(keys, days, period):
    """First/last month (year) bucket that covers only part of its period: observed on < 80% as many
    distinct dates as a full bucket (the median of the middle buckets; with 2 buckets, the bucket's
    length / the mean gap between dates) AND its data starts (ends) more than 2 mean gaps after (before)
    the bucket's start (end). The coverage test keeps full months of weekday-only or sparse dates."""
    distinct = sorted(set().union(*(days[k] for k in keys)))
    mean_gap = max(1.0, (distinct[-1] - distinct[0]).days / (len(distinct) - 1)) if len(distinct) > 1 else 1.0
    middle = median([float(len(days[j])) for j in keys[1:-1]]) if len(keys) >= 3 else None

    def expected(k):
        return middle if middle is not None else ((_bucket_end(k, period) - k).days + 1) / mean_gap

    def stretch(k):
        lead = (min(days[k]) - k).days if k == keys[0] else 0
        trail = (_bucket_end(k, period) - max(days[k])).days if k == keys[-1] else 0
        return max(lead, trail)
    return [k for k in dict.fromkeys((keys[0], keys[-1]))
            if len(days[k]) < 0.8 * expected(k) and stretch(k) > 2 * mean_gap]


def _series(table, value, time, agg="sum", period="raw", by=None, fiscal_start=1):
    """Aggregated series; month/year buckets that cover only part of the period are dropped. Years start
    in fiscal_start (4: April-March 年度, keyed by their first day).

    With by, "groups" maps each group key to its own {keys, x, y, partial} on the same time axis (one
    pass): the periods dropped as partial overall are dropped for every group, and each group also loses
    its own partial first/last bucket (a store opening on the 20th)."""
    vcol = table.column(value)
    gcol = table.column(by) if by else None
    if not time:
        pairs = _rows(vcol)
        keys = [p[0] for p in pairs]
        out = {"kind": "row", "keys": keys, "x": [float(k) for k in keys],
               "y": [p[1] for p in pairs], "rows": len(pairs), "partial": [], "uneven": False}
        if gcol is not None:
            groups = {}
            for row, v in pairs:
                key = gcol.values[row - 1]
                if key is not None:
                    group = groups.setdefault(key, {"keys": [], "x": [], "y": []})
                    group["keys"].append(row)
                    group["x"].append(float(row))
                    group["y"].append(v)
            out["groups"] = groups
        return out
    tcol = table.column(time)
    if tcol.kind not in ("datetime", "numeric"):
        raise DataError("time には日付列または数値列を指定してください")
    rollup = tcol.kind == "datetime" and period in ("month", "year")
    buckets, days, rows, grouped, gdays = {}, {}, 0, {}, {}
    gvalues = gcol.values if gcol is not None else itertools.repeat(None)
    for t, v, g in zip(tcol.values, vcol.values, gvalues):
        if t is None or v is None:
            continue
        if not rollup:
            key = t
        elif period == "month":
            key = datetime.date(t.year, t.month, 1)
        else:
            key = datetime.date(t.year - (t.month < fiscal_start), fiscal_start, 1)
        buckets.setdefault(key, []).append(v)
        if g is not None:
            grouped.setdefault(g, {}).setdefault(key, []).append(v)
            if rollup:
                gdays.setdefault(g, {}).setdefault(key, set()).add(t)
        if rollup:
            days.setdefault(key, set()).add(t)
        rows += 1
    keys = sorted(buckets)
    partial = _partial_edges(keys, days, period) if rollup and len(keys) >= 2 else []
    keys = [k for k in keys if k not in partial]
    rows -= sum(len(buckets[k]) for k in partial)
    counts = [len(buckets[k]) for k in keys]

    def reduce(bucket, ks):
        return [_out(_fsum(bucket[k]) / (len(bucket[k]) if agg == "mean" else 1)) for k in ks]

    def axis(ks):
        return [float((k - keys[0]).days) for k in ks] if tcol.kind == "datetime" else list(ks)

    out = {"kind": tcol.kind, "keys": keys, "x": axis(keys), "y": reduce(buckets, keys), "rows": rows,
           "partial": partial,
           "uneven": rollup and agg == "sum" and bool(counts) and max(counts) > 1.2 * min(counts)}
    if gcol is not None:
        dropped = set(partial)
        out["groups"] = {}
        for g, bucket in grouped.items():
            ks = sorted(k for k in bucket if k not in dropped)
            own = _partial_edges(ks, gdays[g], period) if rollup and len(ks) >= 2 else []
            ks = [k for k in ks if k not in own]
            out["groups"][g] = {"keys": ks, "x": axis(ks), "y": reduce(bucket, ks), "partial": own}
    return out


def _direction(slope, p):
    return ("unknown" if slope is None or p is None else "flat" if p >= 0.05
            else "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat")


def _time_name(table, args):
    return table.column(args["time"]).name if args.get("time") else None


def _period(key, kind, period="raw"):
    if kind == "datetime":
        if period == "month":
            return f"{key.year:04d}-{key.month:02d}"
        if period == "year":     # a fiscal-year bucket (April start) is named by the year it starts in
            return f"{key.year:04d}" if key.month == 1 else f"{key.year:04d}年度"
        return key.isoformat()
    return _key(key)


def _pct(new, old):
    """Relative change in %; None from a zero or negative base (a loss turning into a profit has no
    meaningful percentage change)."""
    return None if new is None or old is None or old <= 0 else (new - old) / old * 100.0


def _months_between(a, b):
    return (b.year - a.year) * 12 + b.month - a.month


@_tool("trend")
def trend(table, args):
    """時系列の推移：始点・終点・変化率・傾き・有意性・年平均成長率・直近の推移。"""
    agg, period = args.get("agg") or "sum", args.get("period") or "raw"
    s = _series(table, args["value"], args.get("time"), agg, period, args.get("by"), args.get("fiscal_start") or 1)
    keys, x, y, kind = s["keys"], s["x"], s["y"], s["kind"]
    n = len(y)
    out = {"value": table.column(args["value"]).name, "time": _time_name(table, args),
           "time_kind": kind, "agg": agg, "period": period, "n": n, "rows_used": s["rows"],
           "partial_periods": [_period(k, kind, period) for k in s["partial"]],
           "uneven_periods": s["uneven"]}
    if args.get("by"):
        out.update(by=table.column(args["by"]).name, groups=[], n_groups=len(s["groups"]), groups_trended=0,
                   groups_skipped=len(s["groups"]), rank_by=None, lowest=None, declining=[], truncated=False)
    if n < 2:
        reason = ("期間全体をカバーする" + ("月" if period == "month" else "年") + "が2つ未満のため推移を判定できません"
                  if s["partial"] else "推移を見るには2時点以上が必要です")
        return {**out, "direction": "unknown", "direction_ja": DIRECTION_LABELS["unknown"], "reason": reason}

    def label(i):
        return _period(keys[i], kind, period)

    reg = linregress(x, y)
    slope, p = reg["slope"], reg["p_value"]
    direction = _direction(slope, p)
    step = median([b - a for a, b in zip(x, x[1:])])
    span = x[-1] - x[0]
    cagr, years = None, None
    if kind == "datetime" and span >= 365:
        years = span / 365.25
    elif kind == "numeric" and span >= 1 and table.column(args["time"]).is_year:
        years = span
    # endpoints only mean something if every period is > 0; a growth rate of a rate (%) is not reported
    if years and table.column(args["value"]).unit != "%" and all(v is not None and v > 0 for v in y):
        try:
            cagr = _out((y[-1] / y[0]) ** (1.0 / years) - 1.0)
        except OverflowError:
            pass
    peak = max(range(n), key=lambda i: (y[i], -i))
    trough = min(range(n), key=lambda i: (y[i], i))
    adjacent = [True] * n      # month/year buckets: "前月比" only against the previous period, not across a gap
    if kind == "datetime" and period in ("month", "year"):
        gaps = [_months_between(a, b) for a, b in zip(keys, keys[1:])]
        usual = 12 if period == "year" else Counter(gaps).most_common(1)[0][0]
        adjacent = [True] + [g == usual for g in gaps]
    elif kind == "numeric" and table.column(args["time"]).is_year:      # 2015, 2017: no 前年比 across the gap
        adjacent = [True] + [b - a == 1 for a, b in zip(keys, keys[1:])]
    out.update(first=y[0], last=y[-1], first_period=label(0), last_period=label(n - 1),
               change=_out(y[-1] - y[0]), pct_change=_pct(y[-1], y[0]),
               slope=slope, slope_unit={"datetime": "day", "numeric": "time", "row": "row"}[kind],
               period_step=step, slope_per_period=_out(slope * step) if slope is not None else None,
               r2=reg["r2"], p_value=p, direction=direction, direction_ja=DIRECTION_LABELS[direction],
               cagr=cagr, cagr_pct=None if cagr is None else cagr * 100.0,
               peak={"period": label(peak), "value": y[peak]},
               trough={"period": label(trough), "value": y[trough]},
               last_periods=[{"period": label(i), "value": y[i],
                              "pct_change": _pct(y[i], y[i - 1]) if i and adjacent[i] else None}
                             for i in range(max(0, n - 6), n)])
    if direction == "unknown":
        out["reason"] = reg.get("reason") or _SMALL
    if args.get("by"):
        out.update(_group_trends(s["groups"], kind, period, step, table.column(args["value"]).unit == "%"))
    return out


MIN_TREND_POINTS = 3


def _group_trends(groups, kind, period, step, rate):
    """Per-group trend summaries ranked by growth. Growth is the fitted change per period as a share of
    the group's mean level (slope_pct), so groups of different size compare fairly; a % column, or any
    group with a zero or negative value, is ranked by the slope per period instead."""
    trended, skipped = [], 0
    for key, g in groups.items():
        y = g["y"]
        if len(y) < MIN_TREND_POINTS or any(v is None for v in y):
            skipped += 1
            continue
        reg = linregress(g["x"], y)
        slope, p = reg["slope"], reg["p_value"]
        direction = _direction(slope, p)
        per_period = _out(slope * step) if slope is not None and step is not None else None
        level = _fsum(y) / len(y)
        positive = min(y) > 0       # a share of the mean level means nothing for a series crossing zero
        trended.append({"key": _key(key), "n": len(y), "first": y[0], "last": y[-1],
                        "partial_periods": [_period(k, kind, period) for k in g.get("partial", [])],
                        "first_period": _period(g["keys"][0], kind, period),
                        "last_period": _period(g["keys"][-1], kind, period),
                        "change": _out(y[-1] - y[0]), "pct_change": _pct(y[-1], y[0]), "slope": slope,
                        "slope_per_period": per_period,
                        "slope_pct": _out(per_period / level * 100.0) if per_period is not None and positive
                        else None,
                        "p_value": p, "r2": reg["r2"], "direction": direction,
                        "direction_ja": DIRECTION_LABELS[direction]})
    metric = "slope_pct" if not rate and all(t["slope_pct"] is not None for t in trended) else "slope_per_period"
    ranked = [t for t in trended if t[metric] is not None]
    ranked.sort(key=lambda t: (-t[metric], _label(t["key"])))
    ranked += [t for t in trended if t[metric] is None]
    counts = Counter(t["direction"] for t in trended)
    valid = [t for t in ranked if t[metric] is not None]
    limit = LIMITS["max_groups"]
    return {"groups": ranked[:limit], "groups_trended": len(trended), "groups_skipped": skipped,
            "rank_by": metric if valid else None, "lowest": valid[-1] if valid else None,
            "declining": [t["key"] for t in reversed(valid) if t["direction"] == "decreasing"][:10],
            "increasing": counts["increasing"], "decreasing": counts["decreasing"], "flat": counts["flat"],
            "unknown": counts["unknown"], "truncated": len(ranked) > limit}


MIN_GROUP_IQR = 5      # quartiles of fewer values per group are not a usable "normal range"


def _bounds(values, method, k):
    """Normal range of one sample -> {lower, upper, scale, q1/q3/iqr or mean/std} or {reason, ...}."""
    if method == "iqr":
        xs = sorted(values)
        q1, q3 = _quantile_sorted(xs, 0.25), _quantile_sorted(xs, 0.75)
        if q3 == q1:       # 0/1 flags, zero-inflated amounts: "normal range 0 to 0" would flag every 1
            return {"q1": q1, "q3": q3, "iqr": 0.0, "reason": "四分位範囲が0のためIQR法では外れ値を判定できません"}
        return {"q1": q1, "q3": q3, "iqr": q3 - q1, "lower": q1 - k * (q3 - q1), "upper": q3 + k * (q3 - q1),
                "scale": q3 - q1}
    m, sd = mean(values), sample_std(values)
    if not sd:
        return {"mean": m, "std": sd, "flat": True, "reason": _FLAT}
    n = len(values)
    if (n - 1) / math.sqrt(n) <= k:     # with the sample SD, |z| can never exceed (n-1)/sqrt(n)
        return {"mean": m, "std": sd,
                "reason": f"データ数（n={n}）が少ないため、|z|>{k:g} の外れ値は原理的に検出できません（IQR法を使ってください）"}
    return {"mean": m, "std": sd, "lower": m - k * sd, "upper": m + k * sd, "scale": sd}


def min_group_size(method, k):
    """Smallest group that outliers(by=...) checks: 5 for IQR; for z-scores the n with (n-1)/sqrt(n) > k."""
    if method == "iqr":
        return MIN_GROUP_IQR
    n = 3
    while (n - 1) / math.sqrt(n) <= k:
        n += 1
    return n


@_tool("outliers")
def outliers(table, args):
    """IQR法またはzスコア法による外れ値（by 指定時はグループごとの基準で判定）。"""
    col = table.column(args["column"])
    method = args.get("method") or "iqr"
    k = args.get("threshold") or (1.5 if method == "iqr" else 3.0)
    if args.get("by"):
        return _outliers_by(col, table.column(args["by"]), method, k)
    rows = _rows(col)
    out = {"column": col.name, "method": method, "threshold": k, "n": len(rows),
           "bounds": None, "count": None, "share": None, "high": None, "low": None, "rows": [],
           "truncated": False}
    if len(rows) < 3:
        return {**out, "reason": _SMALL}
    b = _bounds([v for _, v in rows], method, k)
    out.update({key: b[key] for key in ("q1", "q3", "iqr", "mean", "std") if key in b})
    if "reason" in b:
        if b.get("flat"):
            out.update(count=0, share=0.0, high=0, low=0)
        return {**out, "reason": b["reason"]}
    lower, upper = b["lower"], b["upper"]
    flagged = [(row, v) for row, v in rows if v < lower or v > upper]
    flagged.sort(key=lambda rv: (-max(lower - rv[1], rv[1] - upper), rv[0]))
    high = sum(v > upper for _, v in flagged)
    limit = LIMITS["max_outliers"]
    out.update(bounds={"lower": lower, "upper": upper}, count=len(flagged),
               share=len(flagged) / len(rows), high=high, low=len(flagged) - high,
               rows=[{"row": r, "value": v, "side": "high" if v > upper else "low"}
                     for r, v in flagged[:limit]],
               truncated=len(flagged) > limit)
    if method == "zscore":
        for item in out["rows"]:
            item["z"] = (item["value"] - out["mean"]) / out["std"]
    return out


def _outliers_by(col, bcol, method, k):
    """Outliers against each group's own normal range: a value extreme for its region is found even
    when it is ordinary overall. Groups below min_group_size, or without spread, are skipped and counted."""
    groups, missing_keys = {}, 0
    for row, (key, value) in enumerate(zip(bcol.values, col.values), 1):
        if value is None:
            continue
        if key is None:
            missing_keys += 1
        else:
            groups.setdefault(key, []).append((row, value))
    need = min_group_size(method, k)
    checked, flagged, skipped, skipped_rows, flat = [], [], 0, 0, 0
    for key, rows in groups.items():
        b = _bounds([v for _, v in rows], method, k) if len(rows) >= need else None
        if b is None or "reason" in b:
            skipped, skipped_rows, flat = skipped + 1, skipped_rows + len(rows), flat + (b is not None)
            continue
        lower, upper = b["lower"], b["upper"]
        hits = [(row, v) for row, v in rows if v < lower or v > upper]
        checked.append({"key": _key(key), "n": len(rows), "count": len(hits), "lower": lower, "upper": upper})
        for row, v in hits:
            item = {"row": row, "value": v, "side": "high" if v > upper else "low", "group": _key(key),
                    "lower": lower, "upper": upper}
            if method == "zscore":
                item["z"] = (v - b["mean"]) / b["std"]
            flagged.append((max(lower - v, v - upper) / b["scale"], item))
    flagged.sort(key=lambda d: (-d[0], d[1]["row"]))     # distance in units of the group's IQR / SD
    checked.sort(key=lambda g: (-g["count"], -g["n"], _label(g["key"])))
    n = sum(g["n"] for g in checked)
    high = sum(item["side"] == "high" for _, item in flagged)
    limit, glimit = LIMITS["max_outliers"], LIMITS["max_groups"]
    out = {"column": col.name, "method": method, "threshold": k, "by": bcol.name, "n": n, "bounds": None,
           "count": len(flagged), "share": len(flagged) / n if n else None, "high": high,
           "low": len(flagged) - high, "rows": [item for _, item in flagged[:limit]],
           "truncated": len(flagged) > limit or len(checked) > glimit, "groups": checked[:glimit],
           "n_groups": len(groups), "groups_checked": len(checked), "groups_skipped": skipped,
           "groups_without_spread": flat, "skipped_rows": skipped_rows, "min_group_n": need,
           "missing_keys": missing_keys}
    if not checked:
        out.update(count=None, high=None, low=None,
                   reason=f"グループごとの値が少ない（{need}件未満）か、ばらつきがないため外れ値を判定できません")
    return out


def _find_group(groups, label):
    for key in groups:
        if _label(key) == label:
            return key
    parsed = {}            # parse the label once, not once per group key
    for key in groups:
        kind = bool if isinstance(key, bool) else float if isinstance(key, float) else \
            datetime.date if isinstance(key, datetime.date) else None
        if kind is not None and kind not in parsed:
            parsed[kind] = {bool: parse_bool, float: parse_number, datetime.date: parse_date}[kind](label)
        if kind is not None and parsed[kind] is not None and parsed[kind] == key:
            return key
    matches = [key for key in groups if name_key(_label(key)) == name_key(label)]
    if len(matches) == 1:
        return matches[0]
    raise DataError(f"グループ「{_clip(label, 40)}」が見つかりません")


def _group_summary(groups, key):
    values = groups[key]
    return {"label": _key(key), "n": len(values), "mean": mean(values), "std": sample_std(values),
            "median": median(values)}


def _pairwise(groups, a, b):
    """Welch t-test and Cohen's d of group a against group b."""
    sa, sb = _group_summary(groups, a), _group_summary(groups, b)
    test, d = welch_ttest(groups[a], groups[b]), cohen_d(groups[a], groups[b])
    effect = _effect(d) if d is not None else (None, None)
    p = test["p_value"]
    out = {"a": sa, "b": sb, "diff": _out(sa["mean"] - sb["mean"]), "pct_diff": _pct(sa["mean"], sb["mean"]),
           "t": test["t"], "df": test["df"], "p_value": p, "cohen_d": d, "effect": effect[0],
           "effect_ja": effect[1], "significant": None if p is None else p < 0.05}
    if test.get("reason"):
        out["reason"] = test["reason"]
    return out


@_tool("compare")
def compare(table, args):
    """グループの平均の比較。2グループは Welch の t検定と Cohen の d、a・b を省略して2件以上のグループが
    3つ以上あれば一元配置分散分析（Welch と通常の F、η²）と最大・最小グループの対比。"""
    vcol, bcol = table.column(args["value"]), table.column(args["by"])
    groups = {}
    for key, value in zip(bcol.values, vcol.values):
        if key is not None and value is not None:
            groups.setdefault(key, []).append(value)
    a = _find_group(groups, args["a"]) if args.get("a") is not None else None
    b = _find_group(groups, args["b"]) if args.get("b") is not None else None
    if a is not None and b is not None and a == b:
        raise DataError("a と b には別のグループを指定してください")
    defaulted = args.get("a") is None and args.get("b") is None
    out = {"value": vcol.name, "by": bcol.name, "n_groups": len(groups), "defaulted": defaulted,
           "b_defaulted": args.get("a") is not None and args.get("b") is None}
    if vcol.unit:
        out["unit"] = vcol.unit
    testable = [k for k in groups if len(groups[k]) >= 2]
    if defaulted and len(testable) >= 3:
        return {**out, **_anova(groups, testable)}
    ranked = sorted(groups, key=lambda k: (-len(groups[k]), _label(k)))
    rest = [k for k in ranked if not (a is not None and k == a) and not (b is not None and k == b)]
    if a is None and rest:
        a = rest.pop(0)
    if b is None and rest:
        b = rest.pop(0)
    out["mode"] = "pair"
    if a is None or b is None:
        return {**out, "a": None, "b": None, "diff": None, "pct_diff": None, "t": None, "df": None,
                "p_value": None, "cohen_d": None, "effect": None, "significant": None,
                "reason": "比較できるグループが2つ未満です"}
    return {**out, **_pairwise(groups, a, b)}


def welch_primary(sizes):
    """Welch's F is trusted only when every group has at least max(5, k) values: with many small groups
    its approximation rejects far too often (simulated type-I error at alpha 0.05: 0.18 for 20 groups of 4,
    0.24 for 50 groups of 5, 0.62 for 20 groups of 2), where the classic F stays near 0.05."""
    return bool(sizes) and min(sizes) >= max(WELCH_MIN_N, len(sizes))


WELCH_MIN_N = 5


def _anova(groups, testable):
    """One-way ANOVA over every group with 2+ values. Welch's F (no equal-variance assumption) gives
    the primary p-value when welch_primary allows it; otherwise (many small groups, or a constant group
    that leaves Welch undefined) the classic F does and Welch stays in the output as secondary.
    The highest- and lowest-mean groups are then compared with a Welch t-test whose p-value is also
    Bonferroni-adjusted for the k(k-1)/2 pairs they were picked from."""
    samples = [groups[k] for k in testable]
    classic, welch = anova_oneway(samples), welch_anova(samples)
    if welch["p_value"] is None:
        welch_note = None if classic["p_value"] is None else (
            "constant_group" if any(max(g) == min(g) for g in samples) else "undefined")
    elif not welch_primary([len(g) for g in samples]):
        welch_note = "small_groups"
    else:
        welch_note = None
    primary = welch if welch["p_value"] is not None and welch_note is None else classic
    ranked = sorted(((k, _group_summary(groups, k)) for k in testable),
                    key=lambda ks: (-ks[1]["mean"], _label(ks[0])))      # 2+ finite values: mean is set
    summaries = [summary for _, summary in ranked]
    pair = _pairwise(groups, ranked[0][0], ranked[-1][0])
    k = len(testable)
    pair["comparisons"] = k * (k - 1) // 2
    pair["p_adjusted"] = None if pair["p_value"] is None else min(1.0, pair["p_value"] * pair["comparisons"])
    pair["significant"] = None if pair["p_adjusted"] is None else pair["p_adjusted"] < 0.05
    limit = LIMITS["max_groups"]
    out = {"mode": "anova", "test": "welch_anova" if primary is welch else "anova", "welch_note": welch_note,
           "groups_tested": k, "skipped_groups": len(groups) - k, "n": classic["n"],
           "groups": summaries[:limit], "truncated": k > limit,
           "f": primary["f"], "df1": primary["df1"], "df2": primary["df2"], "p_value": primary["p_value"],
           "significant": None if primary["p_value"] is None else primary["p_value"] < 0.05,
           "eta_squared": classic["eta_squared"],
           "welch": {key: welch[key] for key in ("f", "df1", "df2", "p_value")},
           "classic": {key: classic[key] for key in ("f", "df1", "df2", "p_value")},
           "pairwise": pair}
    for name, result in (("welch", welch), ("classic", classic)):
        if result.get("reason"):
            out[name]["reason"] = result["reason"]
    if primary["p_value"] is None:
        out["reason"] = primary.get("reason") or _SMALL
    return out


def _levels(labels):
    counts = Counter(labels)
    ordered = sorted(counts, key=lambda k: (-counts[k], k))
    limit = LIMITS["max_crosstab_levels"]
    if len(ordered) <= limit:
        return ordered, {k: k for k in ordered}, 0
    kept = ordered[:limit - 1]
    other, i = OTHER, 1
    while other in kept:           # the fold label must never collide with a real level
        other = f"{OTHER}(集約)" if i == 1 else f"{OTHER}(集約{i})"
        i += 1
    mapping = {k: k for k in kept}
    mapping.update({k: other for k in ordered[limit - 1:]})
    return kept + [other], mapping, len(ordered) - len(kept)


@_tool("crosstab")
def crosstab(table, args):
    """2列のクロス集計とカイ二乗検定・CramérのV。"""
    rcol, ccol = table.column(args["row"]), table.column(args["col"])
    pairs = [(_label(r), _label(c)) for r, c in zip(rcol.values, ccol.values)
             if r is not None and c is not None]
    rows, rmap, rfold = _levels([p[0] for p in pairs])
    cols, cmap, cfold = _levels([p[1] for p in pairs])
    rindex, cindex = {k: i for i, k in enumerate(rows)}, {k: i for i, k in enumerate(cols)}
    counts = [[0] * len(cols) for _ in rows]
    for r, c in pairs:
        counts[rindex[rmap[r]]][cindex[cmap[c]]] += 1
    test = chi2_independence(counts)
    row_totals = [sum(r) for r in counts]
    out = {"row": rcol.name, "col": ccol.name, "table": {
               "rows": rows, "cols": cols, "counts": counts, "row_totals": row_totals,
               "col_totals": [sum(c) for c in zip(*counts)] if counts else [],
               "row_shares": [[v / t if t else None for v in r] for r, t in zip(counts, row_totals)]},
           "n": len(pairs), "missing_pairs": table.n_rows - len(pairs),
           "folded": {"rows": rfold, "cols": cfold}, "truncated": bool(rfold or cfold),
           "chi2": test["chi2"], "dof": test["dof"], "p_value": test["p_value"],
           "cramers_v": test["cramers_v"], "low_expected_share": test["low_expected_share"]}
    if test.get("reason"):
        out["reason"] = test["reason"]
    return out


@_tool("top_n")
def top_n(table, args):
    """値の大きい順（または小さい順）の上位n件。"""
    col = table.column(args["column"])
    n, order = args.get("n") or 5, args.get("order") or "desc"
    label = args.get("label")
    if not label:      # a name, else the date (or year) of the row: "which day / quarter" needs it
        others = [c for c in table.columns if c is not col]
        label = next((c.name for c in others if c.kind in ("categorical", "text")), None) or next(
            (c.name for c in others if c.kind == "datetime"), None) or next(
            (c.name for c in others if c.is_year), None)
    lcol = table.column(label) if label else None
    rows = _rows(col)
    ranked = sorted(rows, key=lambda rv: rv[1], reverse=order == "desc")[:n]
    out = {"column": col.name, "order": order, "n": n, "count": len(rows),
           "label_column": lcol.name if lcol else None,
           "rows": [{"row": r, "value": v, "label": None if lcol is None or lcol.values[r - 1] is None
                     else _key(lcol.values[r - 1])} for r, v in ranked]}
    if col.kind == "numeric":
        values = [v for _, v in rows]
        total = _fsum(values)
        if values and total > 0 and min(values) >= 0 and col.unit != "%":
            out["top_share"] = _fsum(v for _, v in ranked) / total
        if col.unit:
            out["unit"] = col.unit
    if not rows:
        out["reason"] = "値がありません"
    return out


def _add_months(day, months, dom=1):
    """Calendar-month step; dom is the day of month (clamped to the month length) or "end"."""
    y, m = divmod(day.year * 12 + day.month - 1 + months, 12)
    last = calendar.monthrange(y, m + 1)[1]
    return datetime.date(y, m + 1, last if dom == "end" else min(dom, last))


def _month_end(day):
    return day.day == calendar.monthrange(day.year, day.month)[1]


def _future(s, periods, period):
    keys, x, kind = s["keys"], s["x"], s["kind"]
    if kind == "datetime":
        first = all(k.day == 1 for k in keys)
        gaps = Counter(_months_between(a, b) for a, b in zip(keys, keys[1:]))
        # monthly or coarser: calendar steps, unless two dates share a month (4-weekly / irregular data)
        if first or (median([b - a for a, b in zip(x, x[1:])]) >= 28 and 0 not in gaps):
            top = max(gaps.values())
            step = max(1, min(g for g, c in gaps.items() if c == top))
            dom = 1 if first else "end" if all(_month_end(k) for k in keys) else \
                Counter(k.day for k in keys).most_common(1)[0][0]
            dates = [_add_months(keys[-1], step * i, dom) for i in range(1, periods + 1)]
        else:
            step = max(1, round(median([b - a for a, b in zip(x, x[1:])])))
            dates = [keys[-1] + datetime.timedelta(days=step * i) for i in range(1, periods + 1)]
        return [(_period(d, kind, period), float((d - keys[0]).days)) for d in dates]
    step = median([b - a for a, b in zip(x, x[1:])])
    return [(_key(float(keys[-1] + step * i)), x[-1] + step * i) for i in range(1, periods + 1)]


@_tool("forecast")
def forecast(table, args):
    """線形トレンドによる将来値と95%予測区間。"""
    periods, period = args.get("periods") or 3, args.get("period") or "raw"
    agg = args.get("agg") or ("mean" if table.column(args["value"]).unit == "%" else "sum")
    s = _series(table, args["value"], args.get("time"), agg, period, fiscal_start=args.get("fiscal_start") or 1)
    x, y = s["x"], s["y"]
    n = len(y)
    out = {"method": "linear_trend", "value": table.column(args["value"]).name,
           "time": _time_name(table, args), "time_kind": s["kind"], "agg": agg, "period": period, "n": n,
           "periods": periods, "level": 0.95, "points": [], "r2": None, "slope": None,
           "p_value": None, "caveat": FORECAST_CAVEAT,
           "partial_periods": [_period(k, s["kind"], period) for k in s["partial"]]}
    if n < 4:
        return {**out, "reason": "予測には4時点以上が必要です"}
    reg = linregress(x, y)
    if reg["residual_std"] is None:
        return {**out, "reason": reg.get("reason") or _SMALL}
    try:
        future = _future(s, periods, period)
    except (ValueError, OverflowError):
        return {**out, "reason": "予測期間が日付の範囲外です"}
    tcrit, sd = t_ppf(0.975, n - 2), reg["residual_std"]
    mx = _fsum(x) / n
    sxx = _sq(x, mx)
    for step, (label, x0) in enumerate(future, 1):
        estimate = reg["intercept"] + reg["slope"] * x0
        half = tcrit * sd * math.sqrt(1.0 + 1.0 / n + (x0 - mx) * (x0 - mx) / sxx)
        out["points"].append({"step": step, "period": label, "estimate": estimate,
                              "lower": estimate - half, "upper": estimate + half})
    out.update(r2=reg["r2"], slope=reg["slope"], p_value=reg["p_value"], residual_std=sd,
               last_period=_period(s["keys"][-1], s["kind"], period), last_value=y[-1])
    return out


@_tool("calculator")
def calculator(table, args):
    """四則演算（neuroquantum_agent.calculate）。"""
    from neuroquantum_agent import calculate
    try:
        return calculate(args["expression"])
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError, RecursionError,
            MemoryError):
        raise DataError("計算式を評価できません（数値と + - * / % のみ使用できます）") from None


_ALL = ["numeric", "datetime", "boolean", "categorical", "text"]
_GROUP_KINDS = ["categorical", "boolean"]
_PERIOD = {"type": "enum", "required": False, "enum": ["raw", "month", "year"], "default": "raw"}
# First month of a period=year bucket (4: 年度). Set by the rule planner, not listed in planner prompts.
_FISCAL = {"type": "int", "required": False, "min": 1, "max": 12, "internal": True}
TOOL_SPECS = {
    "profile": {"description": "行数・列の型・欠損・変換できなかったセルを確認", "label": "データ概要を作成中",
                "args": {}},
    "describe": {"description": "列の要約統計（件数・平均・標準偏差・四分位・最頻値など）",
                 "label": "要約統計を計算中", "args": {"column": {"type": "column", "required": False}}},
    "correlate": {"description": "数値列どうしの相関係数とp値（x,y省略時は相関の強い組を一覧）",
                  "label": "相関を計算中", "args": {
                      "x": {"type": "column", "required": False, "kinds": ["numeric"]},
                      "y": {"type": "column", "required": False, "kinds": ["numeric"]},
                      "method": {"type": "enum", "required": False, "enum": ["pearson", "spearman"],
                                 "default": "pearson"}}},
    "group_by": {"description": "グループ別の集計（合計・平均・中央値・件数・最小・最大）と構成比",
                 "label": "グループ別に集計中", "args": {
                     "by": {"type": "column", "required": True, "kinds": _ALL},
                     "value": {"type": "column", "required": False, "kinds": ["numeric"]},
                     "agg": {"type": "enum", "required": False,
                             "enum": ["sum", "mean", "median", "count", "min", "max"]}}},
    "trend": {"description": "時系列の推移・傾き・変化率・年平均成長率（by でグループ別の推移も）",
              "label": "推移を分析中", "args": {
                  "value": {"type": "column", "required": True, "kinds": ["numeric"]},
                  "time": {"type": "column", "required": False, "kinds": ["datetime", "numeric"]},
                  "agg": {"type": "enum", "required": False, "enum": ["sum", "mean"]},
                  "period": _PERIOD, "fiscal_start": _FISCAL,
                  "by": {"type": "column", "required": False, "kinds": _GROUP_KINDS}}},
    "outliers": {"description": "IQR法またはzスコア法で外れ値を検出（by でグループごとの基準で判定）",
                 "label": "外れ値を検出中", "args": {
                     "column": {"type": "column", "required": True, "kinds": ["numeric"]},
                     "method": {"type": "enum", "required": False, "enum": ["iqr", "zscore"], "default": "iqr"},
                     "threshold": {"type": "number", "required": False, "min": 0.5, "max": 6.0},
                     "by": {"type": "column", "required": False, "kinds": _GROUP_KINDS}}},
    "compare": {"description": "グループの平均を比較（2グループはWelchのt検定・効果量、3グループ以上は分散分析）",
                "label": "グループを比較中",
                "args": {"value": {"type": "column", "required": True, "kinds": ["numeric"]},
                         "by": {"type": "column", "required": True, "kinds": _ALL},
                         "a": {"type": "string", "required": False, "max_length": 500},
                         "b": {"type": "string", "required": False, "max_length": 500}}},
    "crosstab": {"description": "2列のクロス集計とカイ二乗検定", "label": "クロス集計中", "args": {
        "row": {"type": "column", "required": True, "kinds": _ALL},
        "col": {"type": "column", "required": True, "kinds": _ALL}}},
    "top_n": {"description": "値の大きい順（小さい順）に上位n件を表示", "label": "ランキングを作成中", "args": {
        "column": {"type": "column", "required": True, "kinds": ["numeric", "datetime"]},
        "n": {"type": "int", "required": False, "min": 1, "max": LIMITS["max_top_n"], "default": 5},
        "order": {"type": "enum", "required": False, "enum": ["desc", "asc"], "default": "desc"},
        "label": {"type": "column", "required": False, "kinds": _ALL}}},
    "forecast": {"description": "線形トレンドで将来値と95%予測区間を推定", "label": "将来値を予測中", "args": {
        "value": {"type": "column", "required": True, "kinds": ["numeric"]},
        "time": {"type": "column", "required": False, "kinds": ["datetime", "numeric"]},
        "periods": {"type": "int", "required": False, "min": 1,
                    "max": LIMITS["max_forecast_periods"], "default": 3},
        "agg": {"type": "enum", "required": False, "enum": ["sum", "mean"]},
        "period": _PERIOD, "fiscal_start": _FISCAL}},
    "calculator": {"description": "数値と + - * / % による計算", "label": "計算中", "args": {
        "expression": {"type": "string", "required": True, "max_length": 200}}},
}


def _check_arg(table, key, rule, value):
    kind = rule["type"]
    if kind == "column":
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise DataError(f"引数 {key} には列名を文字列で指定してください")
        col = table.column(value)
        if rule.get("kinds") and col.kind not in rule["kinds"]:
            need = "・".join(KIND_LABELS[k] for k in rule["kinds"])
            raise DataError(f"列「{_clip(col.name, 40)}」は{KIND_LABELS[col.kind]}列のため {key} に使えません"
                            f"（{need}列が必要です）")
        return col.name
    if kind == "string":
        if not isinstance(value, str) or not 0 < len(value.strip()) <= rule.get("max_length", 500):
            raise DataError(f"引数 {key} は1〜{rule.get('max_length', 500)}文字の文字列で指定してください")
        return scrub(value.strip())
    if kind == "enum":
        normal = value.strip().casefold() if isinstance(value, str) else None
        if normal not in rule["enum"]:
            raise DataError(f"引数 {key} は {' / '.join(rule['enum'])} のいずれかで指定してください")
        return normal
    if kind == "int" and type(value) is not int:
        raise DataError(f"引数 {key} は整数で指定してください")
    if kind == "number" and (type(value) not in (int, float) or value != value):
        raise DataError(f"引数 {key} は数値で指定してください")
    if not rule["min"] <= value <= rule["max"]:      # exact int/float comparison: a 400-digit int never overflows
        raise DataError(f"引数 {key} は {rule['min']}〜{rule['max']} の範囲で指定してください")
    return float(value) if kind == "number" else value


def _distinct(args, a, b):
    if args.get(a) is not None and args.get(a) == args.get(b):
        raise DataError(f"{a} と {b} には別の列を指定してください")


def validate_args(table, tool, args):
    """Strict schema check; returns normalised args (resolved column names, defaults filled)."""
    if not isinstance(tool, str) or tool not in TOOL_SPECS:
        raise DataError(f"未対応のツールです: {_clip(tool, 40)}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise DataError("引数はオブジェクトで指定してください")
    spec = TOOL_SPECS[tool]["args"]
    unknown = [k for k in args if k not in spec]
    if unknown:
        raise DataError(f"{tool} に未知の引数があります: {_clip(unknown[0], 40)}")
    result = {}
    for key, rule in spec.items():
        value = args.get(key)
        if value is None:
            if rule.get("required"):
                raise DataError(f"{tool} の必須引数 {key} がありません")
            if "default" in rule:
                result[key] = rule["default"]
            continue
        result[key] = _check_arg(table, key, rule, value)
    if tool == "correlate":
        _distinct(result, "x", "y")
    elif tool == "group_by":
        _distinct(result, "by", "value")
        if "agg" not in result:      # rates are averaged, never summed, unless asked
            value = result.get("value")
            result["agg"] = "count" if not value else "mean" if table.column(value).unit == "%" else "sum"
        elif result["agg"] != "count" and not result.get("value"):
            raise DataError("count 以外の集計には value（数値列）が必要です")
    elif tool in ("trend", "forecast"):
        _distinct(result, "value", "time")
        _distinct(result, "value", "by")
        result.setdefault("agg", "mean" if table.column(result["value"]).unit == "%" else "sum")
        if result["period"] != "raw" and (not result.get("time")
                                          or table.column(result["time"]).kind != "datetime"):
            raise DataError("period（month / year）は日付列の time と組み合わせて指定してください")
        if result.get("fiscal_start") == 1:
            del result["fiscal_start"]           # calendar years are the default
        elif "fiscal_start" in result and result["period"] != "year":
            raise DataError("fiscal_start は period=year と組み合わせて指定してください")
    elif tool == "outliers":
        _distinct(result, "column", "by")
        lo, hi, default = (0.5, 5.0, 1.5) if result["method"] == "iqr" else (1.5, 6.0, 3.0)
        result.setdefault("threshold", default)
        if not lo <= result["threshold"] <= hi:
            raise DataError(f"threshold は {result['method']} では {lo}〜{hi} の範囲で指定してください")
    elif tool == "compare":
        _distinct(result, "value", "by")
        if result.get("a") is not None and result.get("a") == result.get("b"):
            raise DataError("a と b には別のグループを指定してください")
    elif tool == "crosstab":
        _distinct(result, "row", "col")
    elif tool == "top_n":
        _distinct(result, "column", "label")
    return {k: result[k] for k in spec if k in result}


def run_tool(table, tool, args):
    normalized = validate_args(table, tool, args)
    return TOOLS[tool](table, normalized)
