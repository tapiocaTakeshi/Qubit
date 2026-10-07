"""Qubit Analyst data tools: loading, statistics (scipy reference values), tools, validation, limits."""
import calendar
import datetime
import json
import math
import random
import time

import pytest

import qubit_analyst_tools as T
from qubit_analyst_tools import DataError, load_table, run_tool, validate_args

X = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 6.1, 2.8, 4.9, 3.7]
Y = [1.0, 2.2, 1.4, 3.9, 3.1, 2.0, 4.4, 2.5, 2.9, 2.6]
Z = [3.9, 4.1, 5.5, 6.0, 4.8, 5.2, 4.4]
TIES = [1, 2, 2, 3, 3, 3, 4, 5, 5, 6]


def close(value, expected, rel=1e-9, abs_=1e-12):
    return value is not None and math.isclose(value, expected, rel_tol=rel, abs_tol=abs_)


def strict_json(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False))


def sales_csv():
    lines = ["日付,店舗,売上,客数,会員,メモ"]
    for month in range(1, 25):
        for i, store in enumerate(["東京", "大阪", "名古屋"]):
            sales = 1000 + 120 * month + 300 * i + (37 * month * (i + 1)) % 90
            member = "はい" if (month + i) % 3 else "いいえ"
            lines.append(f"{2023 + (month - 1) // 12}-{(month - 1) % 12 + 1:02d}-01,{store},"
                         f"\"¥{sales:,}\",{20 + month + 5 * i},{member},note{month * 3 + i}")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def sales():
    return load_table(sales_csv(), name="売上データ")


# ---------------------------------------------------------------- statistics: scipy literals

def test_descriptive_statistics_match_numpy_scipy_reference():
    assert close(T.mean(X), 3.8200000000000003)
    assert close(T.sample_std(X), 1.4148419621207795)
    for q, expected in {0.0: 1.9, 0.1: 2.08, 0.25: 2.925, 0.5: 3.55, 0.9: 5.6499999999999995,
                        1.0: 6.1}.items():
        assert close(T.quantile(X, q), expected)
    assert close(T.median(X), 3.55)
    assert close(T.skewness(X), 0.2698609618676528)


def test_correlation_and_regression_match_scipy_reference():
    p = T.pearson(X, Y)
    assert close(p["r"], 0.9535300638119654) and close(p["p_value"], 1.928590341134559e-05, rel=1e-7)
    assert p["n"] == 10
    s = T.spearman(X, TIES)
    assert close(s["r"], 0.46921626483186957) and close(s["p_value"], 0.17128139641927437, rel=1e-7)
    lr = T.linregress(X, Y)
    expected = {"slope": 0.703263765541741, "intercept": -0.08646758436945046,
                "r": 0.9535300638119654, "stderr": 0.07856605477200163,
                "intercept_stderr": 0.31811006528274893, "p_value": 1.928590341134541e-05}
    for key, value in expected.items():
        assert close(lr[key], value, rel=1e-8), key
    assert close(lr["r2"], 0.9535300638119654 ** 2)


@pytest.mark.parametrize("t,df,sf,cdf", [
    (2.0, 5, 0.050969739414929174, 0.9490302605850708),
    (-1.3, 12, 0.8909914144582429, 0.10900858554175706),
    (0.4, 1, 0.3788810584091566, 0.6211189415908434),
    (4.5, 30, 4.759679696056222e-05, 0.9999524032030395),
    (2.2, 7.5, 0.03059973295305805, 0.9694002670469419),
    (10.0, 3, 0.0010641995292070751, 0.9989358004707929)])
def test_student_t_distribution_matches_scipy(t, df, sf, cdf):
    assert close(T.student_t_sf(t, df), sf)
    assert close(T.student_t_cdf(t, df), cdf)


@pytest.mark.parametrize("p,df,expected", [
    (0.975, 1, 12.706204736174694), (0.975, 2, 4.302652729749462), (0.975, 10, 2.228138851986274),
    (0.975, 30, 2.0422724563012378), (0.025, 8, -2.3060041352041662), (0.995, 4.5, 4.272823993011293),
    (0.6, 20, 0.2567427538545021)])
def test_t_ppf_matches_scipy(p, df, expected):
    assert close(T.t_ppf(p, df), expected)


@pytest.mark.parametrize("x,k,expected", [
    (3.84, 1, 0.05004352124870519), (5.99, 2, 0.05003662708658629), (0.5, 3, 0.9188914116546758),
    (20.0, 10, 0.029252688076961124), (1.0, 0.5, 0.15351359580832247),
    (150.0, 121, 0.03793093530567168)])
def test_chi2_sf_matches_scipy(x, k, expected):
    assert close(T.chi2_sf(x, k), expected)


def test_group_tests_match_scipy_reference():
    w = T.welch_ttest(X, Z)
    assert close(w["t"], -1.918022285813911) and close(w["df"], 14.354683603462354)
    assert close(w["p_value"], 0.0752222196982633)
    assert close(T.cohen_d(X, Z), -0.8533069995040317)
    c = T.chi2_independence([[12, 5, 7], [3, 9, 10]])
    assert close(c["chi2"], 6.998542144130378) and close(c["p_value"], 0.030219403163080322)
    assert c["dof"] == 2 and c["n"] == 46
    assert close(T.cramers_v([[12, 5, 7], [3, 9, 10]]), 0.39005412512185206)


def test_tiny_t_and_centre_quantiles_are_accurate():
    # Cauchy (df=1) has closed forms; these regimes need the complement-aware beta function.
    t = 7.247098285926117e-09
    assert close(T.student_t_sf(t, 1), 0.5 - math.atan(t) / math.pi, rel=1e-14)
    for p in (0.5 + 1e-10, 0.4999999994, 0.3, 0.9, 0.999999):
        assert close(T.t_ppf(p, 1), math.tan(math.pi * (p - 0.5)), rel=1e-9)
    assert T.t_ppf(0.5, 7) == 0.0
    assert close(T.student_t_sf(2.0, 5e7), 0.5 * math.erfc(2.0 / math.sqrt(2)), rel=1e-6)


# ---------------------------------------------------------------- statistics: scipy randomized

def _random_sample(rng, n):
    kind = rng.random()
    if kind < 0.3:
        return [rng.gauss(rng.uniform(-100, 100), rng.uniform(0.1, 50)) for _ in range(n)]
    if kind < 0.5:
        return [rng.expovariate(rng.uniform(0.01, 2)) for _ in range(n)]
    if kind < 0.7:
        return [float(rng.randint(0, 5)) for _ in range(n)]
    return [rng.lognormvariate(3, 1.2) * 1000 for _ in range(n)]


def test_statistics_match_scipy_on_random_samples():
    pytest.importorskip("scipy")
    import warnings
    import numpy as np
    from scipy import stats
    warnings.simplefilter("ignore")

    def same(ours, ref, rel=1e-9):
        assert ours is not None and (abs(ours - ref) <= 1e-12 or abs(ours - ref) <= rel * abs(ref)), (ours, ref)

    for seed in range(250):
        rng = random.Random(seed)
        n = rng.choice([3, 4, 5, 8, 13, 40, 300])
        xs = _random_sample(rng, n)
        a = np.array(xs)
        same(T.mean(xs), float(np.mean(a)))
        same(T.sample_std(xs), float(np.std(a, ddof=1)))
        q = rng.random()
        same(T.quantile(xs, q), float(np.quantile(a, q)))
        if max(xs) != min(xs):
            same(T.skewness(xs), float(stats.skew(a, bias=False)), rel=1e-8)
        ys = [x * rng.uniform(-2, 2) + rng.gauss(0, 30) for x in xs]
        if max(xs) != min(xs):
            p, ref = T.pearson(xs, ys), stats.pearsonr(xs, ys)
            same(p["r"], float(ref[0]))
            assert abs(p["p_value"] - float(ref[1])) < 1e-7
            s, ref = T.spearman(xs, ys), stats.spearmanr(xs, ys)
            same(s["r"], float(ref[0]))
            assert abs(s["p_value"] - float(ref[1])) < 1e-7
            lr, ref = T.linregress(xs, ys), stats.linregress(xs, ys)
            same(lr["slope"], ref.slope)
            same(lr["intercept"], ref.intercept, rel=1e-8)
            same(lr["stderr"], ref.stderr, rel=1e-8)
            assert abs(lr["p_value"] - ref.pvalue) < 1e-7
        zs = _random_sample(rng, rng.choice([2, 3, 6, 30]))
        if max(xs) != min(xs) or max(zs) != min(zs):
            w, ref = T.welch_ttest(xs, zs), stats.ttest_ind(xs, zs, equal_var=False)
            same(w["t"], float(ref.statistic))
            same(w["df"], float(ref.df))
            assert abs(w["p_value"] - float(ref.pvalue)) < 1e-7
        df = rng.choice([0.5, 1, 2, 3.5, 9, 30, 200, 5000, rng.uniform(0.3, 100)])
        t = rng.choice([rng.gauss(0, 2), rng.uniform(-30, 30)])
        same(T.student_t_sf(t, df), float(stats.t.sf(t, df)), rel=1e-9)
        prob = rng.choice([rng.uniform(0.01, 0.99), 0.975, 0.995])
        same(T.t_ppf(prob, df), float(stats.t.ppf(prob, df)), rel=1e-9)
        k = rng.choice([1, 2, 5, 12, 121, rng.uniform(0.5, 40)])
        x = rng.uniform(0, 4 * k + 10)
        same(T.chi2_sf(x, k), float(stats.chi2.sf(x, k)), rel=1e-9)
        width = rng.randint(2, 5)
        obs = [[rng.randint(1, 40) for _ in range(width)] for _ in range(rng.randint(2, 5))]
        c, ref = T.chi2_independence(obs), stats.chi2_contingency(np.array(obs), correction=False)
        same(c["chi2"], float(ref[0]))
        assert c["dof"] == ref[2] and abs(c["p_value"] - float(ref[1])) < 1e-7


# ---------------------------------------------------------------- statistics: edge cases

def test_statistics_edge_cases_return_none_with_reason():
    assert T.mean([]) is None and T.sample_std([1.0]) is None and T.quantile([], 0.5) is None
    assert T.skewness([1, 2]) is None and T.skewness([3, 3, 3, 3]) is None
    assert T.sample_std([5, 5, 5]) == 0.0
    assert T.skewness([0.1] * 7) is None
    for fn in (T.pearson, T.spearman):
        assert fn([1, 2], [3, 4])["r"] is None and fn([1, 2], [3, 4])["reason"]
        flat = fn([1, 2, 3, 4], [7, 7, 7, 7])
        assert flat["r"] is None and flat["p_value"] is None and flat["reason"]
        assert fn([1, 2], [1])["reason"]
    missing = T.pearson([1, None, 3, 4, float("nan"), 6], [2, 5, None, 8, 9, 13])
    assert missing["n"] == 3 and missing["r"] is not None
    assert T.linregress([1, 1, 1], [1, 2, 3])["slope"] is None
    two = T.linregress([1, 2], [3, 5])
    assert two["slope"] == 2.0 and two["p_value"] is None and two["reason"]
    flat_y = T.linregress([1, 2, 3, 4], [5, 5, 5, 5])
    assert flat_y["slope"] == 0.0 and flat_y["r"] == 0.0 and flat_y["p_value"] == 1.0
    assert T.welch_ttest([1], [2, 3])["t"] is None
    assert T.welch_ttest([2, 2, 2], [5, 5])["reason"]
    assert T.welch_ttest([2, 2, 2], [5, 6, 7])["p_value"] is not None
    assert T.cohen_d([1, 1], [1, 1]) is None and T.cohen_d([1], [2, 3]) is None
    assert T.chi2_independence([[5, 0], [0, 0]])["chi2"] is None
    assert T.chi2_independence([[3, 0, 2], [1, 0, 4]])["dof"] == 1   # empty column dropped
    assert T.cramers_v([[4]]) is None
    for bad in (None, float("nan"), 0, -1, True):
        assert T.student_t_sf(1.0, bad) is None and T.t_ppf(0.9, bad) is None
        assert T.chi2_sf(1.0, bad) is None
    assert T.student_t_sf(float("inf"), 3) == 0.0 and T.student_t_sf(float("-inf"), 3) == 1.0
    assert T.student_t_sf(True, 3) is None and T.student_t_cdf("1", 3) is None
    assert T.t_ppf(0, 5) is None and T.t_ppf(1, 5) is None
    assert T.chi2_sf(0, 3) == 1.0 and T.chi2_sf(-2, 3) == 1.0
    assert T.betainc(2, 3, 0) == 0.0 and T.betainc(2, 3, 1) == 1.0 and T.betainc(0, 1, 0.5) is None
    assert T.gammaincc(2, 0) == 1.0 and T.gammaincc(-1, 2) is None
    assert T.rankdata([3, 1, 3, 2]) == [3.5, 1.0, 3.5, 2.0]
    assert T.mean([1e308, 1e308]) is None or math.isfinite(T.mean([1e308, 1e308]))
    assert T.sample_std([1e200, -1e200, 3]) is None
    assert T.skewness([True, False, True]) is None   # bools are not numbers


def test_apqb_mapping_is_consistent():
    for r in (-1.0, -0.3, 0.0, 0.5, 0.97, 1.0, 1.5):
        state = T.apqb(r)
        clipped = max(-1.0, min(1.0, r))
        assert math.isclose(math.cos(2 * state["theta"]), clipped, abs_tol=1e-12)
        assert math.isclose(state["r"] ** 2 + state["T"] ** 2, 1.0, abs_tol=1e-12)
        assert math.isclose(state["T"], abs(math.sin(2 * state["theta"])), abs_tol=1e-12)


# ---------------------------------------------------------------- loading

def test_load_csv_with_bom_currency_percent_and_kinds(sales):
    assert sales.name == "売上データ" and sales.n_rows == 72
    assert sales.kinds() == {"日付": "datetime", "店舗": "categorical", "売上": "numeric",
                             "客数": "numeric", "会員": "boolean", "メモ": "text"}
    assert sales.column("売上").values[0] == 1000 + 120 + 37 % 90 and sales.column("売上").unit == "¥"
    bom = load_table("﻿a,b\n1,2\n")
    assert [c.name for c in bom.columns] == ["a", "b"]


@pytest.mark.parametrize("text", ["a;b;c\n1;x;2024-01-05\n2;y;2024-02-05\n",
                                  "a\tb\tc\n1\tx\t2024-01-05\n2\ty\t2024-02-05\n",
                                  "a|b|c\n1|x|2024-01-05\n2|y|2024-02-05\n",
                                  "a,b,c\r\n1,x,2024-01-05\r\n2,y,2024-02-05\r\n"])
def test_delimiter_is_sniffed(text):
    table = load_table(text)
    assert table.kinds() == {"a": "numeric", "b": "categorical", "c": "datetime"}
    assert table.column("a").values == [1.0, 2.0]


def test_semicolon_csv_keeps_comma_inside_values():
    table = load_table("店舗;売上\n東京;1,234\n大阪;12,345,678.5\n")
    assert table.column("売上").values == [1234.0, 12345678.5]


def test_records_and_column_dict_inputs():
    records = [{"a": 1, "b": True, "d": "2024-01-02"}, {"a": 2.5, "c": "x"},
               {"b": None, "a": float("nan"), "d": datetime.date(2024, 3, 1)}]
    table = load_table(records)
    assert [c.name for c in table.columns] == ["a", "b", "d", "c"]
    assert table.column("a").values == [1.0, 2.5, None] and table.column("a").raw_missing == 1
    assert table.column("b").kind == "boolean" and table.column("b").values == [True, None, None]
    assert table.column("d").values == [datetime.date(2024, 1, 2), None, datetime.date(2024, 3, 1)]
    columns = load_table({"x": [1, 2, 3], "when": [datetime.datetime(2024, 1, 1, 9), "2024/02/01", None]})
    assert columns.column("when").kind == "datetime"
    assert columns.column("when").values[:2] == [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]
    assert load_table(b"a,b\n1,2\n").column("b").values == [2.0]
    assert load_table("売上,地域\n1,東\n".encode("cp932")).column("地域").values == ["東"]


@pytest.mark.parametrize("raw,expected", [
    ("１２３", 123.0), ("1,234", 1234.0), ("¥1,200", 1200.0), ("￥1,200", 1200.0), ("1,200円", 1200.0),
    ("$3.5", 3.5), ("5ドル", 5.0), ("€7", 7.0), ("£2", 2.0), ("12%", 12.0), ("１２％", 12.0),
    ("(123)", -123.0), ("(¥100)", -100.0), ("+5", 5.0), ("-$5", -5.0), ("$-5", -5.0), ("−4", -4.0),
    ("▲300", -300.0), ("3万", 30000.0), ("1.5億", 150000000.0), ("1e3", 1000.0), (".5", 0.5),
    (" 42 ", 42.0), ("-0", 0.0), (7, 7.0), (2.5, 2.5)])
def test_number_parsing(raw, expected):
    assert T.parse_number(raw) == expected


@pytest.mark.parametrize("raw", ["inf", "-Infinity", "nan", "1e999", "1,23", "12 34", "--5", "(5",
                                 "((5))", "(-5)", "¥5円", "1_000", "0x10", "abc", "", True, False,
                                 None, float("inf"), float("nan"), 10 ** 400, [1]])
def test_number_parsing_rejects(raw):
    assert T.parse_number(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("2024-01-05", (2024, 1, 5)), ("2024/1/5", (2024, 1, 5)), ("2024.01.05", (2024, 1, 5)),
    ("2024-03", (2024, 3, 1)), ("2024/3", (2024, 3, 1)), ("2024年1月", (2024, 1, 1)),
    ("２０２４年１月", (2024, 1, 1)), ("2024年1月5日", (2024, 1, 5)), ("2024年", (2024, 1, 1)),
    ("2024-01-05T10:20:30Z", (2024, 1, 5)), ("2024-01-05 10:20:30+09:00", (2024, 1, 5)),
    ("2024/01/05 9:00", (2024, 1, 5))])
def test_date_parsing(raw, expected):
    assert T.parse_date(raw) == datetime.date(*expected)


@pytest.mark.parametrize("raw", ["2024-02-30", "2024-13", "2024-01/05", "2024.01", "20240105", "昨日", 2024])
def test_date_parsing_rejects(raw):
    assert T.parse_date(raw) is None


def test_boolean_parsing():
    table = load_table("b\nTrue\nfalse\nYES\nno\nはい\nいいえ\n○\n×\n")
    assert table.column("b").kind == "boolean"
    assert table.column("b").values == [True, False, True, False, True, False, True, False]


def test_missing_tokens_and_kind_threshold():
    tokens = ["NA", "n/a", "nan", "NULL", "None", "-", "—", "欠損", "不明", "#N/A", "  ", "", "Ｎ／Ａ"]
    table = load_table({"v": tokens + ["5"]})
    assert table.column("v").kind == "numeric" and table.column("v").raw_missing == len(tokens)
    coerced = load_table("x\n" + "\n".join(str(i) for i in range(1, 11)) + "\nfoo\n").column("x")
    assert coerced.kind == "numeric" and coerced.coerced == 1 and coerced.values[-1] is None
    assert coerced.missing == 1 and coerced.raw_missing == 0
    mixed = load_table("x\n1\n2\n3\n4\n5\n6\n7\n8\nfoo\nbar\n").column("x")
    assert mixed.kind == "categorical" and mixed.values[0] == "1"
    dates = load_table("d\n" + "\n".join(f"2024-01-{i:02d}" for i in range(1, 21)) + "\n不正\n2024-02-30\n")
    assert dates.column("d").kind == "datetime" and dates.column("d").coerced == 2
    empty = load_table("a,b\n1,\n2,NA\n").column("b")
    assert empty.kind == "categorical" and empty.values == [None, None] and empty.raw_missing == 2
    assert load_table("p\n12%\n15%\n").column("p").unit == "%"


def test_categorical_versus_text_threshold():
    rows = "\n".join(f"id{i},{'AB'[i % 2]}" for i in range(60))
    table = load_table("name,flag\n" + rows)
    assert table.column("name").kind == "text" and table.column("flag").kind == "categorical"
    assert load_table({"c": [f"v{i}" for i in range(20)]}).column("c").kind == "categorical"
    assert load_table({"c": [f"v{i}" for i in range(21)]}).column("c").kind == "text"


def test_duplicate_blank_and_long_headers():
    table = load_table("name,name,,Name,列3,  \n1,2,3,4,5,6\n")
    assert [c.name for c in table.columns] == ["name", "name_2", "列3", "Name_3", "列3_2", "列6"]
    long = load_table("x" * 200 + ",b\n1,2\n")
    assert len(long.columns[0].name) == 80
    odd = load_table('"a\nb",c\n1,2\n')
    assert odd.columns[0].name == "a b"


def test_rows_blank_lines_and_trailing_delimiters():
    table = load_table("a,b\n1,2\n,\n\n3,4,\n5\n")
    assert table.n_rows == 3 and table.column("b").values == [2.0, 4.0, None]


def test_year_column_stays_numeric():
    table = load_table("年度,year,売上\n2020,2020,1\n2021,2021,2\n2022,2022,3\n")
    assert table.column("年度").kind == "numeric" and table.column("年度").is_year
    assert table.column("year").is_year and not table.column("売上").is_year


def test_column_lookup_exact_then_normalized():
    table = load_table("Sales Amount,地域\n1,東\n2,西\n")
    for name in ("Sales Amount", "sales amount", "ＳＡＬＥＳ　ＡＭＯＵＮＴ", "salesamount", " Sales  Amount "):
        assert table.column(name).name == "Sales Amount"
    manual = T.Table([T.Column("Sales", "numeric", [1.0]), T.Column("sales", "numeric", [2.0])])
    assert manual.column("sales").values == [2.0] and manual.column("Sales").values == [1.0]
    with pytest.raises(DataError):
        manual.column("SALES")          # ambiguous loose match
    for bad in ("売上", None, 3):
        with pytest.raises(DataError):
            table.column(bad)


@pytest.mark.parametrize("data", ["", "   ", "a,b\n", "a,b\n1,2,3\n", [], [1], [{"a": [1]}], {},
                                  {"a": []}, {"a": 1}, {"a": [1], "b": [1, 2]}, 5, None, b"\xff\xfe\xfa\xfb"])
def test_invalid_inputs_raise_data_error(data):
    with pytest.raises(DataError):
        load_table(data)


def test_size_limits_fail_fast():
    started = time.monotonic()
    with pytest.raises(DataError, match="大きすぎ"):
        load_table("a\n" + "1" * T.LIMITS["max_chars"])
    with pytest.raises(DataError, match="行数"):
        load_table("a\n" + "1\n" * (T.LIMITS["max_rows"] + 1))
    with pytest.raises(DataError, match="行数"):
        load_table([{"a": 1}] * (T.LIMITS["max_rows"] + 1))
    with pytest.raises(DataError, match="行数"):
        load_table({"a": [1] * (T.LIMITS["max_rows"] + 1)})
    header = ",".join(f"c{i}" for i in range(T.LIMITS["max_columns"] + 1))
    with pytest.raises(DataError, match="列数"):
        load_table(header + "\n" + ",".join("1" * (T.LIMITS["max_columns"] + 1)))
    with pytest.raises(DataError, match="列数"):
        load_table([{f"c{i}": 1 for i in range(T.LIMITS["max_columns"] + 1)}])
    with pytest.raises(DataError, match="列数"):
        load_table({f"c{i}": [1] for i in range(T.LIMITS["max_columns"] + 1)})
    with pytest.raises(DataError, match="セル"):
        load_table("a,b\n" + "y" * (T.LIMITS["max_cell_chars"] + 1) + ",1\n")
    with pytest.raises(DataError, match="セル"):
        load_table([{"a": "y" * (T.LIMITS["max_cell_chars"] + 1)}])
    with pytest.raises(DataError):
        load_table("a,b\n\"" + "y" * 200_000 + "\",1\n")
    with pytest.raises(DataError, match="大きすぎ"):
        load_table([{f"c{i}": "z" * 400 for i in range(50)}] * 200)
    assert time.monotonic() - started < 5


def test_maximum_table_loads_and_runs_quickly():
    rng = random.Random(3)
    data = {f"c{j}": [rng.randint(0, 999) for _ in range(T.LIMITS["max_rows"])] for j in range(20)}
    started = time.monotonic()
    table = load_table(data)
    run_tool(table, "correlate", {})
    run_tool(table, "describe", {})
    assert time.monotonic() - started < 30


# ---------------------------------------------------------------- tools

def test_profile_shape(sales):
    out = strict_json(run_tool(sales, "profile", {}))
    assert out["rows"] == 72 and out["n_columns"] == 6 and out["coerced"] == 0
    first = out["columns"][0]
    assert set(first) >= {"name", "kind", "missing", "unique", "sample"}
    assert first["sample"] == ["2023-01-01", "2023-02-01", "2023-03-01"]
    money = next(c for c in out["columns"] if c["name"] == "売上")
    assert money["unit"] == "¥" and len(money["sample"]) <= 3


def test_profile_reports_coerced_cells():
    table = load_table("x,y\n" + "\n".join(f"{i},a" for i in range(1, 11)) + "\n?,b\n")
    out = strict_json(run_tool(table, "profile", {}))
    assert out["coerced"] == 1 and out["coerced_columns"] == {"x": 1}
    assert out["columns"][0]["missing"] == 1


def test_describe_shapes(sales):
    out = strict_json(run_tool(sales, "describe", {}))
    assert not out["truncated"] and len(out["columns"]) == 6
    by_name = {c["name"]: c for c in out["columns"]}
    money = by_name["売上"]
    assert set(money) >= {"count", "missing", "mean", "std", "min", "q1", "median", "q3", "max", "skew"}
    values = sales.column("売上").values
    assert money["mean"] == T.round_sig(T.mean(values)) and money["q3"] == T.round_sig(T.quantile(values, .75))
    store = by_name["店舗"]
    assert store["unique"] == 3 and len(store["top"]) == 3
    assert set(store["top"][0]) == {"value", "count", "share"} and store["top"][0]["share"] == T.round_sig(1 / 3)
    assert by_name["会員"]["top"][0]["value"] in (True, False)
    assert by_name["日付"]["min"] == "2023-01-01" and by_name["日付"]["span_days"] == 700
    single = strict_json(run_tool(sales, "describe", {"column": "客数"}))
    assert [c["name"] for c in single["columns"]] == ["客数"]
    wide = load_table({f"c{i}": [i] for i in range(25)})
    assert run_tool(wide, "describe", {})["truncated"] is True


def test_correlate_pair_and_matrix():
    table = load_table({"x": X, "y": Y, "t": TIES, "flat": [1] * 10,
                        "z": [v * -2 for v in X]})
    pair = strict_json(run_tool(table, "correlate", {"x": "x", "y": "y"}))
    assert pair["mode"] == "pair" and pair["r"] == 0.95353 and pair["n"] == 10
    assert pair["p_value"] == T.round_sig(1.928590341134559e-05)
    assert pair["strength"] == "strong" and pair["direction"] == "positive"
    assert set(pair["apqb"]) == {"theta", "r", "T"}
    rank = run_tool(table, "correlate", {"x": "x", "y": "t", "method": "spearman"})
    assert rank["r"] == T.round_sig(0.46921626483186957)
    flat = run_tool(table, "correlate", {"x": "x", "y": "flat"})
    assert flat["r"] is None and flat["reason"]
    matrix = strict_json(run_tool(table, "correlate", {}))
    assert matrix["mode"] == "matrix" and matrix["skipped"] == 4 and matrix["n_pairs"] == 6
    assert matrix["pairs"][0]["r"] == -1.0 and matrix["pairs"][0]["direction"] == "negative"
    assert [abs(p["r"]) for p in matrix["pairs"]] == sorted((abs(p["r"]) for p in matrix["pairs"]), reverse=True)
    anchored = run_tool(table, "correlate", {"x": "y"})
    assert anchored["mode"] == "column" and all("y" in (p["x"], p["y"]) for p in anchored["pairs"])
    for method in ("pearson", "spearman"):
        cached = run_tool(table, "correlate", {"method": method})["pairs"]
        for item in cached:
            fn = T.spearman if method == "spearman" else T.pearson
            direct = fn(table.column(item["x"]).values, table.column(item["y"]).values)
            assert item["r"] == T.round_sig(direct["r"])


def test_correlate_matrix_uses_pairwise_complete_rows():
    table = load_table({"a": [1, 2, None, 4, 5, 6], "b": [2, 4, 6, None, 10, 13], "c": [1, None, None, None, None, 2]})
    out = run_tool(table, "correlate", {})
    assert {(p["x"], p["y"]): p["n"] for p in out["pairs"]} == {("a", "b"): 4}
    assert out["skipped"] == 2


def test_group_by_sum_count_and_truncation(sales):
    out = strict_json(run_tool(sales, "group_by", {"by": "店舗", "value": "売上"}))
    assert out["agg"] == "sum" and out["n_groups"] == 3 and not out["truncated"]
    assert [g["key"] for g in out["groups"]] == ["名古屋", "大阪", "東京"]
    assert math.isclose(sum(g["share"] for g in out["groups"]), 1.0, abs_tol=1e-5)
    assert out["total"] == T.round_sig(sum(sales.column("売上").values))
    counts = run_tool(sales, "group_by", {"by": "会員"})
    assert counts["agg"] == "count" and sum(g["value"] for g in counts["groups"]) == 72
    mean = run_tool(sales, "group_by", {"by": "店舗", "value": "客数", "agg": "mean"})
    assert mean["groups"][0]["share"] is None and [g["value"] for g in mean["groups"]] == [42.5, 37.5, 32.5]
    many = load_table({"k": [f"g{i}" for i in range(70)] + [None], "v": list(range(71))})
    out = run_tool(many, "group_by", {"by": "k", "value": "v"})
    assert len(out["groups"]) == 50 and out["truncated"] and out["other"]["groups"] == 20
    assert out["missing_keys"] == 1 and out["groups"][0]["key"] == "g69"
    years = load_table({"year": [2020, 2020, 2021], "v": [1, 2, 3]})
    assert run_tool(years, "group_by", {"by": "year", "value": "v"})["groups"][0]["key"] in (2020, 2021)


def test_trend_monthly_and_row_order(sales):
    out = strict_json(run_tool(sales, "trend", {"value": "売上", "time": "日付"}))
    assert out["n"] == 24 and out["rows_used"] == 72 and out["direction"] == "increasing"
    assert out["first_period"] == "2023-01-01" and out["last_period"] == "2024-12-01"
    assert out["p_value"] < 0.05 and 0.9 < out["r2"] <= 1 and out["slope_unit"] == "day"
    assert out["cagr"] > 0 and out["cagr_pct"] == pytest.approx(out["cagr"] * 100, rel=1e-5)
    assert len(out["last_periods"]) == 6 and out["last_periods"][0]["pct_change"] is not None
    assert set(out) >= {"first", "last", "change", "pct_change", "slope", "r2", "p_value", "direction"}
    yearly = run_tool(sales, "trend", {"value": "売上", "time": "日付", "period": "year"})
    assert yearly["n"] == 2 and yearly["direction"] == "unknown" and yearly["first_period"] == "2023"
    rows = run_tool(load_table({"v": [5, 4, 3, 2, 1]}), "trend", {"value": "v"})
    assert rows["time_kind"] == "row" and rows["direction"] == "decreasing" and rows["pct_change"] == -80.0
    flat = run_tool(load_table({"v": [3, 3, 3, 3]}), "trend", {"value": "v"})
    assert flat["direction"] == "flat" and flat["slope"] == 0.0
    numeric_time = load_table({"年": [2022, 2020, 2021, 2021], "v": [30, 10, 20, 5]})
    out = run_tool(numeric_time, "trend", {"value": "v", "time": "年", "agg": "mean"})
    assert out["n"] == 3 and out["first"] == 10.0 and out["last_periods"][1]["value"] == 12.5


def test_trend_cagr():
    table = load_table({"d": ["2020-01-01", "2021-01-01", "2022-01-01"], "v": [100, 110, 121]})
    out = run_tool(table, "trend", {"value": "v", "time": "d"})
    assert out["cagr"] == pytest.approx(0.1, rel=1e-3) and out["cagr_pct"] == pytest.approx(10, rel=1e-3)


def test_outliers_iqr_and_zscore():
    values = [10, 11, 9, 10, 12, 11, 10, 95, 9, 10, -40, 11]
    table = load_table({"v": values})
    out = strict_json(run_tool(table, "outliers", {"column": "v"}))
    assert out["count"] == 2 and out["high"] == 1 and out["low"] == 1
    assert [r["row"] for r in out["rows"]] == [8, 11] and out["rows"][0]["value"] == 95
    assert out["bounds"]["lower"] < 9 and out["share"] == T.round_sig(2 / 12)
    z = strict_json(run_tool(table, "outliers", {"column": "v", "method": "zscore", "threshold": 2}))
    assert z["count"] == 1 and z["rows"][0]["row"] == 8 and z["rows"][0]["z"] > 2
    flat = run_tool(load_table({"v": [4, 4, 4, 4]}), "outliers", {"column": "v", "method": "zscore"})
    assert flat["count"] == 0 and flat["reason"]
    tiny = run_tool(load_table({"v": [1, None]}), "outliers", {"column": "v"})
    assert tiny["count"] is None and tiny["reason"]
    many = load_table({"v": list(range(100)) + [10 ** 6] * 30})
    out = run_tool(many, "outliers", {"column": "v", "threshold": 0.5})
    assert len(out["rows"]) == 20 and out["truncated"]


def test_compare_groups_matches_scipy_reference():
    table = load_table({"g": ["A"] * len(X) + ["B"] * len(Z) + ["C"], "v": X + Z + [100]})
    out = strict_json(run_tool(table, "compare", {"value": "v", "by": "g"}))
    assert out["a"]["label"] == "A" and out["b"]["label"] == "B" and out["n_groups"] == 3
    assert out["t"] == T.round_sig(-1.918022285813911) and out["df"] == T.round_sig(14.354683603462354)
    assert out["p_value"] == T.round_sig(0.0752222196982633) and out["significant"] is False
    assert out["cohen_d"] == T.round_sig(-0.8533069995040317) and out["effect"] == "large"
    assert out["diff"] == T.round_sig(T.mean(X) - T.mean(Z))
    flipped = run_tool(table, "compare", {"value": "v", "by": "g", "a": "b", "b": "A"})
    assert flipped["a"]["label"] == "B" and flipped["t"] == -out["t"]
    only_b = run_tool(table, "compare", {"value": "v", "by": "g", "b": "A"})
    assert only_b["a"]["label"] == "B" and only_b["b"]["label"] == "A"
    with pytest.raises(DataError):
        run_tool(table, "compare", {"value": "v", "by": "g", "a": "存在しない"})
    single = run_tool(load_table({"g": ["A", "A"], "v": [1, 2]}), "compare", {"value": "v", "by": "g"})
    assert single["a"] is None and single["p_value"] is None and single["reason"]
    small = run_tool(load_table({"g": ["A", "B", "B"], "v": [1, 2, 3]}), "compare", {"value": "v", "by": "g"})
    assert small["t"] is None and small["reason"] and small["a"]["n"] == 2
    numeric_keys = load_table({"y": [2020, 2020, 2021, 2021], "v": [1, 2, 3, 5]})
    assert run_tool(numeric_keys, "compare", {"value": "v", "by": "y", "a": "2021"})["a"]["label"] == 2021
    assert run_tool(numeric_keys, "compare", {"value": "v", "by": "y", "a": "２０２１.0"})["a"]["label"] == 2021
    flags = load_table({"会員": ["はい", "いいえ"] * 3, "v": [1, 2, 3, 4, 5, 7]})
    out = run_tool(flags, "compare", {"value": "v", "by": "会員", "a": "いいえ", "b": "TRUE"})
    assert out["a"]["label"] is False and out["b"]["label"] is True


def test_crosstab_and_folding():
    rows = ["A"] * 12 + ["B"] * 5 + ["C"] * 7 + ["A"] * 3 + ["B"] * 9 + ["C"] * 10
    cols = ["x"] * 24 + ["y"] * 22
    table = load_table({"r": rows, "c": cols})
    out = strict_json(run_tool(table, "crosstab", {"row": "r", "col": "c"}))
    grid = out["table"]
    assert grid["rows"] == ["C", "A", "B"] and grid["cols"] == ["x", "y"] and out["n"] == 46
    assert out["chi2"] == T.round_sig(6.998542144130378) and out["dof"] == 2
    assert out["p_value"] == T.round_sig(0.030219403163080322)
    assert out["cramers_v"] == T.round_sig(0.39005412512185206)
    assert grid["counts"] == [[7, 10], [12, 3], [5, 9]] and grid["row_totals"] == [17, 15, 14]
    assert grid["col_totals"] == [24, 22] and grid["row_shares"][1] == [0.8, 0.2]
    wide = load_table({"r": [f"r{i % 20}" for i in range(400)], "c": [f"c{i % 15}" for i in range(400)]})
    out = run_tool(wide, "crosstab", {"row": "r", "col": "c"})
    grid = out["table"]
    assert len(grid["rows"]) == 12 and len(grid["cols"]) == 12 and grid["rows"][-1] == "その他"
    assert out["folded"] == {"rows": 9, "cols": 4} and out["truncated"] and sum(grid["row_totals"]) == 400
    clash = load_table({"r": ["その他"] * 30 + [f"r{i}" for i in range(20)], "c": ["x", "y"] * 25})
    assert run_tool(clash, "crosstab", {"row": "r", "col": "c"})["table"]["rows"][-1] == "その他(集約)"
    one = run_tool(load_table({"r": ["a", "a"], "c": ["x", "y"]}), "crosstab", {"row": "r", "col": "c"})
    assert one["chi2"] is None and one["reason"]


def test_top_n(sales):
    out = strict_json(run_tool(sales, "top_n", {"column": "売上", "n": 3}))
    assert out["label_column"] == "店舗" and len(out["rows"]) == 3
    assert out["rows"][0]["value"] == max(sales.column("売上").values) and out["rows"][0]["label"] == "名古屋"
    assert set(out["rows"][0]) == {"row", "value", "label"} and 0 < out["top_share"] < 1
    asc = run_tool(sales, "top_n", {"column": "客数", "order": "asc", "label": "メモ"})
    assert asc["rows"][0] == {"row": 1, "value": 21.0, "label": "note3"} and len(asc["rows"]) == 5
    dates = strict_json(run_tool(sales, "top_n", {"column": "日付", "n": 1}))
    assert dates["rows"][0]["value"] == "2024-12-01"


def test_forecast_matches_ols_prediction_interval():
    ys = [10.0, 12.5, 13.1, 15.8, 16.2, 18.9]
    out = strict_json(run_tool(load_table({"v": ys}), "forecast", {"value": "v", "periods": 2}))
    assert out["method"] == "linear_trend" and out["n"] == 6 and out["caveat"]
    first, second = out["points"]
    assert first["step"] == 1 and first["period"] == 7
    assert first["estimate"] == T.round_sig(20.24666666666667)
    assert first["lower"] == T.round_sig(17.88322484902331) and first["upper"] == T.round_sig(22.61010848431003)
    assert second["lower"] == T.round_sig(19.275370853770482)
    assert out["r2"] == T.round_sig(0.9690119027820676)
    monthly = load_table({"d": [f"2024-{m:02d}-01" for m in range(1, 11)], "v": list(range(10))})
    out = run_tool(monthly, "forecast", {"value": "v", "time": "d", "periods": 4})
    assert [p["period"] for p in out["points"]] == ["2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01"]
    daily = load_table({"d": [f"2024-01-{d:02d}" for d in range(1, 30, 7)], "v": [1, 3, 2, 5, 4]})
    out = run_tool(daily, "forecast", {"value": "v", "time": "d", "periods": 1})
    assert out["points"][0]["period"] == "2024-02-05"
    short = run_tool(load_table({"v": [1, 2, 3]}), "forecast", {"value": "v"})
    assert short["points"] == [] and short["reason"]
    flat = run_tool(load_table({"v": [7, 7, 7, 7, 7]}), "forecast", {"value": "v", "periods": 1})
    assert flat["points"][0] == {"step": 1, "period": 6, "estimate": 7.0, "lower": 7.0, "upper": 7.0}


def test_calculator_tool(sales):
    assert run_tool(sales, "calculator", {"expression": "(12+3)*4"}) == {"expression": "(12+3)*4", "value": 60}
    for expression in ("1/0", "__import__('os')", "2**10"):
        with pytest.raises(DataError):
            run_tool(sales, "calculator", {"expression": expression})


NASTY = {"empty": [None] * 6, "flat": [3] * 6, "one": [None] * 5 + [1],
         "when": [None, "2024-01-01", None, None, None, None], "cat": ["a"] * 6,
         "pct": ["1%", "2%", None, "3%", "x", "4%"]}


@pytest.mark.parametrize("tool", sorted(T.TOOLS))
def test_every_tool_is_json_safe_on_degenerate_data(tool):
    table = load_table(NASTY)
    numeric = [c.name for c in table.columns if c.kind == "numeric"]
    candidates = {
        "profile": [{}], "describe": [{}, *({"column": c} for c in table.names())],
        "correlate": [{}, {"method": "spearman"}, {"x": "flat", "y": "one"}, {"x": "empty", "y": "flat"}],
        "group_by": [{"by": b, "value": v, "agg": a} for b in ("cat", "flat", "empty", "when")
                     for v in numeric for a in ("sum", "mean", "median", "min", "max", "count")],
        "trend": [{"value": v, **t, **b} for v in numeric for t in ({}, {"time": "when"}, {"time": "flat"})
                  for b in ({}, {"by": "cat"})],
        "outliers": [{"column": v, "method": m, **b} for v in numeric for m in ("iqr", "zscore")
                     for b in ({}, {"by": "cat"})],
        "compare": [{"value": v, "by": b} for v in numeric for b in ("cat", "empty", "flat")],
        "crosstab": [{"row": "cat", "col": "flat"}, {"row": "empty", "col": "cat"}],
        "top_n": [{"column": v} for v in numeric] + [{"column": "when"}],
        "forecast": [{"value": v, **t} for v in numeric for t in ({}, {"time": "when"}, {"time": "flat"})],
        "calculator": [{"expression": "1+1"}],
    }[tool]
    for args in candidates:
        try:
            strict_json(run_tool(table, tool, args))
        except DataError:
            pass


def test_tool_registry_matches_specs():
    assert set(T.TOOLS) == set(T.TOOL_SPECS) == {
        "profile", "describe", "correlate", "group_by", "trend", "outliers", "compare", "crosstab",
        "top_n", "forecast", "calculator"}
    for spec in T.TOOL_SPECS.values():
        assert spec["description"] and set(spec) >= {"description", "args"}
        for rule in spec["args"].values():
            assert rule["type"] in ("column", "string", "int", "enum", "number")
            assert isinstance(rule["required"], bool)


# ---------------------------------------------------------------- validation

def test_validate_args_normalises(sales):
    assert validate_args(sales, "correlate", {"x": "売上", "y": " 客数 "}) == {
        "x": "売上", "y": "客数", "method": "pearson"}
    assert validate_args(sales, "group_by", {"by": "店舗"}) == {"by": "店舗", "agg": "count"}
    assert validate_args(sales, "group_by", {"by": "店舗", "value": "売上"})["agg"] == "sum"
    assert validate_args(sales, "outliers", {"column": "売上", "method": "ZScore"}) == {
        "column": "売上", "method": "zscore", "threshold": 3.0}
    assert validate_args(sales, "top_n", {"column": "売上", "n": 10}) == {
        "column": "売上", "n": 10, "order": "desc"}
    assert validate_args(sales, "trend", {"value": "売上", "time": None}) == {
        "value": "売上", "agg": "sum", "period": "raw"}
    assert validate_args(sales, "profile", None) == {}
    assert validate_args(sales, "outliers", {"column": "売上", "threshold": 2})["threshold"] == 2.0


@pytest.mark.parametrize("tool,args", [
    ("delete_rows", {}), (None, {}), ("profile", {"x": 1}), ("profile", []),
    ("describe", {"column": "存在しない"}), ("describe", {"column": 3}), ("describe", {"column": ""}),
    ("correlate", {"x": "店舗"}), ("correlate", {"x": "売上", "y": "売上"}),
    ("correlate", {"method": "kendall"}), ("correlate", {"method": 1}),
    ("group_by", {}), ("group_by", {"by": "店舗", "agg": "sum"}), ("group_by", {"by": "店舗", "value": "店舗"}),
    ("group_by", {"by": "店舗", "value": "売上", "agg": "avg"}), ("group_by", {"by": "売上", "value": "売上"}),
    ("trend", {"value": "日付"}), ("trend", {"value": "売上", "time": "店舗"}),
    ("trend", {"value": "売上", "period": "month"}), ("trend", {"value": "売上", "time": "客数", "period": "year"}),
    ("outliers", {"column": "売上", "threshold": True}), ("outliers", {"column": "売上", "threshold": 0.1}),
    ("outliers", {"column": "売上", "threshold": 5.5}),
    ("outliers", {"column": "売上", "method": "zscore", "threshold": 1}),
    ("outliers", {"column": "売上", "threshold": float("nan")}), ("outliers", {"column": "売上", "threshold": "2"}),
    ("outliers", {"column": "メモ"}), ("outliers", {"column": "売上", "by": "客数"}),
    ("outliers", {"column": "売上", "by": "メモ"}), ("outliers", {"column": "売上", "by": "日付"}),
    ("trend", {"value": "売上", "by": "客数"}), ("trend", {"value": "売上", "by": "日付"}),
    ("trend", {"value": "売上", "by": 1}), ("compare", {"value": "売上"}), ("compare", {"value": "売上", "by": "店舗", "a": 1}),
    ("compare", {"value": "売上", "by": "店舗", "a": "東京", "b": "東京"}), ("compare", {"value": "売上", "by": "店舗", "a": " "}),
    ("crosstab", {"row": "店舗", "col": "店舗"}), ("crosstab", {"row": "店舗"}),
    ("top_n", {"column": "売上", "n": True}), ("top_n", {"column": "売上", "n": 0}), ("top_n", {"column": "売上", "n": 51}),
    ("top_n", {"column": "売上", "n": 5.0}), ("top_n", {"column": "店舗"}), ("top_n", {"column": "売上", "label": "売上"}),
    ("forecast", {"value": "売上", "periods": 13}), ("forecast", {"value": "売上", "periods": False}),
    ("calculator", {}), ("calculator", {"expression": "1" * 201}), ("calculator", {"expression": ["1"]})])
def test_validate_args_rejects(sales, tool, args):
    with pytest.raises(DataError):
        validate_args(sales, tool, args)
    with pytest.raises(DataError):
        run_tool(sales, tool, args)


def test_validation_messages_are_japanese_and_bounded(sales):
    with pytest.raises(DataError) as info:
        validate_args(sales, "describe", {"column": "x" * 150 + "{0}"})
    assert "列" in str(info.value) and len(str(info.value)) < 120
    with pytest.raises(DataError) as info:
        validate_args(sales, "outliers", {"column": "店舗"})
    assert "数値" in str(info.value)


# ---------------------------------------------------------------- JSON helpers

def test_json_safe_and_round_sig():
    out = T.json_safe({"a": float("nan"), "b": [float("inf"), -float("inf"), 1 / 3, -0.0],
                       "c": datetime.date(2024, 1, 2), "d": (1, True, None), 3: 1234567.89,
                       "e": {1, 2}, "f": object()})
    assert out["a"] is None and out["b"] == [None, None, 0.333333, 0.0]
    assert out["c"] == "2024-01-02" and out["d"] == [1, True, None] and out["3"] == 1234568.0
    assert out["e"] == [1, 2] and isinstance(out["f"], str)
    strict_json(out)
    # integer digits are never rounded away: yen totals stay exact, decimals keep 6 significant digits
    assert T.round_sig(123456789) == 123456789.0 and T.round_sig(1.23456789e-12) == 1.23457e-12
    assert T.round_sig(13456789.123) == 13456789.0 and T.round_sig(-98765.4321) == -98765.4
    assert T.round_sig(float("nan")) is None and T.round_sig(True) is None


def test_number_tokens():
    tokens = T.number_tokens({"a": 1.5, "b": [2, True, None, float("nan")], "s": "売上は1,234円、成長率１２．５％",
                              "d": "2024-03-01", "nested": {"x": -7}, "k1": "x"})
    assert tokens[:3] == [1.5, 2.0, 1234.0]
    assert {12.5, 2024.0, 3.0, 1.0, -7.0} <= set(tokens) and 1.0 in tokens
    assert len(tokens) == len(set(tokens))
    assert T.number_tokens([]) == [] and T.number_tokens("no digits") == []


# ---------------------------------------------------------------- review regressions

def daily(start, days, value=lambda i: 100 + 0.2 * i):
    first = datetime.date(*start)
    return load_table({"d": [(first + datetime.timedelta(days=i)).isoformat() for i in range(days)],
                       "v": [value(i) for i in range(days)]})


def test_partial_month_and_year_buckets_are_dropped_not_compared():
    table = daily((2024, 1, 20), 199)          # strictly increasing, 2024-01-20 .. 2024-08-05
    out = run_tool(table, "trend", {"value": "v", "time": "d", "period": "month"})
    assert out["partial_periods"] == ["2024-01", "2024-08"] and out["n"] == 6
    assert out["direction"] == "increasing" and out["pct_change"] > 0 and out["first_period"] == "2024-02"
    assert all(p["pct_change"] is None or p["pct_change"] > 0 for p in out["last_periods"])
    fc = run_tool(table, "forecast", {"value": "v", "time": "d", "period": "month", "periods": 3})
    assert fc["partial_periods"] == ["2024-01", "2024-08"] and fc["slope"] > 0
    assert [p["period"] for p in fc["points"]] == ["2024-08", "2024-09", "2024-10"]
    assert fc["points"][0]["estimate"] > out["last"]          # was 2,870 falling while true totals rise
    months = load_table({"m": [f"{2024 + i // 12}-{i % 12 + 1:02d}" for i in range(15)], "v": list(range(100, 115))})
    yearly = run_tool(months, "trend", {"value": "v", "time": "m", "period": "year"})
    assert yearly["partial_periods"] == ["2025"] and yearly["n"] == 1 and "2つ未満" in yearly["reason"]
    full = run_tool(daily((2023, 1, 1), 730), "trend", {"value": "v", "time": "d", "period": "month"})
    assert full["partial_periods"] == [] and full["n"] == 24 and not full["uneven_periods"]
    weekly = load_table({"d": [(datetime.date(2023, 1, 2) + datetime.timedelta(weeks=i)).isoformat()
                               for i in range(104)], "v": [1000] * 104})
    assert run_tool(weekly, "trend", {"value": "v", "time": "d", "period": "month"})["uneven_periods"] is True


def test_zscore_reports_undetectable_small_samples_and_iqr_zero():
    table = load_table({"v": [100, 102, 98, 101, 99, 103, 97, 100, 5000]})
    z = run_tool(table, "outliers", {"column": "v", "method": "zscore"})
    assert z["count"] is None and "n=9" in z["reason"]          # max |z| = 8/sqrt(9) < 3
    assert run_tool(table, "outliers", {"column": "v"})["count"] == 1
    assert run_tool(load_table({"v": [1, 1, 1, 1, 1000]}), "outliers",
                    {"column": "v", "method": "zscore", "threshold": 2})["count"] is None
    flags = run_tool(load_table({"v": [0] * 20 + [1] * 3}), "outliers", {"column": "v"})
    assert flags["count"] is None and flags["iqr"] == 0 and flags["reason"]


def test_dotted_year_month_labels_are_months_not_decimals():
    table = load_table("年月,売上\n" + "\n".join(f"2024.{m},{100 * m}" for m in range(1, 13)))
    col = table.column("年月")
    assert col.kind == "datetime" and len(set(col.values)) == 12 and col.values[9] == datetime.date(2024, 10, 1)
    assert run_tool(table, "group_by", {"by": "年月", "value": "売上"})["n_groups"] == 12
    assert T.parse_date("2024.01") is None
    assert load_table({"price": ["2019.5", "2020.12", "2021.3"]}).column("price").kind == "numeric"


def test_percent_columns_are_averaged_and_never_given_shares():
    table = load_table("地域,利益率,d\n東,10%,2024-01-01\n東,12%,2024-01-02\n西,20%,2024-01-03\n西,22%,2024-01-04\n")
    assert validate_args(table, "group_by", {"by": "地域", "value": "利益率"})["agg"] == "mean"
    out = run_tool(table, "group_by", {"by": "地域", "value": "利益率", "agg": "sum"})
    assert all(g["share"] is None for g in out["groups"])
    assert "top_share" not in run_tool(table, "top_n", {"column": "利益率"})
    assert validate_args(table, "trend", {"value": "利益率", "time": "d"})["agg"] == "mean"
    assert run_tool(table, "forecast", {"value": "利益率", "time": "d"})["agg"] == "mean"
    assert run_tool(table, "trend", {"value": "利益率", "time": "d"})["cagr"] is None


def test_forecast_steps_month_end_quarter_end_and_mid_month_calendars():
    ends = [datetime.date(2024, m, calendar.monthrange(2024, m)[1]).isoformat() for m in range(1, 13)]
    out = run_tool(load_table({"d": ends, "v": list(range(12))}), "forecast", {"value": "v", "time": "d", "periods": 6})
    assert [p["period"] for p in out["points"]] == ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30",
                                                    "2025-05-31", "2025-06-30"]
    quarters = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    out = run_tool(load_table({"d": quarters, "v": [1, 2, 4, 5]}), "forecast", {"value": "v", "time": "d", "periods": 2})
    assert [p["period"] for p in out["points"]] == ["2025-03-31", "2025-06-30"]
    mid = [f"2024-{m:02d}-15" for m in range(1, 7)]
    out = run_tool(load_table({"d": mid, "v": list(range(6))}), "forecast", {"value": "v", "time": "d", "periods": 2})
    assert [p["period"] for p in out["points"]] == ["2024-07-15", "2024-08-15"]


def test_minority_unit_cells_are_coerced_not_relabelling_the_column():
    col = load_table({"売上": ["¥1,000", "¥1,200", "¥900", "¥1,100", "¥1,050", "¥980", "¥1,010", "¥1,300",
                               "¥1,150", "5%"]}).column("売上")
    assert (col.kind, col.unit, col.coerced, col.values[-1]) == ("numeric", "¥", 1, None)


def test_crosstab_fold_label_never_collides_with_a_real_level():
    rows = [f"L{i}" for i in range(15) for _ in range(30 - i)] + ["その他"] * 100 + ["その他(集約)"] * 90
    out = run_tool(load_table({"r": rows, "c": ["x", "y"] * (len(rows) // 2) + ["x"] * (len(rows) % 2)}),
                   "crosstab", {"row": "r", "col": "c"})
    totals = dict(zip(out["table"]["rows"], out["table"]["row_totals"]))
    assert totals["その他(集約)"] == 90 and out["table"]["rows"][-1] == "その他(集約2)"
    assert sum(totals.values()) == len(rows)


def test_number_parsing_is_linear_on_hostile_cells_and_group_labels():
    attack = "(" + " " * 28 + "1" * 12 + " " * 22 + "x"
    start = time.perf_counter()
    assert T.parse_number(attack) is None and load_table({"v": [1] * 45 + [attack] * 5}).column("v").coerced == 5
    table = load_table({"g": [float(i % 4) for i in range(40)], "v": list(range(40))})
    with pytest.raises(DataError):
        run_tool(table, "compare", {"value": "v", "by": "g", "a": attack})
    assert time.perf_counter() - start < 0.5
    assert T.parse_number(" ( ¥ 1,234.5 ) ") == -1234.5 and T.parse_number("1.5e3") == 1500.0


def test_huge_integer_threshold_is_a_data_error_not_overflow(sales):
    with pytest.raises(DataError):
        validate_args(sales, "outliers", {"column": "売上", "threshold": int("1" * 400)})
    assert validate_args(sales, "outliers", {"column": "売上", "threshold": 2})["threshold"] == 2.0


def test_lone_surrogates_are_scrubbed_from_cells_and_string_arguments():
    table = load_table([{"region": "\ud800east", "sales": 1}, {"region": "west", "sales": 2}] * 3)
    out = run_tool(table, "group_by", {"by": "region", "value": "sales"})
    json.dumps(out, ensure_ascii=False).encode("utf-8")
    assert {g["key"] for g in out["groups"]} == {"?east", "west"}
    assert validate_args(table, "compare", {"value": "sales", "by": "region", "a": "\udfffwest"})["a"] == "?west"


def test_tiny_magnitudes_do_not_divide_by_zero():
    assert T.skewness([0.0, 1e-160, 2e-160, 5e-160]) is None
    test = T.welch_ttest([0.0, 1e-160, 2e-160], [5e-160, 3e-160, 4e-160])
    assert test["df"] == pytest.approx(4.0) and test["p_value"] is not None
    assert T.welch_ttest(X, Z)["df"] == pytest.approx(14.354683603462354, rel=1e-12)
    table = load_table({"g": ["a", "b"] * 4, "tiny": [0.0, 1e-160, 2e-160, 5e-160] * 2})
    assert run_tool(table, "describe", {})["columns"][1]["skew"] is None


def test_cr_only_line_endings_parse():
    table = load_table('date,sales\r2024-01-01,10\r2024-02-01,12\r2024-03-01,"1,500"\r')
    assert table.kinds() == {"date": "datetime", "sales": "numeric"} and table.n_rows == 3
    assert load_table('a,b\r1,"x\ry"\r2,z\r').column("b").values == ["x\ry", "z"]


def test_id_and_year_columns_are_skipped_unless_named():
    table = load_table({"社員ID": list(range(1001, 1021)), "年": [2020 + i % 4 for i in range(20)],
                        "満足度": [3, 4, 2, 5, 4, 3, 3, 4, 5, 2] * 2, "残業": [10, 5, 20, 1, 6, 9, 12, 7, 2, 25] * 2})
    matrix = run_tool(table, "correlate", {})
    assert matrix["columns"] == ["満足度", "残業"] and len(matrix["pairs"]) == 1
    assert run_tool(table, "correlate", {"x": "社員ID", "y": "満足度"})["r"] is not None
    assert "社員ID" not in [c["name"] for c in run_tool(table, "describe", {})["columns"]]
    assert run_tool(table, "describe", {"column": "社員ID"})["columns"][0]["count"] == 20


def test_compare_marks_defaulted_groups():
    table = load_table({"g": list("ABCD") * 6, "v": list(range(24))})
    assert run_tool(table, "compare", {"value": "v", "by": "g"})["defaulted"] is True
    assert run_tool(table, "compare", {"value": "v", "by": "g", "a": "C"})["defaulted"] is False


def test_cagr_for_year_columns_and_never_across_zero():
    years = load_table({"年度": list(range(2018, 2025)), "売上高": [100, 108, 115, 120, 131, 140, 150]})
    assert run_tool(years, "trend", {"value": "売上高", "time": "年度"})["cagr_pct"] == pytest.approx(6.991, rel=1e-3)
    swing = load_table({"年度": list(range(2018, 2025)), "利益": [50, -20, 30, -10, 60, 40, 30]})
    assert run_tool(swing, "trend", {"value": "利益", "time": "年度"})["cagr"] is None
    plain = load_table({"t": list(range(2018, 2025)), "v": [100, 108, 115, 120, 131, 140, 150]})
    assert run_tool(plain, "trend", {"value": "v", "time": "t"})["cagr"] is None    # not a year column


# ---------------------------------------------------------------- ANOVA, per-group outliers and trends

# R: oneway.test(count ~ spray, InsectSprays) -> F = 36.065, num df = 5, denom df = 30.043, p = 7.999e-12
INSECT_SPRAYS = {"A": [10, 7, 20, 14, 14, 12, 10, 23, 17, 20, 14, 13], "B": [11, 17, 21, 11, 16, 14, 17, 17, 19, 21, 7, 13],
                 "C": [0, 1, 7, 2, 3, 1, 2, 1, 3, 0, 1, 4], "D": [3, 5, 12, 6, 4, 3, 5, 5, 5, 5, 2, 4],
                 "E": [3, 5, 3, 5, 3, 6, 1, 1, 3, 2, 6, 4], "F": [11, 9, 15, 22, 15, 16, 13, 10, 26, 26, 24, 13]}


def welch_anova_reference(groups):
    """Welch (1951) written out independently with numpy, for the randomized cross-check."""
    import numpy as np
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    w = n / np.array([np.var(g, ddof=1) for g in groups])
    centre = (w * m).sum() / w.sum()
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    f = ((w * (m - centre) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) * lam / (k * k - 1))
    return f, (k * k - 1) / (3 * lam)


@pytest.mark.parametrize("f,d1,d2,expected", [
    (2.5, 3, 12, 0.10915471239500621), (0.7, 2, 30, 0.5045150193942447), (4.1, 5, 7.3, 0.04372910055385989),
    (15.0, 1, 1, 0.16086124651033248), (1.0, 10, 1000, 0.44136583209978697)])
def test_f_distribution_matches_scipy(f, d1, d2, expected):
    assert close(T.f_sf(f, d1, d2), expected, rel=1e-9)
    assert T.f_sf(0, 3, 4) == 1.0 and T.f_sf(-1, 3, 4) == 1.0 and T.f_sf(1e308, 3, 4) == 0.0
    assert T.f_sf(1.0, 0, 4) is None and T.f_sf(float("nan"), 3, 4) is None and T.f_sf(True, 3, 4) is None


def test_anova_matches_scipy_and_r_reference_values():
    classic = T.anova_oneway([X, Y, Z])          # scipy.stats.f_oneway(X, Y, Z)
    assert close(classic["f"], 8.139607785959031) and close(classic["p_value"], 0.00200245539249616, rel=1e-8)
    assert (classic["df1"], classic["df2"]) == (2, 24) and close(classic["eta_squared"], 0.4041592007384481)
    welch = T.welch_anova([X, Y, Z])
    assert close(welch["f"], 12.502214968840612) and close(welch["df2"], 15.822417853646636)
    assert close(welch["p_value"], 0.0005535653357277769, rel=1e-8)
    sprays = T.welch_anova(list(INSECT_SPRAYS.values()))
    assert round(sprays["f"], 3) == 36.065 and sprays["df1"] == 5 and round(sprays["df2"], 3) == 30.043
    assert close(sprays["p_value"], 7.999e-12, rel=1e-3)
    assert close(T.anova_oneway(list(INSECT_SPRAYS.values()))["f"], 34.7022820554917)
    # two groups: Welch's F is the squared Welch t (same df and p), the classic F the squared Student t
    two, t = T.welch_anova([X, Z]), T.welch_ttest(X, Z)
    assert close(two["f"], t["t"] ** 2) and close(two["df2"], t["df"]) and close(two["p_value"], t["p_value"])
    # affine invariance keeps extreme magnitudes finite
    for scale in (1e45, 1e-200):
        big = T.welch_anova([[v * scale for v in g] for g in (X, Y, Z)])
        assert close(big["f"], welch["f"], rel=1e-9) and close(big["p_value"], welch["p_value"], rel=1e-7)
    assert T.anova_oneway([[1, 1], [1, 1], [1, 1]])["reason"] and T.welch_anova([[1, 1], [2, 2]])["f"] is None
    assert T.anova_oneway([[1, 1], [2, 2], [3, 3]])["eta_squared"] == 1.0
    assert T.welch_anova([[1, 2], [3]])["reason"] and T.anova_oneway([[1], [2]])["reason"]


def test_anova_matches_scipy_and_the_welch_formula_on_random_samples():
    pytest.importorskip("scipy")
    from scipy import stats

    def same(ours, ref, rel=1e-9):
        assert ours is not None and (abs(ours - ref) <= 1e-12 or abs(ours - ref) <= rel * abs(ref)), (ours, ref)

    for seed in range(200):
        rng = random.Random(seed)
        groups = [_random_sample(rng, rng.choice([2, 3, 5, 12, 60])) for _ in range(rng.randint(2, 8))]
        if any(max(g) == min(g) for g in groups):
            continue
        classic, ref = T.anova_oneway(groups), stats.f_oneway(*groups)
        same(classic["f"], float(ref.statistic))
        assert abs(classic["p_value"] - float(ref.pvalue)) < 1e-9
        welch, (f, df2) = T.welch_anova(groups), welch_anova_reference(groups)
        same(welch["f"], float(f))
        same(welch["df2"], float(df2))
        assert abs(welch["p_value"] - float(stats.f.sf(f, len(groups) - 1, df2))) < 1e-9
        d1, d2, x = rng.choice([1, 2, 3.5, 10, 50]), rng.choice([1, 2.5, 7, 30, 500, 20000]), rng.uniform(0, 10)
        same(T.f_sf(x, d1, d2), float(stats.f.sf(x, d1, d2)))


def test_compare_runs_an_anova_for_three_or_more_groups():
    table = load_table({"g": [k for k, v in INSECT_SPRAYS.items() for _ in v],
                        "v": [x for v in INSECT_SPRAYS.values() for x in v]})
    out = strict_json(run_tool(table, "compare", {"value": "v", "by": "g"}))
    assert out["mode"] == "anova" and out["test"] == "welch_anova" and out["welch_note"] is None
    assert out["defaulted"] is True and out["groups_tested"] == 6 and out["skipped_groups"] == 0 and out["n"] == 72
    assert [g["label"] for g in out["groups"]] == ["F", "B", "A", "D", "E", "C"]          # by mean, descending
    assert out["groups"][0] == {"label": "F", "n": 12, "mean": T.round_sig(T.mean(INSECT_SPRAYS["F"])),
                                "std": T.round_sig(T.sample_std(INSECT_SPRAYS["F"])), "median": 15.0}
    assert out["f"] == out["welch"]["f"] == T.round_sig(36.06544389357724)
    assert out["p_value"] == out["welch"]["p_value"] and out["significant"] is True
    assert out["classic"]["f"] == T.round_sig(34.7022820554917) and out["classic"]["df2"] == 66
    assert out["eta_squared"] == T.round_sig(0.724439015562794)
    pair = out["pairwise"]
    assert (pair["a"]["label"], pair["b"]["label"], pair["comparisons"]) == ("F", "C", 15)
    w = T.welch_ttest(INSECT_SPRAYS["F"], INSECT_SPRAYS["C"])
    assert pair["p_value"] == T.round_sig(w["p_value"]) and pair["p_adjusted"] == T.round_sig(min(1, 15 * w["p_value"]))
    assert pair["significant"] is True and pair["t"] == T.round_sig(w["t"])
    # explicit groups keep the 2-group comparison; a single named group too
    for args in ({"a": "A", "b": "B"}, {"a": "C"}):
        pair_mode = run_tool(table, "compare", {"value": "v", "by": "g", **args})
        assert pair_mode["mode"] == "pair" and "groups" not in pair_mode and pair_mode["a"]["label"] == args["a"]
    # groups of one value are skipped and counted; fewer than 3 groups of 2+ values stay pairwise
    extra = load_table({"g": list("AAABBBCCCD"), "v": [1, 2, 3, 5, 6, 7, 9, 9.5, 11, 40]})
    out = run_tool(extra, "compare", {"value": "v", "by": "g"})
    assert out["mode"] == "anova" and out["skipped_groups"] == 1 and out["groups_tested"] == 3
    assert run_tool(load_table({"g": list("AABBC"), "v": [1, 2, 3, 4, 5]}), "compare",
                    {"value": "v", "by": "g"})["mode"] == "pair"


def test_anova_primary_test_falls_back_to_classic_when_welch_is_unreliable():
    noisy = load_table({"g": [f"g{i // 2}" for i in range(40)], "v": [(i * 37) % 11 for i in range(40)]})
    out = run_tool(noisy, "compare", {"value": "v", "by": "g"})       # 20 groups of 2: Welch over-rejects
    assert out["test"] == "anova" and out["welch_note"] == "small_groups"
    assert out["p_value"] == out["classic"]["p_value"] and out["welch"]["p_value"] is not None
    constant = load_table({"g": list("AAABBBCCC"), "v": [1, 1, 1, 4, 5, 6, 8, 9, 10]})
    out = run_tool(constant, "compare", {"value": "v", "by": "g"})
    assert out["test"] == "anova" and out["welch_note"] == "constant_group" and out["welch"]["reason"]
    assert out["p_value"] == out["classic"]["p_value"] and out["significant"] is True
    flat = run_tool(load_table({"g": list("AABBCC"), "v": [2] * 6}), "compare", {"value": "v", "by": "g"})
    assert flat["p_value"] is None and flat["reason"] and flat["significant"] is None
    assert T.welch_primary([5, 5, 5]) and not T.welch_primary([4, 9, 9]) and not T.welch_primary([6] * 7)


def regional_sales(double=("大阪", 8)):
    """Three regions on different scales; one 大阪 month doubled stays inside the pooled range."""
    rows = []
    for m in range(18):
        for region, base in (("東京", 700), ("大阪", 500), ("福岡", 350)):
            value = base + 8 * m + (m * 37 + len(region) * 11) % 60
            if (region, m) == double:
                value *= 2
            rows.append({"月": f"{2024 + m // 12}-{m % 12 + 1:02d}", "地域": region, "売上": value})
    return rows


def test_outliers_within_groups_find_a_value_that_is_ordinary_overall():
    table = load_table(regional_sales())
    row = 8 * 3 + 2                                  # 1-based row of 大阪, month 9
    assert table.column("売上").values[row - 1] > 1000
    pooled = run_tool(table, "outliers", {"column": "売上"})
    assert row not in [r["row"] for r in pooled["rows"]]
    out = strict_json(run_tool(table, "outliers", {"column": "売上", "by": "地域"}))
    assert out["by"] == "地域" and out["bounds"] is None and out["count"] >= 1
    top = out["rows"][0]
    assert (top["row"], top["group"], top["side"]) == (row, "大阪", "high") and top["value"] > top["upper"]
    assert out["groups_checked"] == 3 and out["groups_skipped"] == 0 and out["n"] == 54 and out["min_group_n"] == 5
    osaka = next(g for g in out["groups"] if g["key"] == "大阪")
    values = sorted(table.column("売上").values[i] for i in range(54) if table.column("地域").values[i] == "大阪")
    q1, q3 = T.quantile(values, 0.25), T.quantile(values, 0.75)
    assert osaka["upper"] == T.round_sig(q3 + 1.5 * (q3 - q1)) and osaka["n"] == 18
    z = run_tool(table, "outliers", {"column": "売上", "by": "地域", "method": "zscore", "threshold": 2.5})
    assert z["rows"][0]["row"] == row and z["rows"][0]["z"] > 2.5 and z["min_group_n"] == 9


def test_outliers_by_skips_small_or_flat_groups_and_counts_them():
    values = [10, 11, 9, 10, 12, 11, 10, 95, 9, 10] + [5, 5, 5, 5, 5, 5] + [1, 2, 300]
    table = load_table({"g": ["A"] * 10 + ["B"] * 6 + ["C"] * 3, "v": values})
    out = strict_json(run_tool(table, "outliers", {"column": "v", "by": "g"}))
    assert out["groups_checked"] == 1 and out["groups_skipped"] == 2 and out["groups_without_spread"] == 1
    assert out["skipped_rows"] == 9 and out["n"] == 10 and [r["row"] for r in out["rows"]] == [8]
    small = run_tool(table, "outliers", {"column": "v", "by": "g", "method": "zscore"})    # needs n >= 11
    assert small["count"] is None and small["reason"] and small["groups_checked"] == 0
    keys = load_table({"g": ["A"] * 6 + [None] * 2, "v": [1, 2, 3, 4, 5, 60, 7, 8]})
    out = run_tool(keys, "outliers", {"column": "v", "by": "g"})
    assert out["missing_keys"] == 2 and out["n"] == 6 and out["rows"][0]["row"] == 6
    assert validate_args(keys, "outliers", {"column": "v", "by": "g"}) == {
        "column": "v", "method": "iqr", "threshold": 1.5, "by": "g"}


def test_trend_by_group_summarises_each_group_and_ranks_growth():
    rows = []
    for m in range(12):
        for region, base, growth in (("東", 1000, 30), ("西", 100, 8), ("南", 500, -20)):
            rows.append({"月": f"2024-{m + 1:02d}-01", "地域": region, "売上": base + growth * m + (m * 7) % 5})
    rows += [{"月": "2024-01-01", "地域": "北", "売上": 50}, {"月": "2024-02-01", "地域": "北", "売上": 60}]
    table = load_table(rows)
    out = strict_json(run_tool(table, "trend", {"value": "売上", "time": "月", "period": "month", "by": "地域"}))
    assert out["by"] == "地域" and out["n"] == 12 and out["direction"] == "increasing"   # overall unchanged
    assert out["n_groups"] == 4 and out["groups_trended"] == 3 and out["groups_skipped"] == 1   # 北: 2 points
    assert out["rank_by"] == "slope_pct" and [g["key"] for g in out["groups"]] == ["西", "東", "南"]
    assert (out["increasing"], out["decreasing"], out["flat"]) == (2, 1, 0)
    east = next(g for g in out["groups"] if g["key"] == "東")
    ys = [r["売上"] for r in rows if r["地域"] == "東"]
    xs = [float((datetime.date(2024, m + 1, 1) - datetime.date(2024, 1, 1)).days) for m in range(12)]
    reg = T.linregress(xs, ys)
    assert east["slope"] == T.round_sig(reg["slope"]) and east["p_value"] == T.round_sig(reg["p_value"])
    assert east["first_period"] == "2024-01" and east["last_period"] == "2024-12" and east["n"] == 12
    assert east["pct_change"] == T.round_sig((ys[-1] - ys[0]) / ys[0] * 100)
    assert east["slope_pct"] == T.round_sig(reg["slope"] * out["period_step"] / T.mean(ys) * 100)
    plain = run_tool(table, "trend", {"value": "売上", "time": "月", "period": "month"})
    assert "groups" not in plain and {k: v for k, v in out.items() if k in plain} == strict_json(plain)
    # a rate column (or a non-positive mean) is ranked by the slope itself, in its own unit
    rates = load_table({"月": [f"2024-{m:02d}" for m in range(1, 7)] * 2, "店舗": ["a"] * 6 + ["b"] * 6,
                        "率": [f"{v}%" for v in (10, 11, 12, 13, 14, 15, 50, 52, 54, 56, 58, 60)]})
    out = run_tool(rates, "trend", {"value": "率", "time": "月", "by": "店舗"})
    assert out["rank_by"] == "slope_per_period" and out["groups"][0]["key"] == "b"
    rows_only = run_tool(load_table({"g": list("ab") * 6, "v": list(range(12))}), "trend", {"value": "v", "by": "g"})
    assert rows_only["groups_trended"] == 2 and all(g["direction"] == "increasing" for g in rows_only["groups"])


def test_trend_by_drops_the_partial_periods_of_the_overall_series():
    days = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(0, 100)]
    table = load_table({"日付": [d.isoformat() for d in days] * 2, "店舗": ["a"] * 100 + ["b"] * 100,
                        "売上": [10 + i % 7 for i in range(100)] + [20 + i % 5 for i in range(100)]})
    out = run_tool(table, "trend", {"value": "売上", "time": "日付", "period": "month", "by": "店舗"})
    assert out["partial_periods"] == ["2024-04"] and out["groups_trended"] == 2
    assert all(g["last_period"] == "2024-03" and g["n"] == 3 for g in out["groups"])
    short = load_table({"日付": [d.isoformat() for d in days[:80]] * 2, "店舗": ["a"] * 80 + ["b"] * 80,
                        "売上": list(range(160))})
    out = run_tool(short, "trend", {"value": "売上", "time": "日付", "period": "month", "by": "店舗"})
    assert out["partial_periods"] == ["2024-03"] and out["n"] == 2
    assert out["groups_skipped"] == 2 and out["groups"] == [] and out["rank_by"] is None   # 3 points is the floor
    out = run_tool(table, "trend", {"value": "売上", "time": "日付", "by": "店舗"})
    assert out["groups_trended"] == 2 and all(g["n"] == 100 for g in out["groups"])


def test_per_group_tools_stay_fast_at_the_row_limit():
    rng = random.Random(5)
    n = T.LIMITS["max_rows"]
    days = [(datetime.date(2020, 1, 1) + datetime.timedelta(days=rng.randrange(1500))).isoformat() for _ in range(n)]
    for groups in (3, 1000):
        table = load_table({"日付": days, "店舗": [f"S{rng.randrange(groups)}" for _ in range(n)],
                            "売上": [rng.gauss(1000, 200) for _ in range(n)]})
        assert table.column("店舗").kind == "categorical"
        started = time.monotonic()
        for tool, args in [("outliers", {"column": "売上", "by": "店舗"}),
                           ("outliers", {"column": "売上", "by": "店舗", "method": "zscore"}),
                           ("compare", {"value": "売上", "by": "店舗"}),
                           ("trend", {"value": "売上", "time": "日付", "by": "店舗"}),
                           ("trend", {"value": "売上", "time": "日付", "by": "店舗", "period": "month"})]:
            strict_json(run_tool(table, tool, args))
        assert time.monotonic() - started < 5      # tools are not interruptible: the deadline is checked between them
    ids = load_table({"k": [f"id{i // 2}" for i in range(n)], "v": [rng.gauss(0, 1) for _ in range(n)]})
    started = time.monotonic()
    out = run_tool(ids, "compare", {"value": "v", "by": "k"})
    assert out["groups_tested"] == n // 2 and len(out["groups"]) == T.LIMITS["max_groups"] and out["truncated"]
    assert out["test"] == "anova" and out["welch_note"] == "small_groups" and time.monotonic() - started < 5


# ---------------------------------------------------------------- third review round

def weekdays(start, end):
    days = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]
    return [d.isoformat() for d in days if d.weekday() < 5]


def test_full_weekday_months_and_years_are_not_partial():
    for start, end, period in (((2024, 1, 1), (2024, 2, 29), "month"), ((2022, 1, 1), (2023, 12, 31), "year")):
        days = weekdays(datetime.date(*start), datetime.date(*end))
        out = run_tool(load_table({"d": days, "v": [100 + i for i in range(len(days))]}), "trend",
                       {"value": "v", "time": "d", "period": period})
        assert out["n"] == 2 and out["partial_periods"] == []
    for start, end, partial in (((2024, 1, 15), (2024, 2, 29), "2024-01"), ((2024, 1, 1), (2024, 2, 14), "2024-02")):
        days = weekdays(datetime.date(*start), datetime.date(*end))      # real cut-offs are still caught
        out = run_tool(load_table({"d": days, "v": [1] * len(days)}), "trend", {"value": "v", "time": "d", "period": "month"})
        assert out["partial_periods"] == [partial]


def test_a_sparse_first_month_covering_the_month_is_kept():
    jan = [4, 9, 12, 18, 22, 26]          # 6 dates spanning the month; March has only 5
    other = {2: [1, 5, 8, 10, 13, 16, 19, 22, 26, 28], 3: [2, 9, 15, 22, 29], 4: [1, 5, 9, 13, 18, 23, 26, 30],
             5: [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31], 6: [3, 6, 10, 13, 17, 20, 23, 25, 27, 29]}
    days = [datetime.date(2024, 1, d) for d in jan] + [datetime.date(2024, m, d) for m, ds in other.items() for d in ds]
    out = run_tool(load_table({"d": [d.isoformat() for d in days], "v": [1] * len(days)}), "trend",
                   {"value": "v", "time": "d", "period": "month"})
    assert out["partial_periods"] == [] and out["n"] == 6 and out["rows_used"] == len(days)


def test_year_columns_need_a_year_name_not_just_the_character():
    table = load_table({"部署": list("ABAB"), "年収": [3500, 9500, 5200, 7100], "年商": [2000, 2100, 2050, 2400],
                        "yearly_sales": [2020, 2021, 2022, 2023], "会計年度": [2020, 2021, 2022, 2023],
                        "fiscal_year": [2020, 2021, 2022, 2023], "入社年": [2001, 2010, 2015, 2020]})
    assert not any(table.column(n).is_year for n in ("年収", "年商", "yearly_sales"))
    assert all(table.column(n).is_year for n in ("会計年度", "fiscal_year", "入社年"))


def test_one_stray_percent_cell_does_not_relabel_a_plain_column():
    table = load_table({"region": list("東西南北東西南北"), "sales": [1200, 1300, 900, 950, 700, 650, 400, "5%"]})
    col = table.column("sales")
    assert col.kind == "numeric" and col.unit is None and col.coerced == 1 and col.values[-1] is None
    assert "unit" not in run_tool(table, "group_by", {"by": "region", "value": "sales"})
    rates = load_table({"r": ["5%", "6%", "7%", 8]})       # a unit carried by the majority still applies
    assert rates.column("r").unit == "%" and rates.column("r").values[-1] == 8.0


def test_fiscal_years_bucket_april_to_march():
    months = [f"{2020 + (3 + i) // 12}-{(3 + i) % 12 + 1:02d}-01" for i in range(48)]
    table = load_table({"月": months, "売上": [1000 + 10 * i for i in range(48)]})
    out = run_tool(table, "trend", {"value": "売上", "time": "月", "period": "year", "fiscal_start": 4})
    assert out["n"] == 4 and out["partial_periods"] == [] and out["rows_used"] == 48
    assert (out["first_period"], out["last_period"]) == ("2020年度", "2023年度")
    calendar_years = run_tool(table, "trend", {"value": "売上", "time": "月", "period": "year"})
    assert calendar_years["partial_periods"] == ["2020", "2024"]
    fc = run_tool(table, "forecast", {"value": "売上", "time": "月", "period": "year", "fiscal_start": 4})
    assert [p["period"] for p in fc["points"]] == ["2024年度", "2025年度", "2026年度"]
    assert "fiscal_start" not in validate_args(table, "trend", {"value": "売上", "time": "月", "period": "year",
                                                                "fiscal_start": 1})
    with pytest.raises(DataError):
        validate_args(table, "trend", {"value": "売上", "time": "月", "fiscal_start": 4})


def test_a_store_opening_mid_month_drops_its_own_partial_month():
    rows, day = {"日付": [], "店舗": [], "売上": []}, datetime.date(2024, 1, 1)
    rng = random.Random(2)
    while day <= datetime.date(2024, 6, 30):
        for store in ("A", "B") + (("C",) if day >= datetime.date(2024, 3, 20) else ()):
            rows["日付"].append(day.isoformat())
            rows["店舗"].append(store)
            rows["売上"].append(round(rng.gauss(100, 5)))
        day += datetime.timedelta(days=1)
    out = run_tool(load_table(rows), "trend", {"value": "売上", "time": "日付", "period": "month", "by": "店舗"})
    c = next(g for g in out["groups"] if g["key"] == "C")
    assert c["first_period"] == "2024-04" and c["partial_periods"] == ["2024-03"] and abs(c["slope_pct"]) < 2
    assert out["groups"][0]["key"] != "C" and all(g["partial_periods"] == [] for g in out["groups"] if g["key"] != "C")


def test_amount_names_with_month_are_not_dotted_year_months():
    for name in ("月間売上", "monthly_sales", "月額料金"):
        table = load_table(f"店舗,{name}\nA,1980.5\nB,2010.3\nC,1995.7\nD,2050.1\nE,2003.2\n")
        assert table.column(name).kind == "numeric"
    assert load_table("対象月,v\n2024.1,1\n2024.2,2\n2024.3,3\n").column("対象月").kind == "datetime"


def test_forecast_calendar_steps_use_the_usual_day_and_4_weekly_data_steps_by_days():
    days = ["2023-09-30", "2023-10-30", "2023-11-30", "2023-12-30", "2024-01-30", "2024-02-29"]
    fc = run_tool(load_table({"d": days, "v": [1, 2, 3, 4, 5, 7]}), "forecast", {"value": "v", "time": "d"})
    assert [p["period"] for p in fc["points"]] == ["2024-03-30", "2024-04-30", "2024-05-30"]
    days = [(datetime.date(2024, 1, 1) + datetime.timedelta(days=28 * i)).isoformat() for i in range(13)]
    fc = run_tool(load_table({"d": days, "v": [100 + i for i in range(13)]}), "forecast",
                  {"value": "v", "time": "d", "periods": 12})
    assert fc["points"][-1]["period"] == "2025-11-03" and close(fc["points"][-1]["estimate"], 124.0, abs_=1e-6)


def test_previous_period_change_skips_a_missing_month():
    months = [f"2024-{m:02d}" for m in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)]
    out = run_tool(load_table({"月": months, "v": [100 + 10 * i for i in range(10)] + [300]}), "trend",
                   {"value": "v", "time": "月", "period": "month"})
    assert out["last_periods"][-1]["period"] == "2024-12" and out["last_periods"][-1]["pct_change"] is None
    assert out["last_periods"][-2]["pct_change"] is not None
    quarters = [f"{2022 + i // 4}-{3 * (i % 4) + 1:02d}-01" for i in range(8)]   # the usual gap here is 3 months
    q = run_tool(load_table({"d": quarters, "v": list(range(100, 180, 10))}), "trend", {"value": "v", "time": "d", "period": "month"})
    assert q["last_periods"][-1]["pct_change"] is not None and q["period_step"] >= 89
    years = run_tool(load_table({"年": [2018, 2019, 2020, 2022], "v": [1, 2, 3, 4]}), "trend", {"value": "v", "time": "年"})
    assert [p["pct_change"] is None for p in years["last_periods"]] == [True, False, False, True]


def test_no_percent_change_from_a_negative_base():
    table = load_table({"期": [f"{2020 + i // 4}-{3 * (i % 4) + 3:02d}-01" for i in range(8)],
                        "部門": ["a", "b"] * 4, "営業利益": ["▲125", "40", "▲60", "45", "10", "52", "120", "60"]})
    out = run_tool(table, "trend", {"value": "営業利益", "time": "期"})
    assert out["pct_change"] is None and out["change"] > 0
    by = run_tool(table, "trend", {"value": "営業利益", "time": "期", "by": "部門"})
    assert by["rank_by"] == "slope_per_period" and next(g for g in by["groups"] if g["key"] == "a")["slope_pct"] is None
    cmp = run_tool(table, "compare", {"value": "営業利益", "by": "部門", "a": "b", "b": "a"})
    assert cmp["pct_diff"] is None and cmp["diff"] > 0


def test_trailing_total_rows_are_excluded():
    table = load_table("店舗,売上,客数\n新宿,1200,3400\n渋谷,980,2900\n池袋,1100,3100\n合計,3280,9400\n")
    assert table.n_rows == 3 and table.dropped_rows == [(4, "合計")]
    assert "合計" not in table.column("店舗").values
    kept = load_table("店舗,売上\n新宿,1200\n渋谷,980\n合計,100\n")    # not the sum: a store named 合計 stays
    assert kept.n_rows == 3 and kept.dropped_rows == []
    two = load_table("店舗,売上\nA,10\nB,20\n小計,30\nTotal,30\n")
    assert two.n_rows == 2 and [r for r, _ in two.dropped_rows] == [3, 4]


def test_fiscal_year_strings_are_dates():
    assert T.parse_date("2016年度") == datetime.date(2016, 4, 1) == T.parse_date("FY2016")
    assert T.parse_date("2016年3月期") == datetime.date(2016, 3, 1)
    table = load_table({"年度": [f"{y}年度" for y in range(2016, 2025)], "来場者数": list(range(995, 1400, 45))})
    assert table.column("年度").kind == "datetime"


def test_top_n_labels_rows_by_their_date_without_a_name_column():
    table = load_table({"決算期": ["2024-03-31", "2024-06-30", "2024-09-30"], "営業利益": [195, 223, 262]})
    out = run_tool(table, "top_n", {"column": "営業利益", "n": 1})
    assert out["label_column"] == "決算期" and out["rows"][0]["label"] == "2024-09-30"
    years = run_tool(load_table({"年": [2022, 2023, 2024], "v": [3, 1, 2]}), "top_n", {"column": "v", "n": 1})
    assert years["rows"][0]["label"] == 2022


def test_boolean_columns_keep_their_own_words():
    col = load_table({"離職意向": ["はい", "いいえ", "いいえ", "はい"]}).column("離職意向")
    assert col.kind == "boolean" and col.bool_labels == {True: "はい", False: "いいえ"}
