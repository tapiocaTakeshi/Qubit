"""Bounded analyst SFT: planner decisions and grounded narration with answer-only loss.

Validation is the default; --train explicitly trains a separate candidate from --model-dir.
This never replaces/syncs the serving checkpoint and never downloads a model/dataset.
Prompts are rebuilt with qubit_analyst's runtime builders and every target passes the
inference validators: plan targets parse with parse_plan_decision and never repeat a done
step; narration targets pass the numeric guard against the exact narration prompt (the only
numbers the runtime accepts), and every prompt leaves room for its runtime generation length. External JSONL rows use the same contract: table id, question, language, data,
stage, steps already run (plus remaining for plan rows) and target.
"""
import argparse
import datetime
import hashlib
import json
import random
import types
from collections import Counter
from pathlib import Path

import qubit_analyst as A
import qubit_analyst_tools as T
import train_agent
from neuroquantum_agent_protocol import strict_json
from train_agent import load_training_model

STAGES = ("plan", "narrate")
ROW_KEYS = {"plan": {"table", "question", "language", "data", "stage", "steps", "remaining", "target"},
            "narrate": {"table", "question", "language", "data", "stage", "steps", "target"}}
MAX_ROW_STEPS = A.MAX_PLAN + A.MAX_MODEL_STEPS
MAX_NEW_TOKENS = A.NARRATIVE_NEW_TOKENS    # handler._handle_analyst reserves A.generation_tokens(prompt)
PLANNER_STEPS = 3         # runtime default max_steps, so the first decision sees remaining=3
NARRATE_CHARS = 480       # keeps synthetic narration inside MAX_NEW_TOKENS (<=0.65 tokens/char)
TABLES, QUESTIONS_PER_TABLE = 40, 5


# ---------------------------------------------------------------- synthetic tables

JA_REGIONS = ("東京", "大阪", "名古屋", "福岡", "札幌", "仙台")
EN_REGIONS = ("North", "South", "East", "West", "Central")
STORES = ("渋谷店", "新宿店", "池袋店", "横浜店")
PRODUCTS = {"食品": ("お茶", "コーヒー", "パン", "牛乳", "弁当"),
            "日用品": ("洗剤", "シャンプー", "タオル", "ハンドソープ"),
            "衣料": ("靴下", "Tシャツ", "帽子"), "家電": ("電池", "イヤホン", "充電器")}
AGES = ("20代", "30代", "40代", "50代")
CHANNELS = ("search", "social", "email")


def _columns(names, rows):
    return {name: [row[i] for row in rows] for i, name in enumerate(names)}


def _pick(rng, *synonyms):
    """One name per column role; synonyms vary the schema so the model must copy exact names."""
    return tuple(rng.choice(options) for options in synonyms)


def _months(rng, count):
    start = rng.randrange(2020 * 12, 2024 * 12)
    return [f"{m // 12}-{m % 12 + 1:02d}" for m in range(start, start + count)]


def _days(rng, count):
    start = datetime.date(2024, 1, 1) + datetime.timedelta(days=rng.randrange(300))
    return [start + datetime.timedelta(days=i) for i in range(count)]


def _regional(rng, names, regions):
    """Monthly sales by region: planted growth, ad-spend effect and sometimes one spike."""
    months, picked = _months(rng, rng.randint(12, 24)), rng.sample(regions, rng.randint(3, 4))
    growth, effect = rng.choice((0.03, 0.02, -0.015, 0.0)), rng.choice((0.0, 3.0, 6.0))
    rows = []
    for i, month in enumerate(months):
        for j, region in enumerate(picked):
            ad = round(80 + 4 * i + 20 * j + rng.gauss(0, 12))
            sales = round((900 + 250 * j) * (1 + growth * i) + effect * ad + rng.gauss(0, 80))
            rows.append([month, region, sales, ad, round(sales / rng.uniform(4, 6))])
    if rng.random() < 0.5:
        rows[rng.randrange(len(rows))][2] *= 3
    return _columns(names, rows), dict(zip("TGMDC", names))


def _regional_ja(rng):
    return _regional(rng, _pick(rng, ("月", "年月"), ("地域", "エリア"), ("売上", "売上高", "販売額"),
                                ("広告費", "広告宣伝費"), ("客数", "来店客数")), JA_REGIONS)


def _regional_en(rng):
    return _regional(rng, _pick(rng, ("month", "period"), ("region", "area"), ("sales", "revenue"),
                                ("ad_spend", "marketing"), ("visitors", "footfall")), EN_REGIONS)


def _stores(rng):
    """Daily store sales: weekend traffic, a store price gap, sales = visitors x unit price."""
    names = _pick(rng, ("日付", "営業日"), ("店舗", "店名"), ("客数", "来店客数"), ("単価", "客単価"),
                  ("売上", "売上高"))
    days, stores = _days(rng, rng.randint(28, 40)), rng.sample(STORES, rng.randint(2, 3))
    gap = rng.choice((0, 60, 150))
    rows = []
    for i, day in enumerate(days):
        for j, store in enumerate(stores):
            visitors = round(200 + 70 * (day.weekday() >= 5) + 25 * j + 1.5 * i + rng.gauss(0, 20))
            price = round(900 + gap * j + rng.gauss(0, 40))
            rows.append([day.isoformat(), store, visitors, price, visitors * price])
    return _columns(names, rows), dict(zip("TGDPM", names))


def _products(rng):
    """One row per product: pricier products sell fewer units."""
    names = _pick(rng, ("カテゴリ", "分類"), ("商品", "商品名"), ("売上", "売上金額"), ("販売数", "販売個数"),
                  ("単価", "価格"))
    items = rng.sample([(c, p) for c, ps in PRODUCTS.items() for p in ps], rng.randint(12, 15))
    base = {"食品": 300, "日用品": 600, "衣料": 1500, "家電": 2500}
    rows = []
    for category, product in items:
        price = round(base[category] * rng.uniform(0.7, 1.4), -1)
        units = max(5, round(9000 / price ** 0.8 * rng.uniform(0.6, 1.4)))
        rows.append([category, product, int(price * units), units, int(price)])
    return _columns(names, rows), dict(zip("GLMQP", names))


def _yearly(rng):
    """Annual results: steady growth, sometimes one bad year; headcount follows sales."""
    names = _pick(rng, ("年", "年度"), ("売上高", "売上"), ("営業利益", "経常利益"), ("従業員数", "社員数"))
    start, count = rng.randint(2008, 2013), rng.randint(10, 14)
    sales, staff, dip = rng.randint(800, 1500), rng.randint(40, 90), rng.choice((None, 2020))
    rows = []
    for year in range(start, start + count):
        sales = round(sales * (rng.uniform(1.02, 1.12) if year != dip else 0.8))
        staff = round(staff * rng.uniform(1.0, 1.08))
        rows.append([year, sales, round(sales * rng.uniform(0.05, 0.12)), staff])
    return _columns(names, rows), dict(zip("TMOE", names))


def _campaign(rng):
    """A/B campaign rows: B lifts sales by a random (possibly zero) effect."""
    names = _pick(rng, ("施策", "キャンペーン"), ("地域", "エリア"), ("会員", "会員区分"), ("売上", "購入金額"),
                  ("客数", "来店数"))
    regions, lift = rng.sample(JA_REGIONS, 3), rng.choice((0, 300, 800))
    rows = []
    for _ in range(rng.randint(40, 56)):
        group, member = rng.choice("AB"), rng.random() < 0.4
        visitors = round(rng.gauss(120, 25))
        sales = round(visitors * 25 + lift * (group == "B") + 400 * member + rng.gauss(0, 500))
        rows.append([group, rng.choice(regions), "はい" if member else "いいえ", sales, visitors])
    return _columns(names, rows), dict(zip("XGBMC", names))


def _survey(rng):
    """Survey answers: more use, higher satisfaction; satisfied users tend to continue."""
    names = _pick(rng, ("年代", "年齢層"), ("性別",), ("満足度", "満足度スコア"), ("利用回数", "利用頻度"),
                  ("継続", "継続意向"))
    rows = []
    for _ in range(rng.randint(40, 64)):
        uses = max(0, round(rng.gauss(8, 4)))
        score = min(5, max(1, round(2 + uses / 5 + rng.gauss(0, 0.8))))
        rows.append([rng.choice(AGES), rng.choice(("男性", "女性")), score, uses,
                     "はい" if rng.random() < 0.2 + score * 0.14 else "いいえ"])
    return _columns(names, rows), dict(zip("AGSUK", names))


def _web(rng):
    """Daily web traffic per channel: revenue follows sessions."""
    names = _pick(rng, ("date", "day"), ("channel", "source"), ("sessions", "visits"), ("conversions", "orders"),
                  ("revenue", "sales"))
    days, rate = _days(rng, rng.randint(28, 36)), {c: rng.uniform(0.01, 0.05) for c in CHANNELS}
    rows = []
    for i, day in enumerate(days):
        for channel in CHANNELS:
            sessions = round(rng.gauss(600, 120) + 8 * i)
            conversions = round(sessions * rate[channel] + rng.gauss(0, 3))
            rows.append([day.isoformat(), channel, sessions, max(0, conversions),
                         max(0, conversions) * rng.randint(40, 60)])
    return _columns(names, rows), dict(zip("TGSCR", names))


# Questions with the analyses a careful analyst would run; anything the rule plan already
# covers becomes {"status":"complete"}, the rest become "continue" targets in order.
def _q(question, *ideal):
    return question, list(ideal)


TREND_M = ("trend", {"value": "{M}", "time": "{T}", "period": "month"})
ARCHETYPES = [
    (_regional_ja, "ja", [
        _q("{M}の推移を教えて", TREND_M),
        _q("広告を増やすと{M}は伸びる？", ("correlate", {"x": "{D}", "y": "{M}"})),
        _q("{G}ごとの{M}を比べたい", ("group_by", {"by": "{G}", "value": "{M}"}),
           ("compare", {"value": "{M}", "by": "{G}"})),
        _q("{G}によって{M}は違う？", ("compare", {"value": "{M}", "by": "{G}"})),
        _q("来期の{M}を予測して", ("forecast", {"value": "{M}", "time": "{T}", "period": "month"})),
        _q("{M}が最も多い{G}は？", ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("{M}に異常値はある？", ("outliers", {"column": "{M}"})),
        _q("{M}の月別推移と{G}別の内訳", TREND_M, ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("{G}別の{M}の推移を比べて", ("trend", {"value": "{M}", "time": "{T}", "period": "month", "by": "{G}"})),
        _q("{G}ごとに{M}の異常値を調べて", ("outliers", {"column": "{M}", "by": "{G}"})),
        _q("{G}によって{M}の伸び方は違う？",
           ("trend", {"value": "{M}", "time": "{T}", "period": "month", "by": "{G}"})),
        _q("このデータから何が言える？"),
    ]),
    (_stores, "ja", [
        _q("{M}の推移は？", ("trend", {"value": "{M}", "time": "{T}"})),
        _q("客が多い日ほど売上も多い？", ("correlate", {"x": "{D}", "y": "{M}"})),
        _q("{G}別の{M}", ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("{G}で{P}に違いはある？", ("compare", {"value": "{P}", "by": "{G}"})),
        _q("{M}の上位5日は？", ("group_by", {"by": "{T}", "value": "{M}"})),     # several stores a day: day totals
        _q("{M}の外れ値を調べて", ("outliers", {"column": "{M}"})),
        _q("{G}ごとの{M}の外れ値", ("outliers", {"column": "{M}", "by": "{G}"})),
        _q("{G}の中でどこが一番稼いでいる？", ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("客単価は{G}で違う？", ("compare", {"value": "{P}", "by": "{G}"})),
    ]),
    (_products, "ja", [
        _q("{M}が多い{L}トップ5", ("top_n", {"column": "{M}", "n": 5, "label": "{L}"})),
        _q("{G}別の{M}の構成比", ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("値段が高いと売れにくい？", ("correlate", {"x": "{P}", "y": "{Q}"})),
        _q("{P}と{Q}の相関", ("correlate", {"x": "{P}", "y": "{Q}"})),
        _q("{M}の分布を要約して", ("describe", {"column": "{M}"})),
        _q("どの{G}が一番売れている？", ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("高い商品ほど売上も大きい？", ("correlate", {"x": "{P}", "y": "{M}"})),
    ]),
    (_yearly, "ja", [
        _q("{M}の成長率は？", ("trend", {"value": "{M}", "time": "{T}"})),
        _q("今後3年の{M}を予測", ("forecast", {"value": "{M}", "time": "{T}", "periods": 3})),
        _q("人を増やすと{M}も増える？", ("correlate", {"x": "{E}", "y": "{M}"})),
        _q("{M}と{O}の推移", ("trend", {"value": "{M}", "time": "{T}"}),
           ("trend", {"value": "{O}", "time": "{T}"})),
        _q("{O}が最も高かった年は？", ("top_n", {"column": "{O}", "label": "{T}"})),
        _q("従業員が多い年ほど利益も大きい？", ("correlate", {"x": "{E}", "y": "{O}"})),
    ]),
    (_campaign, "ja", [
        _q("{X}AとBで{M}に差はある？", ("compare", {"value": "{M}", "by": "{X}", "a": "A", "b": "B"})),
        _q("{G}別の{M}", ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("どちらの施策が効果的？", ("compare", {"value": "{M}", "by": "{X}"})),
        _q("{X}と{B}の関係をクロス集計して", ("crosstab", {"row": "{X}", "col": "{B}"})),
        _q("会員と非会員で{M}は違う？", ("compare", {"value": "{M}", "by": "{B}"})),
        _q("Bの施策はAより{M}が高い？", ("compare", {"value": "{M}", "by": "{X}", "a": "B", "b": "A"})),
    ]),
    (_survey, "ja", [
        _q("{A}別の{S}の平均", ("group_by", {"by": "{A}", "value": "{S}", "agg": "mean"})),
        _q("男女で{S}に違いはある？", ("compare", {"value": "{S}", "by": "{G}"})),
        _q("{U}が多い人ほど{S}は高い？", ("correlate", {"x": "{U}", "y": "{S}", "method": "spearman"})),
        _q("{A}と{K}の関係", ("crosstab", {"row": "{A}", "col": "{K}"})),
        _q("{S}の分布", ("describe", {"column": "{S}"})),
        _q("続けている人は{U}が多い？", ("compare", {"value": "{U}", "by": "{K}"})),
    ]),
    (_regional_en, "en", [
        _q("How did {M} change over time?", TREND_M),
        _q("Does ad spend drive {M}?", ("correlate", {"x": "{D}", "y": "{M}"})),
        _q("Which {G} has the highest {M}?", ("group_by", {"by": "{G}", "value": "{M}"})),
        _q("Compare {M} between regions", ("compare", {"value": "{M}", "by": "{G}"})),
        _q("Are there unusual values in {C}?", ("outliers", {"column": "{C}"})),
        _q("What should we expect for {M} next quarter?",
           ("forecast", {"value": "{M}", "time": "{T}", "period": "month"})),
        _q("Do regions differ in {C}?", ("compare", {"value": "{C}", "by": "{G}"})),
        _q("How has {M} trended by {G}?", ("trend", {"value": "{M}", "time": "{T}", "period": "month", "by": "{G}"})),
        _q("Any unusual {M} values within each {G}?", ("outliers", {"column": "{M}", "by": "{G}"})),
    ]),
    (_web, "en", [
        _q("Which {G} brings the most {R}?", ("group_by", {"by": "{G}", "value": "{R}"})),
        _q("Does traffic turn into money?", ("correlate", {"x": "{S}", "y": "{R}"})),
        _q("What are the 5 highest {R} days?", ("group_by", {"by": "{T}", "value": "{R}"})),
        _q("Are {C} different across channels?", ("compare", {"value": "{C}", "by": "{G}"})),
        _q("How is {R} trending?", ("trend", {"value": "{R}", "time": "{T}"})),
        _q("Does social bring more {C} than email?",
           ("compare", {"value": "{C}", "by": "{G}", "a": "social", "b": "email"})),
    ]),
]


def _fill(value, roles):
    if isinstance(value, str):
        return value.format(**roles)
    if isinstance(value, dict):
        return {k: _fill(v, roles) for k, v in value.items()}
    return value


# ---------------------------------------------------------------- records

def _replay(row):
    """Validated (question, language, table, steps); steps run exactly as the controller runs them."""
    stage = row.get("stage") if isinstance(row, dict) else None
    if stage not in STAGES:
        raise ValueError("Invalid stage")
    if set(row) != ROW_KEYS[stage]:
        raise ValueError("Unexpected or missing row fields")
    if not isinstance(row["table"], str) or not 0 < len(row["table"]) <= 80:
        raise ValueError("Invalid table id")
    question, data, params = A.validate_request({"inputs": row["question"], "parameters": {
        "data": row["data"], "language": row["language"]}})
    table = T.load_table(data)
    if not isinstance(row["steps"], list) or len(row["steps"]) > MAX_ROW_STEPS:
        raise ValueError("Invalid steps")
    steps, seen = [], set()
    for item in row["steps"]:
        if not isinstance(item, dict) or set(item) != {"tool", "arguments"}:
            raise ValueError('Each step must be {"tool": name, "arguments": {...}}')
        args = T.validate_args(table, item["tool"], item["arguments"])
        if A._signature(item["tool"], args) in seen:
            raise ValueError("Repeated step; the controller never runs a call twice")
        seen.add(A._signature(item["tool"], args))
        step = {"call_id": f"call_{len(steps) + 1}", "tool": item["tool"], "arguments": args,
                "status": "completed"}
        try:
            step["output"] = T.TOOLS[item["tool"]](table, args)
        except T.DataError as exc:
            step.update(status="failed", output={"error": str(exc)})
        steps.append(step)
    return question, params["language"], table, steps


def compile_record(row):
    """Validate a row and return the exact runtime (prompt, target) pair."""
    question, language, table, steps = _replay(row)
    if row["stage"] == "plan":
        remaining = row["remaining"]
        if type(remaining) is not int or not 1 <= remaining <= A.MAX_MODEL_STEPS:
            raise ValueError("Invalid remaining")
        if not isinstance(row["target"], dict):
            raise ValueError("Plan target must be a decision object")
        target = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
        status, tool, args = A.parse_plan_decision(target, table)
        if status == "continue" and A._signature(tool, args) in {
                A._signature(s["tool"], s["arguments"]) for s in steps}:
            raise ValueError("Plan target repeats a done step")
        return A.planner_prompt(question, table, steps, remaining, language), target
    target = row["target"]
    if not isinstance(target, str) or target != target.strip() or A._narrative_problem(target):
        raise ValueError("Invalid narration target")
    findings = A.build_findings(steps, table, language=language)
    caveats = A.build_caveats(steps, table, language=language, question=question)
    prompt = A.narrative_prompt(question, findings, caveats, language)
    if A.verify_numbers(target, prompt):
        raise ValueError("Narration target contains numbers not found in the narration prompt")
    return prompt, target


def _minimal_args(table, tool, args):
    """Drop arguments whose omission normalises to the same call (defaults the runtime fills)."""
    kept = dict(args)
    for key in args:
        trial = {k: v for k, v in kept.items() if k != key}
        try:
            if T.validate_args(table, tool, trial) == args:
                kept = trial
        except T.DataError:
            pass
    return kept


def _question_records(base, table, ideal):
    rule = [{"tool": p["tool"], "arguments": p["arguments"]} for p in A.rule_plan(base["question"], table)]
    seen, missing = {A._signature(s["tool"], s["arguments"]) for s in rule}, []
    for tool, args in ideal:
        args = T.validate_args(table, tool, args)
        if A._signature(tool, args) not in seen:
            seen.add(A._signature(tool, args))
            missing.append({"tool": tool, "arguments": args})
    rows = []
    for k in range(min(len(missing) + 1, PLANNER_STEPS)):  # the controller stops at max_steps
        target = {"status": "complete"} if k == len(missing) else {
            "status": "continue", "action": missing[k]["tool"],
            "arguments": _minimal_args(table, missing[k]["tool"], missing[k]["arguments"])}
        rows.append({**base, "stage": "plan", "steps": rule + missing[:k],
                     "remaining": PLANNER_STEPS - k, "target": target})
    row = {**base, "stage": "narrate", "steps": rule + missing[:PLANNER_STEPS], "target": ""}
    question, language, table, steps = _replay(row)
    if any(s["status"] != "completed" for s in steps):
        raise ValueError(f"Curriculum step failed for {question!r}")
    findings = A.build_findings(steps, table, language=language)
    caveats = A.build_caveats(steps, table, language=language, question=question)
    row["target"] = A.template_narrative(question, findings, caveats, A.dataset_summary(table), language)
    prompt = A.narrative_prompt(question, findings, caveats, language)
    # Longer reports exceed the generation budget; a template citing a line the prompt had to drop
    # would teach the model to state numbers it was not shown.
    if len(row["target"]) <= NARRATE_CHARS and not A.verify_numbers(row["target"], prompt):
        rows.append(row)
    return rows


def bootstrap_records():
    """Seeded synthetic curriculum (tables with planted trends, correlations, gaps and outliers).

    Not a claim of broad analytic competence: it teaches the JSON planning contract, exact
    column names, when to stop, and number-faithful narration of computed findings.
    """
    rng = random.Random(42)
    records = []
    for index in range(TABLES):
        build, language, templates = ARCHETYPES[index % len(ARCHETYPES)]
        data, roles = build(rng)
        table = T.load_table(data)
        for question, ideal in rng.sample(templates, min(len(templates), QUESTIONS_PER_TABLE)):
            base = {"table": f"t{index + 1:02d}", "question": question.format(**roles),
                    "language": language, "data": data}
            records += _question_records(base, table, [(tool, _fill(args, roles)) for tool, args in ideal])
    return records


def split_records(records):
    """Keep every stage of a (table, question) group in the same split; no post-split replication."""
    train, validation, seen = [], [], set()
    for row in records:
        prompt, target = compile_record(row)
        if (prompt, target) in seen:
            continue
        seen.add((prompt, target))
        group = f"{row['table']}\n{row['question'].strip()}"
        bucket = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 5
        (validation if bucket == 0 else train).append(row)
    if not train or not validation:
        raise ValueError("Need distinct training and validation question groups")
    return train, validation


def encode_record(row, tokenizer, max_length):
    """train_agent's layout and answer-only labels. Mirrors the serving handler: the prompt must leave
    room for the stage's generation length (A.generation_tokens: 96 for plans, 320 for narration) or
    the handler would refuse it, and the answer plus EOS must fit in that length."""
    prompt, answer = compile_record(row)
    prefix = [tokenizer.bof_id, tokenizer.bos_id] + tokenizer.encode(
        f"質問: {prompt}\n回答:", add_special=False)
    target = tokenizer.encode(answer, add_special=False) + [tokenizer.eos_id, tokenizer.eof_id]
    new_tokens = A.generation_tokens(prompt)
    if len(prefix) + len(target) > max_length:
        raise ValueError("Training example exceeds context window; do not silently truncate")
    if len(prefix) + new_tokens > max_length:     # handler: token_count + 2 + max_new_tokens > max_seq_len
        raise ValueError("Training prompt leaves no room for the runtime generation length; "
                         "the serving handler would refuse it")
    if len(target) - 1 > new_tokens:
        raise ValueError("Training answer exceeds the runtime generation length; do not silently truncate")
    return prefix + target, [-100] * len(prefix) + target


def train_candidate(handler, train, validation, *, epochs, max_steps, lr):
    """train_agent.train_candidate (seed 42, AdamW, clipping, held-out loss) run against this
    module's encode_record; the shared loop is rebound, not copied or monkeypatched."""
    loop = types.FunctionType(train_agent.train_candidate.__code__,
                              {**vars(train_agent), "encode_record": encode_record})
    return loop(handler, train, validation, epochs=epochs, max_steps=max_steps, lr=lr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Optional normalized JSONL; default synthetic curriculum")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 3 or not 1 <= args.max_steps <= 200 or not 0 < args.lr <= 1e-4:
        parser.error("epochs 1..3, max-steps 1..200 and lr (0, 1e-4] required")
    if args.train and not args.model_dir:
        parser.error("--train requires --model-dir containing the existing checkpoint and tokenizer")
    if args.dataset:
        if args.dataset.stat().st_size > 20_000_000:
            parser.error("Dataset exceeds 20 MB")
        records = [strict_json(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = bootstrap_records()
    if len(records) > 10000:
        parser.error("Dataset exceeds 10000 records")
    train, validation = split_records(records)
    report = {"trained": False, "source": str(args.dataset or "synthetic-analyst-bootstrap-v1"),
              "train_records": len(train), "validation_records": len(validation),
              "stages": dict(Counter(row["stage"] for row in train + validation)),
              "processed_samples": 0, "analyst_version": A.VERSION, "seed": 42}
    # Exclusive directory creation prevents accidental overwriting of model artifacts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", train), ("validation", validation)):
        with (args.output_dir / f"{name}.jsonl").open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.train:
        import torch
        handler = load_training_model(args.model_dir)
        report.update(train_candidate(handler, train, validation,
                                      epochs=args.epochs, max_steps=args.max_steps, lr=args.lr))
        report["trained"] = True
        report["tokenizer_sha256"] = hashlib.sha256(
            (args.model_dir / "neuroq_tokenizer.model").read_bytes()).hexdigest()
        torch.save({"model_state": handler.model.state_dict(), "config": handler.config,
                    "analyst_training": report, "source_checkpoint": handler.ckpt_path},
                   args.output_dir / "analyst_candidate.pt")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
