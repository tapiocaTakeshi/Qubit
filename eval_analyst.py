"""Model-in-the-loop evaluation of Qubit Analyst planning and narration on the serving path.

Read-only: loads an explicit model directory through handler.EndpointHandler (never the serving
volume), generates with EndpointHandler._analyst_generate exactly as action "analyst" does, and
writes only a new report file. Three suites:
  plan      held-out curriculum planner prompts -> valid JSON / same decision as the target
  narrate   held-out curriculum narration prompts -> accepted by the guard / similarity to target
  scenario  fresh tables and questions run end to end through EndpointHandler._handle_analyst
Accepted narratives are only those the runtime would publish; "accepted" is not "well written".
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import qubit_analyst as A
import train_analyst as TA

SAMPLES = 5             # raw outputs kept per suite for manual reading (clipped)
CLIP = 300


# ---------------------------------------------------------------- fresh scenarios (not in the curriculum)

def _csv(header, rows):
    return "\n".join([",".join(header)] + [",".join(str(v) for v in row) for row in rows])


def scenarios(seed=7):
    rng = random.Random(seed)
    weeks = [f"2025-{1 + i // 4:02d}-{1 + 7 * (i % 4):02d}" for i in range(24)]
    shops = [(w, s, round(80 + 4 * i + (25 if s == "梅田店" else 0) + rng.gauss(0, 6)), round(5 + i * 0.8 + rng.gauss(0, 2), 1))
             for i, w in enumerate(weeks) for s in ("梅田店", "難波店", "京都店")]
    shop_csv = _csv(["週", "店舗", "売上（万円）", "気温"], shops)
    depts = [(d, round(rng.gauss(base, 6), 1), round(rng.gauss(4.2 - base / 40, 0.4), 2))
             for d, base in (("営業", 38), ("開発", 30), ("総務", 18), ("人事", 22)) for _ in range(9)]
    dept_csv = _csv(["部署", "残業時間", "満足度"], depts)
    days = [f"2025-03-{d:02d}" for d in range(1, 29)]
    mkt = [(day, ch, round(spend), round(spend * k + rng.gauss(0, 8)))
           for i, day in enumerate(days) for ch, k in (("search", 0.9), ("social", 0.5))
           for spend in [120 + 3 * i + rng.gauss(0, 10)]]
    mkt_csv = _csv(["date", "channel", "spend", "signups"], mkt)
    years = [(f"{2015 + i}年度", 1200 + 85 * i + round(rng.gauss(0, 30)), round(60 + 9 * i + rng.gauss(0, 12)))
             for i in range(10)]
    fy_csv = _csv(["年度", "売上高(百万円)", "営業利益(百万円)"], years)
    ab = [(g, int(rng.random() < p), round(rng.gauss(3000, 400)) if rng.random() < p else 0)
          for g, p in (("A", 0.08), ("B", 0.13)) for _ in range(150)]
    ab_csv = _csv(["group", "converted", "revenue"], ab)
    ship = [(c, round(max(1, rng.gauss(m, 0.8)), 1)) for c, m in (("Yamato", 2.1), ("Sagawa", 2.6), ("JP", 3.4)) for _ in range(20)]
    ship_csv = _csv(["carrier", "delivery_days"], ship)
    return [
        ("shop-trend", shop_csv, "店舗別の売上の推移を教えて"),
        ("shop-temp", shop_csv, "気温と売上の関係は？"),
        ("dept-compare", dept_csv, "部署によって残業時間に差はある？"),
        ("dept-corr", dept_csv, "残業時間と満足度の関係を教えて"),
        ("mkt-channel", mkt_csv, "Which channel brings more signups?"),
        ("mkt-trend", mkt_csv, "How did spend change over March?"),
        ("fy-growth", fy_csv, "売上高の成長率と来年度の見通しは？"),
        ("fy-profit", fy_csv, "営業利益は増えている？"),
        ("ab-test", ab_csv, "AとBでコンバージョン率に差はある？"),
        ("ab-revenue", ab_csv, "Compare revenue between A and B"),
        ("ship-slowest", ship_csv, "Which carrier has the longest delivery time?"),
        ("ship-outliers", ship_csv, "配送日数に異常値はある？"),
    ]


# ---------------------------------------------------------------- metrics

def _bigrams(text):
    return Counter(text[i:i + 2] for i in range(len(text) - 1))


def similarity(text, reference):
    """Character-bigram F1: a coarse closeness to the template the curriculum teaches."""
    a, b = _bigrams(text), _bigrams(reference)
    overlap = sum((a & b).values())
    return 0.0 if not overlap else 2 * overlap / (sum(a.values()) + sum(b.values()))


def _clip(text):
    return text if len(text) <= CLIP else text[:CLIP] + "…"


def _timed(generate, prompt):
    start = time.monotonic()
    try:
        return generate(prompt), None, time.monotonic() - start
    except ValueError:
        return "", "context", time.monotonic() - start


def eval_plan(rows, generate):
    """Planner decisions. by_target splits exact matches by the target's status, and
    majority_baseline is what always answering the commonest status would score: a planner
    that only ever says "complete" can look accurate while never proposing an analysis."""
    counts, samples, seconds = Counter(), [], 0.0
    by_target = {}
    for row in rows:
        question, language, table, steps = TA._replay(row)
        prompt, target = TA.compile_record(row)
        output, error, spent = _timed(generate, prompt)
        seconds += spent
        expected = A.parse_plan_decision(target, table)
        if error:
            outcome = error
        else:
            try:
                decision = A.parse_plan_decision(output, table)
            except ValueError:
                outcome = "invalid"
            else:
                outcome = ("exact" if decision == expected else
                           "same_status" if decision[0] == expected[0] else "valid_other")
        counts[outcome] += 1
        bucket = by_target.setdefault(expected[0], {"n": 0, "exact": 0})
        bucket["n"] += 1
        bucket["exact"] += outcome == "exact"
        if len(samples) < SAMPLES:
            samples.append({"question": question, "target": target, "output": _clip(output), "outcome": outcome})
    n = len(rows)
    valid = counts["exact"] + counts["same_status"] + counts["valid_other"]
    return {"n": n, "outcomes": dict(counts), "valid_json_rate": valid / n if n else None,
            "exact_rate": counts["exact"] / n if n else None, "by_target": by_target,
            "majority_baseline": max((b["n"] for b in by_target.values()), default=0) / n if n else None,
            "seconds_per_call": seconds / n if n else None, "samples": samples}


def eval_narrate(rows, generate):
    counts, samples, sims, seconds = Counter(), [], [], 0.0
    for row in rows:
        prompt, target = TA.compile_record(row)
        output, error, spent = _timed(generate, prompt)
        seconds += spent
        text = output.strip()
        if error:
            outcome = error
        else:
            outcome = A.narrative_verdict(text, prompt)[0] or "accepted"
            sims.append(similarity(text, target))
        counts[outcome] += 1
        if len(samples) < SAMPLES:
            samples.append({"question": row["question"], "output": _clip(text), "outcome": outcome})
    n = len(rows)
    return {"n": n, "outcomes": dict(counts), "accepted_rate": counts["accepted"] / n if n else None,
            "mean_similarity_to_template": sum(sims) / len(sims) if sims else None,
            "seconds_per_call": seconds / n if n else None, "samples": samples}


def eval_scenarios(cases, endpoint, max_steps):
    results, sources, rejected, stops = [], Counter(), Counter(), Counter()
    for name, data, question in cases:
        start = time.monotonic()
        [result] = endpoint._handle_analyst({"inputs": question, "parameters": {
            "data": data, "max_steps": max_steps, "use_model": True}})
        spent = time.monotonic() - start
        report = result.get("analyst") or {}
        sources[report.get("narrative_source")] += 1
        rejected[report.get("narrative_rejected")] += 1
        stops[report.get("planner_stop")] += 1
        results.append({"name": name, "question": question, "status": report.get("status"),
                        "narrative_source": report.get("narrative_source"),
                        "narrative_rejected": report.get("narrative_rejected"),
                        "planner_stop": report.get("planner_stop"),
                        "model_steps": [s["tool"] for s in report.get("steps", []) if s.get("source") == "model"],
                        "inference_count": report.get("inference_count"), "seconds": round(spent, 2),
                        "text": _clip(result.get("generated_text", result.get("error", "")))})
    n = len(cases)
    return {"n": n, "narrative_source": dict(sources), "narrative_rejected": {str(k): v for k, v in rejected.items()},
            "planner_stop": dict(stops), "model_narrative_rate": sources["model"] / n if n else None,
            "model_steps_total": sum(len(r["model_steps"]) for r in results), "cases": results}


# ---------------------------------------------------------------- entry point

def load_endpoint(model_dir):
    """EndpointHandler on exactly model_dir (checkpoint + neuroq_tokenizer.model), never the serving volume."""
    import handler
    if not (model_dir / "neuroq_tokenizer.model").is_file() or not any(model_dir.glob("*.pt")):
        raise ValueError("--model-dir needs a checkpoint (*.pt) and its neuroq_tokenizer.model")
    handler.NETWORK_VOLUME_PATH = str(model_dir)
    return handler.EndpointHandler(path=str(model_dir))


def held_out(path):
    if path:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            TA.compile_record(row)
        return rows
    return TA.split_records(TA.bootstrap_records())[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="report JSON; must not already exist")
    parser.add_argument("--records", type=Path, help="held-out JSONL (e.g. train_analyst validation.jsonl); "
                        "default: the bootstrap curriculum's validation split")
    parser.add_argument("--plan", type=int, default=40, help="planner prompts to evaluate (0-500)")
    parser.add_argument("--narrate", type=int, default=30, help="narration prompts to evaluate (0-500)")
    parser.add_argument("--scenarios", type=int, default=12, help="end-to-end scenarios (0-12)")
    parser.add_argument("--max-steps", type=int, default=3, help="model planning steps per scenario (0-6)")
    args = parser.parse_args(argv)
    if not (0 <= args.plan <= 500 and 0 <= args.narrate <= 500 and 0 <= args.scenarios <= 12
            and 0 <= args.max_steps <= A.MAX_MODEL_STEPS):
        parser.error("plan/narrate 0..500, scenarios 0..12, max-steps 0..6")
    if args.output.exists():
        parser.error("--output already exists")
    rows = held_out(args.records)
    plan_rows = [r for r in rows if r["stage"] == "plan"][:args.plan]
    narrate_rows = [r for r in rows if r["stage"] == "narrate"][:args.narrate]
    endpoint = load_endpoint(args.model_dir.resolve(strict=True))
    started = time.monotonic()
    report = {"model_dir": str(args.model_dir), "checkpoint": Path(endpoint.ckpt_path).name if endpoint.ckpt_path else None,
              "config": {k: endpoint.config.get(k) for k in ("vocab_size", "embed_dim", "num_layers", "max_seq_len")},
              "records": str(args.records or "bootstrap-validation"),
              "plan": eval_plan(plan_rows, endpoint._analyst_generate),
              "narrate": eval_narrate(narrate_rows, endpoint._analyst_generate),
              "scenario": eval_scenarios(scenarios()[:args.scenarios], endpoint, args.max_steps)}
    report["seconds"] = round(time.monotonic() - started, 1)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    summary = {"plan_valid_json": report["plan"]["valid_json_rate"], "plan_exact": report["plan"]["exact_rate"],
               "plan_by_target": report["plan"]["by_target"], "plan_majority_baseline": report["plan"]["majority_baseline"],
               "narrate_accepted": report["narrate"]["accepted_rate"],
               "narrate_outcomes": report["narrate"]["outcomes"],
               "scenario_model_narratives": report["scenario"]["model_narrative_rate"],
               "scenario_model_steps": report["scenario"]["model_steps_total"], "seconds": report["seconds"]}
    print(json.dumps(summary, ensure_ascii=False))
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
