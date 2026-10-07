"""eval_analyst metrics with scripted generators: no torch, checkpoint or GPU."""
import json

import pytest

import eval_analyst as E
import qubit_analyst as A
import train_analyst as TA


@pytest.fixture(scope="module")
def held_out():
    rows = TA.split_records(TA.bootstrap_records())[1]
    return [r for r in rows if r["stage"] == "plan"][:6], [r for r in rows if r["stage"] == "narrate"][:4]


def test_fresh_scenarios_run_deterministically_and_are_not_curriculum_questions():
    curriculum = {r["question"] for r in TA.bootstrap_records()}
    cases = E.scenarios()
    assert len(cases) == 12 and len({name for name, _, _ in cases}) == 12
    assert E.scenarios() == cases
    for name, data, question in cases:
        assert question not in curriculum
        report = A.run_analyst({"inputs": question, "parameters": {"data": data, "use_model": False}})["analyst"]
        assert report["status"] == "completed" and len(report["findings"]) >= 2, name


def test_plan_suite_scores_targets_exactly_and_garbage_as_invalid(held_out):
    plan_rows, _ = held_out
    targets = {TA.compile_record(r)[0]: TA.compile_record(r)[1] for r in plan_rows}
    perfect = E.eval_plan(plan_rows, lambda prompt: targets[prompt])
    assert perfect["exact_rate"] == perfect["valid_json_rate"] == 1.0
    assert all(b["exact"] == b["n"] for b in perfect["by_target"].values())
    garbage = E.eval_plan(plan_rows, lambda prompt: "もちろん!ここに" * 9)
    assert garbage["valid_json_rate"] == 0.0 and garbage["outcomes"] == {"invalid": len(plan_rows)}
    assert len(garbage["samples"]) <= E.SAMPLES and all(len(s["output"]) <= E.CLIP + 1 for s in garbage["samples"])


def test_plan_suite_counts_context_overflow_separately(held_out):
    def overflow(prompt):
        raise ValueError("Analyst prompt exceeds model context window")
    assert E.eval_plan(held_out[0], overflow)["outcomes"] == {"context": len(held_out[0])}


def test_narrate_suite_accepts_only_what_the_runtime_would_publish(held_out):
    _, narrate_rows = held_out
    targets = {TA.compile_record(r)[0]: TA.compile_record(r)[1] for r in narrate_rows}
    perfect = E.eval_narrate(narrate_rows, lambda prompt: targets[prompt])
    assert perfect["accepted_rate"] == 1.0 and perfect["mean_similarity_to_template"] == pytest.approx(1.0)
    loop = E.eval_narrate(narrate_rows, lambda prompt: "いくつかの" * 50)
    assert loop["outcomes"] == {"repetitive": len(narrate_rows)}
    fabricated = E.eval_narrate(narrate_rows, lambda prompt: "売上は987,654万円でした。")
    assert fabricated["outcomes"] == {"unverified": len(narrate_rows)}


def test_similarity_is_a_bounded_bigram_f1():
    assert E.similarity("売上が増加", "売上が増加") == pytest.approx(1.0)
    assert E.similarity("abc", "xyz") == 0.0
    assert 0 < E.similarity("売上が増加しました", "売上は減少しました") < 1


def test_main_refuses_an_existing_report_and_bad_bounds(tmp_path):
    existing = tmp_path / "report.json"
    existing.write_text("{}")
    for argv in (["--model-dir", str(tmp_path), "--output", str(existing)],
                 ["--model-dir", str(tmp_path), "--output", str(tmp_path / "new.json"), "--scenarios", "13"]):
        with pytest.raises(SystemExit):
            E.main(argv)
    assert json.loads(existing.read_text()) == {}


def test_always_complete_planner_scores_exactly_the_majority_baseline(held_out):
    rows = TA.split_records(TA.bootstrap_records())[1]
    plan_rows = [r for r in rows if r["stage"] == "plan"]
    lazy = E.eval_plan(plan_rows, lambda prompt: '{"status":"complete"}')
    assert lazy["exact_rate"] == pytest.approx(lazy["majority_baseline"])
    assert lazy["by_target"]["continue"]["exact"] == 0 < lazy["by_target"]["continue"]["n"]
