"""Analyst SFT curriculum: runtime-identical prompts, validated targets, held-out groups, safe CLI."""
import json
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import qubit_analyst as A
import qubit_analyst_tools as T
import train_analyst as TA
from train_analyst import bootstrap_records, compile_record, encode_record, split_records, train_candidate

RECORDS = bootstrap_records()
COMPILED = [compile_record(row) for row in RECORDS]
ROOT = Path(__file__).parents[1]


def group(row):
    return row["table"], row["question"]


def replay(row):
    table = T.load_table(row["data"])
    steps = [{"call_id": f"call_{i}", "tool": s["tool"], "arguments": s["arguments"], "status": "completed",
              "output": T.run_tool(table, s["tool"], s["arguments"])} for i, s in enumerate(row["steps"], 1)]
    return table, steps


class CharTokenizer:
    bof_id, bos_id, eos_id, eof_id = 1, 2, 3, 4

    def encode(self, text, add_special=False):
        return [5 + ord(c) % 10 for c in text]


def test_bootstrap_is_deterministic_and_covers_both_stages_and_languages():
    assert json.dumps(bootstrap_records(), ensure_ascii=False) == json.dumps(RECORDS, ensure_ascii=False)
    assert len({row["table"] for row in RECORDS}) == TA.TABLES
    assert {row["stage"] for row in RECORDS} == {"plan", "narrate"}
    assert {row["language"] for row in RECORDS} == {"ja", "en"}
    statuses = [row["target"]["status"] for row in RECORDS if row["stage"] == "plan"]
    assert statuses.count("continue") >= 50 and statuses.count("complete") >= 50
    columns = {name for row in RECORDS for name in row["data"]}
    assert {"月", "日付", "地域", "店舗", "商品", "売上", "販売数", "広告費", "客数", "単価"} <= columns
    # per-group trends / outliers appear in both stages (as done steps the planner must not repeat)
    grouped = [r for r in RECORDS if any(s["tool"] in ("trend", "outliers") and "by" in s["arguments"]
                                         for s in r["steps"])]
    assert {r["stage"] for r in grouped} == {"plan", "narrate"} and {r["language"] for r in grouped} == {"ja", "en"}


def test_every_record_compiles_within_the_prompt_bounds():
    for row, (prompt, target) in zip(RECORDS, COMPILED):
        limit = A.PLANNER_LIMIT if row["stage"] == "plan" else A.NARRATIVE_LIMIT
        assert 0 < len(prompt) <= limit
        assert isinstance(target, str) and target


def test_plan_targets_parse_and_never_repeat_a_done_step():
    for row, (prompt, target) in zip(RECORDS, COMPILED):
        if row["stage"] != "plan":
            continue
        table = T.load_table(row["data"])
        status, tool, args = A.parse_plan_decision(target, table)
        assert target == json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
        assert prompt == A.planner_prompt(row["question"], table, row["steps"], row["remaining"], row["language"])
        if status == "continue":
            done = {A._signature(s["tool"], T.validate_args(table, s["tool"], s["arguments"])) for s in row["steps"]}
            assert A._signature(tool, args) not in done
            spec = T.TOOL_SPECS[tool]["args"]
            assert all(value in table.names() for key, value in row["target"]["arguments"].items()
                       if spec[key]["type"] == "column")      # exact column names, not loose matches


def test_narration_targets_are_the_template_and_pass_the_numeric_guard():
    rows = [(row, target) for row, (_, target) in zip(RECORDS, COMPILED) if row["stage"] == "narrate"]
    assert len(rows) >= 80
    for row, target in rows:
        table, steps = replay(row)
        findings = A.build_findings(steps, table, language=row["language"])
        caveats = A.build_caveats(steps, table, language=row["language"], question=row["question"])
        assert target == A.template_narrative(row["question"], findings, caveats, A.dataset_summary(table),
                                              row["language"])
        # the runtime guard sees only the narration prompt, so the target cites nothing else
        assert A.verify_numbers(target, A.narrative_prompt(row["question"], findings, caveats, row["language"])) == []
        assert len(target) <= TA.NARRATE_CHARS


def test_compiled_prompts_are_exactly_what_the_controller_sends():
    groups = defaultdict(list)
    for row, pair in zip(RECORDS, COMPILED):
        groups[group(row)].append((row, pair))
    accepted = 0
    for items in groups.values():
        first = items[0][0]
        plans = [pair for row, pair in items if row["stage"] == "plan"]
        narration = [pair for row, pair in items if row["stage"] == "narrate"]
        # Without a narration row the controller still asks for one; "" falls back to the template.
        model = Mock(side_effect=[target for _, target in plans + narration] + ([] if narration else [""]))
        result = A.run_analyst({"inputs": first["question"], "parameters": {
            "data": first["data"], "language": first["language"], "max_steps": TA.PLANNER_STEPS}}, model)
        sent = [call.args[0] for call in model.call_args_list]
        assert sent[:len(plans)] == [prompt for prompt, _ in plans]
        assert result["analyst"]["planner_stop"] in ("complete", "max_steps")
        if narration:
            assert sent[len(plans):] == [narration[0][0]]
            assert result["analyst"]["narrative_source"] == "model"
            assert result["generated_text"] == narration[0][1]
            accepted += 1
    assert accepted >= 80


def test_split_keeps_question_groups_together():
    train, validation = split_records(RECORDS)
    assert train and validation
    assert {group(r) for r in train}.isdisjoint(group(r) for r in validation)
    assert {r["stage"] for r in validation} == {"plan", "narrate"}
    # Only exact (prompt, target) repeats are dropped, and no compiled prompt lands in both splits.
    compiled = {id(row): pair for row, pair in zip(RECORDS, COMPILED)}
    assert len(train) + len(validation) == len(set(COMPILED))
    prompts = [{compiled[id(row)][0] for row in rows} for rows in (train, validation)]
    assert prompts[0].isdisjoint(prompts[1])


def test_jsonl_round_trip_compiles_identically():
    for row, compiled in zip(RECORDS, COMPILED):
        assert compile_record(json.loads(json.dumps(row, ensure_ascii=False))) == compiled


def plan_row(status):
    return next(r for r in RECORDS if r["stage"] == "plan" and r["target"]["status"] == status)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(stage="answer"),
    lambda r: r.update(extra=1),
    lambda r: r.pop("remaining"),
    lambda r: r.update(remaining=0),
    lambda r: r.update(remaining=True),
    lambda r: r.update(language="fr"),
    lambda r: r.update(question=""),
    lambda r: r.update(steps=r["steps"] + [r["steps"][-1]]),
    lambda r: r.update(steps=[{"tool": "shell", "arguments": {}}]),
    lambda r: r.update(steps=[{"tool": "describe", "arguments": {"column": "存在しない列"}}]),
    lambda r: r.update(target={"status": "continue", "action": "trend", "arguments": {"value": "存在しない列"}}),
    lambda r: r.update(target={"status": "complete", "answer": "完了"}),
    lambda r: r.update(target="{\"status\":\"complete\"}"),
])
def test_invalid_rows_are_rejected(mutate):
    row = json.loads(json.dumps(plan_row("continue"), ensure_ascii=False))
    mutate(row)
    with pytest.raises(ValueError):
        compile_record(row)


def test_plan_target_repeating_a_done_step_is_rejected():
    row = json.loads(json.dumps(plan_row("complete"), ensure_ascii=False))
    done = row["steps"][-1]
    row["target"] = {"status": "continue", "action": done["tool"], "arguments": done["arguments"]}
    with pytest.raises(ValueError, match="repeats"):
        compile_record(row)


def test_narration_with_an_invented_number_is_rejected():
    row = json.loads(json.dumps(next(r for r in RECORDS if r["stage"] == "narrate"), ensure_ascii=False))
    for bad in (row["target"] + "\n売上は9,876,543に達しました。", "", " " + row["target"], "{\"a\": 1}"):
        with pytest.raises(ValueError):
            compile_record({**row, "target": bad})


def test_encoding_masks_the_prompt_and_refuses_truncation():
    row = plan_row("continue")
    ids, labels = encode_record(row, CharTokenizer(), 10000)
    prompt, answer = compile_record(row)
    prefix = 2 + len(f"質問: {prompt}\n回答:")
    assert len(ids) == len(labels) == prefix + len(answer) + 2
    assert labels[:prefix] == [-100] * prefix and labels[prefix:] == ids[prefix:]
    assert labels[-2:] == [3, 4]
    with pytest.raises(ValueError, match="context window"):
        encode_record(row, CharTokenizer(), prefix + len(answer))
    long_narration = next(r for r in RECORDS if r["stage"] == "narrate" and len(r["target"]) > TA.MAX_NEW_TOKENS)
    with pytest.raises(ValueError, match="generation length"):
        encode_record(long_narration, CharTokenizer(), 10000)


def test_curriculum_fits_the_checked_in_vocabulary(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    pytest.importorskip("google.protobuf")
    from build_tokenizer_from_vocab import build_tokenizer_model
    build_tokenizer_model(ROOT / "neuroq_tokenizer.vocab", tmp_path / "tokenizer.model")
    processor = spm.SentencePieceProcessor(model_file=str(tmp_path / "tokenizer.model"))
    tokenizer = SimpleNamespace(bos_id=2, eos_id=3, bof_id=processor.PieceToId("<bof>"),
                                eof_id=processor.PieceToId("<eof>"),
                                encode=lambda text, add_special=False: processor.EncodeAsIds(text))
    for row, (prompt, _) in zip(RECORDS, COMPILED):
        ids, labels = encode_record(row, tokenizer, 1024)   # serving max_seq_len
        assert len(ids) <= 1024 and any(label != -100 for label in labels)
        prefix = labels.index(next(label for label in labels if label != -100))
        # the serving handler refuses token_count + 2 + generation length > max_seq_len
        assert prefix + A.generation_tokens(prompt) <= 1024, row["question"]


def test_rows_the_serving_handler_would_refuse_are_rejected():
    row = plan_row("complete")
    prompt, answer = compile_record(row)
    prefix = 2 + len(f"質問: {prompt}\n回答:")
    assert A.generation_tokens(prompt) == A.PLANNER_NEW_TOKENS == 96 and len(answer) + 2 < 96
    encode_record(row, CharTokenizer(), prefix + 96)
    with pytest.raises(ValueError, match="serving handler would refuse"):
        encode_record(row, CharTokenizer(), prefix + len(answer) + 2)   # fits 1024-style, not prompt + 96
    narration = next(r for r in RECORDS if r["stage"] == "narrate")
    prompt, answer = compile_record(narration)
    assert A.generation_tokens(prompt) == A.NARRATIVE_NEW_TOKENS == TA.MAX_NEW_TOKENS == 320


def test_train_candidate_reuses_the_agent_loop_with_analyst_encoding(monkeypatch):
    # Stub just enough torch for the shared loop to reach encoding: the analyst encoder must run
    # (train_agent's own encoder would fail on the analyst row shape with a KeyError instead).
    torch, nn, functional = (types.ModuleType(n) for n in ("torch", "torch.nn", "torch.nn.functional"))
    torch.manual_seed, torch.nn, nn.functional = (lambda seed: None), nn, functional
    for module in (torch, nn, functional):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    handler = SimpleNamespace(tokenizer=CharTokenizer(), config={"max_seq_len": 10000})
    rows = [next(r for r in RECORDS if r["stage"] == "narrate" and len(r["target"]) > TA.MAX_NEW_TOKENS)]
    with pytest.raises(ValueError, match="generation length"):
        train_candidate(handler, rows, rows, epochs=1, max_steps=1, lr=1e-5)


def test_cli_validation_writes_splits_and_refuses_an_existing_directory(tmp_path):
    out = tmp_path / "run"
    report = TA.main(["--output-dir", str(out)])
    assert report["trained"] is False and report["seed"] == 42
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "train.jsonl", "validation.jsonl"]
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report
    lines = {name: (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() for name in ("train", "validation")}
    assert len(lines["train"]) == report["train_records"] and len(lines["validation"]) == report["validation_records"]
    with pytest.raises(FileExistsError):
        TA.main(["--output-dir", str(out)])
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("\n".join(lines["train"] + lines["validation"]) + "\n", encoding="utf-8")
    again = TA.main(["--dataset", str(dataset), "--output-dir", str(tmp_path / "again")])
    assert (again["train_records"], again["validation_records"]) == (report["train_records"], report["validation_records"])
    assert (tmp_path / "again" / "train.jsonl").read_text(encoding="utf-8").splitlines() == lines["train"]


@pytest.mark.parametrize("argv", [["--train"], ["--epochs", "4"], ["--lr", "0.01"], ["--max-steps", "0"]])
def test_cli_rejects_unsafe_options_before_writing(tmp_path, argv):
    out = tmp_path / "run"
    with pytest.raises(SystemExit):
        TA.main(["--output-dir", str(out), *argv])
    assert not out.exists()


def test_importing_the_trainer_does_not_import_torch():
    code = "import sys, train_analyst; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
