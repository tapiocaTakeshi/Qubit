"""Analyst wiring: handler route, model adapter and RunPod progress, without torch or a GPU."""
import ast
import copy
import json
import sys
import typing
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

import neuroquantum_agent_progress
from neuroquantum_agent_progress import dispatch_job
from qubit_analyst import run_analyst

ROOT = Path(__file__).parents[1]
HANDLER_TREE = ast.parse((ROOT / "handler.py").read_text(encoding="utf-8"))
ENDPOINT = next(n for n in HANDLER_TREE.body if isinstance(n, ast.ClassDef) and n.name == "EndpointHandler")
CSV = "month,region,sales,cost\n" + "\n".join(
    f"2024-{m:02d}-01,{'東' if m % 2 else '西'},{100 + m * 10 + (m % 3) * 4},{60 + m * 3}" for m in range(1, 13))
NARRATIVE = "売上は増加傾向です。詳細は所見を参照してください。"
SAFE_DECODING = {"temperature": 0.2, "max_new_tokens": 320, "repetition_penalty": 1.0,
                 "no_repeat_ngram_size": 0, "presence_penalty": 0, "frequency_penalty": 0,
                 "repeat_span_blocking": False, "deduplicate_output": False, "min_new_tokens": 0}


def method(name):
    node = next(n for n in ENDPOINT.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"Dict": typing.Dict, "List": typing.List, "Any": typing.Any}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "handler.py", "exec"), namespace)
    return namespace[name]


HANDLE_ANALYST = method("_handle_analyst")


class FakeEndpoint(SimpleNamespace):
    """Stands in for EndpointHandler; anything routed through __call__ is recorded."""

    def __call__(self, data):
        self.routed.append(data)
        return [{"generated_text": "{}"}]


def model_reply(prompt):
    return '{"status":"complete"}' if '"remaining"' in prompt else NARRATIVE


def endpoint(reply=model_reply, *, max_seq_len=1024, tokens=lambda text: len(text) // 4):
    ep = FakeEndpoint(routed=[], config={"max_seq_len": max_seq_len})
    ep.tokenizer = SimpleNamespace(encode=Mock(side_effect=lambda text, add_special=True: [7] * tokens(text)))
    ep._handle_inference = Mock(side_effect=lambda data: [{"generated_text": reply(data["inputs"])}])
    ep._handle_analyst = MethodType(HANDLE_ANALYST, ep)
    return ep


def request(**params):
    return {"action": "analyst", "inputs": "売上の推移を教えて", "parameters": {"data": CSV, **params}}


def template_text():
    return run_analyst(request(use_model=False))["generated_text"]


# ---------------------------------------------------------------- handler._handle_analyst

def test_model_calls_go_straight_to_inference_with_analyst_safe_decoding():
    ep, events = endpoint(), []
    [result] = ep._handle_analyst(request(), on_event=events.append)
    report = result["analyst"]
    assert report["status"] == "completed"
    assert report["planner_stop"] == "complete"
    assert report["narrative_source"] == "model"
    assert result["generated_text"] == NARRATIVE
    assert ep.routed == []
    assert ep._handle_inference.call_count == report["inference_count"] == 2
    for call, encoded in zip(ep._handle_inference.call_args_list, ep.tokenizer.encode.call_args_list):
        payload = call.args[0]
        assert set(payload) == {"inputs", "parameters"}
        assert payload["parameters"] == SAFE_DECODING
        assert not payload["inputs"].startswith("質問:")
        assert encoded.args == (f"質問: {payload['inputs']}\n回答:",)
        assert encoded.kwargs == {"add_special": False}
    assert events == report["events"]


@pytest.mark.parametrize("max_seq_len,fits", [(422, True), (421, False)])
def test_token_budget_reserves_max_new_tokens(max_seq_len, fits):
    # 100 prompt tokens + BOF/BOS + 320 new tokens must fit in the context window.
    ep = endpoint(max_seq_len=max_seq_len, tokens=lambda text: 100)
    [result] = ep._handle_analyst(request())
    assert ep._handle_inference.call_count == (2 if fits else 0)
    assert result["analyst"]["narrative_source"] == ("model" if fits else "template")


def test_context_overflow_still_completes_with_the_template_narrative():
    ep = endpoint(max_seq_len=400, tokens=len)
    [result] = ep._handle_analyst(request())
    report = result["analyst"]
    ep._handle_inference.assert_not_called()
    assert (report["status"], report["stop_reason"]) == ("completed", "completed")
    assert report["planner_stop"] == "context_overflow"
    assert report["narrative_source"] == "template"
    assert result["generated_text"] == template_text()
    assert report["findings"] and report["warnings"]
    assert "exceeds" not in json.dumps(result)


def test_planner_overflow_does_not_block_a_short_narrative():
    ep = endpoint(tokens=lambda text: 2000 if '"remaining"' in text else 10)
    [result] = ep._handle_analyst(request())
    assert ep._handle_inference.call_count == 1
    assert result["analyst"]["planner_stop"] == "context_overflow"
    assert result["analyst"]["narrative_source"] == "model"


def test_garbage_model_output_never_reaches_the_report():
    ep = endpoint(reply=lambda prompt: "売上は12345.6円増えました。")
    [result] = ep._handle_analyst(request())
    report = result["analyst"]
    assert report["status"] == "completed"
    assert report["planner_stop"] == "invalid_decision"
    assert report["narrative_source"] == "template"
    assert report["unverified_numbers"] == ["12345.6"]
    assert result["generated_text"] == template_text()
    assert ep.routed == []


def test_invalid_request_is_an_error_without_model_calls():
    ep = endpoint()
    [result] = ep._handle_analyst({"action": "analyst", "inputs": "推移は？", "parameters": {}})
    assert set(result) == {"error"} and "parameters.data" in result["error"]
    ep._handle_inference.assert_not_called()


# ---------------------------------------------------------------- RunPod progress

def install_progress(monkeypatch):
    publish = Mock()
    monkeypatch.setitem(sys.modules, "runpod.serverless.modules.rp_progress",
                        SimpleNamespace(progress_update=publish))
    return publish


def test_dispatch_streams_analyst_events_with_job_local_replayable_snapshots(monkeypatch):
    publish = install_progress(monkeypatch)
    ep = endpoint()
    ep._resolve_action = lambda data: data["action"]
    ep._handle_agent = Mock()
    first_job = {"id": "first"}
    [result] = dispatch_job(ep, request(), first_job)
    snapshots = [json.loads(c.args[1]) for c in publish.call_args_list]
    assert all(c.args[0] is first_job for c in publish.call_args_list)
    assert all(set(s) == {"analyst_event", "analyst_events"} for s in snapshots)
    assert [len(s["analyst_events"]) for s in snapshots] == list(range(1, len(snapshots) + 1))
    assert snapshots[0]["analyst_event"]["type"] == "started"
    assert snapshots[-1]["analyst_event"]["type"] == "finished"
    assert snapshots[-1]["analyst_events"] == result["analyst"]["events"]
    assert "分析を開始しました" in publish.call_args_list[0].args[1]
    ep._handle_agent.assert_not_called()
    assert ep.routed == []
    publish.reset_mock()
    dispatch_job(ep, request(), {"id": "second"})
    assert len(json.loads(publish.call_args_list[0].args[1])["analyst_events"]) == 1


def test_progress_outage_does_not_lose_the_analyst_report(monkeypatch):
    install_progress(monkeypatch).side_effect = RuntimeError("secret-progress")
    ep = endpoint()
    ep._resolve_action = lambda data: "analyst"
    [result] = dispatch_job(ep, request(), {"id": "job"})
    assert result["analyst"]["status"] == "completed"
    assert len(result["analyst"]["warnings"]) == 1
    assert "secret-progress" not in json.dumps(result)


def test_agent_snapshots_are_byte_identical(monkeypatch):
    publish = install_progress(monkeypatch)
    def agent(data, on_event):
        on_event({"sequence": 1, "type": "started", "label": "開始"})
        return [{"generated_text": "ok"}]
    handler = SimpleNamespace(_resolve_action=lambda data: "agent", _handle_agent=Mock(side_effect=agent),
                              _handle_analyst=Mock())
    job = {"id": "job"}
    assert dispatch_job(handler, {"action": "agent"}, job) == [{"generated_text": "ok"}]
    publish.assert_called_once_with(job, '{"agent_event": {"sequence": 1, "type": "started", "label": "開始"}, '
                                         '"agent_events": [{"sequence": 1, "type": "started", "label": "開始"}]}')
    handler._handle_analyst.assert_not_called()


@pytest.mark.parametrize("action", ["inference", "train", "status", "jev_judge", "Analyst"])
def test_dispatch_keeps_other_actions_unchanged(action, monkeypatch):
    publish = install_progress(monkeypatch)
    handler = Mock(return_value=[{"generated_text": "chat"}])
    handler._resolve_action.return_value = action
    data = {"action": action, "inputs": "hi"}
    assert dispatch_job(handler, data, {"id": "job"}) == [{"generated_text": "chat"}]
    handler.assert_called_once_with(data)
    handler._handle_agent.assert_not_called()
    handler._handle_analyst.assert_not_called()
    publish.assert_not_called()


# ---------------------------------------------------------------- RunPod entrypoints

RECORDS = [{"month": f"2024-{m:02d}", "sales": 100 + m * 10, "cost": 60 + m} for m in range(1, 13)]
COLUMNS = {"month": [r["month"] for r in RECORDS], "sales": [r["sales"] for r in RECORDS]}


@pytest.mark.parametrize("payload", [CSV, RECORDS, COLUMNS], ids=["csv", "records", "columns"])
@pytest.mark.parametrize("filename,function,instance_name", [
    ("runpod_handler.py", "run_handler", "handler"),
    ("handler.py", "_runpod_handler", "_global_handler"),
])
def test_both_runpod_entrypoints_forward_analyst_data_intact(filename, function, instance_name, payload,
                                                             monkeypatch):
    source = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == function)
    instance = object()
    namespace = {instance_name: instance}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    dispatch = Mock(return_value=[{"generated_text": "ok"}])
    monkeypatch.setattr(neuroquantum_agent_progress, "dispatch_job", dispatch)
    params = {"data": payload, "use_model": False, "max_steps": 0, "language": "ja", "table_name": "売上"}
    job = {"id": "job", "input": {"action": "analyst", "prompt": "売上の推移は？", "parameters": params}}
    original = copy.deepcopy(job)
    assert namespace[function](job) == {"generated_text": "ok"}
    handler_arg, forwarded, job_arg = dispatch.call_args.args
    assert handler_arg is instance and job_arg is job
    assert forwarded == {"action": "analyst", "inputs": "売上の推移は？",
                         "parameters": original["input"]["parameters"]}
    assert forwarded["parameters"]["data"] is payload
    assert job == original
    report = run_analyst(forwarded)["analyst"]
    assert report["status"] == "completed"
    assert report["dataset"]["name"] == "売上" and report["dataset"]["rows"] == 12


# ---------------------------------------------------------------- handler.py source

def call_tables():
    call = next(n for n in ENDPOINT.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    tables = {}
    for node in ast.walk(call):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            tables[node.targets[0].id] = {k.value: (v.value.id, v.attr)
                                          for k, v in zip(node.value.keys, node.value.values)}
    return tables


def test_analyst_is_a_routed_data_action_not_a_training_action():
    tables = call_tables()
    assert tables["_routes"]["analyst"] == ("self", "_handle_analyst")
    assert tables["_routes"]["agent"] == ("self", "_handle_agent")
    assert "analyst" not in tables["_routes_no_data"]
    assert not "analyst".startswith("train")  # __call__ normalises only train* payloads
    node = next(n for n in ENDPOINT.body if isinstance(n, ast.FunctionDef) and n.name == "_handle_analyst")
    assert [a.arg for a in node.args.args] == ["self", "data", "on_event"]
    assert isinstance(node.args.defaults[0], ast.Constant) and node.args.defaults[0].value is None
    calls = [n.func for n in ast.walk(node) if isinstance(n, ast.Call)]
    assert not any(isinstance(f, ast.Name) and f.id == "self" for f in calls)
    assert not any(isinstance(f, ast.Attribute) and f.attr == "__call__" for f in calls)


def test_action_lists_document_analyst():
    runpod_tree = ast.parse((ROOT / "runpod_handler.py").read_text(encoding="utf-8"))
    functions = {n.name: n for tree in (HANDLER_TREE, runpod_tree) for n in tree.body
                 if isinstance(n, ast.FunctionDef)}
    call = next(n for n in ENDPOINT.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    docs = [ast.get_docstring(HANDLER_TREE), ast.get_docstring(ENDPOINT), ast.get_docstring(call),
            ast.get_docstring(functions["_runpod_handler"]), ast.get_docstring(functions["run_handler"])]
    assert all("analyst" in doc for doc in docs)
