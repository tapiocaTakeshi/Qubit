"""Multi-step controller and live progress contracts, without a GPU or network."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from neuroquantum_agent import run_agent, validate_request
from neuroquantum_agent_protocol import parse_decision
from neuroquantum_agent_progress import dispatch_job


def action(name, **arguments):
    return json.dumps({"status": "continue", "action": name, "arguments": arguments})


def final(answer):
    return json.dumps({"status": "complete", "answer": answer})


def request(**params):
    return {"prompt": "検索して計算してください", "parameters": {"protocol": 3, **params}}


def test_search_observation_changes_next_action_and_final_needs_no_extra_generation():
    events = []
    def search(query):
        assert events[-1]["type"] == "action"
        return [{"title": "料金", "url": "https://example.org", "text": "1個120円"}]
    def generate(prompt):
        if '"tool":"calculator"' in prompt:
            assert '240' in prompt
            return final("2個で240円です。")
        if '"tool":"web_search"' in prompt:
            assert '1個120円' in prompt
            return action("calculator", expression="120*2")
        return action("web_search", query="料金")
    model = Mock(side_effect=generate)
    result = run_agent(request(), model, search, on_event=events.append)
    assert result["generated_text"] == "2個で240円です。"
    assert model.call_count == 3
    assert [s["tool"] for s in result["agent"]["steps"]] == ["web_search", "calculator"]
    assert [e["type"] for e in events] == ["started", "decision", "action", "observation",
        "decision", "action", "observation", "decision", "finished"]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert result["agent"]["events"] == events


def test_invalid_json_retries_with_controller_feedback_and_consumes_budget():
    model = Mock(side_effect=["invalid", action("calculator", expression="2*3"), final("6")])
    result = run_agent(request(max_steps=3), model)
    assert "controller_error" in model.call_args_list[1].args[0]
    assert result["agent"]["status"] == "completed"
    assert result["agent"]["decision_count"] == 3
    assert any(e["type"] == "retry" for e in result["agent"]["events"])


def test_retry_is_bounded_and_does_not_expose_raw_output():
    model = Mock(side_effect=["private-invalid-output", "invalid", "不明です"])
    result = run_agent(request(), model)
    assert result["agent"]["status"] == "fallback"
    assert model.call_count == 3
    assert "private-invalid-output" not in json.dumps(result)


def test_rethink_is_bounded_and_never_claims_a_tool_ran():
    model = Mock(side_effect=[action("rethink"), action("rethink"), "不明です"])
    result = run_agent(request(max_steps=2), model)
    assert result["agent"]["status"] == "limited"
    assert result["agent"]["stop_reason"] == "max_steps"
    assert not result["agent"]["steps"]
    assert model.call_count == 3


def test_ten_actions_are_bounded_by_ten_decisions_and_one_synthesis():
    model = Mock(side_effect=[action("calculator", expression=f"{i}+1") for i in range(10)] + ["完了"])
    result = run_agent(request(), model)
    assert len(result["agent"]["steps"]) == 10
    assert result["agent"]["decision_count"] == 10
    assert result["agent"]["inference_count"] == 11
    assert result["agent"]["stop_reason"] == "max_steps"


def test_tool_failure_is_observed_and_can_be_corrected():
    model = Mock(side_effect=[action("calculator", expression="1/0"),
                             action("calculator", expression="2+2"), final("4")])
    result = run_agent(request(), model)
    assert [s["status"] for s in result["agent"]["steps"]] == ["failed", "completed"]
    assert '"status":"failed"' in model.call_args_list[1].args[0]


def test_required_tool_cannot_be_bypassed_by_direct_final():
    result = run_agent(request(tool_choice="calculator"), Mock(return_value=final("計算済みです")))
    assert result["agent"]["status"] == "failed"
    assert result["generated_text"] != "計算済みです"


def test_optional_thought_is_discarded():
    decision = json.dumps({"status": "continue", "action": "calculator",
                          "arguments": {"expression": "2+2"}, "thought": "private reasoning"})
    events = []
    result = run_agent(request(), Mock(side_effect=[decision, final("4")]), on_event=events.append)
    assert "private reasoning" not in json.dumps(result)
    assert "thought" not in json.dumps(events)


@pytest.mark.parametrize("raw", [
    '{"status":"complete","answer":""}', '{"status":"complete","answer":42}',
    '{"status":"complete","answer":"yes","action":"shell"}',
    '{"status":"complete","status":"continue","answer":"yes"}',
    action("shell", command="touch file"), action("calculator", expression="2", extra="x"),
    action("web_search", query="unconfigured"), action("rethink", extra="x"),
    '{"status":"continue","action":"calculator","arguments":null}',
])
def test_structured_decisions_are_strictly_validated(raw):
    with pytest.raises(ValueError):
        parse_decision(raw, ["calculator"])


@pytest.mark.parametrize("params", [{"max_steps": 11}, {"max_seconds": 0},
    {"max_seconds": True}, {"max_seconds": float("nan")}, {"max_seconds": 241}])
def test_invalid_controller_limits_are_rejected(params):
    with pytest.raises(ValueError):
        validate_request(request(**params))


def test_cancellation_after_observation_prevents_next_inference():
    stopped = [False]
    def progress(event):
        if event["type"] == "observation":
            stopped[0] = True
    model = Mock(return_value=action("calculator", expression="1+1"))
    result = run_agent(request(), model, on_event=progress, cancelled=lambda: stopped[0])
    assert result["agent"]["stop_reason"] == "cancelled"
    assert len(result["agent"]["steps"]) == 1
    assert model.call_count == 1


def test_deadline_during_inference_prevents_tool_execution_and_final_generation():
    now = [0]
    search = Mock()
    def generate(prompt):
        now[0] = 2
        return action("web_search", query="x")
    model = Mock(side_effect=generate)
    result = run_agent(request(max_seconds=1), model, search, clock=lambda: now[0])
    search.assert_not_called()
    assert model.call_count == 1
    assert result["agent"]["stop_reason"] == "timeout"


def test_progress_callback_cannot_mutate_execution_and_failure_is_nonfatal():
    def broken(event):
        event["type"] = "fake"
        raise RuntimeError("secret")
    result = run_agent(request(), Mock(return_value=final("ok")), on_event=broken)
    assert result["agent"]["status"] == "completed"
    assert result["agent"]["events"][0]["type"] == "started"
    assert len(result["agent"]["warnings"]) == 1
    assert "secret" not in json.dumps(result)


def test_model_failure_is_terminal_and_sanitized():
    result = run_agent(request(), Mock(side_effect=RuntimeError("secret-path")))
    assert result["agent"]["status"] == "failed"
    assert result["agent"]["events"][-1]["type"] == "finished"
    assert "secret-path" not in json.dumps(result)


def test_runpod_dispatch_progress_is_job_local_with_replayable_snapshots(monkeypatch):
    publish = Mock()
    monkeypatch.setitem(sys.modules, "runpod.serverless.modules.rp_progress",
                        SimpleNamespace(progress_update=publish))
    def handle(data, on_event):
        return [run_agent(data, Mock(side_effect=[action("calculator", expression="1+1"), final("2")]),
                          on_event=on_event)]
    handler = SimpleNamespace(_resolve_action=lambda data: "agent", _handle_agent=handle)
    first_job = {"id": "first"}
    dispatch_job(handler, request(), first_job)
    snapshots = [json.loads(c.args[1]) for c in publish.call_args_list]
    assert all(c.args[0] is first_job for c in publish.call_args_list)
    assert len(snapshots[0]["agent_events"]) == 1
    assert snapshots[-1]["agent_event"]["type"] == "finished"
    assert len(snapshots[-1]["agent_events"]) == len(snapshots)
    publish.reset_mock()
    dispatch_job(handler, request(), {"id": "second"})
    assert len(json.loads(publish.call_args_list[0].args[1])["agent_events"]) == 1


def test_dispatch_keeps_non_agent_actions_unchanged():
    handler = Mock(return_value=[{"generated_text": "chat"}])
    handler._resolve_action.return_value = "inference"
    data = {"prompt": "hi"}
    assert dispatch_job(handler, data, {"id": "job"}) == [{"generated_text": "chat"}]
    handler.assert_called_once_with(data)
    handler._handle_agent.assert_not_called()


@pytest.mark.parametrize("filename,function,instance_name", [
    ("runpod_handler.py", "run_handler", "handler"),
    ("handler.py", "_runpod_handler", "_global_handler"),
])
def test_both_runpod_entrypoints_forward_the_actual_job(filename, function, instance_name, monkeypatch):
    import neuroquantum_agent_progress
    source = ast.parse((Path(__file__).parents[1] / filename).read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == function)
    instance = object()
    namespace = {instance_name: instance}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    dispatch = Mock(return_value=[{"generated_text": "ok"}])
    monkeypatch.setattr(neuroquantum_agent_progress, "dispatch_job", dispatch)
    job = {"id": "job", "input": {"action": "agent", "prompt": "hi", "parameters": {"protocol": 3}}}
    assert namespace[function](job) == {"generated_text": "ok"}
    assert dispatch.call_args.args == (instance,
        {"action": "agent", "inputs": "hi", "parameters": {"protocol": 3}}, job)
