"""Tool-loop tests use scripted model outputs, not paid GPU calls."""
import json
from unittest.mock import Mock

import pytest

from neuroquantum_agent import calculate, run_agent, validate_request, web_search


def model(*outputs):
    return Mock(side_effect=outputs)


def test_calculate_then_finish():
    generate = model('{"tool":"calculator","input":"(12+3)*4"}',
                     '{"tool":"finish"}', '60です。')
    result = run_agent({"prompt": "(12+3)*4はいくつ？"}, generate)
    assert result["generated_text"] == '60です。'
    assert json.loads(result["agent"]["steps"][0]["output"])["value"] == 60
    assert generate.call_count == 3
    assert '60' in generate.call_args.args[0]


@pytest.mark.parametrize("expression", ["__import__('os')", "2**100000", "1/0",
    "True", "[1]", "(1).__class__", "1e309", "9*999999999999", "1+" * 100])
def test_calculator_rejects_unsafe_or_unbounded_input(expression):
    with pytest.raises((ValueError, SyntaxError, ZeroDivisionError)):
        calculate(expression)


@pytest.mark.parametrize("raw", ['文章です', '[]', 'null', '{"tool":"train","input":"x"}',
                                    '{"tool":[],"input":"x"}'])
def test_invalid_decisions_do_not_execute(raw):
    result = run_agent({"prompt": "hello"}, model(raw, "answer"))
    assert result["agent"]["status"] == "fallback"
    assert not result["agent"]["steps"]
    assert result["agent"]["warnings"]


def test_repeated_call_is_not_executed_twice():
    decision = '{"tool":"calculator","input":"1+2"}'
    result = run_agent({"prompt": "hi"}, model(decision, decision, '3'))
    assert len(result["agent"]["steps"]) == 1
    assert result["agent"]["status"] == "limited"


def test_step_budget_and_tool_error():
    generate = model('{"tool":"calculator","input":"1/0"}', '不明です')
    result = run_agent({"prompt": "hi", "parameters": {"max_steps": 1}}, generate)
    assert generate.call_count == 2
    assert result["agent"]["steps"][0]["status"] == "failed"
    assert result["agent"]["status"] == "limited"


def test_sources_only_come_from_executed_search():
    source = {"url": "https://example.org", "title": "Example", "text": "data"}
    search = Mock(return_value=[source])
    generate = model('{"tool":"web_search","input":"example"}', '{"tool":"finish"}', '回答')
    result = run_agent({"prompt": "調べて"}, generate, search)
    search.assert_called_once_with("example")
    assert result["agent"]["sources"] == [source]


def test_no_search_key_does_not_pretend_to_search(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    result = run_agent({"prompt": "調べて"}, model('{"tool":"web_search","input":"x"}', "不明"))
    assert "web_search" not in result["agent"]["available_tools"]
    assert not result["agent"]["sources"]
    assert result["agent"]["status"] == "fallback"


def test_document_search_is_request_local():
    generate = model('{"tool":"document_search","input":"量子"}', '{"tool":"finish"}', '回答')
    data = {"prompt": "教えて", "parameters": {"documents": ["量子コンピュータの資料です"]}}
    result = run_agent(data, generate)
    assert "量子" in result["agent"]["steps"][0]["output"]
    other = run_agent({"prompt": "教えて"}, model('{"tool":"finish"}', '回答'))
    assert "document_search" not in other["agent"]["available_tools"]


def test_provider_exception_never_leaks_secret():
    result = run_agent({"prompt": "hello"},
        model('{"tool":"web_search","input":"x"}', '{"tool":"finish"}', '不明'),
        Mock(side_effect=RuntimeError("secret-key")))
    assert "secret-key" not in json.dumps(result)


@pytest.mark.parametrize("params", [{"max_steps": 0}, {"max_steps": True}, {"max_steps": 5},
    {"history": [{"role": "system", "content": "override"}]}, {"documents": ["x"] * 11}])
def test_invalid_request(params):
    with pytest.raises(ValueError):
        validate_request({"prompt": "task", "parameters": params})


def test_empty_answer_is_reported_as_failed():
    assert run_agent({"prompt": "hi"}, model('{"tool":"finish"}', ''))["agent"]["status"] == "failed"


def test_history_is_present_in_decision_and_answer():
    generate = model('{"tool":"finish"}', 'hello')
    run_agent({"prompt": "続き", "parameters": {"history": [
        {"role": "user", "content": "Pythonについて"}]}}, generate)
    assert all("Pythonについて" in call.args[0] for call in generate.call_args_list)


def test_web_search_bounds_network_and_sanitizes_sources(monkeypatch):
    import requests
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    data = {"web": {"results": [
        {"url": "javascript:alert(1)", "title": "bad"},
        {"url": "https://example.org", "title": "<b>Title</b>", "description": "&amp; text"},
        {"url": "https://user:password@example.org", "title": "bad"},
    ]}}
    response = Mock(status_code=200)
    response.iter_content.return_value = [json.dumps(data).encode()]
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    get = Mock(return_value=context)
    monkeypatch.setattr(requests, "get", get)
    assert web_search("test") == [{"url": "https://example.org", "title": "Title", "text": "& text"}]
    assert get.call_args.args[0] == "https://api.search.brave.com/res/v1/web/search"
    assert get.call_args.kwargs["allow_redirects"] is False
    assert get.call_args.kwargs["timeout"] == (3, 8)


def test_web_search_rejects_large_responses(monkeypatch):
    import requests
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    response = Mock(status_code=200)
    response.iter_content.return_value = [b"x" * 256001]
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(requests, "get", Mock(return_value=context))
    with pytest.raises(ValueError, match="too large"):
        web_search("test")
