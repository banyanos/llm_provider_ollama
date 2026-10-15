# -*- coding: utf-8 -*-
"""Ollama plugin-contract tests (LLM Provider Plugin plan §6.2 / A.6).

Single-turn tool branches, raw-HTTP invoke on the pooled client (D7),
auth per request from ctx.credentials, usage extraction (Q6) and
neutral→native conversions — verified against recorded HTTP fixtures via
``httpx.MockTransport`` (no network, no model).
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from llm_provider_base import LlmProviderHandler

from llm_provider_ollama import OllamaProvider


def _handler(transport=None, **config_overrides):
    configuration = {"model": "llama3", "base_url": "https://ollama.example.com"}
    configuration.update(config_overrides)
    agent = {
        "instructions": "You are terse.",
        "configuration": configuration,
        "llm": {"llm_name": "ollama"},
        "tool_call_role": "tool",
    }
    ctx = SimpleNamespace(
        http_client=httpx.Client(
            transport=transport or httpx.MockTransport(
                lambda request: httpx.Response(200, json={"message": {"content": "ok"}})
            )
        ),
        credentials={"api_key": "sk-ollama"},
        endpoint="https://ollama.example.com",
        model_identifier="llama3",
        options={},
        trace_id="tid-1",
        emit=None,
        is_cancelled=None,
        pool_name="httpx_llm_ollama",
    )
    return OllamaProvider(logging.getLogger("kit"), agent, ctx=ctx)


# ---------------------------------------------------------------- metadata


def test_handler_is_plugin_base_subclass():
    assert issubclass(OllamaProvider, LlmProviderHandler)


def test_plugin_metadata():
    assert OllamaProvider.api_protocol == "ollama"
    assert OllamaProvider.display_name == "Ollama"
    assert OllamaProvider.contract_version == "1"
    assert OllamaProvider.default_pool_name == "httpx_llm_ollama"
    md = OllamaProvider.plugin_metadata()
    assert isinstance(md["config_schema"], dict)
    # Q8/CREDS：凭据不入 schema（本 schema 为 A.6 新增）
    assert "api_key" not in md["config_schema"].get("properties", {})
    assert "headers" not in md["config_schema"].get("properties", {})


# ---------------------------------------------------------------- __init__


def test_init_requires_pooled_client():
    # 直接构造：ctx 缺 http_client → 构造期 fail-fast（plan §5.2.1）
    agent = {"instructions": "x", "configuration": {"model": "m", "base_url": "u"}, "llm": {}, "tool_call_role": "tool"}
    with pytest.raises(Exception, match="ctx.http_client"):
        OllamaProvider(
            logging.getLogger("k"), agent, ctx=SimpleNamespace(http_client=None, credentials={})
        )


def test_init_requires_base_url():
    agent = {"instructions": "x", "configuration": {"model": "m"}, "llm": {}, "tool_call_role": "tool"}
    ctx = SimpleNamespace(
        http_client=httpx.Client(), credentials={}, endpoint=None,
        model_identifier="m", options={}, trace_id="t", emit=None,
        is_cancelled=None, pool_name="httpx_llm_ollama",
    )
    with pytest.raises(Exception, match="base_url"):
        OllamaProvider(logging.getLogger("k"), agent, ctx=ctx)


def test_init_excludes_client_only_keys():
    handler = _handler(api_key="leaky", headers={"X-Custom": "v"})
    assert "api_key" not in handler.model_setting
    assert "headers" not in handler.model_setting


# ------------------------------------------------------ invoke_model (D7)


def test_invoke_model_raw_http_non_stream():
    captured = {}

    def handler_transport(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello"},
                "prompt_eval_count": 5,
                "eval_count": 7,
                "done_reason": "stop",
            },
        )

    handler = _handler(transport=httpx.MockTransport(handler_transport))
    result = handler.invoke_model(**{"input_messages": [{"role": "user", "content": "hi"}], "stream": False})

    assert captured["url"] == "https://ollama.example.com/api/chat"
    assert captured["auth"] == "Bearer sk-ollama"
    assert captured["body"]["model"] == "llama3"
    # system instruction prepended from instructions (native behaviour kept)
    assert captured["body"]["messages"][0]["role"] == "system"
    assert result["message"]["content"] == "hello"


def test_invoke_model_web_tools_become_typed_dicts():
    captured = {}

    def handler_transport(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    handler = _handler(
        transport=httpx.MockTransport(handler_transport),
        web_tools_enabled={"web_search": True, "web_fetch": True},
    )
    handler.invoke_model(**{"input_messages": [], "stream": False})
    tools = captured["body"].get("tools") or []
    assert {"type": "web_search"} in tools
    assert {"type": "web_fetch"} in tools


def test_invoke_model_raw_http_stream_ndjson():
    body = (
        '{"message": {"role": "assistant", "content": "hel"}, "done": false}\n'
        '{"message": {"role": "assistant", "content": "lo"}, "done": true,'
        ' "prompt_eval_count": 5, "eval_count": 7, "done_reason": "stop"}\n'
    ).encode("utf-8")
    handler = _handler(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    )
    chunks = list(
        handler.invoke_model(**{"input_messages": [], "stream": True})
    )
    assert len(chunks) == 2
    assert chunks[-1]["done"] is True
    assert chunks[-1]["eval_count"] == 7


def test_invoke_model_http_error_duck_typed():
    handler = _handler(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="unauthorized")
        )
    )
    with pytest.raises(Exception) as exc_info:
        handler.invoke_model(**{"input_messages": [], "stream": False})
    # §5.4 duck-typed .response for adapter mapping
    status = getattr(exc_info.value, "response", None)
    assert status is not None and int(getattr(status, "status_code")) == 401


# ------------------------------------------------- neutral → native (§5.5)


def test_to_native_drops_system_role():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert all(m["role"] != "system" for m in native)
    assert native == [{"role": "user", "content": "hi"}]


def test_to_native_tool_shapes():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "find_order", "arguments": '{"q": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
    )
    assert native[1]["tool_calls"] == [
        {"function": {"name": "find_order", "arguments": {"q": 1}}}
    ]
    tool_msg = native[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_name"] == ""
    # 字符串内容原样透传（不做 JSON 解码）
    assert tool_msg["content"] == "result"


def test_to_native_strips_provider_state():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {
                "role": "assistant",
                "content": "x",
                "provider_state": {"api_protocol": "anthropic", "blocks": [1]},
            }
        ]
    )
    assert "provider_state" not in native[0]


# ------------------------------------------- single-turn handle_response


def test_handle_response_tool_branch_single_turn():
    handler = _handler()
    response = {
        "message": {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {"function": {"name": "find_order", "arguments": {"q": "x"}}}
            ],
        },
        "prompt_eval_count": 10,
        "eval_count": 4,
        "done_reason": "stop",
    }
    handler.handle_response(response, [])

    calls = handler.final_output["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"]  # synthesized for OpenAI pairing
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "find_order"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "x"}
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "done_reason": "stop",
    }


def test_handle_response_empty_retries():
    handler = _handler()
    responses = [
        {"message": {"role": "assistant", "content": ""}},
        {
            "message": {"role": "assistant", "content": "finally"},
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    ]

    def fake_invoke(**kwargs):
        return responses.pop(0)

    import unittest.mock as mock

    with mock.patch.object(handler, "invoke_model", side_effect=fake_invoke):
        handler.handle_response(
            {"message": {"role": "assistant", "content": ""}}, []
        )
    assert handler.final_output["content"] == "finally"
    assert handler.final_output["usage"]["total_tokens"] == 2


def test_handle_response_plain_answer_usage():
    handler = _handler()
    handler.handle_response(
        {
            "message": {"role": "assistant", "content": "answer"},
            "prompt_eval_count": 3,
            "eval_count": 9,
            "done_reason": "stop",
        },
        [],
    )
    assert handler.final_output["content"] == "answer"
    assert handler.final_output["finish_reason"] == "stop"
    assert handler.final_output["usage"]["total_tokens"] == 12


# --------------------------------------------- single-turn handle_stream


def _emit_capture(handler):
    emitted = []
    handler.ctx = SimpleNamespace(**{**handler.ctx.__dict__, "emit": lambda c: emitted.append(dict(c))})
    return emitted


def test_handle_stream_tool_branch_single_turn():
    handler = _handler()
    emitted = _emit_capture(handler)
    handler.run = {"run_uuid": "run-1", "thread_uuid": "thread-1"}

    chunks = [
        {
            "message": {
                "role": "assistant",
                "content": "ch",
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": {"q": 2}}}
                ],
            },
            "done": False,
        },
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": 8,
            "eval_count": 3,
            "done_reason": "stop",
        },
    ]
    finished = __import__("threading").Event()
    handler.handle_stream(iter(chunks), input_messages=[], stream_event=finished)

    assert finished.is_set()
    calls = handler.final_output["tool_calls"]
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": 2}
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"]["total_tokens"] == 11
    end_markers = [c for c in emitted if c.get("is_message_end")]
    assert end_markers, f"no message-end marker in {emitted!r}"


def test_handle_stream_final_usage_and_finish():
    handler = _handler()
    _emit_capture(handler)
    handler.run = {"run_uuid": "run-1", "thread_uuid": "thread-1"}

    chunks = [
        {"message": {"role": "assistant", "content": "hel"}, "done": False},
        {
            "message": {"role": "assistant", "content": "lo"},
            "done": True,
            "prompt_eval_count": 2,
            "eval_count": 5,
            "done_reason": "stop",
        },
    ]
    handler.handle_stream(iter(chunks), input_messages=[], stream_event=None)

    assert handler.final_output["content"] == "hello"
    assert handler.final_output["finish_reason"] == "stop"
    assert handler.final_output["usage"]["total_tokens"] == 7

def test_endpoint_normalised_to_native_root():
    """Records saved for the old OLLAMA builtin end in /v1 (plan D8, G3)."""
    from llm_provider_ollama import OllamaProvider

    assert OllamaProvider._native_root("http://h:11434/v1/") == "http://h:11434"
    assert OllamaProvider._native_root("https://ollama.com/api") == "https://ollama.com"
    assert OllamaProvider._native_root("http://h:11434") == "http://h:11434"


def test_replaces_builtin_protocol():
    from llm_provider_ollama import OllamaProvider

    assert OllamaProvider.plugin_metadata()["replaces_protocol"] == "OLLAMA"


def test_list_models_uses_native_tags_endpoint():
    from types import SimpleNamespace

    import httpx

    from llm_provider_ollama import OllamaProvider

    seen = {}

    def _respond(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"models": [{"name": "glm-5.3-flash"}]})

    ctx = SimpleNamespace(
        http_client=httpx.Client(transport=httpx.MockTransport(_respond)),
        endpoint="https://ollama.com/v1", credentials={"api_key": "k"}, timeout_seconds=5,
    )
    assert OllamaProvider.list_models(ctx) == [{"model_id": "glm-5.3-flash", "name": "glm-5.3-flash"}]
    assert seen["url"] == "https://ollama.com/api/tags"
