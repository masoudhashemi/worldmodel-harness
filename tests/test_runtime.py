import asyncio
import json

from pydantic import SecretStr
from stirrup.core.models import AssistantMessage, ToolCall, ToolMessage

from worldmodel_harness import runtime
from worldmodel_harness.config import ModelSettings


def test_response_wait_returns_an_empty_message():
    class Slow:
        async def generate(self, messages, tools):
            await asyncio.sleep(5)

    async def scenario():
        message = await runtime.BoundedClient(Slow(), 0.05).generate([], {})
        assert message.blocks[0].text == ""
        assert message.token_usage.answer == 0

    asyncio.run(scenario())


def test_client_routing_and_explicit_limits(monkeypatch):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(runtime, "LiteLLMClient", capture)
    runtime.build_client(ModelSettings(), SecretStr("offline-secret"))
    assert captured["model"] == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    assert captured["api_key"] == "offline-secret"
    assert captured["kwargs"]["api_base"] == "https://openrouter.ai/api/v1"
    assert captured["kwargs"]["num_retries"] == 0
    assert "extra_body" not in captured["kwargs"]
    runtime.build_client(ModelSettings(reasoning_effort="low"), SecretStr("offline-secret"))
    assert captured["kwargs"]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "response_format" not in captured["kwargs"]
    runtime.build_client(ModelSettings(json_mode=True), SecretStr("offline-secret"))
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}


def test_parse_gemma_calls_handles_unquoted_tags_and_nested_structures():
    text = (
        'Leading thought\n'
        '<|tool_call>call:execute_decision{"action": "a1", "goal": "test goal"}<tool_call|>\n'
        '<|tool_call>call:execute_decision{action: <tool_call|>, goal: <|"|>hello<|"|>}<tool_call|>\n'
        'Trailing response text'
    )
    clean, calls = runtime.parse_gemma_calls(text)
    assert "Leading thought" in clean
    assert "Trailing response text" in clean
    assert "<|tool_call>" not in clean
    assert len(calls) == 2
    assert calls[0] == ("execute_decision", {"action": "a1", "goal": "test goal"})
    assert calls[1][0] == "execute_decision"
    assert calls[1][1]["goal"] == "hello"



def test_gemma_json_arguments_preserve_escaped_values():
    text = '<|tool_call>call:api_search{"query":"line one\\nline two"}<tool_call|>'
    _, calls = runtime.parse_gemma_calls(text)
    assert calls == [("api_search", {"query": "line one\nline two"})]


def test_parse_gemma_calls_handles_unclosed_braces():
    text = '<|tool_call>call:execute_decision{action:{arguments:{summary:<|"|>Task finished<|"|>},tool:<|"|>finish<|"|>}'
    _, calls = runtime.parse_gemma_calls(text)
    assert len(calls) == 1
    assert calls[0][0] == "execute_decision"
    assert calls[0][1]["action"]["tool"] == "finish"
    assert calls[0][1]["action"]["arguments"]["summary"] == "Task finished"
