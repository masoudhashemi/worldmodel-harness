import asyncio
import json
from pathlib import Path

from pydantic import SecretStr
from stirrup.core.models import AssistantMessage, ToolCall, ToolMessage

from worldmodel_harness import runtime, smoke
from worldmodel_harness.config import ModelSettings


def test_client_routing_and_explicit_limits(monkeypatch):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(runtime, "LiteLLMClient", capture)
    runtime.build_client(ModelSettings(), SecretStr("offline-secret"))
    assert captured["model"] == "openrouter/qwen/qwen3.8-27b"
    assert captured["api_key"] == "offline-secret"
    assert captured["kwargs"]["api_base"] == "https://openrouter.ai/api/v1"
    assert captured["kwargs"]["num_retries"] == 0
    assert "extra_body" not in captured["kwargs"]
    runtime.build_client(ModelSettings(reasoning_effort="low"), SecretStr("offline-secret"))
    assert captured["kwargs"]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "response_format" not in captured["kwargs"]
    runtime.build_client(ModelSettings(json_mode=True), SecretStr("offline-secret"))
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}


class ScriptedClient:
    """Exercise the real Stirrup loop without a model, network, or credentials."""

    model_slug = "offline-scripted-client"
    context_window_tokens = 32768

    async def generate(self, messages, tools):
        assert set(tools) == {"create_ticket", "lookup_ticket", "finish"}
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            prompt = next(m.content for m in messages if m.role == "user")
            external_id = prompt.split("external_id ")[1].split(".")[0]
            name, args = "create_ticket", {"external_id": external_id}
        elif len(results) == 1:
            previous = json.loads(results[-1].content)
            assert previous["outcome"] == "unknown"
            name, args = "lookup_ticket", previous["arguments"]
        else:
            previous = json.loads(results[-1].content)
            name, args = "finish", {"ticket_id": previous["result"]["tickets"][0]["id"]}
        return AssistantMessage(blocks=[ToolCall(name=name, arguments=json.dumps(args))])


def test_real_stirrup_loop_with_scripted_client(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    source = Path(__file__).resolve().parents[1] / "configs/model-free-test.toml"
    (tmp_path / "configs/model-free-test.toml").write_text(source.read_text())
    monkeypatch.delenv("WORLDMODEL_MODEL_CONFIG", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-secret")
    monkeypatch.setattr(smoke, "build_client", lambda *_: ScriptedClient())
    report = asyncio.run(smoke.run_smoke(tmp_path))
    assert report["passed"]
    assert report["model"] == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    assert report["environment_calls"] == 2
    assert report["model_calls_with_usage"] == 3
    assert report["message_count"] >= 8
    assert report["token_usage"] == {"input": 0, "answer": 0, "reasoning": 0}
    output = tmp_path / "runs" / report["run_id"]
    events = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert [event["outcome"] for event in events] == ["unknown", "success"]
    assert "offline-secret" not in (output / "summary.json").read_text()
