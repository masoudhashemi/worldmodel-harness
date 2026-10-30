import asyncio
import json

import pytest
from stirrup.core.models import AssistantMessage, ToolCall

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import dataset


def _call(name, **arguments):
    return AssistantMessage(blocks=[ToolCall(name=name, arguments=json.dumps(arguments))])


def _run(tmp_path, monkeypatch, review: bool, rendered: str):
    from pydantic import SecretStr

    from worldmodel_harness import experiment
    from worldmodel_harness.config import ModelSettings
    from worldmodel_harness.experiment import TrialSettings, run_trial

    class ScriptedClient:
        model_slug = "scripted"
        context_window_tokens = 262144

        def __init__(self):
            self.seen = []

        async def generate(self, messages, tools):
            self.seen.append(messages)
            if len(self.seen) == 1:
                return _call("api_fetch", method="GET", url="https://slack.com/api/conversations.list")
            return _call("finish", summary="done")

    client = ScriptedClient()
    observed = []
    monkeypatch.setattr(experiment, "load_settings", lambda _: ModelSettings())
    monkeypatch.setattr(experiment, "load_api_key", lambda *_: SecretStr("offline-secret"))
    monkeypatch.setattr(experiment, "build_client", lambda *_: client)
    monkeypatch.setattr(experiment, "observe", lambda _ledger, *call: observed.append(call))
    monkeypatch.setattr(experiment, "render", lambda _ledger: rendered)
    name = json.loads(dataset("simple")[0]["info"])["task_name"]
    settings = TrialSettings(task_name=name, review=review)
    result = asyncio.run(run_trial(tmp_path, settings, tmp_path / "trial"))
    return result, client.seen, observed


def test_first_finish_is_held_once_with_the_review(tmp_path, monkeypatch):
    result, seen, observed = _run(tmp_path, monkeypatch, review=True, rendered="ROW -> nothing written")
    assert result["finished"]
    # fetch, held finish, accepted finish
    assert len(seen) == 3
    assert "ROW -> nothing written" in seen[2][-1].content
    assert result["review"]["text"] == "ROW -> nothing written"
    assert observed and observed[0][0]["url"].endswith("conversations.list")


def test_empty_review_and_baseline_finish_at_once(tmp_path, monkeypatch):
    result, seen, _ = _run(tmp_path, monkeypatch, review=True, rendered="")
    assert result["finished"] and len(seen) == 2 and result["review"]["text"] == ""
    result, seen, observed = _run(tmp_path, monkeypatch, review=False, rendered="ROW")
    assert result["finished"] and len(seen) == 2 and "review" not in result and not observed
