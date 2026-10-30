import asyncio
import json

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.advisory import build_advisory_block, task_services
from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.learning import Learner
from worldmodel_harness.store import KnowledgeStore

INBOX = "https://gmail.googleapis.com/gmail/v1/users/me/messages"


def _learn_inbox_read(store, row_index):
    """One task that discovers Gmail endpoints and reads the inbox successfully."""
    from automationbench.tools.api import api_fetch, api_search

    adapter = AutomationBenchAdapter(
        dataset("simple")[row_index], "simple", store, Ledger(), Learner(store), store
    )

    async def run():
        await adapter.execute(api_search, {"query": "gmail list messages", "top_k": 5})
        await adapter.execute(api_search, {"query": "gmail create draft", "top_k": 5})
        await adapter.execute(api_fetch, {"method": "GET", "url": INBOX, "params": '{"q":"in:inbox"}'})

    asyncio.run(run())
    return adapter


def test_note_lists_routes_confirmed_in_two_tasks_and_names_gaps(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    first = _learn_inbox_read(store, 0)
    text = "Draft an email to the board summarizing the inbox. Then post it in Slack #general."
    block, info = build_advisory_block(store, first.scope, text)
    # One task is not enough evidence for a route.
    assert "gmail.users.messages.list" not in info["endpoints"]
    second = _learn_inbox_read(store, 1)
    block, info = build_advisory_block(store, second.scope, text)
    assert "gmail.users.messages.list" in info["endpoints"]
    assert f"GET {INBOX.replace('/me/', '/{userId}/')}" in block
    # The task drafts; a confirmed inbox read must not pass for a drafting route.
    assert "gmail draft" in info["uncovered"]
    assert "incomplete" in block
    assert len(block) < 2000
    store.close()


def test_services_follow_task_literals_not_only_names():
    known = {"gmail", "sheets", "google_drive", "slack", "zoom"}
    text = "Traffic is in ss_traffic. Send recommendations to loc@company.example.com and #ops."
    assert task_services(text, known) == ["gmail", "google_drive", "sheets", "slack"]


def test_advisory_session_uses_baseline_tools_and_withdraws_note_on_a_loop(tmp_path, monkeypatch):
    from pydantic import SecretStr
    from stirrup.core.models import AssistantMessage, ToolCall

    from worldmodel_harness import experiment
    from worldmodel_harness.config import ModelSettings
    from worldmodel_harness.experiment import ExperimentSettings

    knowledge = tmp_path / "knowledge.sqlite"
    store = KnowledgeStore(knowledge)
    for i in (0, 1):
        _learn_inbox_read(store, i)
    store.close()

    class ScriptedClient:
        model_slug = "scripted"
        context_window_tokens = 262144
        calls = 0
        seen = []

        async def generate(self, messages, tools):
            self.calls += 1
            self.seen.append((sorted(tools), messages[0].content))
            if self.calls <= 3:
                call, args = "api_search", {"query": "gmail inbox"}
            else:
                call, args = "finish", {"summary": "done"}
            return AssistantMessage(blocks=[ToolCall(name=call, arguments=json.dumps(args))])

    client = ScriptedClient()
    monkeypatch.setattr(experiment, "load_settings", lambda _: ModelSettings())
    monkeypatch.setattr(experiment, "load_api_key", lambda *_: SecretStr("offline-secret"))
    monkeypatch.setattr(experiment, "build_client", lambda *_: client)
    name = json.loads(dataset("simple")[2]["info"])["task_name"]
    settings = ExperimentSettings(task_names=(name,), memory="graph", phase="test", advisory=True)
    result = asyncio.run(experiment.run_automationbench(tmp_path, settings, knowledge))["results"][0]
    assert result["finished"]
    tools, first_system = client.seen[0]
    assert tools == ["api_fetch", "api_search", "base64_encode", "finish"]
    assert "gmail.users.messages.list" in result["advisory"]["endpoints"]
    assert "Advisory Memory" in first_system
    assert client.seen[1][1] == first_system  # fixed note, not regrown each turn
    # The same call three times withdraws the note for the rest of the episode.
    assert "Advisory Memory" not in client.seen[3][1]
    assert result["advisory"]["withdrawn"]["reason"] == "repeated_action"
    assert "decision_audit" not in result
