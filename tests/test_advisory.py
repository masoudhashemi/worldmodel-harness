import asyncio
import json

import pytest
from stirrup.core.models import AssistantMessage, ToolCall

pytest.importorskip("automationbench")

from worldmodel_harness.advisory import build_note, detect_loop, task_services
from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.store import KnowledgeStore

INBOX = "https://gmail.googleapis.com/gmail/v1/users/me/messages"


def _learn_inbox_read(store, row_index):
    """One task that discovers Gmail endpoints and reads the inbox successfully."""
    from automationbench.tools.api import api_fetch, api_search

    adapter = AutomationBenchAdapter(dataset("simple")[row_index], "simple", Ledger(), store)

    async def run():
        for query in ("gmail list messages", "gmail create draft", "gmail send message"):
            await adapter.execute(api_search, {"query": query, "top_k": 5})
        await adapter.execute(api_fetch, {"method": "GET", "url": INBOX, "params": '{"q":"in:inbox"}'})

    asyncio.run(run())
    return adapter


def _call(name, **arguments):
    return AssistantMessage(blocks=[ToolCall(name=name, arguments=json.dumps(arguments))])


def test_note_lists_routes_confirmed_in_two_tasks_and_names_gaps(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    text = "Draft an email to the board summarizing the inbox. Then post it in Slack #general."
    first = _learn_inbox_read(store, 0)
    _, info = build_note(store, first.scope, text)
    # One task is not enough evidence for a route.
    assert "gmail.users.messages.list" not in info["endpoints"]
    second = _learn_inbox_read(store, 1)
    note, info = build_note(store, second.scope, text)
    assert "gmail.users.messages.list" in info["endpoints"]
    assert f"GET {INBOX.replace('/me/', '/{userId}/')}" in note
    # The task drafts; a confirmed inbox read must not pass for a drafting route.
    assert "gmail draft" in info["uncovered"]
    assert "incomplete" in note
    assert len(note) < 2000
    store.close()


def test_services_follow_task_literals_not_only_names():
    known = {"gmail", "sheets", "google_drive", "slack", "zoom"}
    text = "Traffic is in ss_traffic. Send recommendations to loc@company.example.com and #ops."
    assert task_services(text, known) == ["gmail", "google_drive", "sheets", "slack"]


def test_write_endpoints_carry_their_request_format(tmp_path):
    from worldmodel_harness.advisory import _endpoint_line

    send = {
        "id": "gmail.users.messages.send",
        "method": "POST",
        "url": "https://gmail.googleapis.com/gmail/v1/users/{userId}/messages/send",
        "request": "Message with raw (base64url-encoded RFC 2822) or payload with headers[]",
    }
    assert "request: Message with raw" in _endpoint_line(send)
    read = {**send, "method": "GET", "id": "gmail.users.messages.list"}
    assert "request:" not in _endpoint_line(read)


def test_loop_is_the_same_call_three_times_in_the_last_six():
    search = _call("api_search", query="policy spreadsheet id")
    other = [_call("api_search", query=f"q{i}") for i in range(4)]
    assert detect_loop([search, *other, search]) is None
    loop = detect_loop([search, other[0], search, other[1], search])
    assert loop["tool"] == "api_search" and loop["count"] == 3
    assert detect_loop([_call("finish", summary="done")] * 3) is None


def test_advisory_trial_uses_baseline_tools_and_withdraws_the_note_on_a_loop(tmp_path, monkeypatch):
    from pydantic import SecretStr

    from worldmodel_harness import experiment
    from worldmodel_harness.config import ModelSettings
    from worldmodel_harness.experiment import TrialSettings, run_trial

    knowledge = tmp_path / "knowledge.sqlite"
    store = KnowledgeStore(knowledge)
    for i in (0, 1):
        _learn_inbox_read(store, i)
    store.close()

    class ScriptedClient:
        model_slug = "scripted"
        context_window_tokens = 262144
        seen = []

        async def generate(self, messages, tools):
            self.seen.append((sorted(tools), messages[0].content))
            if len(self.seen) <= 3:
                return _call("api_search", query="gmail inbox")
            return _call("finish", summary="done")

    client = ScriptedClient()
    monkeypatch.setattr(experiment, "load_settings", lambda _: ModelSettings())
    monkeypatch.setattr(experiment, "load_api_key", lambda *_: SecretStr("offline-secret"))
    monkeypatch.setattr(experiment, "build_client", lambda *_: client)
    name = json.loads(dataset("simple")[2]["info"])["task_name"]
    settings = TrialSettings(task_name=name, advisory=True)
    result = asyncio.run(run_trial(tmp_path, settings, tmp_path / "trial", knowledge))
    assert result["finished"]
    tools, first_system = client.seen[0]
    assert tools == ["api_fetch", "api_search", "base64_encode", "finish"]
    assert "gmail.users.messages.list" in result["advisory"]["endpoints"]
    assert "Advisory Memory" in first_system
    assert client.seen[1][1] == first_system  # a fixed note, not regrown each turn
    # The same call three times withdraws the note for the rest of the task.
    assert "Advisory Memory" not in client.seen[3][1]
    assert result["advisory"]["withdrawn"]["reason"] == "repeated_action"
    assert (tmp_path / "trial" / "result.json").exists()


def test_baseline_trial_has_no_store_and_no_note(tmp_path, monkeypatch):
    from pydantic import SecretStr

    from worldmodel_harness import experiment
    from worldmodel_harness.config import ModelSettings
    from worldmodel_harness.experiment import TrialSettings, run_trial

    class ScriptedClient:
        model_slug = "scripted"
        context_window_tokens = 262144
        systems = []

        async def generate(self, messages, tools):
            self.systems.append(messages[0].content)
            return _call("finish", summary="done")

    client = ScriptedClient()
    monkeypatch.setattr(experiment, "load_settings", lambda _: ModelSettings())
    monkeypatch.setattr(experiment, "load_api_key", lambda *_: SecretStr("offline-secret"))
    monkeypatch.setattr(experiment, "build_client", lambda *_: client)
    name = json.loads(dataset("simple")[0]["info"])["task_name"]
    result = asyncio.run(run_trial(tmp_path, TrialSettings(task_name=name), tmp_path / "trial"))
    assert result["finished"] and "advisory" not in result
    assert "Advisory Memory" not in client.systems[0]
    assert not list(tmp_path.glob("**/*.sqlite"))


def test_user_placement_leaves_the_system_prompt_unchanged():
    from stirrup.core.models import SystemMessage, UserMessage

    from worldmodel_harness.experiment import _with_note

    messages = [SystemMessage(content="agent"), SystemMessage(content="rules"), UserMessage(content="task")]
    system = _with_note(messages, "hint", "system")
    assert system[0].content.endswith("### Advisory Memory\nhint") and len(system) == 3
    user = _with_note(messages, "hint", "user")
    assert [m.content for m in user[:3]] == ["agent", "rules", "task"]
    assert user[3].role == "user" and user[3].content == "### Advisory Memory\nhint"
