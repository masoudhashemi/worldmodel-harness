import asyncio
from types import SimpleNamespace

import pytest
from stirrup.core.models import TokenUsage, Tool, ToolMessage, ToolResult

from worldmodel_harness.baseline_fallback import BaselineFallback, detect_trap
from worldmodel_harness.experiment import KnowledgeClient


def _row(tool, arguments, allowed=True):
    return {
        "decision": {"action": {"tool": tool, "arguments": arguments}},
        "gate": {"allowed": allowed},
    }


def test_identical_action_recurring_in_window_is_a_trap():
    query = {"query": "meeting priority policy spreadsheet id", "top_k": 5}
    lookup = {"query": "meeting priority policy spreadsheet id", "max_chars": 4000}
    # The E014 zoom_calendar_conflict loop: search and lookup alternate on one query.
    audit = []
    for _ in range(2):
        audit += [_row("api_search", query), _row("lookup_knowledge", lookup)]
    assert detect_trap(audit) is None
    audit.append(_row("api_search", query))
    trap = detect_trap(audit)
    assert trap["reason"] == "repeated_action"
    assert trap["tool"] == "api_search" and trap["count"] == 3


def test_varied_progress_and_spread_out_repeats_are_not_traps():
    audit = [_row("api_search", {"query": f"q{i}"}) for i in range(10)]
    assert detect_trap(audit) is None
    read = {"method": "GET", "url": "https://x/records/1"}
    spread = []
    for i in range(3):
        spread += [_row("api_fetch", read)] + [_row("api_search", {"query": f"s{i}{j}"}) for j in range(3)]
    assert detect_trap(spread) is None


def test_consecutive_gate_blocks_are_a_trap():
    audit = [_row("api_fetch", {"url": f"https://x/{i}"}, allowed=False) for i in range(2)]
    assert detect_trap(audit) is None
    audit.append(_row("api_fetch", {"url": "https://x/3"}, allowed=False))
    assert detect_trap(audit)["reason"] == "blocked_streak"


def test_disabled_fallback_never_engages():
    fallback = BaselineFallback(enabled=False)
    audit = [_row("api_search", {"query": "same"})] * 5
    assert fallback.observe(audit) == ""
    assert not fallback.active


pytest.importorskip("automationbench")


def test_trapped_operational_episode_continues_as_the_baseline(tmp_path):
    from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
    from worldmodel_harness.budget import Ledger
    from worldmodel_harness.decisions import CompactDecision, ProposedAction
    from worldmodel_harness.goal_checks import CheckedFinish
    from worldmodel_harness.learning import Learner
    from worldmodel_harness.operational_tools import operational_tools
    from worldmodel_harness.store import KnowledgeStore

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", store, Ledger(), Learner(store))
    raw = adapter.tools()
    finish = Tool(
        name="finish",
        description="Finish",
        parameters=CheckedFinish,
        executor=lambda _: ToolResult(content="Finished"),
    )
    adapter.baseline_fallback = BaselineFallback(True, baseline_tools=raw, finish_tool=finish)
    audit = []
    actor = operational_tools(
        adapter, raw, adaptive=False, allow_experimental=False, audit=audit, refresh_audit=[]
    )[0]
    search = CompactDecision(
        action=ProposedAction(tool="api_search", arguments={"query": "playbook spreadsheet id"})
    )
    results = [asyncio.run(actor.executor(search)) for _ in range(3)]
    assert "disengaged" not in results[1].content
    assert "Operational layer disengaged" in results[2].content
    assert adapter.baseline_fallback.report()["trigger"]["reason"] == "repeated_action"

    class Response:
        token_usage = TokenUsage()

    class FakeClient:
        async def generate(self, sent, tools):
            self.sent, self.tools = sent, dict(tools)
            return Response()

    client = FakeClient()
    settings = SimpleNamespace(operational=True, memory="graph", advisory=False)
    wrapper = KnowledgeClient(client, adapter, store, settings)
    live_tools = {"execute_decision": actor}
    history = [ToolMessage(content="x" * 5000, tool_call_id=str(i), name="api_search") for i in range(6)]
    asyncio.run(wrapper.generate(history, live_tools))
    # The agent's live tool map now holds the raw tools; no memory, no pruning.
    assert set(live_tools) == {t.name for t in raw} | {"finish"}
    assert live_tools["finish"].executor(None).content == "Finished"
    assert client.sent == history
    assert wrapper.retrievals == []
    # Raw tools still record evidence for the learner.
    before = len(store.episodes(adapter.scope))
    asyncio.run(live_tools["api_search"].executor(
        live_tools["api_search"].parameters(query="google drive list files")
    ))
    assert len(store.episodes(adapter.scope)) == before + 1
    store.close()


def test_real_session_switches_to_baseline_tools_after_a_trap(tmp_path, monkeypatch):
    import json

    from pydantic import SecretStr
    from stirrup.core.models import AssistantMessage, ToolCall

    from worldmodel_harness import experiment
    from worldmodel_harness.config import ModelSettings
    from worldmodel_harness.experiment import ExperimentSettings

    looped = {"action": {"tool": "api_search", "arguments": {"query": "policy spreadsheet id"}}}

    class ScriptedClient:
        model_slug = "scripted"
        context_window_tokens = 262144
        calls = 0
        offered = []

        async def generate(self, messages, tools):
            self.calls += 1
            self.offered.append(sorted(tools))
            if self.calls <= 3:
                call, args = "execute_decision", looped
            elif self.calls == 4:
                call, args = "api_search", {"query": "google drive list files"}
            else:
                call, args = "finish", {"summary": "done"}
            return AssistantMessage(blocks=[ToolCall(name=call, arguments=json.dumps(args))])

    client = ScriptedClient()
    monkeypatch.setattr(experiment, "load_settings", lambda _: ModelSettings())
    monkeypatch.setattr(experiment, "load_api_key", lambda *_: SecretStr("offline-secret"))
    monkeypatch.setattr(experiment, "build_client", lambda *_: client)
    settings = ExperimentSettings(
        task_names=("simple.email_sf_contact_phone_update",),
        operational=True,
        phase="test",
        baseline_fallback=True,
    )
    result = asyncio.run(experiment.run_automationbench(tmp_path, settings))["results"][0]
    assert result["finished"]
    assert result["baseline_fallback"]["trigger"]["reason"] == "repeated_action"
    assert "execute_decision" in client.offered[2]
    assert client.offered[3] == ["api_fetch", "api_search", "base64_encode", "finish"]
    # Three looped searches plus the direct baseline search.
    assert result["ledger"]["environment_calls"] == 4
