import asyncio
import json
from types import SimpleNamespace

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.goal_checks import GoalTracker
from worldmodel_harness.knowledge import (
    Episode,
    LiveState,
    OutputExpectation,
    Scope,
    response_coverage,
)
from worldmodel_harness.observable_predictions import compare_expectations, inspect_path
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.verification_tools import CheckEffect, simple_check_tool


def test_actual_adapter_array_can_verify_and_become_stale(tmp_path):
    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", store, Ledger())
    task = "Set record-1 status to done."
    adapter.task = SimpleNamespace(prompt=[{"role": "user", "content": task}])
    adapter.goal_tracker = GoalTracker(adapter)

    def api_fetch(method, url):
        return json.dumps([{"id": "record-1", "status": "done"}])

    asyncio.run(adapter.execute(api_fetch, {"method": "GET", "url": "https://example.test/records"}))
    event = store.episodes(adapter.scope)[0]
    assert event.coverage == ("0",)
    assert inspect_path(event, (0, "status"))["value"] == "done"
    request = CheckEffect(requirement_id="status", task_quote=task,
                          episode_id=event.episode_id, path=(0, "status"),
                          expected="done", anchor_text="record-1")
    result = asyncio.run(simple_check_tool(adapter).executor(request))
    assert result.success, result.content
    adapter.live.invalidate({fact.key for fact in adapter.live.facts()}, "later-write")
    assert adapter.goal_tracker.assess(adapter.goal_tracker.checks["status"]) == "stale_evidence"
    store.close()


def test_array_coverage_does_not_invent_unobserved_elements():
    scope = Scope(environment="test", version="1", task_id="task")
    event = Episode(task_id="task", scope=scope, operation="api_fetch", arguments={},
                    response=[{"value": 7}, {"value": 8}], coverage=("0",), outcome="success")
    for path in ((0, "value"), ("0", "value")):
        assert inspect_path(event, path)["result"] == "observed"
        predicted = event.model_copy(update={"output_expectations": (
            OutputExpectation(path=path, expected=7),
        )})
        assert compare_expectations(predicted)[0]["result"] == "matched"
    assert inspect_path(event, (1, "value"))["result"] == "uncovered_evidence_path"
    assert inspect_path(event, (2, "value"))["result"] == "invalid_evidence_path"
    assert response_coverage([]) == ()
    live = LiveState("task")
    live.observe(event)
    assert len(live.facts()) == 1
