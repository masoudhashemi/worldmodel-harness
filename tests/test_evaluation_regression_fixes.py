import pytest
from worldmodel_harness.decisions import CompactDecision, ProposedAction
from worldmodel_harness.goal_checks import CheckedFinish, GoalTracker
from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.operational_requirements import api_requirements
from worldmodel_harness.store import KnowledgeStore


def test_exact_literal_template_beats_generic_wildcard_in_api_requirements(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    learner = Learner(store)
    scope = Scope(environment="test", version="1", tenant="a", task_id="t1")
    search = Episode(
        task_id="t1",
        scope=scope,
        operation="api_search",
        arguments={},
        response={"results": []},
        outcome="success",
    )
    store.append_episode(search)
    generic = {
        "id": "airtable.records.list",
        "method": "GET",
        "url": "https://api.airtable.com/v0/{baseId}/{tableId}",
    }
    literal = {
        "id": "airtable.meta.bases",
        "method": "GET",
        "url": "https://api.airtable.com/v0/meta/bases",
    }
    for payload in (generic, literal):
        learner.document(
            search,
            relation="endpoint_contract",
            participants=(payload["id"], payload["url"], payload["method"]),
            statement="Contract",
            payload=payload,
            scope=Scope(environment="test", version="1"),
        )

    # Calling the exact literal URL must match the literal contract cleanly
    reqs = api_requirements(
        store,
        scope,
        ProposedAction(tool="api_fetch", arguments={"method": "GET", "url": "https://api.airtable.com/v0/meta/bases"}),
    )
    assert reqs.operation_id == "airtable.meta.bases"
    store.close()


def test_compact_decision_auto_truncates_llm_dict_rationale():
    # LLM passes tool arguments as a dict with long rationale
    data = {
        "action": {"tool": "api_search", "arguments": {"query": "test query"}},
        "goal": "Search endpoint",
        "rationale": "R" * 500,
    }
    d = CompactDecision.model_validate(data)
    assert len(d.rationale) == 320
    assert d.rationale == "R" * 320


def test_goal_tracker_second_finish_call_allows_soft_completion():
    class DummyAdapter:
        task = None
        entity_tracker = None

    tracker = GoalTracker(DummyAdapter())
    # First finish call: returns needs_verification with guidance
    res1 = tracker.finish(CheckedFinish(summary="All mutations completed"))
    assert not res1.success
    assert tracker.finish_audit[-1]["success"] is False

    # Second finish call: allows completion
    res2 = tracker.finish(CheckedFinish(summary="All mutations completed"))
    assert res2.success
    assert tracker.completion_status in ("checked", "unverified")
