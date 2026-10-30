import asyncio
import json
from types import SimpleNamespace

from stirrup.core.models import TokenUsage, ToolMessage

from worldmodel_harness.decisions import CompactDecision, ProposedAction
from worldmodel_harness.experiment import KnowledgeClient
from worldmodel_harness.goal_checks import MAX_VERIFICATION_REFUSALS, CheckedFinish, GoalTracker
from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.operational_requirements import api_requirements
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.trajectory_graph import SAMPLING_SALT, TransitionEdge


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

    # Second finish call: nothing is outstanding, so completion is allowed
    res2 = tracker.finish(CheckedFinish(summary="All mutations completed"))
    assert res2.success
    assert tracker.completion_status == "unverified"


class UnverifiedWriteTracker(GoalTracker):
    """A PATCH wrote `status`, but no check reads it back."""

    def _written_slots(self):
        return set(), {"status"}


def test_unverified_write_gets_bounded_specific_reminders_before_soft_completion():
    class DummyAdapter:
        task = None
        entity_tracker = None

    tracker = UnverifiedWriteTracker(DummyAdapter())
    for _ in range(MAX_VERIFICATION_REFUSALS):
        result = tracker.finish(CheckedFinish(summary="done"))
        assert not result.success
        feedback = json.loads(result.content)
        assert feedback["unverified_written_fields"] == ["status"]
        assert "status" in feedback["next"]
        assert "call finish again" not in feedback["next"]

    # Verification cannot spiral: after the bounded reminders the finish is accepted,
    # and the run reports it as unverified rather than checked.
    final = tracker.finish(CheckedFinish(summary="done"))
    assert final.success
    assert tracker.completion_status == "unverified"


def test_entity_refusal_does_not_count_as_verification_reminder():
    class Entity:
        name, badge_id, entity_id = "Avery", None, "e1"

    class Tracker:
        def __init__(self):
            self.open = [Entity()]

        def unresolved(self):
            return []

        def pending_entities(self):
            return self.open

    class DummyAdapter:
        task = None
        entity_tracker = Tracker()

    adapter = DummyAdapter()
    tracker = GoalTracker(adapter)
    assert json.loads(tracker.finish(CheckedFinish(summary="done")).content)["status"] == (
        "pending_entity_obligations"
    )
    adapter.entity_tracker.open = []
    # The entity refusal was a different guard; verification still gets its reminder.
    assert not tracker.finish(CheckedFinish(summary="done")).success
    assert tracker.finish(CheckedFinish(summary="done")).success


def test_thompson_draws_are_stable_within_a_salt_and_vary_across_salts():
    edge = TransitionEdge(source="a", target="b", success_count=3, failure_count=2)
    token = SAMPLING_SALT.set("0:task")
    try:
        first = edge.sample_beta()
        assert edge.sample_beta() == first
        SAMPLING_SALT.set("1:task")
        assert edge.sample_beta() != first
    finally:
        SAMPLING_SALT.reset(token)


def test_baseline_keeps_full_tool_history_while_operational_arm_prunes():
    long_output = "x" * 5000
    messages = [ToolMessage(content=long_output, tool_call_id=str(i), name="api_fetch") for i in range(6)]

    class Response:
        token_usage = TokenUsage()

    class FakeClient:
        def __init__(self):
            self.seen = None

        async def generate(self, sent, tools):
            self.seen = sent
            return Response()

    for operational, expect_pruned in ((False, False), (True, True)):
        client = FakeClient()
        settings = SimpleNamespace(operational=operational, memory="none", advisory=False)
        wrapper = KnowledgeClient(client, SimpleNamespace(), None, settings)
        wrapper.online_learner = None
        asyncio.run(wrapper.generate(messages, []))
        pruned = len(client.seen[0].content) < len(long_output)
        assert pruned is expect_pruned
