import json
from types import SimpleNamespace

from worldmodel_harness.context import PREFIX, render_context, render_episodes, select_procedures
from worldmodel_harness.knowledge import (
    Claim,
    Dependency,
    Episode,
    Layer,
    LiveState,
    Scope,
    WorkingGraph,
)
from worldmodel_harness.procedures import Procedure, Step
from worldmodel_harness.store import KnowledgeStore


def artifact(**updates):
    base = {
        "procedure_id": "salesforce-query",
        "scope": Scope(environment="test", version="1"),
        "goal_pattern": "Salesforce query",
        "parameter_types": {},
        "required_claims": (),
        "status": "active",
        "steps": (Step(name="stop", kind="stop"),),
    }
    return Procedure(**(base | updates))


def test_selection_excludes_stale_inapplicable_and_irrelevant():
    good = artifact()
    unavailable = artifact(procedure_id="unavailable", prerequisites={"permission": True})
    suspended = artifact(procedure_id="suspended", status="suspended")
    irrelevant = artifact(procedure_id="email", goal_pattern="Send email")
    store = SimpleNamespace(procedures=lambda _: [good, unavailable, suspended, irrelevant])
    chosen, rejected = select_procedures(store, good.scope, LiveState("task"), "Salesforce query")
    assert chosen == [good]
    assert rejected == {
        "unavailable": "unresolved_prerequisite",
        "suspended": "inactive_or_out_of_scope",
    }


def test_linked_procedure_is_offered_without_shared_goal_words():
    from worldmodel_harness.knowledge import Dependency

    linked = artifact(
        procedure_id="bind-message",
        goal_pattern="Read the first mailbox entry",
        required_claims=(Dependency(claim_id="gmail-get", version=1),),
    )
    unrelated = artifact(procedure_id="email", goal_pattern="Send email")
    claim = Claim(
        claim_id="gmail-get",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("gmail",),
        statement="Fetch one Gmail message",
        scope=linked.scope,
        status="supported",
    )
    def claim_for(claim_id):
        if claim_id != "gmail-get":
            raise KeyError(claim_id)
        return claim

    store = SimpleNamespace(
        procedures=lambda _: [linked, unrelated],
        claim=claim_for,
        dependency_compatible=lambda dependency: True,
    )
    chosen, rejected = select_procedures(
        store,
        linked.scope,
        LiveState("task"),
        "quarterly revenue workbook",
        linked_claim_ids={"gmail-get"},
    )
    assert chosen == [linked]
    assert rejected == {}


def test_whole_message_budget_includes_artifacts_and_metadata():
    procedure = artifact(goal_pattern="Salesforce " * 1000)
    working = WorkingGraph(scope=procedure.scope, gaps=tuple("missing" for _ in range(1000)))
    content, report = render_context(working, [procedure], max_chars=700)
    assert len(content) <= 700
    assert report["artifact_candidates"] == []
    assert "1000" in content
    assert render_context(working, [procedure], max_chars=1)[0] == ""


def test_dependency_bundle_is_not_split_by_rendering_budget():
    scope = Scope(environment="test", version="1")
    dependency = Claim(
        claim_id="contract",
        layer=Layer.INTERFACE,
        relation="schema",
        participants=("tool",),
        statement="Schema",
        scope=scope,
    )
    consumer = Claim(
        claim_id="consumer",
        layer=Layer.BEHAVIOR,
        relation="produces",
        participants=("tool",),
        statement="Produces ID",
        scope=scope,
        dependencies=(Dependency(claim_id="contract", version=1),),
    )
    working = WorkingGraph(scope=scope, claims=(consumer, dependency))
    for budget in (500, 1000, 2000, 5000):
        content, report = render_context(working, [], max_chars=budget)
        assert len(content) <= budget
        if "consumer" in report["injected_claims"]:
            assert "contract" in report["injected_claims"]
        if content:
            assert json.loads(content[len(PREFIX) :])["claims"] is not None


def test_historical_episode_retrieval_is_not_live_state(tmp_path):
    store = KnowledgeStore(tmp_path / "memory.sqlite")
    previous = Scope(environment="test", version="1", tenant="old", task_id="old-task")
    episode = Episode(
        task_id="old-task",
        scope=previous,
        operation="lookup",
        arguments={},
        response={"id": "old-record"},
        outcome="success",
        coverage=("id",),
    )
    store.append_episode(episode)
    current = Scope(environment="test", version="1", tenant="new", task_id="new-task")
    content, selected = render_episodes(store, current, "lookup record", max_chars=2000)
    assert selected["injected_episodes"] == [episode.episode_id]
    assert "NOT current task state" in content
    assert store.episodes(current) == []
    assert render_episodes(store, current, "lookup", max_chars=1)[0] == ""
    different = current.model_copy(update={"version": "2"})
    assert render_episodes(store, different, "lookup", max_chars=2000)[1]["injected_episodes"] == []
    store.close()
