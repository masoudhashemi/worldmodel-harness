import sqlite3
from datetime import timedelta

import pytest

from worldmodel_harness.knowledge import (
    Claim,
    Dependency,
    Episode,
    Layer,
    LiveState,
    Prediction,
    Scope,
    changed_payload_fields,
    utc_now,
)
from worldmodel_harness.learning import Learner, assess_prediction
from worldmodel_harness.retrieval import retrieve
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def store(tmp_path):
    result = KnowledgeStore(tmp_path / "knowledge.sqlite")
    yield result
    result.close()


def observation(**kwargs):
    defaults = {
        "task_id": "t1",
        "scope": Scope(environment="fixture", version="1", task_id="t1"),
        "operation": "tool_schema",
        "arguments": {},
        "response": {"id": "123"},
        "outcome": "success",
        "coverage": ("id",),
    }
    return Episode(**(defaults | kwargs))


def supported(store, claim_id="lookup", **kwargs):
    episode = observation()
    store.append_episode(episode)
    defaults = {
        "claim_id": claim_id,
        "layer": Layer.INTERFACE,
        "relation": "contract",
        "participants": ("lookup",),
        "statement": "lookup ticket",
        "status": "supported",
        "supporting_episodes": (episode.episode_id,),
        "scope": Scope(environment="fixture", version="1"),
    }
    claim = Claim(**(defaults | kwargs))
    store.put_claim(claim)
    return claim


def test_append_only_versions_and_unique_events(store):
    claim = supported(store)
    with pytest.raises(ValueError, match="contiguous"):
        store.put_claim(claim)
    store.put_claim(claim.model_copy(update={"version": 2, "status": "disputed"}))
    assert store.claim(claim.claim_id, 1).status == "supported"
    assert store.claim(claim.claim_id).status == "disputed"
    episode = store.episode(claim.supporting_episodes[0])
    with pytest.raises(sqlite3.IntegrityError):
        store.append_episode(episode.model_copy(update={"episode_id": "duplicate"}))


def test_dependency_compatibility_allows_evidence_only_updates(store):
    claim = supported(store, payload={"parameters": {"q": "string", "limit": 1}})
    dep = Dependency(claim_id=claim.claim_id, version=1, fields=("parameters.q",))
    assert store.dependency_compatible(dep)
    store.put_claim(claim.model_copy(update={"version": 2, "last_verified_at": utc_now()}))
    assert store.dependency_compatible(dep)
    store.put_claim(
        claim.model_copy(
            update={
                "version": 3,
                "payload": {"parameters": {"q": "string", "limit": 2}},
            }
        )
    )
    assert store.dependency_compatible(dep)
    assert not store.dependency_compatible(Dependency(claim_id=claim.claim_id, version=1))


@pytest.mark.parametrize(
    "change",
    [
        {"payload": {"parameters": {"q": "integer"}}},
        {"payload": {"parameters": {}}},
        {"conditions": {"authorized": True}},
        {"needs_review": True},
        {"status": "disputed"},
    ],
)
def test_dependency_compatibility_rejects_changed_or_reverted_semantics(store, change):
    claim = supported(store, payload={"parameters": {"q": "string"}})
    dep = Dependency(claim_id=claim.claim_id, version=1, fields=("parameters.q",))
    store.put_claim(claim.model_copy(update={"version": 2, **change}))
    assert not store.dependency_compatible(dep)
    store.put_claim(claim.model_copy(update={"version": 3}))
    assert not store.dependency_compatible(dep)


def test_missing_dependency_field_cannot_match_itself(store):
    claim = supported(store, payload={})
    assert not store.dependency_compatible(
        Dependency(claim_id=claim.claim_id, version=1, fields=("missing",))
    )


def test_retrieval_and_rendering_preserve_compatible_pinned_dependencies(store):
    from worldmodel_harness.context import render_context

    schema = supported(store, "schema", payload={"q": "string", "limit": 1})
    supported(
        store, "consumer", dependencies=(Dependency(claim_id="schema", version=1, fields=("q",)),)
    )
    store.put_claim(
        schema.model_copy(update={"version": 2, "payload": {"q": "string", "limit": 2}})
    )
    working = retrieve(store, schema.scope, "ticket")
    assert {c.claim_id for c in working.claims} == {"schema", "consumer"}
    _, selection = render_context(working, [], max_chars=12000, store=store)
    assert set(selection["injected_claims"]) == {"schema", "consumer"}
    store.put_claim(schema.model_copy(update={"version": 3, "payload": {"q": "integer"}}))
    working = retrieve(store, schema.scope, "ticket")
    assert "consumer" not in {c.claim_id for c in working.claims}


def test_nested_field_changes_include_removals_and_atomic_replacements():
    assert changed_payload_fields(
        {"parameters": {"q": "string", "limit": 1}, "removed": None},
        {"parameters": {"q": "string", "limit": True}, "added": None},
    ) == {"parameters.limit", "removed", "added"}
    assert changed_payload_fields({"parameters": {"q": "string"}}, {"parameters": []}) == {
        "parameters"
    }


def test_scope_does_not_leak_across_tasks_tenants_or_versions(store):
    scope = Scope(environment="fixture", version="1", tenant="a", task_id="t1")
    assert not scope.applies_to(Scope(environment="fixture", version="1", tenant="b", task_id="t1"))
    assert not scope.applies_to(Scope(environment="fixture", version="1", tenant="a", task_id="t2"))
    assert not scope.applies_to(Scope(environment="fixture", version="2", tenant="a", task_id="t1"))
    supported(store)
    assert not store.claims(Scope(environment="other", version="1"))


def test_frozen_store_cannot_mutate(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    writer = KnowledgeStore(path)
    claim = supported(writer)
    writer.close()
    reader = KnowledgeStore(path, frozen=True)
    assert reader.claim(claim.claim_id) == claim
    with pytest.raises(PermissionError):
        reader.append_episode(observation())
    reader.close()


def test_missing_field_and_timeout_are_not_negative_evidence():
    prediction = Prediction(claim_id="c", claim_version=1, path="id", expected="123")
    assert assess_prediction(prediction, observation()) == "supported"
    assert assess_prediction(prediction, observation(outcome="unknown")) == "unobserved"
    assert assess_prediction(prediction, observation(response={}, coverage=())) == "unobserved"
    assert assess_prediction(prediction, observation(response={"id": "456"})) == "contradicted"


def test_dependency_bundle_never_silently_truncated(store):
    dependency = supported(store, "schema", participants=("schema",), statement="schema")
    supported(store, "procedure", dependencies=(Dependency(claim_id="schema", version=1),))
    scope = dependency.scope
    selected = retrieve(store, scope, "ticket", max_chars=10000)
    assert {c.claim_id for c in selected.claims} == {"schema", "procedure"}
    too_small = retrieve(store, scope, "ticket", max_chars=1)
    assert too_small.blocked
    assert not too_small.claims
    assert any(gap.startswith("budget:") for gap in too_small.gaps)


def test_supported_endpoint_contract_survives_unrelated_wording(store):
    contract = supported(
        store,
        "gmail-get",
        relation="endpoint_contract",
        participants=("gmail.users.messages.get", "https://gmail.example/messages/{id}", "GET"),
        statement="Fetch one Gmail message",
    )
    supported(
        store,
        "behavior",
        relation="observed_output_type",
        layer=Layer.BEHAVIOR,
        participants=("gmail.users.messages.get",),
        statement="Observed output type",
    )
    working = retrieve(store, contract.scope, "assign the quarterly revenue workbook")
    assert {c.claim_id for c in working.claims} == {"gmail-get"}


def test_stale_claim_requires_refresh(store):
    claim = supported(store, valid_until=utc_now() - timedelta(seconds=1))
    working = retrieve(store, claim.scope, "ticket")
    assert working.blocked
    assert f"refresh:{claim.claim_id}" in working.gaps


def test_scoped_field_change_does_not_invalidate_unrelated_or_monitor(store):
    root = supported(store, "root")
    supported(
        store, "consumer", dependencies=(Dependency(claim_id="root", version=1, fields=("id",)),)
    )
    supported(
        store, "unrelated", dependencies=(Dependency(claim_id="root", version=1, fields=("name",)),)
    )
    supported(
        store, "monitor", dependencies=(Dependency(claim_id="root", version=1, mode="monitors"),)
    )
    affected = Learner(store).reconsider(root.claim_id, {"id"})
    assert affected == {"consumer"}
    assert store.claim("consumer").needs_review
    assert not store.claim("unrelated").needs_review
    assert not store.claim("monitor").needs_review


def test_live_state_is_task_local_and_event_invalidated():
    state = LiveState("t1")
    state.observe(observation())
    assert len(state.facts()) == 1
    state.invalidate({"tool_schema:id"}, "changed")
    assert not state.facts()
    with pytest.raises(ValueError):
        state.observe(observation(task_id="t2"))


def test_nested_field_change_reconsiders_consumers(store):
    supported(store, "root")
    supported(
        store,
        "nested",
        dependencies=(Dependency(claim_id="root", version=1, fields=("parameters.q",)),),
    )
    assert Learner(store).reconsider("root", {"parameters"}) == {"nested"}
