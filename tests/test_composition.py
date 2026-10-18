from worldmodel_harness.composition import CompositionLearner
from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.store import KnowledgeStore


def setup(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    learner = Learner(store)
    scope = Scope(environment="fixture", version="1")
    for endpoint in (
        {"id": "records.list", "url": "https://test/records", "method": "GET"},
        {"id": "records.update", "url": "https://test/records/{id}", "method": "PATCH"},
    ):
        event = Episode(
            task_id="schema",
            scope=scope,
            operation="tool_schema",
            arguments={},
            response=endpoint,
            outcome="success",
        )
        store.append_episode(event)
        learner.document(
            event,
            relation="endpoint_contract",
            participants=(endpoint["id"],),
            statement=endpoint["id"],
            payload=endpoint,
            scope=scope,
        )
    return store, CompositionLearner(learner), scope


def pair(store, scope, *, value="private-record-123", source_changes=None, target_changes=None):
    source = Episode(
        **{
            "task_id": scope.task_id,
            "scope": scope,
            "operation": "api_fetch",
            "arguments": {"method": "GET", "url": "https://test/records"},
            "response": {"records": [{"id": value}]},
            "coverage": ("records",),
            "outcome": "success",
            **(source_changes or {}),
        }
    )
    store.append_episode(source)
    target = Episode(
        **{
            "task_id": scope.task_id,
            "scope": scope,
            "operation": "api_fetch",
            "arguments": {"method": "PATCH", "url": "https://test/records/" + value},
            "response": {},
            "outcome": "success",
            "pre_evidence": (source.episode_id,),
            **(target_changes or {}),
        }
    )
    store.append_episode(target)
    return source, target


def test_observed_connection_retains_paths_and_both_endpoints_not_values(tmp_path):
    store, learner, scope = setup(tmp_path)
    source, target = pair(store, scope.model_copy(update={"tenant": "a", "task_id": "a"}))
    claim = learner.observe(target)[0]
    assert claim.status == "hypothesis"
    assert claim.payload["output_path"] == "records.*.id"
    assert claim.payload["input_path"] == "path.id"
    assert claim.supporting_episodes == (source.episode_id, target.episode_id)
    assert len(claim.dependencies) == 2
    assert "private-record-123" not in claim.model_dump_json()
    assert learner.observe(target) == []
    store.close()


def test_only_independent_tasks_and_distinct_values_support_regularities(tmp_path):
    store, learner, scope = setup(tmp_path)
    for task, value in (
        ("a", "private-record-123"),
        ("a", "private-record-456"),
        ("b", "private-record-789"),
    ):
        _, target = pair(
            store, scope.model_copy(update={"tenant": task, "task_id": task}), value=value
        )
        claim = learner.observe(target)[0]
        assert claim.status == ("supported" if task == "b" else "hypothesis")
    assert "private-record" not in claim.model_dump_json()
    assert "not proof" in claim.payload["interpretation"]
    store.close()


def test_constant_reused_across_tasks_does_not_promote(tmp_path):
    store, learner, scope = setup(tmp_path)
    for task in ("a", "b"):
        _, target = pair(store, scope.model_copy(update={"tenant": task, "task_id": task}))
        assert learner.observe(target)[0].status == "hypothesis"
    store.close()


def test_coverage_success_and_pre_action_evidence_are_mandatory(tmp_path):
    store, learner, scope = setup(tmp_path)
    scope = scope.model_copy(update={"tenant": "a", "task_id": "a"})
    for changes in (
        {"coverage": ()},
        {"outcome": "unknown"},
        {"outcome": "error"},
        {"response": {"records": [{"id": "private-record-123"}, {"id": "private-record-123"}]}},
    ):
        _, target = pair(store, scope, source_changes=changes)
        assert learner.observe(target) == []
    for changes in ({"pre_evidence": ()}, {"outcome": "error"}, {"outcome": "unknown"}):
        _, target = pair(store, scope, target_changes=changes)
        assert learner.observe(target) == []
    store.close()


def test_other_tasks_cannot_supply_bindings(tmp_path):
    store, learner, scope = setup(tmp_path)
    a = scope.model_copy(update={"tenant": "a", "task_id": "a"})
    b = scope.model_copy(update={"tenant": "b", "task_id": "b"})
    source, _ = pair(store, a)
    _, target = pair(store, b, target_changes={"pre_evidence": (source.episode_id,)})
    assert learner.observe(target) == []
    store.close()


def test_structured_body_matches_but_substrings_are_not_inferred(tmp_path):
    store, learner, scope = setup(tmp_path)
    scope = scope.model_copy(update={"tenant": "a", "task_id": "a"})
    arguments = {
        "method": "PATCH",
        "url": "https://test/records/unrelated-id",
        "body": '{"external_id":"private-record-123"}',
    }
    _, target = pair(store, scope, target_changes={"arguments": arguments})
    claim = learner.observe(target)[0]
    assert claim.payload["input_path"] == "body.external_id"
    arguments = {**arguments, "body": '{"q":"SELECT Id WHERE Id = private-record-123"}'}
    _, target = pair(store, scope, target_changes={"arguments": arguments})
    assert learner.observe(target) == []
    store.close()


def test_connection_is_reconsidered_when_either_endpoint_changes(tmp_path):
    store, learner, scope = setup(tmp_path)
    _, target = pair(store, scope.model_copy(update={"tenant": "a", "task_id": "a"}))
    claim = learner.observe(target)[0]
    for dependency in claim.dependencies:
        affected = Learner(store).reconsider(dependency.claim_id, {"url"})
        assert claim.claim_id in affected
        assert store.claim(claim.claim_id).needs_review
    store.close()
