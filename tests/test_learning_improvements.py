import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from stirrup.core.models import AssistantMessage, TextBlock, Tool, ToolResult

from worldmodel_harness.budget import Ledger
from worldmodel_harness.decisions import CompactDecision
from worldmodel_harness.goal_checks import (
    CheckedFinish,
    GoalCheck,
    GoalTracker,
    GoalUpdate,
    OutcomeContract,
)
from worldmodel_harness.knowledge import (
    Claim,
    Dependency,
    Episode,
    LiveState,
    OutputExpectation,
    Record,
    Scope,
    utc_now,
)
from worldmodel_harness.opportunities import trace_slices
from worldmodel_harness.procedures import ArtifactInput, Procedure, Step, artifact_tool
from worldmodel_harness.retrieval import entity_match, retrieve
from worldmodel_harness.rule_tests import observe_rules
from worldmodel_harness.semantic_learning import (
    SemanticProposal,
    evidence_catalog,
    retain_hypotheses,
)
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.structured_output import generate_checked


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="arbitrary", version="1", tenant="tenant", task_id="initial")
    yield store, scope
    store.close()


def event(scope, **changes):
    return Episode(
        **{
            "task_id": scope.task_id,
            "scope": scope,
            "operation": "read_anything",
            "arguments": {"collection": "widgets"},
            "response": {"value": 3},
            "outcome": "success",
            "coverage": ("value",),
            **changes,
        }
    )


def hypothesis(store, scope, statement="Possible effect"):
    source = event(scope)
    store.append_episode(source)
    proposal = SemanticProposal.model_validate(
        {
            "hypotheses": [
                {
                    "layer": "behavior",
                    "relation": "possible_effect",
                    "participants": ["widgets"],
                    "statement": statement,
                    "citations": [next(iter(evidence_catalog([source])))],
                    "alternatives": ["Could be incidental"],
                    "distinguishing_observation": "Read another task",
                    "response_rule": {
                        "operation": "read_anything",
                        "argument_conditions": {"collection": "widgets"},
                        "prediction": {
                            "path": ["value"],
                            "predicate": "type",
                            "expected": "integer",
                        },
                    },
                }
            ]
        }
    )
    claim_id = retain_hypotheses(proposal, [source], store, scope)[0]
    return store.claim(claim_id)


def observe(store, scope, **changes):
    e = event(scope, timestamp=utc_now() + timedelta(seconds=1), **changes)
    store.append_episode(e)
    return e, observe_rules(store, e)


def test_compact_schema_no_alternatives_and_bounded_rationale():
    d = CompactDecision(action={"tool": "a1", "arguments": {}})
    assert d.rationale == ""
    assert "alternatives" not in CompactDecision.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        CompactDecision(action=d.action, rationale="x" * 321)


def test_single_costed_repair_does_not_execute_tools():
    class Client:
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            assert tools == {}
            return AssistantMessage(
                blocks=[TextBlock(text="bad" if self.calls == 1 else '{"ok":true}')]
            )

    client, accounting = Client(), {}
    result = asyncio.run(generate_checked(client, [], lambda d: d, accounting))
    assert result == {"ok": True}
    assert client.calls == accounting["returned_model_responses"] == 2
    assert len(accounting["invalid_outputs"]) == 1


def test_repair_stops_after_second_invalid_response():
    class Client:
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            return AssistantMessage(blocks=[TextBlock(text="bad")])

    client = Client()
    with pytest.raises(ValueError):
        asyncio.run(generate_checked(client, [], lambda d: d, {}))
    assert client.calls == 2


def test_focus_matches_endpoint_templates_and_preserves_dependency_bundle(world):
    store, scope = world
    assert entity_match("https://example.test/items/42?x=1", "https://example.test/items/{id}")
    assert not entity_match("https://other.test/items/42", "https://example.test/items/{id}")
    source = event(scope)
    store.append_episode(source)
    dep = Claim(
        claim_id="dep",
        layer="interface",
        relation="schema",
        participants=("other",),
        statement="Unrelated words",
        scope=scope,
        status="supported",
        supporting_episodes=(source.episode_id,),
    )
    target = Claim(
        claim_id="target",
        layer="behavior",
        relation="mapping",
        participants=("https://example.test/items/{id}",),
        statement="Useful route",
        scope=scope,
        status="supported",
        supporting_episodes=(source.episode_id,),
        dependencies=(Dependency(claim_id="dep", version=1),),
    )
    store.put_claim(dep)
    store.put_claim(target)
    graph = retrieve(
        store, scope, "no lexical match", focus_entities=("https://example.test/items/42",)
    )
    assert {c.claim_id for c in graph.claims} == {"target", "dep"}
    tiny = retrieve(
        store, scope, "", focus_entities=("https://example.test/items/42",), max_chars=1
    )
    assert not tiny.claims
    assert "budget:target" in tiny.gaps


def test_response_rule_promotes_only_after_two_future_tasks_then_disputes(world):
    store, scope = world
    h = hypothesis(store, scope)
    # Proposal's source observation cannot retrospectively verify the rule.
    assert observe_rules(store, store.episode(h.supporting_episodes[0])) == []
    one = scope.model_copy(update={"task_id": "one"})
    e, changed = observe(store, one)
    assert changed == []
    # Repeated same-event processing and multiple reads in one task are not two trials.
    assert observe_rules(store, e) == []
    assert observe(store, one)[1] == []
    _, changed = observe(store, scope.model_copy(update={"task_id": "two"}))
    tested = store.claim(changed[0])
    assert tested.status == "supported" and tested.relation == "tested_response_rule"
    assert store.claim(h.claim_id).status == "hypothesis"
    _, changed = observe(
        store, scope.model_copy(update={"task_id": "three"}), response={"value": "wrong type"}
    )
    assert store.claim(tested.claim_id).status == "disputed"
    assert store.claim(tested.claim_id).contradicting_episodes
    observe(store, scope.model_copy(update={"task_id": "four"}))
    assert store.claim(tested.claim_id).status == "disputed"


def test_relogged_generating_event_is_not_prospective_support(world):
    import sqlite3

    store, scope = world
    h = hypothesis(store, scope)
    original = store.episode(h.supporting_episodes[0])
    for task in ("relog-one", "relog-two"):
        clone = original.model_copy(update={
            "episode_id": task, "task_id": task,
            "scope": scope.model_copy(update={"task_id": task}),
            "timestamp": utc_now() + timedelta(seconds=1),
        })
        with pytest.raises(sqlite3.IntegrityError):
            store.append_episode(clone)
        with pytest.raises((KeyError, ValueError)):
            observe_rules(store, clone)
    assert not store.claim(h.claim_id).payload.get("prospective_tests")


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "unknown"},
        {"outcome": "blocked"},
        {"coverage": ()},
        {"response": {}},
        {"arguments": {"collection": "elsewhere"}},
        {"purpose": "validation"},
    ],
)
def test_unobserved_or_inapplicable_data_never_promotes(world, changes):
    store, scope = world
    hypothesis(store, scope)
    for task in ("one", "two", "three"):
        observe(store, scope.model_copy(update={"task_id": task}), **changes)
    assert not any(c.relation == "tested_response_rule" for c in store.claims())


def test_canonical_rule_merges_paraphrases_but_not_tenants(world):
    store, scope = world
    original = hypothesis(store, scope)
    again = hypothesis(store, scope, "Same prediction with completely different words")
    assert original.claim_id == again.claim_id
    assert len(again.supporting_episodes) == 2
    other = hypothesis(store, scope.model_copy(update={"tenant": "elsewhere"}))
    assert other.claim_id != original.claim_id


def test_frozen_rules_and_forged_observations_cannot_update_knowledge(world, tmp_path):
    store, scope = world
    original = hypothesis(store, scope)
    observed = event(scope.model_copy(update={"task_id": "next"}))
    store.append_episode(observed)
    forged = observed.model_copy(update={"response": {"value": "fabricated"}})
    with pytest.raises(ValueError, match="exact retained"):
        observe_rules(store, forged)
    frozen = KnowledgeStore(tmp_path / "knowledge.sqlite", frozen=True)
    try:
        assert observe_rules(frozen, observed) == []
        assert frozen.claim(original.claim_id) == original
    finally:
        frozen.close()


def test_counterevidence_suspends_dependent_procedure(world):
    store, scope = world
    hypothesis(store, scope)
    observe(store, scope.model_copy(update={"task_id": "one"}))
    _, changed = observe(store, scope.model_copy(update={"task_id": "two"}))
    supported = store.claim(changed[0])
    dependent = Procedure(
        procedure_id="dependent",
        goal_pattern="Use prediction",
        scope=supported.scope,
        parameter_types={},
        required_claims=(Dependency(claim_id=supported.claim_id, version=supported.version),),
        status="experimental",
        steps=(
            Step(name="read", kind="query", tool="read", next_step="verify"),
            Step(
                name="verify",
                kind="verify",
                value={"$ref": "steps.read.data.value"},
                predicate="type",
                expected="integer",
                next_step="stop",
            ),
            Step(name="stop", kind="stop"),
        ),
    )
    tool = Tool(
        name="read",
        parameters=Record,
        executor=lambda _: ToolResult(content='{"value":3}'),
        description="Read",
    )
    store.put_procedure(dependent, tools={"read": tool})
    assert store.procedures()[0].status == "experimental"
    observe(store, scope.model_copy(update={"task_id": "three"}), response={"value": "wrong"})
    assert not store.dependency_compatible(dependent.required_claims[0])
    assert store.procedures()[0].status == "suspended"


def test_goal_checks_reject_missing_stale_cross_task_and_discovery(world):
    store, scope = world
    live = LiveState(scope.task_id)
    adapter = SimpleNamespace(
        evidence=store,
        scope=scope,
        live=live,
        task=SimpleNamespace(prompt=({"content": "Set the widget value to 3 exactly."},)),
    )
    tracker = GoalTracker(adapter)
    assert not tracker.finish(CheckedFinish(summary="done")).success
    e = event(scope)
    store.append_episode(e)
    live.observe(e)
    check = GoalCheck(
        requirement_id="value",
        task_quote="Set the widget value to 3 exactly.",
        outcome=OutcomeContract(target=OutputExpectation(path=("value",), expected=3)),
        episode_id=e.episode_id,
        observation=OutputExpectation(path=("value",), expected=3),
    )
    assert asyncio.run(tracker.tool().executor(GoalUpdate(checks=(check,)))).success
    assert tracker.finish(CheckedFinish(summary="done")).success
    live.invalidate({"read_anything:value"}, "write")
    assert not tracker.finish(CheckedFinish(summary="done")).success
    assert tracker.finish(CheckedFinish(summary="could not verify", unresolved=True)).success
    assert tracker.completion_status == "unresolved"
    other = event(scope.model_copy(update={"task_id": "another"}))
    store.append_episode(other)
    assert (
        tracker.assess(check.model_copy(update={"episode_id": other.episode_id}))
        == "incompatible_evidence"
    )
    invented = check.model_copy(update={"task_quote": "A requirement absent from the task"})
    assert not asyncio.run(tracker.tool().executor(GoalUpdate(checks=(invented,)))).success


def test_trace_slices_include_neighbors_without_other_tasks_or_validation(world):
    store, scope = world
    before = event(scope, outcome="error")
    middle = event(scope)
    validation = event(scope, purpose="validation")
    unrelated = event(scope.model_copy(update={"task_id": "other"}))
    for e in (before, middle, validation, unrelated):
        store.append_episode(e)
    rows = trace_slices(store, (middle.episode_id,))
    assert {r["episode_id"] for r in rows} == {before.episode_id, middle.episode_id}
    from worldmodel_harness.context_compression import MandatoryContextOverflow

    with pytest.raises(MandatoryContextOverflow):
        trace_slices(store, (middle.episode_id,), max_chars=1)


def test_goal_context_is_lossless_even_beyond_old_character_cap(world):
    store, scope = world
    adapter = SimpleNamespace(evidence=store, scope=scope, live=LiveState(scope.task_id))
    tracker = GoalTracker(adapter)
    for i in range(32):
        tracker.checks[str(i)] = GoalCheck(
            requirement_id=str(i), task_quote=f"Requirement {i}: " + "long " * 150
        )
    encoded = tracker.context().split(": ", 1)[1]
    packed = json.loads(encoded)
    assert len(encoded) > 2000
    decoded = [dict(zip(packed["columns"], row, strict=True)) for row in packed["rows"]]
    assert decoded == tracker.report()
    assert len(decoded) == 32


def test_goal_context_overflow_is_explicit_and_prevents_model_call(world):
    from worldmodel_harness.context_compression import MandatoryContextOverflow
    from worldmodel_harness.experiment import ExperimentSettings, KnowledgeClient

    store, scope = world
    adapter = SimpleNamespace(
        evidence=store,
        scope=scope,
        live=LiveState(scope.task_id),
        task=SimpleNamespace(prompt=({"content": "A public task"},)),
        ledger=Ledger(),
        prediction_assessments=[],
    )
    adapter.goal_tracker = GoalTracker(adapter)
    adapter.goal_tracker.checks["all"] = GoalCheck(requirement_id="all", task_quote="A public task")

    class Client:
        async def generate(self, messages, tools):
            pytest.fail("Must not issue a model call after dropping required context")

    wrapper = KnowledgeClient(
        Client(), adapter, store, ExperimentSettings(operational=True, retrieval_chars=100)
    )
    with pytest.raises(MandatoryContextOverflow):
        asyncio.run(wrapper.generate([], {}))
    assert wrapper.retrievals[-1]["error"] == "mandatory_context_overflow"
    assert wrapper.retrievals[-1]["goal_check_count"] == 1


def test_artifact_reuse_is_measured_and_experiment_is_bounded(world):
    store, scope = world
    ledger, audit = Ledger(), []

    class Args(Record):
        key: str

    def read(params):
        ledger.charge_environment("task")
        return ToolResult(content=json.dumps({"key": params.key}))

    tools = {"read": Tool(name="read", parameters=Args, executor=read, description="Read a record")}
    procedure = Procedure(
        procedure_id="learned",
        goal_pattern="Read a key",
        scope=scope,
        parameter_types={"key": "string"},
        required_claims=(),
        status="experimental",
        steps=(
            Step(
                name="read",
                kind="query",
                tool="read",
                arguments={"key": {"$ref": "inputs.key"}},
                next_step="check",
            ),
            Step(
                name="check",
                kind="verify",
                value={"$ref": "steps.read.data.key"},
                expected={"$ref": "inputs.key"},
                next_step="stop",
            ),
            Step(name="stop", kind="stop"),
        ),
    )
    store.put_procedure(procedure, tools=tools)
    tool = artifact_tool(
        store,
        scope,
        LiveState(scope.task_id),
        tools,
        allow_experimental=True,
        audit=audit,
        ledger=ledger,
    )
    params = ArtifactInput(procedure_id="learned", inputs={"key": "current"})
    assert asyncio.run(tool.executor(params)).success
    assert asyncio.run(tool.executor(params)).success
    assert not asyncio.run(tool.executor(params)).success
    assert len(audit) == ledger.environment_calls == 2
    assert all(r["completed"] and r["environment_calls"] == 1 for r in audit)
    assert all(r["measured_saving_seconds"] is None for r in audit)
    assert audit[0]["execution_id"] != audit[1]["execution_id"]


def test_checked_finish_stays_on_sqlite_owner_thread_in_real_stirrup(world, tmp_path):
    from stirrup import Agent
    from stirrup.core.models import ToolCall

    store, scope = world
    live = LiveState(scope.task_id)
    source = event(scope)
    store.append_episode(source)
    live.observe(source)
    tracker = GoalTracker(SimpleNamespace(evidence=store, scope=scope, live=live))
    tracker.checks["value"] = GoalCheck(
        requirement_id="value",
        task_quote="Set the value to 3 exactly",
        outcome=OutcomeContract(target=OutputExpectation(path=("value",), expected=3)),
        episode_id=source.episode_id,
        observation=OutputExpectation(path=("value",), expected=3),
    )

    class Client:
        model_slug = "offline-scripted"
        context_window_tokens = 32000

        async def generate(self, messages, tools):
            return AssistantMessage(
                blocks=[ToolCall(name="finish", arguments='{"summary":"Checked value"}')]
            )

    async def run():
        agent = Agent(
            client=Client(),
            name="checked-finish",
            tools=[],
            max_turns=2,
            finish_tool=tracker.finish_tool(),
        )
        async with agent.session(output_dir=tmp_path / "finish-session") as session:
            finish, _, _ = await session.run(
                "Finish the synthetic task using its existing evidence."
            )
        return finish

    assert asyncio.run(run()).summary == "Checked value"
    assert tracker.completion_status == "checked"


def test_parameterized_url_and_value_kind_rule_transfers_across_task_tenants(world):
    store, base_scope = world
    scope_1 = base_scope.model_copy(update={"tenant": "task_1", "task_id": "task_1"})
    ep1 = event(
        scope_1,
        operation="api_fetch",
        arguments={
            "method": "PATCH",
            "url": "https://example.test/sobjects/Contact/003001",
            "body": {"Email": "alice@example.org"},
        },
        response={"status": 204},
        coverage=("status",),
    )
    store.append_episode(ep1)
    proposal = SemanticProposal.model_validate(
        {
            "hypotheses": [
                {
                    "layer": "behavior",
                    "relation": "possible_effect",
                    "participants": ["https://example.test/sobjects/Contact/{id}"],
                    "statement": "PATCH Contact with email returns 204",
                    "citations": [next(iter(evidence_catalog([ep1])))],
                    "alternatives": ["May return 200"],
                    "distinguishing_observation": "PATCH another Contact record",
                    "response_rule": {
                        "operation": "api_fetch",
                        "argument_conditions": {
                            "method": "PATCH",
                            "url": "https://example.test/sobjects/Contact/{id}",
                            "body": {"Email": "alice@example.org"},
                        },
                        "prediction": {
                            "path": ["status"],
                            "predicate": "equals",
                            "expected": 204,
                        },
                        "value_roles": [
                            {
                                "path": ["argument_conditions", "body", "Email"],
                                "role": "parameter",
                            }
                        ],
                    },
                }
            ]
        }
    )
    claim_id = retain_hypotheses(proposal, [ep1], store, scope_1)[0]
    claim = store.claim(claim_id)
    assert claim.scope.tenant is None
    assert claim.payload["response_rule"]["argument_conditions"]["body"]["Email"] == {
        "$kind": "email"
    }

    scope_2 = base_scope.model_copy(update={"tenant": "task_2", "task_id": "task_2"})
    _, changed_2 = observe(
        store,
        scope_2,
        operation="api_fetch",
        arguments={
            "method": "PATCH",
            "url": "https://example.test/sobjects/Contact/003002",
            "body": {"Email": "bob@company.io"},
        },
        response={"status": 204},
        coverage=("status",),
    )
    assert changed_2 == []

    scope_3 = base_scope.model_copy(update={"tenant": "task_3", "task_id": "task_3"})
    _, changed_3 = observe(
        store,
        scope_3,
        operation="api_fetch",
        arguments={
            "method": "PATCH",
            "url": "https://example.test/sobjects/Contact/003003",
            "body": {"Email": "carol@domain.net"},
        },
        response={"status": 204},
        coverage=("status",),
    )
    assert len(changed_3) == 1
    tested = store.claim(changed_3[0])
    assert tested.status == "supported"
    assert tested.relation == "tested_response_rule"
    assert tested.scope.tenant is None


def test_learn_semantics_prefers_non_discovery_and_downgrades_untestable_rule(world):
    from worldmodel_harness.semantic_learning import learn_semantics

    store, scope = world
    discovery = event(
        scope,
        operation="api_search",
        purpose="discovery",
        arguments={"query": "contact"},
        response={"endpoints": [{"id": "patch_contact"}]},
        coverage=("endpoints",),
    )
    execution = event(
        scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://example.test/sobjects/Contact/003001"},
        response={"Id": "003001", "Email": "alice@example.org"},
        coverage=("Id", "Email"),
    )
    store.append_episode(discovery)
    store.append_episode(execution)

    class Client:
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            prompt = json.loads(messages[1].content)
            assert [e["episode_id"] for e in prompt["episodes"]] == [execution.episode_id]
            ref = next(iter(prompt["evidence_catalog"]))
            payload = {
                "hypotheses": [
                    {
                        "layer": "behavior",
                        "relation": "possible_effect",
                        "participants": ["api_fetch"],
                        "statement": "Untestable rule should downgrade to null without retrying",
                        "citations": [ref],
                        "alternatives": ["Other"],
                        "distinguishing_observation": "Check response",
                        "response_rule": {
                            "operation": "api_fetch",
                            "argument_conditions": {"method": "POST"},
                            "prediction": {
                                "path": ["NonExistentField"],
                                "predicate": "equals",
                                "expected": "nope",
                            },
                        },
                    }
                ]
            }
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(payload))])

    client, accounting = Client(), {}
    ids = asyncio.run(learn_semantics(client, store, scope, accounting=accounting))
    assert client.calls == 1
    assert len(ids) == 1
    assert store.claim(ids[0]).payload["response_rule"] is None


def test_http_method_path_and_dotted_arg_response_rule_normalizes_and_promotes(world):
    store, base_scope = world
    scope_1 = base_scope.model_copy(update={"tenant": "task_1", "task_id": "task_1"})
    ep1 = event(
        scope_1,
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://mailchimp.test/3.0/lists/list_001/members",
            "body": {"email_address": "alex@example.com", "status": "subscribed"},
        },
        response={
            "id": "hash_1",
            "email_address": "alex@example.com",
            "status": "subscribed",
        },
        coverage=("id", "email_address", "status"),
    )
    store.append_episode(ep1)
    proposal = SemanticProposal.model_validate(
        {
            "hypotheses": [
                {
                    "layer": "behavior",
                    "relation": "possible_effect",
                    "participants": ["https://mailchimp.test/3.0/lists/{id}/members"],
                    "statement": "POST /lists/{list_id}/members returns created subscriber with status",
                    "citations": [next(iter(evidence_catalog([ep1])))],
                    "alternatives": ["May return only id"],
                    "distinguishing_observation": "POST another subscriber",
                    "response_rule": {
                        "operation": "POST /lists/{list_id}/members",
                        "argument_conditions": {
                            "body.email_address": "string (email format)",
                            "body.status": "subscribed",
                        },
                        "prediction": {
                            "path": ["status"],
                            "predicate": "equals",
                            "expected": "subscribed",
                        },
                        "value_roles": [
                            {
                                "path": ["argument_conditions", "body", "email_address"],
                                "role": "parameter",
                            }
                        ],
                    },
                }
            ]
        }
    )
    claim_id = retain_hypotheses(proposal, [ep1], store, scope_1)[0]
    claim = store.claim(claim_id)
    rule = claim.payload["response_rule"]
    assert rule is not None
    assert rule["operation"] == "api_fetch"
    assert rule["argument_conditions"]["method"] == "POST"
    assert rule["argument_conditions"]["body"]["email_address"] == {"$kind": "email"}

    for idx in (2, 3):
        scope_next = base_scope.model_copy(
            update={"tenant": f"task_{idx}", "task_id": f"task_{idx}"}
        )
        _, changed = observe(
            store,
            scope_next,
            operation="api_fetch",
            arguments={
                "method": "POST",
                "url": f"https://mailchimp.test/3.0/lists/list_00{idx}/members",
                "body": {
                    "email_address": f"user{idx}@company.org",
                    "status": "subscribed",
                    "merge_fields": {"FNAME": "User"},
                },
            },
            response={
                "id": f"hash_{idx}",
                "email_address": f"user{idx}@company.org",
                "status": "subscribed",
            },
            coverage=("id", "email_address", "status"),
        )
    assert len(changed) == 1
    assert store.claim(changed[0]).relation == "tested_response_rule"
    assert store.claim(changed[0]).status == "supported"


def test_checkable_paths_extracted_for_verifiable_episodes(world):
    from worldmodel_harness.operational_tools import _checkable_paths

    _, scope = world
    ep = event(
        scope,
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://app.asana.com/api/1.0/tasks"},
        response={
            "status": 201,
            "data": {"gid": "task_900", "name": "Enable dark mode", "completed": False},
        },
        coverage=("status", "data"),
    )
    paths = _checkable_paths(ep)
    assert ["data", "gid"] in paths
    assert ["data", "name"] in paths
    assert ["status"] not in paths


def test_retrieval_prioritizes_supported_and_compact_context_strips_prose_noise(world):
    from worldmodel_harness.context import PREFIX, render_context

    store, scope = world
    ep = event(scope)
    store.append_episode(ep)
    generic_claim = Claim(
        claim_id="generic-note",
        layer="behavior",
        relation="note",
        participants=("mailchimp",),
        statement="mailchimp members list subscribe status subscribed",
        scope=scope,
        status="supported",
        needs_review=False,
        payload={
            "dimension": "INV",
            "alternatives": ["alt1"],
            "distinguishing_observation": "obs1",
            "interpretation": "Note with extra metadata",
        },
        supporting_episodes=(ep.episode_id,),
    )
    supported_contract = Claim(
        claim_id="contract-1",
        layer="interface",
        relation="endpoint_contract",
        participants=("mailchimp",),
        statement="mailchimp subscribe",
        scope=scope,
        status="supported",
        needs_review=False,
        supporting_episodes=(ep.episode_id,),
    )
    store.put_claim(generic_claim)
    store.put_claim(supported_contract)
    graph = retrieve(store, scope, "mailchimp subscribe status")
    assert graph.claims[0].claim_id == "contract-1"
    rendered, _ = render_context(graph, [], max_chars=4000)
    payload = json.loads(rendered[len(PREFIX) :])
    by_id = {c["claim_id"]: c for c in payload["claims"]}
    assert "conditions" not in by_id["contract-1"]
    assert "alternatives" not in by_id["generic-note"]["payload"]
    assert "distinguishing_observation" not in by_id["generic-note"]["payload"]


def test_workspace_identity_mapping_generalizes_as_dom_workspace_fact(world):
    from worldmodel_harness.semantic_learning import hypothesis_context

    store, base_scope = world
    scope_1 = base_scope.model_copy(update={"tenant": "task_1", "task_id": "task_1"})
    ep1 = event(
        scope_1,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://slack.com/api/conversations.list"},
        response={"channels": [{"id": "CMETRICS", "name": "metrics"}]},
        coverage=("channels",),
    )
    store.append_episode(ep1)
    proposal = SemanticProposal.model_validate(
        {
            "hypotheses": [
                {
                    "layer": "configuration",
                    "relation": "possible_identity_mapping",
                    "participants": ["#metrics", "CMETRICS"],
                    "statement": "Channel #metrics maps to channel ID CMETRICS in this workspace",
                    "citations": [next(iter(evidence_catalog([ep1])))],
                    "alternatives": ["Channel ID may differ in another workspace"],
                    "distinguishing_observation": "Verify channel name via conversations.info",
                }
            ]
        }
    )
    claim_id = retain_hypotheses(proposal, [ep1], store, scope_1)[0]
    claim = store.claim(claim_id)
    assert claim.scope.tenant is None
    assert claim.payload["dimension"] == "DOM"
    assert claim.payload["workspace_fact"] is True
    assert claim.payload["source_tenant"] == "task_1"

    scope_2 = base_scope.model_copy(update={"tenant": "task_2", "task_id": "task_2"})
    assert any(c.claim_id == claim_id for c in store.claims(scope_2))
    hint = hypothesis_context(store, scope_2, "post summary to #metrics channel")
    assert "CMETRICS" in hint
    assert '"dimension": "DOM"' in hint
    assert '"workspace_fact": true' in hint


def test_adaptation_attribution_verdict_and_static_trigger_gate(world):
    from worldmodel_harness.adaptation import attribute_adaptation
    from worldmodel_harness.procedures import ValidationCase, validate_candidate

    _, scope = world
    c_pass = ValidationCase(
        task_id="t1", task_name="t1", passed=True, isolated_execution=True, evidence_ids=("e1",)
    )
    c_fail = ValidationCase(
        task_id="t2", task_name="t2", passed=False, isolated_execution=True, evidence_ids=("e2",)
    )
    assert attribute_adaptation((c_pass, c_pass)) == "helped"
    assert attribute_adaptation((c_fail, c_fail)) == "hurt"
    assert attribute_adaptation((c_pass, c_fail)) == "no_observed_effect"
    assert attribute_adaptation((c_pass, c_pass), concurrent_changes=2) == "confounded"

    empty_surface = Procedure(
        procedure_id="empty-surface",
        goal_pattern="   ",
        trigger="",
        scope=scope,
        parameter_types={},
        required_claims=(),
        steps=(
            Step(
                name="verify",
                kind="verify",
                value={"$ref": "steps.verify.data.ok"},
                expected=True,
                next_step="stop",
            ),
            Step(name="stop", kind="stop"),
        ),
    )
    with pytest.raises(ValueError, match="goal_pattern or trigger"):
        validate_candidate(empty_surface, {})


