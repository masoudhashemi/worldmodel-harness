import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from stirrup.core.models import Tool

from worldmodel_harness.checkpoint import checkpoint_hash, copy_checkpoint
from worldmodel_harness.decision_runtime import dispatch_decision
from worldmodel_harness.decisions import ActionRequirements, Decision, UnresolvedRequirements
from worldmodel_harness.experiment import ExperimentSettings
from worldmodel_harness.goal_checks import (
    CheckedFinish,
    GoalCheck,
    GoalTracker,
    GoalUpdate,
    OutcomeContract,
)
from worldmodel_harness.knowledge import Claim, Episode, LiveState, OutputExpectation, Record, Scope
from worldmodel_harness.observable_predictions import compare_expectations
from worldmodel_harness.operational_requirements import api_requirements
from worldmodel_harness.resource_invalidation import invalidation_keys
from worldmodel_harness.rule_tests import (
    ResponseRule,
    applicable,
    rule_signature,
    validate_testability,
)
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "store.sqlite")
    scope = Scope(environment="test", version="1", tenant="tenant", task_id="task")
    yield store, scope
    store.close()


def observed(scope, **updates):
    return Episode(
        **{
            "task_id": scope.task_id,
            "scope": scope,
            "operation": "api_fetch",
            "arguments": {
                "method": "GET",
                "url": "https://crm.test/query",
                "params": '{"q":"items"}',
            },
            "response": {"records": [{"value": 3}]},
            "coverage": ("records",),
            "outcome": "success",
            **updates,
        }
    )


@pytest.mark.parametrize(
    "path,result",
    [
        (("records", 0, "value"), "matched"),
        (("records", "0", "value"), "matched"),
        (("records", 1, "value"), "unobserved"),
        (("records", "-1", "value"), "unobserved"),
        (("records", 0, "absent"), "unobserved"),
        (("records", "__class__"), "unobserved"),
    ],
)
def test_typed_array_paths(world, path, result):
    _, scope = world
    e = observed(scope, output_expectations=(OutputExpectation(path=path, expected=3),))
    assert compare_expectations(e)[0]["result"] == result
    assert compare_expectations(e.model_copy(update={"coverage": ()}))[0]["result"] == "unobserved"


@pytest.mark.parametrize("index", [-1, True, 1.2])
def test_invalid_index_types_rejected(index):
    with pytest.raises(ValidationError):
        OutputExpectation(path=("records", index))


def test_object_numeric_key_is_not_an_array_index(world):
    _, scope = world
    e = observed(scope, response={"records": {"0": 3}})
    assert (
        compare_expectations(
            e.model_copy(
                update={
                    "output_expectations": (OutputExpectation(path=("records", "0"), expected=3),)
                }
            )
        )[0]["result"]
        == "matched"
    )
    assert (
        compare_expectations(
            e.model_copy(
                update={
                    "output_expectations": (OutputExpectation(path=("records", 0), expected=3),)
                }
            )
        )[0]["result"]
        == "unobserved"
    )


def test_historical_read_is_not_current_state_and_cannot_downgrade(world):
    store, scope = world
    e = observed(scope)
    store.append_episode(e)
    live = LiveState(scope.task_id)
    live.observe(e)
    tracker = GoalTracker(
        SimpleNamespace(
            evidence=store,
            scope=scope,
            live=live,
            task=SimpleNamespace(prompt=[{"content": "Find the item with value 3 exactly."}]),
        )
    )
    current = GoalCheck(
        requirement_id="item",
        task_quote="Find the item with value 3 exactly.",
        outcome=OutcomeContract(
            target=OutputExpectation(path=("records", 0, "value"), expected=3),
            resource_url="https://crm.test/query",
        ),
        episode_id=e.episode_id,
        observation=OutputExpectation(path=("records", 0, "value"), expected=3),
    )
    assert tracker.assess(current) == "matched"
    assert asyncio.run(tracker.tool().executor(GoalUpdate(checks=(current,)))).success
    live.invalidate({f.key for f in live.facts()}, "write")
    assert tracker.assess(current) == "stale_evidence"
    historical = current.model_copy(update={"evidence_mode": "historical_observation"})
    assert tracker.assess(historical) == "matched"
    assert not asyncio.run(tracker.tool().executor(GoalUpdate(checks=(historical,)))).success
    write = observed(scope, arguments={"method": "PATCH", "url": "https://crm.test/items/1"})
    store.append_episode(write)
    assert (
        tracker.assess(historical.model_copy(update={"episode_id": write.episode_id}))
        == "outcome_resource_mismatch"
    )


@pytest.mark.parametrize(
    "path,result",
    [
        ((0, "value"), "invalid_evidence_path"),
        (("records", 9, "value"), "invalid_evidence_path"),
        (("records", 0, "missing"), "invalid_evidence_path"),
        (("records", 0, "value"), "stale_evidence"),
    ],
)
def test_goal_path_diagnosis_precedes_freshness(world, path, result):
    store, scope = world
    event = observed(scope)
    store.append_episode(event)
    tracker = GoalTracker(
        SimpleNamespace(evidence=store, scope=scope, live=LiveState(scope.task_id))
    )
    target = OutputExpectation(path=path, expected=3)
    check = GoalCheck(
        requirement_id="value",
        task_quote="Set to 3 exactly",
        episode_id=event.episode_id,
        observation=target,
        outcome=OutcomeContract(target=target, resource_url="https://crm.test/query"),
    )
    diagnosis = tracker.diagnose(check)
    assert diagnosis["result"] == result
    if result == "invalid_evidence_path":
        assert "failed_at" in diagnosis
    else:
        assert "permitted current read" in diagnosis["next"]


@pytest.mark.parametrize(
    "predicate,path", [("present", ()), ("present", ("records",)), ("type", ("records",))]
)
def test_weak_predicates_cannot_finish(world, predicate, path):
    store, scope = world
    event = observed(scope)
    store.append_episode(event)
    live = LiveState(scope.task_id)
    live.observe(event)
    tracker = GoalTracker(SimpleNamespace(evidence=store, scope=scope, live=live))
    check = GoalCheck(
        requirement_id="value",
        task_quote="Set to 3 exactly",
        episode_id=event.episode_id,
        observation=OutputExpectation(path=path, predicate=predicate, expected="array"),
    )
    tracker.checks["value"] = check
    assert tracker.assess(check) == "insufficient_outcome_predicate"
    assert not tracker.finish(CheckedFinish(summary="done")).success
    assert tracker.finish(CheckedFinish(summary="not verified", unresolved=True)).success


def test_grounded_contract_cannot_be_replaced_or_self_confirmed(world):
    store, scope = world
    source = observed(scope, response={"message": "Please use new-value"}, coverage=("message",))
    target = observed(scope, response={"value": "new-value"}, coverage=("value",))
    for e in (source, target):
        store.append_episode(e)
    live = LiveState(scope.task_id)
    live.observe(target)
    tracker = GoalTracker(
        SimpleNamespace(
            evidence=store,
            scope=scope,
            live=live,
            task=SimpleNamespace(prompt=[{"content": "Use the requested value."}]),
        )
    )
    predicate = OutputExpectation(path=("value",), expected="new-value")
    contract = OutcomeContract(
        target=predicate,
        source_episode_id=source.episode_id,
        source_path=("message",),
        resource_url="https://crm.test/query",
    )
    check = GoalCheck(
        requirement_id="value",
        task_quote="Use the requested value.",
        episode_id=target.episode_id,
        observation=predicate,
        outcome=contract,
    )
    assert asyncio.run(tracker.tool().executor(GoalUpdate(checks=(check,)))).success
    assert tracker.finish(CheckedFinish(summary="done")).success
    wrong_path = OutputExpectation(path=("wrong",), expected="new-value")
    tracker.checks["value"] = check.model_copy(
        update={
            "observation": wrong_path,
            "outcome": contract.model_copy(update={"target": wrong_path}),
        }
    )
    assert tracker.assess(tracker.checks["value"]) == "invalid_evidence_path"
    assert asyncio.run(tracker.tool().executor(GoalUpdate(checks=(check,)))).success
    assert tracker.assess(tracker.checks["value"]) == "matched"
    circular = check.model_copy(
        update={
            "outcome": contract.model_copy(
                update={"source_episode_id": target.episode_id, "source_path": ("value",)}
            )
        }
    )
    assert tracker.assess(circular) == "circular_outcome_grounding"
    assert not asyncio.run(tracker.tool().executor(GoalUpdate(checks=(circular,)))).success
    assert tracker.assess(check.model_copy(update={"outcome": None})) == "missing_outcome_contract"
    changed = check.model_copy(
        update={"observation": OutputExpectation(path=("value",), expected="other")}
    )
    assert tracker.assess(changed) == "outcome_contract_mismatch"
    unsupported = check.model_copy(
        update={"outcome": OutcomeContract(target=predicate, resource_url="https://crm.test/query")}
    )
    assert tracker.assess(unsupported) == "ungrounded_outcome_value"
    uncovered = target.model_copy(update={"coverage": ()})
    # A separate observation with a real but uncovered path is not a stale fact.
    uncovered = uncovered.model_copy(update={"episode_id": "uncovered", "event_id": "uncovered"})
    store.append_episode(uncovered)
    assert (
        tracker.assess(check.model_copy(update={"episode_id": uncovered.episode_id}))
        == "uncovered_evidence_path"
    )


@pytest.mark.parametrize(
    "method,url,value,result",
    [
        ("PATCH", "https://crm.test/items/1", 3, "outcome_verification_requires_read"),
        ("GET", "https://crm.test/items/2", 3, "outcome_resource_mismatch"),
        ("GET", "https://crm.test/items/1", 4, "contradicted"),
        ("GET", "https://crm.test/items/1", 3, "matched"),
    ],
)
def test_outcome_requires_target_resource_read_and_value(world, method, url, value, result):
    store, scope = world
    event = observed(
        scope,
        arguments={"method": method, "url": url},
        response={"value": value},
        coverage=("value",),
    )
    store.append_episode(event)
    live = LiveState(scope.task_id)
    live.observe(event)
    tracker = GoalTracker(SimpleNamespace(evidence=store, scope=scope, live=live))
    target = OutputExpectation(path=("value",), expected=3)
    check = GoalCheck(
        requirement_id="value",
        task_quote="Set value to 3.",
        episode_id=event.episode_id,
        observation=target,
        outcome=OutcomeContract(target=target, resource_url="https://crm.test/items/1"),
    )
    assert tracker.assess(check) == result


def test_outcome_literal_is_not_a_substring_of_another_value(world):
    store, scope = world
    tracker = GoalTracker(
        SimpleNamespace(evidence=store, scope=scope, live=LiveState(scope.task_id))
    )
    check = GoalCheck(
        requirement_id="value",
        task_quote="Set value to 33.",
        outcome=OutcomeContract(target=OutputExpectation(path=("value",), expected=3)),
    )
    assert tracker.grounding(check) == "ungrounded_outcome_value"


def contract(store, scope, name, method, url):
    source = observed(scope, operation="api_search", arguments={"query": name})
    store.append_episode(source)
    store.put_claim(
        Claim(
            claim_id=name,
            layer="interface",
            relation="endpoint_contract",
            participants=(name,),
            statement=name,
            scope=scope,
            status="supported",
            supporting_episodes=(source.episode_id,),
            payload={"id": name, "method": method, "url": url},
        )
    )


def test_invalidation_covers_queries_and_cross_origin_dependencies(world):
    store, scope = world
    live = LiveState(scope.task_id)
    for url in (
        "https://crm.test/query",
        "https://crm.test/items/1",
        "https://mail.test/messages/1",
        "https://index.test/search",
    ):
        e = observed(scope, arguments={"method": "GET", "url": url})
        store.append_episode(e)
        live.observe(e)
    contract(store, scope, "write", "PATCH", "https://crm.test/items/{id}")
    contract(store, scope, "source", "GET", "https://crm.test/query")
    contract(store, scope, "target", "GET", "https://index.test/search")
    store.put_claim(
        Claim(
            layer="behavior",
            relation="observed_value_flow",
            participants=("source", "target"),
            statement="Observed dependency",
            scope=scope,
            payload={"source_endpoint": "source", "target_endpoint": "target"},
        )
    )
    adapter = SimpleNamespace(live=live, evidence=store, knowledge=store, scope=scope)
    keys = invalidation_keys(adapter, {"method": "PATCH", "url": "https://crm.test/items/1"})
    assert len(keys) == 3
    assert not any("mail.test" in k for k in keys)
    assert any("/query" in k for k in keys) and any("index.test" in k for k in keys)
    assert (
        len(invalidation_keys(adapter, {"method": "DELETE", "url": "https://unknown.test/1"})) == 4
    )


def test_phone_rule_matches_a_different_phone_number_without_storing_it():
    from worldmodel_harness.knowledge import Episode, Scope
    from worldmodel_harness.rule_tests import ValueRole, applicable, normalize_rule, rule_signature

    literal = "+1-415-555-0100"
    rule = ResponseRule(
        operation="api_fetch",
        argument_conditions={"body": {"Phone": literal}},
        prediction=OutputExpectation(path=("Phone",), expected=literal),
        value_roles=(
            ValueRole(path=("argument_conditions", "body", "Phone"), role="parameter"),
            ValueRole(path=("prediction", "expected"), role="example"),
        ),
    )
    original = rule.model_dump_json()
    abstracted = normalize_rule(rule)
    assert rule.model_dump_json() == original
    episode = Episode(
        scope=Scope(environment="test", version="1", task_id="task"),
        task_id="task",
        operation="api_fetch",
        arguments={"method": "PATCH", "body": {"Phone": "(415) 555-0199"}},
        response={"Phone": "(415) 555-0199"},
        coverage=("Phone",),
        outcome="success",
    )
    assert applicable(abstracted, episode)
    assert literal not in rule_signature(rule)
    assert rule.model_dump_json() == original
    assert literal not in abstracted.model_dump_json()


@pytest.mark.parametrize("value,other", [
    ("2026-01-01", "2027-01-01"),
    ("https://billing.test/a", "https://billing.test/b"),
    ("a@example.test", "b@example.test"),
    ("+1-415-555-0100", "+1-415-555-0199"),
])
def test_undeclared_and_constraint_literals_are_not_generalized(value, other):
    from worldmodel_harness.rule_tests import ValueRole, normalize_rule

    base = ResponseRule(operation="read", argument_conditions={"cutoff": value},
                        prediction=OutputExpectation(path=("value",), expected=value))
    assert normalize_rule(base).prediction.expected == value
    different = base.model_copy(update={"argument_conditions": {"cutoff": other}})
    assert rule_signature(base) != rule_signature(different)
    for role in ("constraint", "unknown"):
        declared = base.model_copy(update={"value_roles": (
            ValueRole(path=("prediction", "expected"), role=role),)})
        assert normalize_rule(declared).prediction.expected == value


def test_role_identity_and_invalid_role_paths():
    from worldmodel_harness.rule_tests import ValueRole, normalize_rule

    base = ResponseRule(operation="read", argument_conditions={"date": "2026-01-01"},
                        prediction=OutputExpectation(path=("value",), expected="2026-01-01"))
    parameter = base.model_copy(update={"value_roles": (
        ValueRole(path=("prediction", "expected"), role="parameter"),)})
    assert rule_signature(base) != rule_signature(parameter)
    assert normalize_rule(parameter).prediction.expected == {"$kind": "date"}
    for path in (("prediction", "path"), ("argument_conditions", "absent"),
                 ("argument_conditions",)):
        with pytest.raises(ValueError):
            normalize_rule(base.model_copy(update={"value_roles": (
                ValueRole(path=path, role="example"),)}))


def test_rule_normalization_and_testability(world):
    _, scope = world
    e = observed(scope)
    rule = ResponseRule(
        operation="api_fetch",
        argument_conditions={"params": {"q": "items"}},
        prediction=OutputExpectation(path=("records", 0, "value"), expected=3),
    )
    assert applicable(rule, e)
    validate_testability(rule, [e])
    serialized = rule.model_copy(update={"argument_conditions": {"params": '{"q": "items"}'}})
    assert rule_signature(rule) == rule_signature(serialized)
    assert isinstance(e.arguments["params"], str)  # Never rewrite raw provenance.
    for invalid in (
        rule.model_copy(update={"argument_conditions": {"invented": 1}}),
        rule.model_copy(update={"prediction": OutputExpectation(path=("absent",), expected=3)}),
    ):
        with pytest.raises(ValueError, match="not testable"):
            validate_testability(invalid, [e])
    # Contradicted but covered is still testable, not automatically supported.
    validate_testability(
        rule.model_copy(
            update={"prediction": OutputExpectation(path=("records", 0, "value"), expected=4)}
        ),
        [e],
    )


def test_gate_returns_schema_field_errors_without_invalid_values(world):
    store, scope = world

    class Inputs(Record):
        count: int

    tool = Tool(
        name="read",
        description="read",
        parameters=Inputs,
        executor=lambda _: pytest.fail("Invalid call executed"),
    )
    decision = Decision(
        goal="inspect",
        rationale="inspect",
        action={"tool": "read", "arguments": {"count": "secret-invalid-value"}},
    )
    audit = []
    result = asyncio.run(
        dispatch_decision(
            decision,
            tools={"read": tool},
            requirements_for=lambda _: ActionRequirements(),
            store=store,
            scope=scope,
            live=LiveState(scope.task_id),
            audit=audit,
        )
    )
    response = json.loads(result.content)
    assert response["repair"]["validation_errors"][0]["field"] == ["count"]
    assert response["repair"]["checked_schema"]["properties"]["count"]["type"] == "integer"
    assert "secret-invalid-value" not in result.content
    assert not audit[0]["invoked"]


def test_missing_contract_exposes_discovery_target(world):
    store, scope = world
    from worldmodel_harness.decisions import ProposedAction

    with pytest.raises(UnresolvedRequirements) as exc:
        api_requirements(
            store,
            scope,
            ProposedAction(
                tool="api_fetch", arguments={"method": "GET", "url": "https://crm.test/items/1"}
            ),
        )
    assert exc.value.details["reason"] == "missing_contract"
    assert exc.value.details["discovery_query"] == "GET https://crm.test/items/1"


def test_checkpoint_isolation_and_overlap_protocol(tmp_path):
    source = tmp_path / "source.sqlite"
    store = KnowledgeStore(source)
    store.register_learning_task("already-learned", "online")
    store.close()
    digest = checkpoint_hash(source)
    for i in range(2):
        clone = tmp_path / f"trial-{i}.sqlite"
        assert copy_checkpoint(source, clone) == digest
        trial = KnowledgeStore(clone)
        assert "only-other-trial" not in trial.learning_tasks()
        trial.register_learning_task("only-other-trial", "checkpoint")
        trial.close()
    assert checkpoint_hash(source) == digest
    ExperimentSettings(phase="checkpoint", operational=True, task_names=("already-learned",))
    for changes in (
        {"limit": 2},
        {"semantic_learning": True},
        {"adaptive": True},
        {"task_names": ()},
    ):
        with pytest.raises(ValidationError):
            ExperimentSettings(
                **{
                    "phase": "checkpoint",
                    "operational": True,
                    "task_names": ("already-learned",),
                    **changes,
                }
            )


def test_reliability_averages_per_task_and_requires_complete_trials():
    from worldmodel_harness.reliability import reliability

    result = reliability({"a": [True, True, True], "b": [True, False, False]})
    assert result["pass_all"] == {"1": 2 / 3, "2": 0.5, "3": 0.5}
    assert result["pass_any"]["3"] == 1
    assert result["pass_any"]["2"] == pytest.approx(5 / 6)
    with pytest.raises(ValueError):
        reliability({"a": [True]})


def test_real_checkpoint_trials_are_isolated_and_frozen_overlap_stays_guarded(
    tmp_path, monkeypatch
):
    from pydantic import SecretStr
    from stirrup.core.models import AssistantMessage, ToolCall

    from worldmodel_harness import experiment
    from worldmodel_harness.config import ModelSettings

    name = "simple.email_sf_contact_phone_update"
    source = tmp_path / "source.sqlite"
    store = KnowledgeStore(source)
    store.register_learning_task(name, "online")
    store.close()
    digest = checkpoint_hash(source)

    class ScriptedClient:
        model_slug = "scripted"
        context_window_tokens = 262144
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                call, args = (
                    "execute_decision",
                    {
                        "action": {
                            "tool": "api_search",
                            "arguments": {"query": "gmail list messages", "top_k": 3},
                        }
                    },
                )
            else:
                call, args = "finish", {"summary": "diagnostic only", "unresolved": True}
            return AssistantMessage(blocks=[ToolCall(name=call, arguments=json.dumps(args))])

    monkeypatch.setattr(experiment, "load_settings", lambda _: ModelSettings())
    monkeypatch.setattr(experiment, "load_api_key", lambda *_: SecretStr("offline-secret"))
    monkeypatch.setattr(experiment, "build_client", lambda *_: ScriptedClient())
    for _ in range(2):
        report = asyncio.run(
            experiment.run_automationbench(
                tmp_path,
                ExperimentSettings(phase="checkpoint", task_names=(name,), operational=True),
                source,
            )
        )
        assert report["results"][0]["finished"]
        assert report["results"][0]["ledger"]["environment_calls"] == 1
        assert report["results"][0]["retrievals"][0]["injected_claims"] == []
        assert report["knowledge_claims"] > 0
        assert checkpoint_hash(source) == digest
    with pytest.raises(ValueError, match="overlaps"):
        asyncio.run(
            experiment.run_automationbench(
                tmp_path,
                ExperimentSettings(phase="frozen", task_names=(name,), operational=True),
                source,
            )
        )


@pytest.mark.parametrize(
    "bad_arguments",
    [
        '{"action":"{bad"}',
        '{"action":null}',
        '{"action":{"tool":"read","arguments":{}},"unknown":"private-marker"}',
    ],
)
def test_real_stirrup_outer_schema_repairs_without_execution(tmp_path, bad_arguments):
    from stirrup.core.models import AssistantMessage, ToolCall, ToolResult, UserMessage

    from worldmodel_harness.decisions import CompactDecision
    from worldmodel_harness.repairing_agent import RepairingAgent

    invocations = []

    async def execute(params):
        invocations.append(params)
        return ToolResult(content="done")

    tool = Tool(
        name="execute_decision",
        description="checked data",
        parameters=CompactDecision,
        executor=execute,
    )
    finish = Tool(
        name="finish",
        description="finish",
        parameters=Record,
        executor=lambda _: ToolResult(content="finished"),
    )

    class Client:
        model_slug = "scripted"
        context_window_tokens = 32768
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, args = "execute_decision", bad_arguments
            elif self.calls == 2:
                assert invocations == []
                failure = next(m for m in reversed(messages) if m.role == "tool")
                details = json.loads(failure.content)
                assert details["origin"] == "outer_tool_schema"
                assert details["environment_called"] is False
                assert details["validation_errors"] and details["checked_schema"]
                assert "private-marker" not in failure.content
                name, args = "execute_decision", '{"action":{"tool":"read","arguments":{}}}'
            else:
                name, args = "finish", "{}"
            return AssistantMessage(blocks=[ToolCall(name=name, arguments=args)])

    async def run():
        agent = RepairingAgent(
            client=Client(),
            name="outer-repair",
            tools=[tool],
            finish_tool=finish,
            max_turns=4,
            repair_schemas={tool.name: tool.parameters, finish.name: finish.parameters},
        )
        async with agent.session(output_dir=tmp_path / "session") as session:
            result, _, _ = await session.run(
                [UserMessage(content="Exercise invalid and corrected calls")]
            )
        assert result is not None
        assert len(agent.outer_validation_audit) == 1

    asyncio.run(run())
    assert len(invocations) == 1


def test_terminal_brace_suggestion_requires_resubmission_before_execution(tmp_path):
    from stirrup.core.models import AssistantMessage, ToolCall, ToolResult, UserMessage

    from worldmodel_harness.decisions import CompactDecision
    from worldmodel_harness.repairing_agent import RepairingAgent

    original = {"tool": "api_fetch", "arguments": {"method": "POST", "url": "https://example.test/tasks",
                "body": json.dumps({"data": {"name": "Quoted 'task'", "workspace": "workspace_7"}})}}
    malformed = json.dumps({"action": json.dumps(original)[:-1], "goal": "Create requested task"})
    invocations = []

    async def execute(params):
        invocations.append(params.action)
        return ToolResult(content="created")

    tool = Tool(name="execute_decision", description="guarded", parameters=CompactDecision, executor=execute)
    finish = Tool(name="finish", description="done", parameters=Record, executor=lambda _: ToolResult(content="done"))

    class Client:
        model_slug = "scripted"
        context_window_tokens = 32768
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, args = tool.name, malformed
            elif self.calls == 2:
                assert invocations == []
                failure = next(m for m in reversed(messages) if m.role == "tool")
                details = json.loads(failure.content)
                assert details["environment_called"] is False
                assert details["suggested_arguments"]["action"] == original
                name, args = tool.name, json.dumps(details["suggested_arguments"])
            else:
                assert len(invocations) == 1
                name, args = "finish", "{}"
            return AssistantMessage(blocks=[ToolCall(name=name, arguments=args)])

    async def run():
        agent = RepairingAgent(client=Client(), name="brace-repair", tools=[tool], finish_tool=finish,
                               max_turns=4, repair_schemas={tool.name: tool.parameters, finish.name: finish.parameters})
        async with agent.session(output_dir=tmp_path/'session') as session:
            result, _, _ = await session.run([UserMessage(content="Exercise syntax repair")])
        assert result is not None
        assert len(agent.outer_validation_audit) == 1

    asyncio.run(run())
    assert len(invocations) == 1
    assert invocations[0].arguments == original["arguments"]


@pytest.mark.parametrize("action", ['{"tool":"api_fetch","arguments":{"body":"unfinished',
                                    '{"tool":"api_fetch" "arguments":{}}',
                                    '{"tool":"api_fetch","arguments":{',
                                    '{"tool":"api_fetch"',
                                    '{"tool":"api_fetch","arguments":{}}'])
def test_suggestion_does_not_guess_missing_values_or_interior_syntax(action):
    from worldmodel_harness.decisions import CompactDecision
    from worldmodel_harness.repairing_agent import suggested_arguments

    assert suggested_arguments(CompactDecision, json.dumps({"action": action})) is None


def test_normalize_assistant_tool_calls_parses_gemma_raw_tokens():
    from stirrup.core.models import AssistantMessage, TextBlock, ToolCall
    from worldmodel_harness.repairing_agent import normalize_assistant_tool_calls

    raw_text = (
        '<|tool_call>call:execute_decision{action:{arguments:{body:<|"|>{"task": "123"}<|"|>,'
        'method:<|"|>POST<|"|>,url:<|"|>https://app.asana.com/api/1.0/sections/sec_feb/addTask<|"|>},'
        'goal:<|"|>Add task<|"|>}}<tool_call|>'
    )
    msg = AssistantMessage(blocks=[TextBlock(text=raw_text)])
    normalized = normalize_assistant_tool_calls(msg)

    tool_calls = [b for b in normalized.blocks if isinstance(b, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "execute_decision"
    parsed_args = json.loads(tool_calls[0].arguments)
    assert parsed_args["action"]["arguments"]["url"] == "https://app.asana.com/api/1.0/sections/sec_feb/addTask"


def test_compact_decision_truncates_oversized_goal():
    from worldmodel_harness.decisions import CompactDecision

    d = CompactDecision(
        action={"tool": "api_fetch", "arguments": {}},
        goal="A" * 300,
    )
    assert len(d.goal) == 240
    assert d.goal == "A" * 240


def test_consecutive_blocked_queries_escalate_and_guide_monday_items_find(world):
    from worldmodel_harness.decision_runtime import count_consecutive_empty_queries, dispatch_decision
    from worldmodel_harness.knowledge import Episode

    store, scope = world

    # Add 2 empty episodes for a monday search query
    ep1 = Episode(
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://api.monday.com/v2/items:find", "body": '{"board_id": "brd_wrong", "name": "Conveyor"}'},
        response={"items": []},
        outcome="success",
    )
    ep2 = Episode(
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://api.monday.com/v2/items:find", "body": '{"board_id": "brd_wrong", "name": "Conveyor"}'},
        response={"items": []},
        outcome="success",
    )
    store.append_episode(ep1)
    store.append_episode(ep2)

    # Empty count from store alone should be 2
    count_2 = count_consecutive_empty_queries(
        store, scope, "api_fetch", {"method": "POST", "url": "https://api.monday.com/v2/items:find", "body": '{"board_id": "brd_wrong", "name": "Conveyor"}'}
    )
    assert count_2 == 2

    # If audit has a blocked attempt for the same query, total count escalates to 3
    audit = [
        {
            "decision": {
                "action": {
                    "tool": "api_fetch",
                    "arguments": {"method": "POST", "url": "https://api.monday.com/v2/items:find", "body": '{"board_id": "brd_wrong", "name": "Conveyor"}'},
                }
            },
            "invoked": False,
        }
    ]
    count_3 = count_consecutive_empty_queries(
        store, scope, "api_fetch", {"method": "POST", "url": "https://api.monday.com/v2/items:find", "body": '{"board_id": "brd_wrong", "name": "Conveyor"}'}, audit=audit
    )
    assert count_3 == 3


