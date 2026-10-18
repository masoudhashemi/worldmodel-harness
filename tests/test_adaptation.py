import asyncio
import json

import pytest
from stirrup.core.models import AssistantMessage, TextBlock, Tool, ToolResult

from worldmodel_harness.adaptation import (
    RepairRequest,
    ValidationEnvironment,
    ValidationTrial,
    propose,
    validate_and_adopt,
)
from worldmodel_harness.budget import Ledger
from worldmodel_harness.knowledge import Claim, Dependency, Episode, Layer, LiveState, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.procedures import Procedure, Step
from worldmodel_harness.store import KnowledgeStore


def setup(tmp_path, *, wrong_output=False):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="fixture", version="1")
    seed = Episode(
        task_id="training",
        scope=scope,
        operation="tool_schema",
        arguments={},
        response={"schema": "lookup"},
        outcome="success",
    )
    store.append_episode(seed)
    claim = Claim(
        claim_id="lookup-contract",
        layer=Layer.INTERFACE,
        relation="contract",
        participants=("lookup",),
        statement="Lookup records",
        scope=scope,
        status="supported",
        supporting_episodes=(seed.episode_id,),
    )
    store.put_claim(claim)
    candidate = Procedure(
        procedure_id="verify-lookup",
        scope=scope,
        goal_pattern="lookup",
        parameter_types={},
        required_claims=(Dependency(claim_id=claim.claim_id, version=1),),
        training_tasks=("training",),
        steps=(
            Step(name="read", kind="query", tool="lookup", next_step="verify"),
            Step(
                name="verify",
                kind="verify",
                value={"$ref": "steps.read.data.found"},
                expected=True,
                next_step="stop",
            ),
            Step(name="stop", kind="stop"),
        ),
    )

    def factory(trial, ledger, evidence):
        target = scope.model_copy(update={"task_id": trial.task_name, "tenant": trial.task_name})

        def lookup(_):
            ledger.charge_environment("validation")
            response = {getattr(factory, "response_field", "found"): not wrong_output}
            evidence.append_episode(
                Episode(
                    task_id=trial.task_name,
                    scope=target,
                    operation="lookup",
                    arguments={},
                    response=response,
                    outcome="success",
                    coverage=tuple(response),
                    purpose="validation",
                )
            )
            return ToolResult(content=json.dumps(response))

        return ValidationEnvironment(
            task_id=trial.task_name,
            scope=target,
            live=LiveState(trial.task_name),
            tools={"lookup": Tool(name="lookup", description="Lookup", executor=lookup)},
        )

    return store, scope, candidate, factory


def test_actual_isolated_validation_adopts_and_charges_once(tmp_path):
    store, scope, candidate, factory = setup(tmp_path)
    ledger = Ledger()
    trials = tuple(ValidationTrial(task_name=name, inputs={}) for name in ("v1", "v2"))
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            trials,
            factory,
            store,
            ledger=ledger,
            estimated_reuses=100,
            estimated_saving_per_use=1,
        )
    )
    assert result.adopted
    assert ledger.environment_calls == 2
    assert ledger.by_purpose == {"validation": 2}
    assert len(result.cases) == 2
    assert store.procedures(scope)[0].status == "active"
    assert store.procedures(scope)[0].version == 2
    assert store.learning_tasks() == {"v1", "v2"}
    store.close()


def test_failed_execution_is_retained_without_adoption(tmp_path):
    store, scope, candidate, factory = setup(tmp_path, wrong_output=True)
    trials = tuple(ValidationTrial(task_name=name, inputs={}) for name in ("v1", "v2"))
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            trials,
            factory,
            store,
            ledger=Ledger(),
            estimated_reuses=100,
            estimated_saving_per_use=1,
        )
    )
    assert not result.adopted
    assert all(not case.passed for case in result.cases)
    assert store.procedures(scope)[0].status == "candidate"
    store.close()


def test_validation_budget_exhaustion_cannot_adopt(tmp_path):
    store, _, candidate, factory = setup(tmp_path)
    ledger = Ledger(max_environment_calls=1)
    trials = tuple(ValidationTrial(task_name=name, inputs={}) for name in ("v1", "v2"))
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            trials,
            factory,
            store,
            ledger=ledger,
            estimated_reuses=100,
            estimated_saving_per_use=1,
        )
    )
    assert not result.adopted
    assert ledger.environment_calls == 1
    assert result.execution_reasons[-1] == "validation_exception:BudgetExceeded"
    store.close()


def test_training_validation_overlap_rejected(tmp_path):
    store, _, candidate, factory = setup(tmp_path)
    trials = tuple(ValidationTrial(task_name=name, inputs={}) for name in ("training", "v2"))
    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(
            validate_and_adopt(
                candidate,
                trials,
                factory,
                store,
                ledger=Ledger(),
                estimated_reuses=100,
                estimated_saving_per_use=1,
            )
        )
    assert not store.procedures()
    store.close()


def test_model_cannot_self_activate_or_invent_validation(tmp_path):
    store, scope, candidate, factory = setup(tmp_path)
    environment = factory(ValidationTrial(task_name="v1", inputs={}), Ledger(), store)

    class Client:
        async def generate(self, messages, tools):
            assert tools == {}
            assert "schema" in messages[1].content
            data = candidate.model_dump(mode="json")
            data.update(status="active", estimated_reuses=9999, validation=[{"forged": True}])
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(data))])

    proposed = asyncio.run(
        propose(Client(), store, scope, environment.tools, ("training",), goal="lookup")
    )
    assert proposed.status == "candidate"
    assert proposed.validation == ()
    assert proposed.estimated_reuses == 0
    store.close()


@pytest.mark.parametrize("use_hint_as_requirement", [False, True])
def test_composition_hints_are_visible_but_cannot_become_supported_requirements(
    tmp_path, use_hint_as_requirement
):
    store, scope, candidate, factory = setup(tmp_path)
    store.put_claim(
        Claim(
            claim_id="flow-hint",
            layer=Layer.BEHAVIOR,
            relation="observed_value_flow",
            participants=("lookup",),
            statement="lookup output binding",
            scope=scope,
            dependencies=(Dependency(claim_id="lookup-contract", version=1),),
        )
    )
    environment = factory(ValidationTrial(task_name="v1", inputs={}), Ledger(), store)

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            assert prompt["observational_hints"]["claims"][0]["claim_id"] == "flow-hint"
            assert "flow-hint" not in {c["claim_id"] for c in prompt["claims"]}
            data = candidate.model_dump(mode="json")
            if use_hint_as_requirement:
                data["required_claims"] = [{"claim_id": "flow-hint", "version": 1}]
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(data))])

    if use_hint_as_requirement:
        with pytest.raises(ValueError, match="supplied knowledge"):
            asyncio.run(
                propose(Client(), store, scope, environment.tools, ("training",), goal="lookup")
            )
    else:
        proposed = asyncio.run(
            propose(Client(), store, scope, environment.tools, ("training",), goal="lookup")
        )
        assert proposed.status == "candidate"
    store.close()


def suspended_fixture(tmp_path):
    store, scope, candidate, factory = setup(tmp_path)
    trials = tuple(ValidationTrial(task_name=name, inputs={}) for name in ("v1", "v2"))
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            trials,
            factory,
            store,
            ledger=Ledger(),
            estimated_reuses=100,
            estimated_saving_per_use=1,
        )
    )
    assert result.adopted
    old = store.claim("lookup-contract")
    store.put_claim(old.model_copy(update={"version": 2, "payload": {"schema_revision": 2}}))
    Learner(store).reconsider(old.claim_id, {"schema_revision"})
    suspended = store.procedures()[0]
    assert suspended.status == "suspended"
    repaired = candidate.model_copy(
        update={"required_claims": (Dependency(claim_id="lookup-contract", version=2),)}
    )
    request = RepairRequest(
        procedure_id=suspended.procedure_id,
        suspended_version=suspended.version,
        reason="Verified schema revision",
    )
    return store, scope, repaired, factory, request


def test_repair_reexecutes_and_preserves_version_history(tmp_path):
    store, scope, candidate, factory, request = suspended_fixture(tmp_path)
    ledger = Ledger()
    trials = tuple(ValidationTrial(task_name=name, inputs={}) for name in ("r1", "r2"))
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            trials,
            factory,
            store,
            ledger=ledger,
            estimated_reuses=100,
            estimated_saving_per_use=1,
            repair=request,
        )
    )
    assert result.adopted
    assert ledger.environment_calls == 2
    latest = store.procedures(scope)[0]
    assert latest.status == "active"
    assert latest.version == 5
    assert latest.repair_from_version == 3
    assert latest.repair_reason == request.reason
    assert {c.task_name for c in latest.validation} == {"r1", "r2"}
    history = [
        Procedure.model_validate_json(row[0]).status
        for row in store.db.execute("SELECT body FROM procedures ORDER BY version")
    ]
    assert history == ["candidate", "active", "suspended", "candidate", "active"]
    store.close()


@pytest.mark.parametrize(
    "attack", ["no_request", "stale_version", "drop_dependency", "old_pin", "interface"]
)
def test_repair_rejects_unsafe_shortcuts_before_environment_calls(tmp_path, attack):
    store, _, candidate, factory, request = suspended_fixture(tmp_path)
    if attack == "no_request":
        request = None
    elif attack == "stale_version":
        request = request.model_copy(update={"suspended_version": 2})
    elif attack == "drop_dependency":
        candidate = candidate.model_copy(update={"required_claims": ()})
    elif attack == "old_pin":
        candidate = candidate.model_copy(
            update={"required_claims": (Dependency(claim_id="lookup-contract", version=1),)}
        )
    elif attack == "interface":
        candidate = candidate.model_copy(update={"parameter_types": {"new": "string"}})
    ledger = Ledger()
    with pytest.raises(ValueError):
        asyncio.run(
            validate_and_adopt(
                candidate,
                tuple(ValidationTrial(task_name=name, inputs={}) for name in ("r1", "r2")),
                factory,
                store,
                ledger=ledger,
                estimated_reuses=100,
                estimated_saving_per_use=1,
                repair=request,
            )
        )
    assert ledger.environment_calls == 0
    assert store.procedures()[0].status == "suspended"
    store.close()


def test_failed_repair_remains_suspended_and_can_be_retried(tmp_path):
    store, _, candidate, factory, request = suspended_fixture(tmp_path)
    # A real second execution still occurs, but insufficient reuse cannot authorize adoption.
    trials = tuple(ValidationTrial(task_name=name, inputs={}) for name in ("r1", "r2"))
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            trials,
            factory,
            store,
            ledger=Ledger(),
            estimated_reuses=0,
            estimated_saving_per_use=1,
            repair=request,
        )
    )
    assert not result.adopted
    assert store.procedures()[0].status == "suspended"
    request = request.model_copy(update={"suspended_version": 5})
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            tuple(ValidationTrial(task_name=name, inputs={}) for name in ("retry1", "retry2")),
            factory,
            store,
            ledger=Ledger(),
            estimated_reuses=100,
            estimated_saving_per_use=1,
            repair=request,
        )
    )
    assert result.adopted
    assert store.procedures()[0].repair_from_version == 5
    store.close()


def test_shifted_output_requires_actual_program_repair(tmp_path):
    store, _, candidate, factory, request = suspended_fixture(tmp_path)
    factory.response_field = "exists"
    ledger = Ledger()
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            tuple(ValidationTrial(task_name=name, inputs={}) for name in ("shift1", "shift2")),
            factory,
            store,
            ledger=ledger,
            estimated_reuses=100,
            estimated_saving_per_use=1,
            repair=request,
        )
    )
    assert not result.adopted
    assert ledger.environment_calls == 2
    assert all(not case.passed for case in result.cases)
    # Merely repinning knowledge did not fix the changed output binding.
    candidate = candidate.model_copy(
        update={
            "steps": (
                candidate.steps[0],
                candidate.steps[1].model_copy(update={"value": {"$ref": "steps.read.data.exists"}}),
                candidate.steps[2],
            )
        }
    )
    request = request.model_copy(update={"suspended_version": 5})
    result = asyncio.run(
        validate_and_adopt(
            candidate,
            tuple(ValidationTrial(task_name=name, inputs={}) for name in ("fixed1", "fixed2")),
            factory,
            store,
            ledger=ledger,
            estimated_reuses=100,
            estimated_saving_per_use=1,
            repair=request,
        )
    )
    assert result.adopted
    assert ledger.environment_calls == 4
    assert all(
        store.episode(eid).response == {"exists": True}
        for case in result.cases
        for eid in case.evidence_ids
    )
    store.close()
