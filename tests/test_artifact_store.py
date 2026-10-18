import asyncio

import pytest
from stirrup.core.models import Tool, ToolResult

from worldmodel_harness.knowledge import Claim, Dependency, Episode, Layer, LiveState, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.procedures import (
    ArtifactInput,
    Procedure,
    Step,
    ValidationCase,
    applicability,
    artifact_tool,
    execute,
)
from worldmodel_harness.store import KnowledgeStore


def validated(store, *, dependency_mode="requires"):
    scope = Scope(environment="fixture", version="1")
    cases = []
    for task in ("validation-a", "validation-b"):
        episode = Episode(
            task_id=task,
            scope=scope,
            operation="lookup",
            arguments={},
            response={"exists": True},
            coverage=("exists",),
            outcome="success",
            purpose="validation",
        )
        store.append_episode(episode)
        cases.append(
            ValidationCase(
                task_id=task,
                passed=True,
                isolated_execution=True,
                evidence_ids=(episode.episode_id,),
            )
        )
    claim = Claim(
        claim_id="contract",
        layer=Layer.INTERFACE,
        relation="contract",
        participants=("lookup",),
        statement="Lookup contract",
        scope=scope,
        status="supported",
        supporting_episodes=cases[0].evidence_ids,
    )
    store.put_claim(claim)
    artifact = Procedure(
        procedure_id="verify",
        goal_pattern="verify",
        scope=scope,
        parameter_types={},
        required_claims=(Dependency(claim_id="contract", version=1, mode=dependency_mode),),
        status="active",
        validation=tuple(cases),
        estimated_reuses=5,
        estimated_saving_per_use=2,
        build_and_validation_cost=1,
        steps=(
            Step(name="lookup", kind="query", tool="lookup", next_step="check"),
            Step(
                name="check",
                kind="verify",
                value={"$ref": "steps.lookup.data.exists"},
                expected=True,
                next_step="stop",
            ),
            Step(name="stop", kind="stop"),
        ),
    )
    tools = {
        "lookup": Tool(
            name="lookup",
            description="Lookup",
            executor=lambda _: ToolResult(content='{"exists":true}'),
        )
    }
    return artifact, tools


def test_active_artifact_is_revalidated_and_history_preserved(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    artifact, tools = validated(store)
    store.put_procedure(artifact, tools=tools)
    Learner(store).reconsider("contract", {"id"})
    latest = store.procedures()[0]
    assert latest.status == "suspended"
    assert latest.version == 2
    original = store.db.execute("SELECT body FROM procedures WHERE version=1").fetchone()[0]
    assert Procedure.model_validate_json(original).status == "active"
    store.close()


def test_monitor_survives_disputed_dependency(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    artifact, tools = validated(store, dependency_mode="monitors")
    store.put_procedure(artifact, tools=tools)
    Learner(store).reconsider("contract", {"id"})
    assert store.procedures()[0].status == "active"
    store.close()


def test_experimental_artifact_requires_opt_in_and_suspends_after_failure(tmp_path):
    async def scenario():
        store = KnowledgeStore(tmp_path / "knowledge.sqlite")
        artifact, tools = validated(store)
        artifact = artifact.model_copy(update={"status": "experimental", "validation": ()})
        store.put_procedure(artifact, tools=tools)
        ordinary = artifact_tool(store, artifact.scope, LiveState("test"), tools)
        assert not (
            await ordinary.executor(ArtifactInput(procedure_id="verify", inputs={}))
        ).success
        enabled = artifact_tool(
            store, artifact.scope, LiveState("test"), tools, allow_experimental=True
        )
        assert (await enabled.executor(ArtifactInput(procedure_id="verify", inputs={}))).success
        tools["lookup"] = Tool(
            name="lookup",
            description="Lookup",
            executor=lambda _: ToolResult(content='{"exists":false}'),
        )
        assert not (await enabled.executor(ArtifactInput(procedure_id="verify", inputs={}))).success
        assert store.procedures()[0].status == "suspended"
        store.close()

    asyncio.run(scenario())


def test_activation_cannot_skip_independent_validation(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    artifact, tools = validated(store)
    with pytest.raises(ValueError, match="two independent"):
        store.put_procedure(artifact.model_copy(update={"validation": ()}), tools=tools)
    with pytest.raises(ValueError, match="tool and evidence"):
        store.put_procedure(artifact)
    store.close()


def test_frozen_artifact_store_is_read_only(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    store = KnowledgeStore(path)
    artifact, tools = validated(store)
    store.put_procedure(artifact, tools=tools)
    store.close()
    frozen = KnowledgeStore(path, frozen=True)
    assert frozen.procedures()[0] == artifact
    with pytest.raises(PermissionError):
        frozen.suspend_dependents("contract", set())
    frozen.close()


def test_runtime_uses_registered_tools_and_suspension_falls_back(tmp_path):
    async def scenario():
        store = KnowledgeStore(tmp_path / "knowledge.sqlite")
        artifact, tools = validated(store)
        calls = []

        def lookup(_):
            calls.append("lookup")
            return ToolResult(content='{"exists":true}')

        tools["lookup"] = Tool(name="lookup", description="Lookup", executor=lookup)
        store.put_procedure(artifact, tools=tools)
        tool = artifact_tool(store, artifact.scope, LiveState("test"), tools)
        result = await tool.executor(ArtifactInput(procedure_id="verify", inputs={}))
        assert result.success
        assert calls == ["lookup"]
        store.suspend_dependents("contract", set())
        result = await tool.executor(ArtifactInput(procedure_id="verify", inputs={}))
        assert not result.success
        assert calls == ["lookup"]
        unknown = await tool.executor(ArtifactInput(procedure_id="arbitrary-code", inputs={}))
        assert not unknown.success
        assert calls == ["lookup"]
        store.close()

    asyncio.run(scenario())


def test_inflight_suspension_stops_before_next_step(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    artifact, tools = validated(store)
    calls = []

    def lookup(_):
        calls.append("lookup")
        # Invalidation can arrive during an ordinary registered tool call.
        Learner(store).reconsider("contract", set())
        return ToolResult(content='{"exists":true}')

    tools["lookup"] = Tool(name="lookup", description="Lookup", executor=lookup)
    store.put_procedure(artifact, tools=tools)
    result = asyncio.run(execute(artifact, {}, tools, store, artifact.scope, LiveState("test")))
    assert not result.completed
    assert result.executed_steps == ("lookup",)
    assert calls == ["lookup"]
    assert store.procedures()[0].status == "suspended"
    store.close()


@pytest.mark.parametrize("changed_field,usable", [("limit", True), ("q", False)])
def test_nested_contract_changes_preserve_only_unaffected_artifacts(
    tmp_path, changed_field, usable
):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    artifact, tools = validated(store)
    learner = Learner(store)

    def document(payload):
        episode = Episode(
            task_id="schema",
            scope=artifact.scope,
            operation="tool_schema",
            arguments={},
            response=payload,
            coverage=tuple(payload),
            outcome="success",
        )
        store.append_episode(episode)
        return learner.document(
            episode,
            relation="schema",
            participants=("lookup",),
            statement="Lookup schema",
            payload=payload,
            scope=artifact.scope,
        )

    claim = document({"parameters": {"q": "string", "limit": "integer"}})
    artifact = artifact.model_copy(
        update={
            "required_claims": (
                Dependency(claim_id=claim.claim_id, version=1, fields=("parameters.q",)),
            )
        }
    )
    store.put_procedure(artifact, tools=tools)
    payload = {"parameters": {"q": "string", "limit": "integer"}}
    payload["parameters"][changed_field] = "changed"
    document(payload)
    assert (store.procedures()[0].status == "active") is usable
    assert (applicability(artifact, store, artifact.scope, LiveState("test")) is None) is usable
    # Pinned evidence is not rewritten to pretend the artifact used newer evidence.
    assert store.procedures()[0].required_claims[0].version == 1
    store.close()
