import asyncio
import json

import pytest
from pydantic import BaseModel, ValidationError
from stirrup.core.models import Tool, ToolResult

from worldmodel_harness.knowledge import LiveState, Scope
from worldmodel_harness.procedures import (
    CURRENT_ARTIFACT,
    Procedure,
    Step,
    execute,
    resolve,
    validate_candidate,
)
from worldmodel_harness.store import KnowledgeStore


class Lookup(BaseModel):
    external_id: str


def procedure(**changes):
    base = {
        "procedure_id": "lookup-verify",
        "goal_pattern": "verify ticket",
        "scope": Scope(environment="fixture", version="1"),
        "parameter_types": {"key": "string"},
        "required_claims": (),
        "status": "active",
        "steps": (
            Step(
                name="lookup",
                kind="query",
                tool="lookup",
                arguments={"external_id": {"$ref": "inputs.key"}},
                next_step="check",
            ),
            Step(
                name="check",
                kind="verify",
                value={"$ref": "steps.lookup.data.external_id"},
                expected={"$ref": "inputs.key"},
                next_step="done",
            ),
            Step(name="done", kind="stop"),
        ),
    }
    return Procedure(**(base | changes))


def run(tmp_path, artifact, executor):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    tool = Tool(name="lookup", description="Lookup", parameters=Lookup, executor=executor)
    try:
        return asyncio.run(
            execute(
                artifact, {"key": "x"}, {"lookup": tool}, store, artifact.scope, LiveState("task")
            )
        )
    finally:
        store.close()


def test_typed_binding_and_observable_verification(tmp_path):
    result = run(
        tmp_path,
        procedure(),
        lambda p: ToolResult(content=json.dumps({"external_id": p.external_id})),
    )
    assert result.completed
    assert result.executed_steps == ("lookup", "check", "done")


def test_unknown_outcome_does_not_satisfy_verification(tmp_path):
    result = run(
        tmp_path, procedure(), lambda _: ToolResult(content='{"error":"timeout"}', success=False)
    )
    assert not result.completed
    assert result.reason == "tool_failed_or_outcome_unknown"


def test_failed_completion_check_is_not_success(tmp_path):
    result = run(tmp_path, procedure(), lambda _: ToolResult(content='{"external_id":"wrong"}'))
    assert not result.completed


def test_constants_cannot_verify_environment_state(tmp_path):
    original = procedure()
    steps = (
        original.steps[0],
        Step(name="check", kind="verify", value=True, expected=True, next_step="done"),
        original.steps[2],
    )
    calls = []
    with pytest.raises(ValueError, match="explicit observed-result reference"):
        run(tmp_path, procedure(steps=steps), lambda _: calls.append("should-not-execute"))
    assert calls == []


def test_no_unregistered_tools_or_invalid_branches():
    with pytest.raises(ValueError, match="allowed toolset"):
        validate_candidate(procedure(), {})
    with pytest.raises(ValidationError, match="Unknown branch"):
        procedure(steps=(Step(name="start", kind="stop", next_step="nonexistent"),))


def test_binding_is_data_access_not_code_execution():
    with pytest.raises(KeyError):
        resolve({"$ref": "inputs.__class__"}, {"inputs": {"key": "x"}})


def test_unsupported_interpolation_rejected_before_execution(tmp_path):
    base = procedure()
    steps = (
        base.steps[0].model_copy(update={"arguments": {"external_id": "{{inputs.key}}"}}),
        *base.steps[1:],
    )
    with pytest.raises(ValueError, match="Unsupported string interpolation"):
        run(tmp_path, procedure(steps=steps), lambda _: ToolResult(content="must not run"))


def test_execution_provenance_is_scoped_and_reset(tmp_path):
    contexts = []

    def lookup(_):
        contexts.append(CURRENT_ARTIFACT.get())
        return ToolResult(content='{"external_id":"x"}')

    assert run(tmp_path, procedure(), lookup).completed
    assert contexts[0]["artifact_id"] == "lookup-verify"
    assert contexts[0]["artifact_step"] == "lookup"
    assert contexts[0]["artifact_execution_id"]
    assert CURRENT_ARTIFACT.get() is None


def test_url_component_transform_cannot_inject_path_or_query(tmp_path):
    base = procedure()
    artifact = procedure(
        steps=(
            Step(
                name="encode",
                kind="transform",
                transform="url_encode",
                value="record/id?admin=true",
                next_step="lookup",
            ),
            *base.steps,
        )
    )
    result = run(
        tmp_path, artifact, lambda p: ToolResult(content=json.dumps({"external_id": p.external_id}))
    )
    assert result.completed
    assert result.outputs["encode"] == "record%2Fid%3Fadmin%3Dtrue"
