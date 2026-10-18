"""Bounded, data-only execution artifacts; no generated Python or eval.

The same registered tools execute both individual and procedure calls, so
permission checks, evidence logging, and environment budgets cannot be bypassed.
"""

import inspect
import json
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Literal
from urllib.parse import quote
from uuid import uuid4

from pydantic import Field, model_validator
from stirrup.core.models import Tool, ToolResult

from .knowledge import Dependency, LiveState, Record, Scope, utc_now
from .store import KnowledgeStore

CURRENT_ARTIFACT: ContextVar[dict | None] = ContextVar("current_artifact", default=None)


class Step(Record):
    name: str
    kind: Literal["call", "query", "transform", "branch", "verify", "stop"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    # A reference is {"$ref": "inputs.foo"} or {"$ref": "steps.lookup.data.id"}.
    value: Any = None
    expected: Any = None
    predicate: Literal["equals", "type"] = "equals"
    transform: Literal["identity", "json", "concat", "url_encode"] = "identity"
    next_step: str | None = None
    on_false: str | None = None
    on_error: str | None = None


class ValidationCase(Record):
    task_id: str
    task_name: str | None = None
    passed: bool
    isolated_execution: bool
    evidence_ids: tuple[str, ...]


class Procedure(Record):
    procedure_id: str
    version: int = Field(default=1, ge=1)
    surface: Literal["procedure", "input_adapter", "retrieval_recipe", "verification"] = "procedure"
    goal_pattern: str
    trigger: str = ""
    expected_runtime_effect: str = ""
    stop_condition: str = ""
    scope: Scope
    parameter_types: dict[str, Literal["string", "integer", "number", "boolean", "object", "array"]]
    required_claims: tuple[Dependency, ...]
    prerequisites: dict[str, Any] = Field(default_factory=dict)
    steps: tuple[Step, ...] = Field(min_length=1)
    max_steps: int = Field(default=32, ge=1, le=256)
    training_tasks: tuple[str, ...] = ()
    validation: tuple[ValidationCase, ...] = ()
    status: Literal["candidate", "experimental", "active", "suspended", "retired"] = "candidate"
    learned_opportunity_id: str | None = None
    estimated_reuses: int = Field(default=0, ge=0)
    estimated_saving_per_use: float = 0
    build_and_validation_cost: float = Field(default=0, ge=0)
    repair_from_version: int | None = Field(default=None, ge=1)
    repair_reason: str | None = None

    @model_validator(mode="after")
    def structure(self):
        names = {step.name for step in self.steps}
        if len(names) != len(self.steps):
            raise ValueError("Duplicate step name")
        for step in self.steps:
            for target in (step.next_step, step.on_false, step.on_error):
                if target is not None and target not in names:
                    raise ValueError("Unknown branch target")
            if step.kind in ("call", "query") and step.tool is None:
                raise ValueError("Calls require an explicit registered tool")
        if not any(step.kind == "stop" for step in self.steps):
            raise ValueError("Procedure requires an explicit stop")
        return self


class ExecutionResult(Record):
    completed: bool
    reason: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    executed_steps: tuple[str, ...] = ()
    execution_id: str | None = None


class ArtifactInput(Record):
    procedure_id: str
    inputs: dict[str, Any]


def artifact_tool(
    store: KnowledgeStore,
    scope: Scope,
    live: LiveState,
    tools: dict[str, Tool],
    *,
    allow_experimental: bool = False,
    audit: list | None = None,
    ledger=None,
) -> Tool:
    """Expose only stored artifacts; generated code and inline programs are not accepted."""

    experimental_uses = 0

    async def run(params: ArtifactInput) -> ToolResult:
        nonlocal experimental_uses
        artifact = next(
            (p for p in store.procedures(scope) if p.procedure_id == params.procedure_id), None
        )
        if artifact is None:
            return ToolResult(
                content="Unknown or out-of-scope procedure; use ordinary tools.", success=False
            )
        if artifact.status == "experimental":
            if experimental_uses >= 2:
                return ToolResult(
                    content="Experimental invocation allowance exhausted; use ordinary tools.",
                    success=False,
                )
            experimental_uses += 1
        started = perf_counter()
        before = ledger.environment_calls if ledger else None
        row = {
            "procedure_id": artifact.procedure_id,
            "version": artifact.version,
            "status": artifact.status,
            "task_id": scope.task_id,
            "estimated_saving_seconds_per_use": artifact.estimated_saving_per_use,
            "measured_saving_seconds": None,
        }
        try:
            result = await execute(
                artifact,
                params.inputs,
                tools,
                store,
                scope,
                live,
                allow_experimental=allow_experimental,
            )
            row.update(
                completed=result.completed,
                reason=result.reason,
                executed_steps=list(result.executed_steps),
                execution_id=result.execution_id,
            )
        except BaseException as exc:
            row.update(completed=False, error_type=type(exc).__name__)
            raise
        finally:
            row["elapsed_seconds"] = perf_counter() - started
            row["environment_calls"] = ledger.environment_calls - before if ledger else None
            if audit is not None:
                audit.append(row)
        if (
            artifact.status == "experimental"
            and allow_experimental
            and result.executed_steps
            and not result.completed
            and not store.frozen
        ):
            latest = next(
                p for p in store.procedures(scope) if p.procedure_id == artifact.procedure_id
            )
            if latest.status == "experimental":
                store.put_procedure(
                    latest.model_copy(update={"version": latest.version + 1, "status": "suspended"})
                )
        return ToolResult(content=result.model_dump_json(), success=result.completed)

    return Tool(
        name="run_procedure",
        parameters=ArtifactInput,
        executor=run,
        description="Run a stored validated procedure. Its completion is not task completion. "
        "On unmet prerequisites or suspension, use ordinary tools to investigate.",
    )


def resolve(value: Any, context: dict) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            current = context
            for part in value["$ref"].split("."):
                if isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                elif isinstance(current, dict):
                    current = current[part]
                else:
                    raise ValueError("Invalid binding path")
            return current
        return {key: resolve(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, context) for item in value]
    return value


def validate_candidate(procedure: Procedure, tools: dict[str, Tool]) -> None:
    def bindings(value):
        if isinstance(value, str) and ("{{inputs." in value or "{{steps." in value):
            raise ValueError("Unsupported string interpolation; use $ref and transform steps")
        if isinstance(value, dict):
            if "$ref" in value:
                if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
                    raise ValueError("Invalid reference object")
                parts = value["$ref"].split(".")
                if len(parts) < 2 or parts[0] not in ("inputs", "steps"):
                    raise ValueError("Invalid reference root")
                available = (
                    procedure.parameter_types
                    if parts[0] == "inputs"
                    else {s.name for s in procedure.steps}
                )
                if parts[1] not in available:
                    raise ValueError("Unknown binding reference")
            for item in value.values():
                bindings(item)
        if isinstance(value, list):
            for item in value:
                bindings(item)

    for step in procedure.steps:
        for value in (step.arguments, step.value, step.expected):
            bindings(value)
        if step.kind == "verify" and (
            not isinstance(step.value, dict)
            or set(step.value) != {"$ref"}
            or not step.value["$ref"].startswith("steps.")
        ):
            raise ValueError("Verification requires an explicit observed-result reference")
        if step.kind in ("call", "query"):
            if step.tool not in tools:
                raise ValueError("Artifact calls a tool outside the allowed toolset")
            allowed = set(tools[step.tool].parameters.model_fields)
            if set(step.arguments) - allowed:
                raise ValueError("Artifact has unknown tool arguments")
    if not (procedure.goal_pattern or procedure.trigger).strip():
        raise ValueError("Procedure requires a non-empty goal_pattern or trigger surface")
    if not any(step.kind == "verify" for step in procedure.steps):
        raise ValueError("Adopted procedures require an observable completion check")


def adopt(procedure: Procedure, tools: dict[str, Tool], store: KnowledgeStore) -> Procedure:
    validate_candidate(procedure, tools)
    independent = {
        case.task_name or case.task_id
        for case in procedure.validation
        if case.passed
        and case.isolated_execution
        and case.evidence_ids
        and case.task_id not in procedure.training_tasks
        and case.task_name not in procedure.training_tasks
    }
    if len(independent) < 2 or not all(case.passed for case in procedure.validation):
        raise ValueError("Need successful isolated validation on two independent development tasks")
    for case in procedure.validation:
        for episode_id in case.evidence_ids:
            episode = store.episode(episode_id)
            if episode.task_id != case.task_id or episode.purpose != "validation":
                raise ValueError("Validation evidence does not match its execution case")
    benefit = procedure.estimated_reuses * procedure.estimated_saving_per_use
    if benefit <= procedure.build_and_validation_cost:
        raise ValueError("Estimated reuse does not justify artifact construction")
    return procedure.model_copy(update={"status": "active"})


def applicability(
    procedure: Procedure,
    store: KnowledgeStore,
    scope: Scope,
    live: LiveState,
    *,
    allow_experimental: bool = False,
) -> str | None:
    allowed = procedure.status == "active" or (
        allow_experimental and procedure.status == "experimental"
    )
    if not allowed or not procedure.scope.applies_to(scope):
        return "inactive_or_out_of_scope"
    # A call may trigger invalidation while this execution holds an older object.
    latest = next(
        (p for p in store.procedures(scope) if p.procedure_id == procedure.procedure_id), None
    )
    if latest is not None and latest.status in ("suspended", "retired"):
        return "inactive_or_out_of_scope"
    if latest is not None and latest.version != procedure.version:
        return "superseded_artifact"
    facts = {fact.key: fact.value for fact in live.facts()}
    if any(
        key not in facts or facts[key] != value for key, value in procedure.prerequisites.items()
    ):
        return "unresolved_prerequisite"
    for dependency in procedure.required_claims:
        if dependency.mode == "monitors":
            continue
        try:
            claim = store.claim(dependency.claim_id)
        except KeyError:
            return "missing_dependency"
        if (
            not store.dependency_compatible(dependency)
            or claim.status != "supported"
            or claim.needs_review
            or not claim.scope.applies_to(scope)
            or (claim.valid_until is not None and claim.valid_until <= utc_now())
        ):
            return "dependency_needs_revalidation"
        if any(key not in facts or facts[key] != value for key, value in claim.conditions.items()):
            return "unresolved_claim_condition"
    return None


async def execute(
    procedure: Procedure,
    inputs: dict,
    tools: dict[str, Tool],
    store: KnowledgeStore,
    scope: Scope,
    live: LiveState,
    *,
    allow_experimental: bool = False,
) -> ExecutionResult:
    validate_candidate(procedure, tools)
    types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    if set(inputs) != set(procedure.parameter_types):
        return ExecutionResult(completed=False, reason="invalid_parameters")
    for key, kind in procedure.parameter_types.items():
        if not isinstance(inputs[key], types[kind]) or (
            kind in ("integer", "number") and isinstance(inputs[key], bool)
        ):
            return ExecutionResult(completed=False, reason="invalid_parameter_type")
    context = {"inputs": inputs, "steps": {}}
    visited: list[str] = []
    steps = {step.name: step for step in procedure.steps}
    current = procedure.steps[0].name
    execution_id = uuid4().hex
    verified = False

    def result(completed, reason):
        return ExecutionResult(
            completed=completed,
            reason=reason,
            outputs=context["steps"],
            executed_steps=tuple(visited),
            execution_id=execution_id,
        )

    for _ in range(procedure.max_steps):
        problem = applicability(
            procedure, store, scope, live, allow_experimental=allow_experimental
        )
        if problem:
            return result(False, problem)
        step = steps[current]
        visited.append(current)
        target = step.next_step
        try:
            if step.kind in ("call", "query"):
                tool = tools[step.tool]
                params = tool.parameters.model_validate(resolve(step.arguments, context))
                token = CURRENT_ARTIFACT.set(
                    {
                        "artifact_id": procedure.procedure_id,
                        "artifact_version": procedure.version,
                        "artifact_step": step.name,
                        "artifact_execution_id": execution_id,
                    }
                )
                try:
                    response = tool.executor(params)
                    if inspect.isawaitable(response):
                        response = await response
                finally:
                    CURRENT_ARTIFACT.reset(token)
                try:
                    data = json.loads(response.content)
                except (TypeError, json.JSONDecodeError):
                    data = response.content
                context["steps"][current] = {"data": data, "success": response.success}
                # Any further call invalidates the previous completion check.
                verified = False
                if not response.success:
                    if step.on_error is None:
                        return result(False, "tool_failed_or_outcome_unknown")
                    target = step.on_error
            elif step.kind == "transform":
                value = resolve(step.value, context)
                if step.transform == "json":
                    value = json.dumps(value)
                elif step.transform == "concat":
                    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                        return result(False, "invalid_transform_input")
                    value = "".join(value)
                elif step.transform == "url_encode":
                    if not isinstance(value, str):
                        return result(False, "invalid_transform_input")
                    value = quote(value, safe="")
                context["steps"][current] = value
            elif step.kind in ("branch", "verify"):
                actual = resolve(step.value, context)
                expected = resolve(step.expected, context)
                if step.predicate == "type":
                    from .behavior import value_type

                    matches = value_type(actual) == expected
                else:
                    matches = actual == expected
                if step.kind == "verify":
                    # Completion checks must reference actual tool output, not a constant or input.
                    reference = (
                        str(step.value.get("$ref", "")) if isinstance(step.value, dict) else ""
                    )
                    parts = reference.split(".")
                    if (
                        len(parts) < 4
                        or parts[0] != "steps"
                        or parts[2] != "data"
                        or parts[1] not in steps
                        or steps[parts[1]].kind not in ("call", "query")
                        or not context["steps"].get(parts[1], {}).get("success")
                    ):
                        return result(False, "verification_requires_observation")
                    verified = matches
                if not matches:
                    if step.on_false is None:
                        return result(False, "predicate_not_established")
                    target = step.on_false
            elif step.kind == "stop":
                return result(verified, "verified" if verified else "completion_unverified")
        except (ValueError, KeyError, IndexError, TypeError):
            return result(False, "invalid_binding_or_schema")
        if target is None:
            return result(False, "missing_transition")
        current = target
    return result(False, "step_budget_exhausted")
