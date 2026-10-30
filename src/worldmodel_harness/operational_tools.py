"""Stirrup tool surface for structured decisions, guarded reuse, and live refresh."""

import json

from pydantic import Field
from stirrup.core.models import Tool, ToolResult

from .decision_runtime import dispatch_decision
from .decisions import CompactDecision, Decision, ProposedAction
from .live_refresh import RefreshEstimate, refresh_live_fact
from .observable_predictions import CURRENT_EXPECTATIONS
from .operation_semantics import is_observation_read
from .operational_requirements import api_requirements
from .procedures import artifact_tool


class RefreshInput(RefreshEstimate):
    key: str = Field(min_length=1)


def _checkable_paths(event) -> list[list[str | int]]:
    """Extract up to 8 covered leaf paths on a verifiable episode response."""
    if not isinstance(event.response, dict) or not event.coverage:
        return []
    paths: list[list[str | int]] = []
    for top in event.coverage:
        if top not in event.response or top == "status":
            continue
        val = event.response[top]
        if isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, (str, int, float, bool)):
                    paths.append([str(top), str(k2)])
                elif isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                    for k3, v3 in v2[0].items():
                        if isinstance(v3, (str, int, float, bool)):
                            paths.append([str(top), str(k2), 0, str(k3)])
                else:
                    paths.append([str(top), str(k2)])
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            for k2, v2 in val[0].items():
                if isinstance(v2, (str, int, float, bool)):
                    paths.append([str(top), 0, str(k2)])
        elif isinstance(val, (str, int, float, bool)):
            paths.append([str(top)])
        if len(paths) >= 8:
            break
    return paths[:8]


def operational_tools(
    adapter,
    base_tools,
    *,
    adaptive,
    allow_experimental,
    audit,
    refresh_audit,
    hypothesis_policy="off",
    max_hypothesis_probes=3,
    lookup_audit=None,
    retrieval_chars=12000,
    expand=True,
    information_audit=None,
):
    """No raw-action bypass: procedures use the same guarded primitive boundary."""
    raw = {t.name: t for t in base_tools}
    from .task_state import TaskState
    from .goal_checks import GoalTracker

    adapter.goal_tracker = GoalTracker(adapter)
    adapter.entity_tracker = TaskState(adapter)
    adapter.evidence_aliases = {}
    adapter.artifact_audit = []

    async def dispatch(decision, toolset):
        index = len(audit)
        token = CURRENT_EXPECTATIONS.set((decision.action.tool, decision.output_expectations))
        try:
            return await dispatch_decision(
                decision,
                tools=toolset,
                requirements_for=lambda action: api_requirements(
                    adapter.knowledge, adapter.scope, action
                ),
                store=adapter.knowledge,
                evidence_store=adapter.evidence,
                scope=adapter.scope,
                live=adapter.live,
                audit=audit,
                entity_tracker=getattr(adapter, "entity_tracker", None),
            )
        finally:
            CURRENT_EXPECTATIONS.reset(token)
            # Each nested procedure dispatch charges only its own gate, not the
            # environment call or the gates charged by its children.
            if len(audit) > index:
                adapter.ledger.compute_seconds += audit[index]["gate_seconds"]

    guarded = {}
    for name, tool in raw.items():

        async def run(params, tool_name=name):
            return await dispatch(
                Decision(
                    goal="Execute a checked procedure step",
                    rationale="Recheck current prerequisites",
                    action=ProposedAction(tool=tool_name, arguments=params.model_dump()),
                ),
                raw,
            )

        guarded[name] = Tool(
            name=name, description=tool.description, parameters=tool.parameters, executor=run
        )
    available = dict(raw)
    from .knowledge_lookup import knowledge_lookup_tool

    lookup = knowledge_lookup_tool(
        adapter,
        max_chars=retrieval_chars,
        expand=expand,
        adaptive=adaptive,
        allow_experimental=allow_experimental,
        audit=lookup_audit if lookup_audit is not None else [],
    )
    available[lookup.name] = lookup
    if adaptive:
        artifact = artifact_tool(
            adapter.knowledge,
            adapter.scope,
            adapter.live,
            guarded,
            allow_experimental=allow_experimental,
            audit=adapter.artifact_audit,
            ledger=adapter.ledger,
        )
        available[artifact.name] = artifact

    from .verification_tools import simple_check_tool

    goals = adapter.goal_tracker.tool()
    available[goals.name] = goals
    target_tool = adapter.entity_tracker.tool()
    available[target_tool.name] = target_tool
    check_tool = simple_check_tool(adapter)
    available[check_tool.name] = check_tool
    aliases = {f"a{i + 1}": name for i, name in enumerate(available)}
    for i, name in enumerate(available):
        aliases[f"a{i + 1}:{name}"] = name
        aliases[f"{name}:{name}"] = name

    def resolve_tool(name: str) -> tuple[str | None, str | None]:
        if name in aliases:
            return aliases[name], None
        if ":" in name:
            return None, f"Unregistered tool alias or prefix '{name}'. Registered aliases: {sorted(aliases.keys())}"
        if name in available:
            return name, None
        return None, f"Unrecognized tool '{name}'. Available tools: {sorted(available.keys())}"

    async def decide(decision):
        data = decision.model_dump()
        data["rationale"] = data.get("rationale") or "Execute next checked action"
        action_tool = (data["action"].get("tool") or "").strip()
        root_tool = (getattr(decision, "root_tool", None) or "").strip()

        canonical_action = None
        if action_tool:
            canonical_action, err_action = resolve_tool(action_tool)
            if err_action:
                return ToolResult(
                    content=json.dumps({
                        "status": "blocked",
                        "origin": "alias_registry",
                        "tool": action_tool,
                        "message": err_action,
                    }),
                    success=False,
                )

        canonical_root = None
        if root_tool:
            canonical_root, err_root = resolve_tool(root_tool)
            if err_root:
                return ToolResult(
                    content=json.dumps({
                        "status": "blocked",
                        "origin": "alias_registry",
                        "tool": root_tool,
                        "message": err_root,
                    }),
                    success=False,
                )

        if canonical_root and canonical_action and canonical_root != canonical_action:
            return ToolResult(
                content=json.dumps({
                    "status": "blocked",
                    "origin": "tool_conflict",
                    "message": (
                        f"Conflicting tool specifications: root '{root_tool}' "
                        f"(canonical: '{canonical_root}') != action '{action_tool}' "
                        f"(canonical: '{canonical_action}')"
                    ),
                }),
                success=False,
            )

        data["action"]["tool"] = canonical_action or canonical_root or ""
        if data["action"]["tool"] == "record_goal_checks":
            for check in data["action"]["arguments"].get("checks", []):
                if isinstance(check, dict):
                    eid = check.get("episode_id")
                    check["episode_id"] = adapter.evidence_aliases.get(eid, eid)
                    outcome = check.get("outcome")
                    if isinstance(outcome, dict):
                        source = outcome.get("source_episode_id")
                        if isinstance(source, str):
                            outcome["source_episode_id"] = adapter.evidence_aliases.get(
                                source, source
                            )
        data["evidence_ids"] = tuple(
            adapter.evidence_aliases.get(e, e) for e in data.get("evidence_ids", ())
        )
        data.pop("root_tool", None)
        decision = Decision.model_validate(data)
        adapter.active_decision = decision
        result = await dispatch(decision, available)
        fallback = getattr(adapter, "baseline_fallback", None)
        if fallback is not None:
            notice = fallback.observe(audit)
            if notice:
                return ToolResult(content=result.content + notice, success=result.success)
        known = set(adapter.evidence_aliases.values())
        new_verifiable = None
        for event in adapter.evidence.episodes(adapter.scope):
            if event.episode_id not in known:
                alias = f"e{len(adapter.evidence_aliases) + 1}"
                adapter.evidence_aliases[alias] = event.episode_id
                known.add(event.episode_id)
                if hasattr(adapter, "entity_tracker") and adapter.entity_tracker is not None:
                    task = getattr(adapter, "task", None)
                    task_text = " ".join(m["content"] for m in task.prompt) if task else ""
                    method = event.arguments.get("method", "GET").upper() if isinstance(event.arguments, dict) else "GET"
                    if event.operation == "api_fetch" and method == "GET":
                        adapter.entity_tracker.register_from_read(event, task_text=task_text)
                    else:
                        adapter.entity_tracker.record_action(event)
                if (
                    event.origin == "environment"
                    and event.outcome == "success"
                    and event.purpose not in ("discovery", "validation")
                    and event.coverage
                    and isinstance(event.response, dict)
                    and (
                        event.operation != "api_fetch"
                        or is_observation_read(adapter, event)

                    )
                ):
                    new_verifiable = (alias, event)
        menu = list(adapter.evidence_aliases.items())
        if menu and result.success:
            suffix = "\nTask evidence references: " + json.dumps(dict(menu))
            if new_verifiable is not None:
                alias, event = new_verifiable
                paths = _checkable_paths(event)
                if paths:
                    suffix += f"\nCheckable paths on {alias}: {json.dumps(paths)}"
                    suffix += "\nFor task effects call check_effect directly with this episode reference, modified field path, exact intended value from task/source, and an exact task quote and identity anchor. Reading a record alone does not register verification."
            if decision.action.tool == "api_fetch":
                method = str(decision.action.arguments.get("method", "GET")).upper()
                if method not in ("GET", "HEAD"):
                    suffix += "\nNote: API execution confirmed service receipt; it does not verify that the chosen target record or recipient matches all task constraints."
            result = ToolResult(
                content=result.content + suffix,
                success=result.success,
            )
        return result

    async def refresh(params: RefreshInput):
        estimate = RefreshEstimate.model_validate(params.model_dump(exclude={"key"}))
        result = await refresh_live_fact(adapter, params.key, estimate)
        refresh_audit.append({"request": params.model_dump(), "result": result.model_dump()})
        return ToolResult(content=result.model_dump_json(), success=result.refreshed)

    catalog = {
        alias: {
            "name": name,
            "description": available[name].description,
            "parameters": available[name].parameters.model_json_schema(),
        }
        for alias, name in aliases.items()
        if name not in ("record_goal_checks", "check_effect", "select_task_target")
    }
    exposed = [
        Tool(
            name="execute_decision",
            parameters=CompactDecision,
            executor=decide,
            description="Execute the next action step. Format 'action' as an object: {'tool': '<tool_name>', 'arguments': {<parameters>}}, e.g. for api_search: {'tool': 'api_search', 'arguments': {'query': '...'}}. Do not omit arguments for tools that require them. Keep rationale "
            "empty unless needed. Reads are observations, not work assignments. "
            "Use the separate check_effect tool for task verification. Include known "
            "dependencies, bindings, and evidence. Missing prerequisites block execution. "
            "Use discovery to learn unknown contracts. api_search searches endpoint documentation, not records or messages. "
            "Use explicit workspace/project/record IDs supplied in the task as inputs; do not assume they need translation into other IDs. "
            "After discovery yields an applicable contract, execute it. Repeating an unproductive discovery query does not resolve an already supplied ID. "
            "Where useful, supply output_expectations for observable response fields before "
            "the call; missing fields remain inconclusive, and a matched response is not proof "
            "of hidden effects or task completion. "
            "Available action tools (names and parameter schemas): " + json.dumps(catalog),
        ),
        Tool(
            name="refresh_live_fact",
            parameters=RefreshInput,
            executor=refresh,
            description="Recheck a task-local fact using its previously observed successful read. "
            "Supply explicit stale-decision-loss and check-cost estimates in the same units. "
            "Mark mandatory only when the next decision requires the fact. Uses the shared "
            "environment budget. Cannot invent a lookup route or retry an unknown write.",
        ),
        check_tool,
    ]
    if hypothesis_policy != "off":
        from .hypothesis_probe import hypothesis_probe_tool
        from .information import tracked_information_tool

        exposed.append(
            tracked_information_tool(
                hypothesis_probe_tool(
                    adapter, dispatch, policy=hypothesis_policy, max_calls=max_hypothesis_probes
                ),
                information_audit if information_audit is not None else [],
            )
        )
    return exposed
