"""Flat typed verification, without nested action JSON or duplicated predicates."""

import json
import re
from typing import Any, Literal

from pydantic import Field
from stirrup.core.models import Tool, ToolResult

from .behavior import SEMANTIC_KINDS, text_formats, value_format, value_kind
from .goal_checks import GoalCheck, GoalUpdate, OutcomeContract, SourceBinding, anchored
from .knowledge import OutputExpectation, Record
from .observable_predictions import inspect_path


class VerifyGoal(Record):
    requirement_id: str = Field(min_length=1, max_length=80)
    task_quote: str = Field(min_length=1, max_length=1000)
    kind: Literal["source", "effect"]
    episode_id: str | None = None
    path: tuple[str | int, ...] = ()
    anchor_text: str | None = None
    anchor_path: tuple[str | int, ...] = ()
    expected: Any = None
    resource_url: str | None = None
    source_requirement_id: str | None = None
    identity_path: tuple[str | int, ...] = ()


def matched_source_for(tracker, expected, target, identity_path, value_path):
    """Infer only exact scalar-value and same-record identity bindings, never prose."""
    kind = value_kind(expected)
    if (
        kind not in SEMANTIC_KINDS
        or not identity_path
        or identity_path[:-1] != value_path[:-1]
        or target.scope != tracker.adapter.scope
        or target.outcome != "success"
        or target.operation != "api_fetch"
        or target.arguments.get("method", "GET").upper() not in ("GET", "HEAD")
    ):
        return None
    identity = inspect_path(target, identity_path)
    if identity["result"] != "observed":
        return None
    found = []
    for check in tracker.checks.values():
        if check.kind != "source" or tracker.assess(check) != "matched":
            continue
        if not check.episode_id or check.observation is None:
            continue
        try:
            event = tracker.adapter.evidence.episode(check.episode_id)
        except KeyError:
            continue
        inspected = inspect_path(event, check.observation.path)
        if inspected["result"] != "observed":
            continue
        content = inspected["value"]
        if (
            value_kind(content) != kind
            or not check.source_binding
            or not anchored(check.source_binding.anchor_text, identity["value"])
        ):
            continue
        if type(content) is type(expected) and content == expected:
            found.append(check)
    return found[0] if len(found) == 1 else None


def annotate_repair(result, params, tracker):
    """Name the one absent field and the ids to resend. A shared value kind is not success."""
    if result.success:
        return result
    try:
        body = json.loads(result.content)
    except (TypeError, json.JSONDecodeError):
        return result
    if not isinstance(body, dict) or "result" not in body:
        return result
    field = body.get("field")
    if body["result"] == "missing_source_fields":
        field = next(
            (
                name
                for name in ("path", "anchor_text", "anchor_path")
                if not getattr(params, name)
            ),
            field,
        )
    elif body["result"] == "missing_effect_fields":
        if not params.path:
            field = "path"
        elif not params.resource_url:
            field = "resource_url"
        elif "expected" not in params.model_fields_set:
            field = "expected"
    elif body["result"] == "missing_target_identity":
        field = "identity_path"
    if field:
        body["field"] = field
        if body["result"] in {"missing_source_fields", "missing_effect_fields"}:
            body["required"] = [field]
    body["requirement_id"] = body.get("requirement_id") or params.requirement_id
    body["episode_id"] = params.episode_id
    matched = [
        check
        for check in tracker.checks.values()
        if check.kind == "source"
        and check.requirement_id != params.requirement_id
        and tracker.assess(check) == "matched"
    ]
    copy = {"requirement_id": params.requirement_id, "episode_id": params.episode_id}
    if len(matched) == 1:
        copy["source_requirement_id"] = matched[0].requirement_id
        body["matched_source_id"] = matched[0].requirement_id
    elif len(matched) > 1:
        body["matched_source_ids"] = [check.requirement_id for check in matched]
    body["copy"] = copy
    return ToolResult(content=json.dumps(body), success=False)


def verification_tool(adapter):
    tracker = adapter.goal_tracker

    async def verify(params):
        original_request = params.model_dump(mode="json")
        inferred_binding = None
        aliases = getattr(adapter, "evidence_aliases", {})
        eid = aliases.get(params.episode_id, params.episode_id)
        pending = GoalCheck(
            requirement_id=params.requirement_id,
            task_quote=params.task_quote,
            kind=params.kind,
            evidence_mode="historical_observation" if params.kind == "source" else "current_state",
        )
        # Preserve a pending requirement even when its subsequent evidence is bad.
        if params.requirement_id not in tracker.checks:
            registered = await tracker.tool().executor(GoalUpdate(checks=(pending,)))
            if not registered.success:
                tracker.verification_audit.append(
                    {
                        "request": params.model_dump(mode="json"),
                        "success": False,
                        "feedback": registered.content,
                    }
                )
                return registered
        if not eid:
            result = ToolResult(
                content=json.dumps(
                    {
                        "result": "missing_evidence",
                        "field": "episode_id",
                        "next": "Supply the task-local episode reference from a successful read.",
                    }
                ),
                success=False,
            )
        elif params.kind == "source":
            if not params.anchor_text or not params.anchor_path or not params.path:
                result = ToolResult(
                    content=json.dumps(
                        {
                            "result": "missing_source_fields",
                            "required": ["anchor_text", "anchor_path", "path"],
                            "next": "Quote a task entity as anchor_text; anchor_path locates it in the read; path locates useful source content. No expected whole-body value is needed.",
                        }
                    ),
                    success=False,
                )
            elif params.anchor_path[:-1] != params.path[:-1]:
                result = ToolResult(
                    content=json.dumps(
                        {
                            "result": "identity_record_mismatch",
                            "next": "Source content and identity paths must select fields on the same record.",
                        }
                    ),
                    success=False,
                )
            else:
                check = pending.model_copy(
                    update={
                        "episode_id": eid,
                        "observation": OutputExpectation(path=params.path, predicate="present"),
                        "source_binding": SourceBinding(
                            anchor_text=params.anchor_text, anchor_path=params.anchor_path
                        ),
                    }
                )
                result = await tracker.tool().executor(GoalUpdate(checks=(check,)))
        else:
            if not params.source_requirement_id:
                try:
                    target_event = adapter.evidence.episode(eid)
                except KeyError:
                    target_event = None
                attached = matched_source_for(
                    tracker, params.expected, target_event, params.identity_path, params.path
                ) if target_event else None
                if attached is not None:
                    inferred_binding = {
                        "method": "unique_exact_scalar_and_same_record_identity",
                        "source_requirement_id": attached.requirement_id,
                        "source_episode_id": attached.episode_id,
                        "source_path": list(attached.observation.path),
                        "target_episode_id": eid,
                        "identity_path": list(params.identity_path),
                    }
                    params = params.model_copy(
                        update={"source_requirement_id": attached.requirement_id}
                    )
            source = tracker.checks.get(params.source_requirement_id)
            if (
                not params.path
                or not params.resource_url
                or "expected" not in params.model_fields_set
            ):
                result = ToolResult(
                    content=json.dumps(
                        {
                            "result": "missing_effect_fields",
                            "required": ["path", "resource_url", "expected"],
                            "next": "Supply the intended value and a current target read; not a write acknowledgment.",
                        }
                    ),
                    success=False,
                )
            elif params.source_requirement_id and (
                source is None or source.kind != "source" or tracker.assess(source) != "matched"
            ):
                result = ToolResult(
                    content=json.dumps(
                        {
                            "result": "source_requirement_unverified",
                            "field": "source_requirement_id",
                            "next": "Verify the cited source requirement first.",
                        }
                    ),
                    success=False,
                )
            elif not params.identity_path or (
                not params.source_requirement_id
                and (
                    not params.anchor_text
                    or params.anchor_text not in " ".join(m["content"] for m in adapter.task.prompt)
                )
            ):
                result = ToolResult(
                    content=json.dumps(
                        {
                            "result": "missing_target_identity",
                            "field": "identity_path",
                            "next": "Point to the target identity field matching the source task anchor, within the same record as path.",
                        }
                    ),
                    success=False,
                )
            elif params.identity_path[:-1] != params.path[:-1]:
                result = ToolResult(
                    content=json.dumps(
                        {
                            "result": "identity_record_mismatch",
                            "next": "Value and identity paths must select fields on the same record.",
                        }
                    ),
                    success=False,
                )
            else:
                target = OutputExpectation(path=params.path, expected=params.expected)
                contract = OutcomeContract(
                    target=target,
                    resource_url=params.resource_url,
                    source_requirement_id=params.source_requirement_id,
                    source_episode_id=source.episode_id if source else None,
                    source_path=source.observation.path if source else (),
                    identity_path=params.identity_path,
                    identity_anchor=params.anchor_text if not source else None,
                )
                check = pending.model_copy(
                    update={"episode_id": eid, "observation": target, "outcome": contract}
                )
                result = await tracker.tool().executor(GoalUpdate(checks=(check,)))
        if result.success:
            diagnosis = tracker.diagnose(tracker.checks[params.requirement_id])
            result = ToolResult(
                content=json.dumps({"requirement_id": params.requirement_id, **diagnosis}),
                success=diagnosis["result"] == "matched",
            )
        else:
            result = annotate_repair(result, params, tracker)
        tracker.verification_audit.append(
            {
                "request": params.model_dump(mode="json"),
                "original_request": original_request,
                "inferred_binding": inferred_binding,
                "success": result.success,
                "feedback": result.content,
            }
        )
        return result

    return Tool(
        name="verify_goal",
        parameters=VerifyGoal,
        executor=verify,
        description="Direct typed goal verification. Call this tool directly, not inside execute_decision. "
        "One requirement at a time. Reuse its requirement_id when repairing it; do not create duplicates. "
        "Quote the task exactly. kind=source: episode_id is a successful read, "
        "path is useful source content, anchor_text is a task-quoted identity, anchor_path finds that identity "
        "in the read (case/punctuation normalized). Use an identity already named in the public task, "
        "not a newly discovered answer/value. No expected source-body value or second source read. "
        "kind=effect: supply current GET/HEAD episode_id, resource_url, path and expected desired value. "
        "If value comes from source content, cite its verified source_requirement_id and identity_path "
        "on the target record; identity and value paths must share their parent. Otherwise expected must "
        "be literal in task_quote and supply task-quoted anchor_text plus identity_path. "
        "Short episode IDs are accepted. Omit episode_id to register pending work. "
        "Desired values/resources stay fixed; correct path errors or obtain fresh reads. Never retry a write "
        "just to verify it. Unmatched checks remain unresolved. Model-selected task coverage is not independently proven.",
    )


class CheckEffect(Record):
    requirement_id: str = Field(min_length=1, max_length=80)
    task_quote: str = Field(min_length=1, max_length=1000)
    episode_id: str = Field(min_length=1)
    path: tuple[str | int, ...] = Field(min_length=1)
    expected: Any
    anchor_text: str = Field(min_length=1, max_length=200)


def _container(value, path):
    for key in path[:-1]:
        if isinstance(value, dict) and isinstance(key, str) and key in value:
            value = value[key]
        elif isinstance(value, list):
            idx = int(key) if isinstance(key, str) and key.isdigit() else key
            if type(idx) is int and 0 <= idx < len(value):
                value = value[idx]
            else:
                return None
        else:
            return None
    return value if isinstance(value, dict) else None


def _normalize_list_indices(value, path: tuple[str | int, ...]) -> tuple[str | int, ...]:
    out: list[str | int] = []
    cur = value
    for step in path:
        if isinstance(cur, list) and isinstance(step, str) and step.isdigit():
            idx = int(step)
            out.append(idx)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        elif (
            isinstance(cur, list)
            and type(step) is int
            and 0 <= step < len(cur)
            or isinstance(cur, dict)
            and isinstance(step, str)
            and step in cur
        ):
            out.append(step)
            cur = cur[step]
        else:
            out.append(step)
            cur = None
    return tuple(out)


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in list(value.items())[:64]:
            yield from _leaves(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value[:64]):
            yield from _leaves(item, (*path, index))
    elif path:
        yield path, value


def representative_name(keys):
    """Select a canonical identity field from a group of candidate keys."""
    folded = {str(key).casefold(): key for key in keys}
    for preferred in ("name", "fullname", "displayname", "firstname", "lastname", "id", "key", "email"):
        if preferred in folded:
            return folded[preferred]
    return min(keys, key=lambda key: str(key).casefold())


def _quoted(task_quote: str, expected) -> bool:
    literal = (
        expected
        if isinstance(expected, str)
        else (json.dumps(expected) if type(expected) in (int, float, bool) else "")
    )
    if bool(literal.strip()) and (
        re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", task_quote) is not None
    ):
        return True
    if isinstance(expected, str):
        num_pat = r"(?<![A-Za-z0-9_])\d+(?:,\d{3})*(?:\.\d+)?"
        quote_nums = [s.replace(",", "") for s in re.findall(num_pat, task_quote)]
        expected_nums = [s.replace(",", "") for s in re.findall(num_pat, expected)]
        if len(quote_nums) >= 2 and set(quote_nums) <= set(expected_nums):
            return True
    return False


def _scalar_matches(actual, expected) -> bool:
    if type(actual) is type(expected) and actual == expected:
        return True
    if type(actual) in (int, float, bool) and isinstance(expected, str):
        return json.dumps(actual) == expected.strip()
    if isinstance(actual, str) and type(expected) in (int, float, bool):
        return actual.strip() == json.dumps(expected)
    return False


def expected_format(expected, field_name: str | None = None) -> str | None:
    """Resolve the normalized value format for a slot, independent of its literal characters."""
    if isinstance(expected, dict) and set(expected) == {"$kind"}:
        label = str(expected["$kind"]).upper()
        return label if label in {kind.upper() for kind in SEMANTIC_KINDS} else None
    return value_format(expected, field_name)


def resolve_effect(adapter, params: CheckEffect):
    """Resolve resource, record identity, and source read from structural value types and roles."""
    target_field = params.path[-1] if isinstance(params.path[-1], str) else None
    slot = expected_format(params.expected, target_field)
    if isinstance(params.expected, dict) and set(params.expected) == {"$kind"} and slot is None:
        return None, {
            "result": "unknown_format",
            "field": "expected",
            "next": "Specify a recognized semantic format in $kind.",
        }
    aliases = getattr(adapter, "evidence_aliases", {})
    episode_id = aliases.get(params.episode_id, params.episode_id)
    try:
        target = adapter.evidence.episode(episode_id)
    except KeyError:
        return None, {
            "result": "missing_evidence",
            "field": "episode_id",
            "next": "Cite a task-local episode from a successful current read.",
        }
    method = target.arguments.get("method", "GET").upper()
    if (
        target.scope != adapter.scope
        or target.outcome != "success"
        or target.operation != "api_fetch"
        or (method not in ("GET", "HEAD") and not (method == "POST" and bool(target.coverage)))
    ):
        return None, {
            "result": "outcome_verification_requires_read",
            "next": "Cite a successful GET or HEAD of the target record, not a write.",
        }
    effective_path = _normalize_list_indices(target.response, params.path)
    observed = inspect_path(target, effective_path)
    if observed["result"] != "observed" and isinstance(target.response, dict):
        candidates = [
            (key, 0, *params.path)
            for key, item in target.response.items()
            if key in target.coverage
            and isinstance(item, list)
            and len(item) == 1
            and inspect_path(target, (key, 0, *params.path))["result"] == "observed"
        ]
        if len(candidates) == 1:
            effective_path = candidates[0]
            observed = inspect_path(target, effective_path)
    if observed["result"] != "observed":
        return None, observed
    parent = _container(target.response, effective_path)
    if parent is None or effective_path[-1] not in parent:
        return None, {
            "result": "invalid_evidence_path",
            "field": "path",
            "available_keys": list(parent) if isinstance(parent, dict) else None,
        }
    field_formats = {}
    for key, item in parent.items():
        fmt = value_format(item, key)
        if fmt:
            field_formats[key] = fmt
    identity_keys = [
        key for key, fmt in field_formats.items() if fmt == "NAME" and key != effective_path[-1]
    ]
    if not identity_keys and field_formats.get(effective_path[-1]) == "NAME":
        identity_keys = [effective_path[-1]]
    if not identity_keys:
        for candidate_fmt in ("IDENTIFIER", "EMAIL"):
            identity_keys = [
                key
                for key, fmt in field_formats.items()
                if fmt == candidate_fmt and key != effective_path[-1]
            ]
            if identity_keys:
                break
    if not identity_keys:
        identity_keys = [
            key
            for key, item in parent.items()
            if key != effective_path[-1]
            and (
                anchored(params.anchor_text, item)
                or (isinstance(item, str) and len(item.strip()) >= 3 and anchored(item, params.anchor_text))
            )
        ]
    target_val = parent.get(effective_path[-1])
    if not identity_keys and (
        field_formats.get(effective_path[-1]) in ("EMAIL", "IDENTIFIER")
        or (
            any(k != effective_path[-1] for k in parent)
            and (
                anchored(params.anchor_text, target_val)
                or (isinstance(target_val, str) and len(target_val.strip()) >= 3 and anchored(target_val, params.anchor_text))
            )
        )
    ):
        identity_keys = [effective_path[-1]]
    if not identity_keys:
        return None, {
            "result": "missing_target_identity",
            "field": "anchor_text",
            "available_keys": list(parent),
            "field_formats": field_formats,
            "next": "Cite a record object that contains an identity field alongside the target slot.",
        }
    identity_key = representative_name(identity_keys)
    identity_format = field_formats.get(identity_key) or params.anchor_text
    identity_value = parent.get(identity_key)
    identities = [(*effective_path[:-1], identity_key)]
    task_obj = getattr(adapter, "task", None)
    if task_obj:
        user_msgs = [m["content"] for m in task_obj.prompt if m.get("role") != "system"]
        public_task = " ".join(user_msgs or [m["content"] for m in task_obj.prompt])
    else:
        public_task = params.task_quote
    grounding_text = public_task if params.task_quote in public_task else params.task_quote
    expected_value = (
        observed["value"]
        if (
            slot is None
            and isinstance(params.expected, str)
            and (
                (
                    type(observed["value"]) in (int, float, bool)
                    and json.dumps(observed["value"]) == params.expected.strip()
                )
                or (
                    isinstance(observed["value"], str)
                    and _quoted(grounding_text, params.expected)
                    and _quoted(observed["value"], params.expected)
                    and _quoted(grounding_text, observed["value"])
                )
            )
        )
        else params.expected
    )
    sources = []
    for event in adapter.evidence.episodes(adapter.scope):
        if (
            event.episode_id == target.episode_id
            or event.scope != adapter.scope
            or event.outcome != "success"
            or event.operation != "api_fetch"
            or event.arguments.get("method", "GET").upper() not in ("GET", "HEAD")
        ):
            continue
        groups: dict[tuple, list] = {}
        for leaf_path, leaf in _leaves(event.response):
            if leaf_path[0] not in event.coverage:
                continue
            groups.setdefault(leaf_path[:-1], []).append((leaf_path, leaf))
        for group in groups.values():
            exact_values = [
                leaf_path
                for leaf_path, leaf in group
                if (
                    value_format(leaf, leaf_path[-1] if leaf_path else None) == slot
                    if slot
                    else _scalar_matches(leaf, expected_value)
                )
            ]
            text_values = [
                leaf_path
                for leaf_path, leaf in group
                if isinstance(leaf, str)
                and (slot in text_formats(leaf) if slot else _quoted(leaf, expected_value))
            ]
            values = exact_values or text_values
            matched_identity = [
                leaf
                for leaf_path, leaf in group
                if value_format(leaf, leaf_path[-1] if leaf_path else None) == identity_format
                or (isinstance(leaf, str) and identity_format in text_formats(leaf))
                or (isinstance(identity_value, str) and anchored(identity_value, leaf))
            ]
            if values and matched_identity:
                sources.append((event, values[0]))
    distinct_sources = [
        item
        for item in sources
        if item[0].arguments.get("url") != target.arguments.get("url")
    ]
    pool = distinct_sources or sources
    source_episode_id = None
    source_path: tuple = ()
    if pool:
        source_episode_id = pool[-1][0].episode_id
        source_path = pool[-1][1]
    elif slot and value_format(parent.get(effective_path[-1]), target_field) == slot:
        source_episode_id = None
        source_path = ()
    elif not _quoted(grounding_text, expected_value):
        return None, {
            "result": "ungrounded_outcome_value",
            "field": "expected",
            "next": "Ground the expected field in a recognized value format, a prior source read, "
            "or a literal quote from the public task.",
        }
    return {
        "episode_id": episode_id,
        "path": effective_path,
        "resource_url": target.arguments.get("url"),
        "identity_path": identities[0],
        "identity_format": identity_format,
        "value_format": slot,
        "expected_value": expected_value,
        "source_episode_id": source_episode_id,
        "source_path": source_path,
    }, None


def simple_check_tool(adapter):
    tracker = adapter.goal_tracker

    async def check(params: CheckEffect):
        resolved, error = resolve_effect(adapter, params)
        if error is not None:
            result = ToolResult(
                content=json.dumps(
                    {
                        "requirement_id": params.requirement_id,
                        "episode_id": params.episode_id,
                        **error,
                    }
                ),
                success=False,
            )
        else:
            target = (
                OutputExpectation(
                    path=resolved["path"], predicate="format", expected=resolved["value_format"]
                )
                if resolved["value_format"]
                else OutputExpectation(path=resolved["path"], expected=resolved["expected_value"])
            )
            contract = OutcomeContract(
                target=target,
                resource_url=resolved["resource_url"],
                source_episode_id=resolved["source_episode_id"],
                source_path=resolved["source_path"],
                identity_path=resolved["identity_path"],
                identity_anchor=resolved["identity_format"],
            )
            goal = GoalCheck(
                requirement_id=params.requirement_id,
                task_quote=params.task_quote,
                episode_id=resolved["episode_id"],
                observation=target,
                outcome=contract,
            )
            prior = tracker.checks.pop(params.requirement_id, None)
            result = await tracker.tool().executor(GoalUpdate(checks=(goal,)))
            if not result.success and prior is not None:
                tracker.checks[params.requirement_id] = prior
            if result.success:
                diagnosis = tracker.diagnose(tracker.checks[params.requirement_id])
                result = ToolResult(
                    content=json.dumps(
                        {
                            "requirement_id": params.requirement_id,
                            "resource_url": resolved["resource_url"],
                            "identity_path": list(resolved["identity_path"]),
                            "source_episode_id": resolved["source_episode_id"],
                            **diagnosis,
                        }
                    ),
                    success=diagnosis["result"] == "matched",
                )
        tracker.verification_audit.append(
            {
                "tool": "check_effect",
                "request": params.model_dump(mode="json"),
                "success": result.success,
                "feedback": result.content,
            }
        )
        return result

    return Tool(
        name="check_effect",
        parameters=CheckEffect,
        executor=check,
        description="Check one current effect on a target record. Cite the latest GET or HEAD "
        "read of the target record, the modified field path, the expected value or semantic format, "
        "and the task mention as anchor_text. "
        "The resource URL, record identity field, and source grounding are resolved structurally "
        "by value type and record shape rather than literal value memorization. "
        "Reuse requirement_id to repair a path. Quote task_quote from the public task. "
        "Call finish after all modified fields match.",
    )
