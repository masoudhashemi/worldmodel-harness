"""Task-local requirement/evidence ledger. No hidden grader access."""

import json
import re
from typing import Literal

from pydantic import Field
from stirrup.core.models import Tool, ToolResult

from .context_compression import encode_records
from .grounding import GroundedDerivation, verify_derivation
from .knowledge import OutputExpectation, Record
from .observable_predictions import compare_expectations, inspect_path
from .operation_semantics import is_observation_read


class OutcomeContract(Record):
    """Fixed target, separate from the observation used to verify it."""

    target: OutputExpectation
    resource_url: str | None = None
    source_episode_id: str | None = None
    source_path: tuple[str | int, ...] = ()
    source_requirement_id: str | None = None
    identity_path: tuple[str | int, ...] = ()
    source_identity_path: tuple[str | int, ...] = ()
    identity_anchor: str | None = None
    derivation: GroundedDerivation | None = None


class SourceBinding(Record):
    anchor_text: str = Field(min_length=1)
    anchor_path: tuple[str | int, ...]


def anchored(text, value):
    """Conservative lexical identity match, not semantic/entity resolution."""
    if not isinstance(value, str):
        return False
    words = re.findall(r"\w+", text.casefold())
    observed = re.findall(r"\w+", value.casefold())
    return bool(words) and any(observed[i : i + len(words)] == words for i in range(len(observed)))


TRANSPORT_PAYLOAD_KEYS = {
    "range",
    "values",
    "majordimension",
    "raw",
    "requests",
    "data",
}


class GoalCheck(Record):
    requirement_id: str = Field(min_length=1, max_length=80)
    task_quote: str = Field(min_length=1, max_length=1000)
    episode_id: str | None = None
    observation: OutputExpectation | None = None
    evidence_mode: Literal["current_state", "historical_observation"] = "current_state"
    outcome: OutcomeContract | None = None
    kind: Literal["effect", "source"] = "effect"
    source_binding: SourceBinding | None = None


class GoalUpdate(Record):
    checks: tuple[GoalCheck, ...] = Field(min_length=1, max_length=16)


class CheckedFinish(Record):
    summary: str
    unresolved: bool = False


class GoalTracker:
    def __init__(self, adapter):
        self.adapter = adapter
        self.checks = {}
        self.verification_audit = []
        self.finish_audit = []
        self._entity_finish_warning_given = False

    def assess(self, check):
        return self.diagnose(check)["result"]

    def diagnose(self, check):
        result = self._assess(check)
        if isinstance(result, dict):
            return result
        return {"result": result}

    def _assess(self, check):
        if not check.episode_id or check.observation is None:
            return "missing_evidence"
        try:
            event = self.adapter.evidence.episode(check.episode_id)
        except KeyError:
            return "missing_evidence"
        if event.scope != self.adapter.scope or event.outcome != "success":
            return "incompatible_evidence"
        path = inspect_path(event, check.observation.path)
        if path["result"] != "observed":
            return path
        if check.kind == "source":
            if check.evidence_mode != "historical_observation":
                return "source_requires_historical_mode"
            if not is_observation_read(self.adapter, event):
                return "source_requires_read"
            if not check.observation.path or path["value"] in (None, "", [], {}):
                return "missing_source_content"
            binding = check.source_binding
            task = getattr(self.adapter, "task", None)
            public_task = " ".join(m["content"] for m in task.prompt) if task else check.task_quote
            if binding is None or not binding.anchor_path or binding.anchor_text not in public_task:
                return {
                    "result": "missing_task_anchor",
                    "field": "anchor_text",
                    "public_task": public_task,
                    "next": "Reuse this requirement_id. Copy an identity/name literally from the public task, not a newly discovered answer/value. Set anchor_path to a field of the source read containing that identity (case/punctuation normalized). The identity may occur outside the requirement's shorter task_quote.",
                }
            anchor = inspect_path(event, binding.anchor_path)
            if anchor["result"] != "observed":
                return {**anchor, "field": "anchor_path"}
            return (
                "matched"
                if anchored(binding.anchor_text, anchor["value"])
                else "source_anchor_mismatch"
            )
        if check.observation.predicate != "equals" or not check.observation.path:
            return "insufficient_outcome_predicate"
        if check.outcome is None:
            return "missing_outcome_contract"
        if check.outcome.target != check.observation:
            return "outcome_contract_mismatch"
        if event.operation == "api_fetch":
            if not check.outcome.resource_url:
                return "missing_outcome_resource"
            if event.arguments.get("url") != check.outcome.resource_url:
                return "outcome_resource_mismatch"
            if not is_observation_read(self.adapter, event):
                return "outcome_verification_requires_read"
        grounding = self.grounding(check)
        if grounding != "grounded":
            return grounding
        if check.outcome.source_requirement_id or check.outcome.identity_anchor:
            anchor_text = check.outcome.identity_anchor
            if check.outcome.source_requirement_id:
                source_check = self.checks[check.outcome.source_requirement_id]
                anchor_text = source_check.source_binding.anchor_text
            identity = inspect_path(event, check.outcome.identity_path)
            if not check.outcome.identity_path or identity["result"] != "observed":
                return {
                    **identity,
                    "result": "invalid_target_identity_path",
                    "field": "identity_path",
                }
            if check.outcome.source_identity_path and check.outcome.source_requirement_id:
                source_event = self.adapter.evidence.episode(source_check.episode_id)
                source_identity = inspect_path(source_event, check.outcome.source_identity_path)
                if (source_identity["result"] != "observed"
                        or not (source_identity["value"] == identity["value"]
                                or anchored(identity["value"], source_identity["value"]))):
                    return "target_identity_mismatch"
            elif not anchored(anchor_text, identity["value"]):
                return "target_identity_mismatch"
        if check.evidence_mode == "historical_observation":
            # Past reads can establish what was observed, never a present-state effect.
            if not is_observation_read(self.adapter, event):
                return "historical_evidence_requires_read"
            return compare_expectations(
                event.model_copy(update={"output_expectations": (check.observation,)})
            )[0]["result"]
        # Discovery and old observations are not evidence of current task effects.
        namespace = event.operation + (
            ":" + str(event.arguments["url"]) if "url" in event.arguments else ""
        )
        facts = {f.key: f for f in self.adapter.live.facts()}
        required = check.observation.path[:1] or event.coverage
        if (
            event.operation == "api_search"
            or not required
            or any(
                namespace + ":" + str(key) not in facts
                or facts[namespace + ":" + str(key)].episode_id != event.episode_id
                for key in required
            )
        ):
            return {
                "result": "stale_evidence"
                if event.operation != "api_search"
                else "nonlive_evidence",
                "path": list(check.observation.path),
                "next": "Obtain a permitted current read; do not retry the write merely to verify it.",
            }
        return compare_expectations(
            event.model_copy(
                update={
                    "output_expectations": (check.observation,),
                }
            )
        )[0]["result"]

    def grounding(self, check):
        contract = check.outcome
        if contract is None:
            return "missing_outcome_contract"
        expected = contract.target.expected
        def contains_kind(value):
            if isinstance(value, dict):
                return "$kind" in value or any(contains_kind(v) for v in value.values())
            return isinstance(value, list) and any(contains_kind(v) for v in value)

        if contains_kind(expected):
            return "insufficient_outcome_predicate"
        if contract.target.predicate != "equals" or not contract.target.path:
            return "insufficient_outcome_predicate"
        if contract.derivation is not None:
            try:
                target = self.adapter.evidence.episode(check.episode_id)
            except KeyError:
                return "missing_evidence"
            return verify_derivation(self.adapter, target, contract.target.path, expected,
                                     check.task_quote, contract.identity_anchor,
                                     contract.derivation)
        if contract.source_episode_id:
            if contract.source_requirement_id:
                source_check = self.checks.get(contract.source_requirement_id)
                if (
                    source_check is None
                    or source_check.kind != "source"
                    or self.assess(source_check) != "matched"
                ):
                    return "source_requirement_unverified"
                if (
                    source_check.episode_id != contract.source_episode_id
                    or source_check.observation.path != contract.source_path
                ):
                    return "source_binding_mismatch"
            if contract.source_episode_id == check.episode_id:
                return "circular_outcome_grounding"
            try:
                source = self.adapter.evidence.episode(contract.source_episode_id)
            except KeyError:
                return "missing_outcome_source"
            if (
                source.scope != self.adapter.scope
                or source.outcome != "success"
                or (
                    not is_observation_read(self.adapter, source)
                )
            ):
                return "incompatible_outcome_source"
            inspected = inspect_path(source, contract.source_path)
            if inspected["result"] != "observed":
                return "invalid_outcome_source_path"
            value = inspected["value"]
            if type(value) is type(expected) and value == expected:
                return "grounded"
            text = value if isinstance(value, str) else ""
        else:
            # Only literal public-task support; do not guess semantic equivalence.
            task_obj = getattr(self.adapter, "task", None)
            if task_obj:
                user_msgs = [m["content"] for m in task_obj.prompt if m.get("role") != "system"]
                public_task = " ".join(user_msgs or [m["content"] for m in task_obj.prompt])
            else:
                public_task = check.task_quote
            text = public_task if check.task_quote in public_task else check.task_quote
        literal = (
            expected
            if isinstance(expected, str)
            else (json.dumps(expected) if type(expected) in (int, float, bool) else "")
        )
        if literal.strip() and re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", text):
            return "grounded"
        return "ungrounded_outcome_value"

    def report(self):
        return [{**c.model_dump(mode="json"), **self.diagnose(c)} for c in self.checks.values()]

    def context(self):
        rows = self.report()
        prefix = "Complete registered task checks, lossless columns/rows (not grader feedback; check task coverage): "
        return prefix + encode_records(rows)

    def tool(self):
        async def update(params):
            aliases = getattr(self.adapter, "evidence_aliases", {})
            normalized = []
            for check in params.checks:
                changes = {"episode_id": aliases.get(check.episode_id, check.episode_id)}
                if check.outcome:
                    changes["outcome"] = check.outcome.model_copy(update={
                        "source_episode_id": aliases.get(check.outcome.source_episode_id, check.outcome.source_episode_id)})
                normalized.append(check.model_copy(update=changes))
            params = params.model_copy(update={"checks": tuple(normalized)})
            if len({c.requirement_id for c in params.checks}) != len(params.checks):
                return ToolResult(content="Duplicate requirement IDs in update.", success=False)
            task = " ".join(m["content"] for m in self.adapter.task.prompt)
            for check in params.checks:
                state = getattr(self.adapter, "entity_tracker", None)
                if state is not None and hasattr(state, "effect_update_error"):
                    error = state.effect_update_error(check)
                    if error:
                        return ToolResult(content=error, success=False)
                if check.task_quote not in task:
                    return ToolResult(
                        content=json.dumps({"result": "invalid_task_quote", "field": "task_quote",
                            "public_task": task, "requirement_id": check.requirement_id,
                            "next": "Copy an exact substring of public_task; reuse requirement_id."}), success=False
                    )
                old = self.checks.get(check.requirement_id)
                if old and old.kind != check.kind:
                    return ToolResult(
                        content="Cannot change a requirement between source and effect.",
                        success=False,
                    )
                if (
                    old
                    and old.source_binding
                    and self.assess(old) == "matched"
                    and old.source_binding.anchor_text
                    != (check.source_binding.anchor_text if check.source_binding else None)
                ):
                    return ToolResult(
                        content="Cannot change the registered task identity anchor.", success=False
                    )
                if old and old.task_quote != check.task_quote:
                    return ToolResult(
                        content="Cannot replace an existing requirement; update its evidence.",
                        success=False,
                    )
                if old and old.evidence_mode != check.evidence_mode:
                    return ToolResult(
                        content="Cannot weaken or change a registered requirement's evidence mode.",
                        success=False,
                    )
                if old and old.outcome is not None:
                    # Locators may be repaired; desired value, predicate, resource,
                    # and its grounding must not change to fit the observation.
                    corrected = old.outcome
                    if check.outcome is not None:
                        corrected = corrected.model_copy(
                            update={
                                "target": corrected.target.model_copy(
                                    update={"path": check.outcome.target.path}
                                ),
                                "identity_path": check.outcome.identity_path,
                            }
                        )
                    if corrected != check.outcome:
                        return ToolResult(
                            content="Cannot change a registered outcome value, predicate, resource or grounding. Only the target path can be corrected.",
                            success=False,
                        )
                if check.outcome is not None and self.grounding(check) != "grounded":
                    return ToolResult(
                        content=json.dumps(
                            {
                                "requirement_id": check.requirement_id,
                                "result": self.grounding(check),
                                "next": "Ground a non-root equals target in the exact task quote or a separate successful source read. Do not derive the desired value from the verification response.",
                            }
                        ),
                        success=False,
                    )
            if len(self.checks.keys() | {c.requirement_id for c in params.checks}) > 32:
                return ToolResult(content="Requirement allowance exhausted.", success=False)
            self.checks.update({c.requirement_id: c for c in params.checks})
            return ToolResult(content=json.dumps(self.report()))

        return Tool(
            name="record_goal_checks",
            parameters=GoalUpdate,
            executor=update,
            description="Before acting, register requirements quoted from the public task. "
            "Use current_state for final effects (default). Use historical_observation only "
            "for a requirement to find/read past evidence, not to establish current state. "
            "The mode cannot change after registration. Paths use string object keys and "
            "integer array indexes, e.g. [records, 0, Phone]. "
            "Set outcome.target to a non-root equals predicate for the requested value, "
            "grounded literally in task_quote or in a separate source_episode_id/source_path "
            "(a source string may contain the desired value). Set observation to that same target. "
            "For API evidence set outcome.resource_url to the intended verification resource; "
            "use a successful known non-mutating read, not a write acknowledgment. "
            "The outcome value, resource and grounding are immutable once supplied; target "
            "paths can be corrected to match response shape. Presence/type checks cannot "
            "establish task completion. Register pending requirements without evidence first. "
            "Update each with current episode IDs and observable predicates. Missing or "
            "contradicted evidence needs a permitted read/check, not a blind write retry. "
            "Checks validate observations, not the completeness of your interpretation.",
        )

    def value_slot_format(self, check):
        """Format of the field a current effect check verifies, not its identity field."""
        from .behavior import value_format

        obs = check.observation
        if check.kind != "effect" or check.evidence_mode != "current_state":
            return None
        if obs is None or not obs.path:
            return None
        if obs.predicate == "format":
            return obs.expected if isinstance(obs.expected, str) else None
        if obs.predicate != "equals":
            return None
        field = obs.path[-1] if isinstance(obs.path[-1], str) else None
        return value_format(obs.expected, field)

    def _written_slots(self) -> tuple[set[str], set[str]]:
        from .behavior import value_format

        evidence = getattr(self.adapter, "evidence", None)
        scope = getattr(self.adapter, "scope", None)
        if evidence is None or scope is None:
            return set(), set()
        formats: set[str] = set()
        fields: set[str] = set()
        for event in evidence.episodes(scope):
            method = event.arguments.get("method", "GET").upper()
            if (
                event.scope != scope
                or event.outcome != "success"
                or event.operation != "api_fetch"
                or is_observation_read(self.adapter, event)
                or method not in ("POST", "PUT", "PATCH")
            ):
                continue
            body = event.arguments.get("body")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    continue
            if isinstance(body, dict):
                for key, item in body.items():
                    if method in ("PUT", "PATCH") and str(key).casefold() not in TRANSPORT_PAYLOAD_KEYS:
                        fields.add(str(key))
                    fmt = value_format(item, str(key))
                    if fmt:
                        formats.add(fmt)
        return formats, fields

    def requested_formats(self):
        from .behavior import requested_change_formats

        task = getattr(self.adapter, "task", None)
        text = " ".join(m["content"] for m in task.prompt) if task else ""
        written_formats, _ = self._written_slots()
        return sorted(requested_change_formats(text) | written_formats)

    def finish(self, params):
        rows = self.report()
        all_matched = bool(rows) and all(r["result"] == "matched" for r in rows)
        required = self.requested_formats()
        _, written_fields = self._written_slots()
        matched_checks = [
            check
            for check in self.checks.values()
            if self.assess(check) == "matched"
        ]
        covered = {self.value_slot_format(check) for check in matched_checks}
        covered_fields = {
            str(check.observation.path[-1]).casefold()
            for check in matched_checks
            if check.kind == "effect"
            and check.evidence_mode == "current_state"
            and check.observation
            and check.observation.path
        }
        missing = [fmt for fmt in required if fmt not in covered]
        unverified_written = [
            field
            for field in sorted(written_fields)
            if field.casefold() not in covered_fields
        ]
        verified = all_matched and not missing and not unverified_written
        slots = {
            "required_value_formats": required,
            "matched_value_formats": sorted(f for f in covered if f),
            "missing_value_formats": missing,
        }
        if unverified_written:
            slots["unverified_written_fields"] = unverified_written

        entity_tracker = getattr(self.adapter, "entity_tracker", None)
        pending = (
            entity_tracker.unresolved()
            if entity_tracker is not None and hasattr(entity_tracker, "unresolved")
            else []
        )
        pending_entities = entity_tracker.pending_entities() if entity_tracker is not None else []
        if pending_entities and not getattr(self, "_entity_finish_warning_given", False):
            self._entity_finish_warning_given = True
            unprocessed = [e.name or e.badge_id or e.entity_id for e in pending_entities]
            result = ToolResult(
                content=json.dumps(
                    {
                        "status": "pending_entity_obligations",
                        "unprocessed_entities": unprocessed,
                        "message": (
                            f"Eligible entities {unprocessed} still have pending obligations. "
                            "The task requires processing all eligible entities identified in the records before completing. "
                            "Please process the remaining entities to complete all task obligations."
                        ),
                        "next_step": f"Process obligations for {unprocessed[0]}.",
                    }
                ),
                success=False,
            )
            self.finish_audit.append(
                {
                    "request": params.model_dump(mode="json"),
                    "success": False,
                    "reason": "pending_entity_obligations",
                    "unprocessed_entities": unprocessed,
                    **slots,
                }
            )
            return result
        if not verified and not params.unresolved:
            result = ToolResult(
                content=json.dumps(
                    {
                        "status": "needs_verification",
                        **slots,
                        "checks": rows,
                        "verification_tool": "check_effect",
                        "public_task": " ".join(m["content"] for m in getattr(getattr(self.adapter, "task", None), "prompt", []) if m.get("role") != "system"),
                        "next": (
                            "The task changes a "
                            + " and ".join(missing or unverified_written)
                            + " value. A matched identity/NAME check does not verify it. "
                            "Check the field holding that value on the current record."
                            if all_matched and (missing or unverified_written)
                            else "Call check_effect directly before finish: supply requirement_id, exact task_quote, latest known non-mutating read episode_id, modified field path, exact expected value from task/source, and task-quoted identity anchor_text. A read alone does not register verification. Do not repeat a successful write."
                        ),
                    }
                ),
                success=False,
            )
        else:
            self.completion_status = "unresolved" if params.unresolved else "checked"
            result = ToolResult(
                content=json.dumps({"status": self.completion_status, **slots, "checks": rows})
            )
        self.finish_audit.append(
            {
                "request": params.model_dump(mode="json"),
                "success": result.success,
                "shadow_unresolved": pending,
                **slots,
            }
        )
        return result

    def finish_tool(self):
        # Stirrup offloads synchronous executors to a worker. Keep SQLite reads
        # on the same event-loop thread that owns the task evidence connection.
        async def finish_checked(params):
            return self.finish(params)

        return Tool(
            name="finish",
            parameters=CheckedFinish,
            executor=finish_checked,
            description="Finish the task and summarize the verified outcome after completing all task requirements and checks.",
        )
