"""Task-local observations and explicitly proposed, evidence-bound obligations.

Reading a record never creates work. The actor proposes task coverage; this ledger
checks provenance and declared conditions, not the completeness of natural language
interpretation. Reusable learning remains in the interaction/knowledge pipeline.
"""

import json

from pydantic import Field, model_validator
from stirrup.core.models import Tool, ToolResult

from .goal_checks import GoalCheck, GoalUpdate, SourceBinding
from .knowledge import OutputExpectation, Record
from .observable_predictions import compare_expectations, inspect_path
from .operation_semantics import is_observation_read


class TargetProposal(Record):
    target_id: str = Field(min_length=1, max_length=80)
    source_episode_id: str
    record_path: tuple[str | int, ...] = ()
    selection_checks: tuple[GoalCheck, ...] = Field(min_length=1, max_length=8)
    effects: tuple[GoalCheck, ...] = Field(min_length=1, max_length=16)


class SelectionCondition(Record):
    requirement_id: str = Field(min_length=1, max_length=80)
    task_quote: str = Field(min_length=1, max_length=1000)
    anchor_text: str = Field(min_length=1)
    anchor_path: tuple[str | int, ...]
    observation: OutputExpectation

    @model_validator(mode="after")
    def explicit_expectation(self):
        if (
            self.observation.predicate != "present"
            and "expected" not in self.observation.model_fields_set
        ):
            raise ValueError(
                "Supply observation.expected explicitly, or use predicate='present' for existence only."
            )
        return self


class RequiredEffect(Record):
    requirement_id: str = Field(min_length=1, max_length=80)
    task_quote: str = Field(min_length=1, max_length=1000)


class SelectTarget(Record):
    target_id: str = Field(min_length=1, max_length=80)
    source_episode_id: str
    record_path: tuple[str | int, ...] = ()
    selection_checks: tuple[SelectionCondition, ...] = Field(min_length=1, max_length=8)
    effects: tuple[RequiredEffect, ...] = Field(min_length=1, max_length=8)

    def proposal(self):
        return TargetProposal(
            target_id=self.target_id,
            source_episode_id=self.source_episode_id,
            record_path=self.record_path,
            selection_checks=tuple(
                GoalCheck(
                    requirement_id=c.requirement_id,
                    task_quote=c.task_quote,
                    kind="source",
                    evidence_mode="historical_observation",
                    episode_id=self.source_episode_id,
                    observation=c.observation,
                    source_binding=SourceBinding(
                        anchor_text=c.anchor_text, anchor_path=c.anchor_path
                    ),
                )
                for c in self.selection_checks
            ),
            effects=tuple(
                GoalCheck(requirement_id=e.requirement_id, task_quote=e.task_quote)
                for e in self.effects
            ),
        )


class TaskState:
    def __init__(self, adapter):
        self.adapter = adapter
        self.observations = {}
        self.targets = {}
        self.action_evidence = set()
        self._repair_checks = {}
        self._blocked_attempts = {}

    def register_from_read(self, episode, task_text=""):
        if (
            episode.scope == self.adapter.scope
            and episode.origin == "environment"
            and episode.outcome == "success"
            and is_observation_read(self.adapter, episode)
        ):
            self.observations[episode.episode_id] = episode
        return []  # No targets or obligations are inferred from record shape.

    def record_action(self, episode):
        if episode.scope == self.adapter.scope and episode.origin == "environment":
            self.action_evidence.add(episode.episode_id)
        # An acknowledgment is evidence of an attempt, not verified requirement completion.

    def active_entity(self):
        return None

    def pending_entities(self):
        return []

    def selection_status(self, proposal):
        tracker = self.adapter.goal_tracker
        results = {}
        for declared in proposal.selection_checks:
            check = tracker.checks.get(declared.requirement_id, declared)
            if check.episode_id != proposal.source_episode_id:
                result = "target_source_mismatch"
            elif not check.source_binding or not check.observation:
                result = "missing_target_binding"
            elif any(
                path[:-1] != proposal.record_path
                for path in (check.source_binding.anchor_path, check.observation.path)
            ):
                result = "different_record"
            else:
                result = tracker.assess(check)
                if result == "matched":
                    episode = self.adapter.evidence.episode(check.episode_id)
                    result = compare_expectations(
                        episode.model_copy(update={"output_expectations": (check.observation,)})
                    )[0]["result"]
            results[check.requirement_id] = result
        return results

    @staticmethod
    def condition_contract(check):
        """Locators can be repaired; meaning cannot be replaced to fit an observation."""
        return check.model_copy(
            update={
                "episode_id": None,
                "observation": check.observation.model_copy(update={"path": ()})
                if check.observation
                else None,
                "source_binding": check.source_binding.model_copy(update={"anchor_path": ()})
                if check.source_binding
                else None,
            }
        )

    def diagnostic(self, proposal):
        statuses = self.selection_status(proposal)
        aliases = getattr(self.adapter, "evidence_aliases", {})
        ref = next(
            (a for a, eid in aliases.items() if eid == proposal.source_episode_id),
            proposal.source_episode_id,
        )
        source = self.adapter.evidence.episode(proposal.source_episode_id)
        failures = []
        for check in proposal.selection_checks:
            status = statuses[check.requirement_id]
            if status == "matched":
                continue
            locators = {}
            for label, path in (
                ("anchor", check.source_binding.anchor_path if check.source_binding else ()),
                ("observation", check.observation.path if check.observation else ()),
            ):
                inspected = inspect_path(source, path)
                entry = {"path": path, "result": inspected["result"]}
                if inspected["result"] == "observed":
                    entry["value"] = str(inspected["value"])[:120]
                else:
                    # Suggest observed locations with the same field name, never a desired value.
                    candidates = []

                    def walk(value, prefix=(), candidates=candidates, path=path):
                        if len(candidates) >= 4 or len(prefix) > 8:
                            return
                        items = (
                            value.items()
                            if isinstance(value, dict)
                            else enumerate(value[:20])
                            if isinstance(value, list)
                            else ()
                        )
                        for key, child in items:
                            current = (*prefix, key)
                            if (
                                path
                                and str(key) == str(path[-1])
                                and inspect_path(source, current)["result"] == "observed"
                            ):
                                candidates.append(current)
                            walk(child, current)

                    walk(source.response)
                    entry["candidate_paths"] = candidates[:4]
                locators[label] = entry
            failures.append({"requirement_id": check.requirement_id, "result": status, **locators})
        return {
            "target_id": proposal.target_id,
            "source": ref,
            "record_path": proposal.record_path,
            "failures": failures,
            "repair": "Resubmit select_task_target with corrected absolute paths/common record_path or a new read source. Preserve IDs, task quotes, anchors, predicates, expected values and effects. A lookup proves only its declared condition, not eligibility of another record.",
        }

    def action_error(self, action):
        if action.tool != "api_fetch":
            return None
        from .operation_semantics import known_effect

        effect, _ = known_effect(self.adapter, action.arguments)
        if effect == "read":
            return None
        proposal = self.targets.get(action.target_id)
        if proposal is not None and all(
            v == "matched" for v in self.selection_status(proposal).values()
        ):
            return None
        detail = (
            self.diagnostic(proposal)
            if proposal
            else {
                "target_id": action.target_id,
                "available_targets": list(self.targets),
                "repair": "Call select_task_target on a task-local read establishing the intended record's eligibility. Then set action.target_id to the accepted target.",
            }
        )
        signature = json.dumps([action.model_dump(mode="json"), detail], sort_keys=True)
        count = self._blocked_attempts.get(signature, 0) + 1
        self._blocked_attempts[signature] = count
        return {
            "status": "unchanged_blocked_retry" if count > 1 else "target_selection_required",
            "instruction": "Do not retry this write until selection is accepted; repair selection or obtain new evidence.",
            **detail,
            "unchanged_attempts": count,
        }

    def effect_update_error(self, check):
        for proposal in self.targets.values():
            for declared in proposal.selection_checks:
                if check.requirement_id == declared.requirement_id:
                    if self._repair_checks.get(check.requirement_id) == check:
                        continue
                    # Selection contracts cannot be weakened through the generic ledger.
                    if check.model_copy(update={"episode_id": declared.episode_id}) != declared:
                        return "cannot_change_target_selection"
                    if check.episode_id != declared.episode_id and all(
                        v == "matched" for v in self.selection_status(proposal).values()
                    ):
                        return "cannot_replace_selected_source"
            if check.requirement_id not in {c.requirement_id for c in proposal.effects}:
                continue
            if check.kind != "effect" or check.evidence_mode != "current_state":
                return "target_effect_requires_current_state"
            if check.outcome is not None:
                selection_ids = {c.requirement_id for c in proposal.selection_checks}
                if check.outcome.source_requirement_id not in selection_ids:
                    return "effect_requires_selected_source"
                if (
                    check.outcome.source_identity_path
                    and check.outcome.source_identity_path[: len(proposal.record_path)]
                    != proposal.record_path
                ):
                    return "identity_outside_selected_record"
        return None

    def summary(self):
        tracker = self.adapter.goal_tracker
        rows = []
        for proposal in self.targets.values():
            selection = self.selection_status(proposal)
            selected = all(value == "matched" for value in selection.values())
            effects = {
                c.requirement_id: tracker.assess(tracker.checks.get(c.requirement_id, c))
                for c in proposal.effects
            }
            rows.append(
                {
                    "target_id": proposal.target_id,
                    "source_episode_id": proposal.source_episode_id,
                    "record_path": proposal.record_path,
                    "selected": selected,
                    "selection_checks": selection,
                    "effects": effects,
                    "complete": selected and all(v == "matched" for v in effects.values()),
                }
            )
        return {
            "observation_count": len(self.observations),
            "targets": rows,
            "coverage": "Actor-declared requirements; full task coverage is not inferred.",
        }

    def unresolved(self):
        return [row for row in self.summary()["targets"] if not row["complete"]]

    def focus(self):
        for target in self.targets.values():
            for check in (*target.selection_checks, *target.effects):
                live = self.adapter.goal_tracker.checks.get(check.requirement_id, check)
                if self.adapter.goal_tracker.assess(live) != "matched":
                    return check.task_quote
        return ""

    def tool(self):
        async def select(params):
            if isinstance(params, SelectTarget):
                params = params.proposal()
            aliases = getattr(self.adapter, "evidence_aliases", {})

            def resolve(check):
                return check.model_copy(
                    update={"episode_id": aliases.get(check.episode_id, check.episode_id)}
                )

            proposal = params.model_copy(
                update={
                    "source_episode_id": aliases.get(
                        params.source_episode_id, params.source_episode_id
                    ),
                    "selection_checks": tuple(resolve(c) for c in params.selection_checks),
                }
            )
            try:
                source = self.adapter.evidence.episode(proposal.source_episode_id)
            except KeyError:
                return ToolResult(content="Unknown target source episode", success=False)
            if (
                source.scope != self.adapter.scope
                or source.origin != "environment"
                or source.outcome != "success"
                or not is_observation_read(self.adapter, source)
            ):
                return ToolResult(
                    content="Target requires a task-local successful read", success=False
                )
            ids = [c.requirement_id for c in (*proposal.selection_checks, *proposal.effects)]
            if len(ids) > 16:
                return ToolResult(content="At most 16 total checks per target", success=False)
            if len(set(ids)) != len(ids):
                return ToolResult(
                    content="Use distinct selection and effect requirement IDs", success=False
                )
            for other_id, other in self.targets.items():
                if other_id != proposal.target_id and set(ids) & {
                    c.requirement_id for c in (*other.selection_checks, *other.effects)
                }:
                    return ToolResult(
                        content="Requirement already bound to another target", success=False
                    )
            if any(
                c.kind != "source" or c.evidence_mode != "historical_observation"
                for c in proposal.selection_checks
            ):
                return ToolResult(
                    content="Selection checks must be historical source checks", success=False
                )
            if any(
                c.kind != "effect" or c.evidence_mode != "current_state" for c in proposal.effects
            ):
                return ToolResult(
                    content="Each required effect needs its own current-state check", success=False
                )
            old = self.targets.get(proposal.target_id)
            if old and old != proposal:
                old_conditions = tuple(self.condition_contract(c) for c in old.selection_checks)
                new_conditions = tuple(
                    self.condition_contract(c) for c in proposal.selection_checks
                )
                if (
                    all(v == "matched" for v in self.selection_status(old).values())
                    or old_conditions != new_conditions
                    or old.effects != proposal.effects
                ):
                    return ToolResult(
                        content="Cannot weaken or replace bound target requirements", success=False
                    )
            selection_ids = {c.requirement_id for c in proposal.selection_checks}
            if any(
                c.outcome and c.outcome.source_requirement_id not in selection_ids
                for c in proposal.effects
            ):
                return ToolResult(
                    content="Effects must bind to a declared selection source", success=False
                )
            tracker = self.adapter.goal_tracker
            self._repair_checks = {c.requirement_id: c for c in proposal.selection_checks}
            try:
                # Preserve any effect evidence already registered while repairing selection.
                effects = tuple(tracker.checks.get(c.requirement_id, c) for c in proposal.effects)
                result = await tracker.tool().executor(
                    GoalUpdate(checks=(*proposal.selection_checks, *effects))
                )
            finally:
                self._repair_checks = {}
            if not result.success:
                return result
            self.targets[proposal.target_id] = proposal
            statuses = self.selection_status(proposal)
            return ToolResult(
                content=json.dumps(
                    {
                        "target_id": proposal.target_id,
                        "selected": all(v == "matched" for v in statuses.values()),
                        "selection_checks": statuses,
                        "diagnostic": self.diagnostic(proposal)
                        if any(v != "matched" for v in statuses.values())
                        else None,
                    }
                ),
                success=all(v == "matched" for v in statuses.values()),
            )

        return Tool(
            name="select_task_target",
            parameters=SelectTarget,
            executor=select,
            description="Before acting on a record, propose its task-local source episode and record_path. "
            "Declare source selection_checks for ALL task conditions (identity, eligibility, policy, recency) "
            "using exact task quotes, anchor_text/anchor_path, and an observation predicate. Paths are absolute within the source response; record_path is their common parent. Register each required effect separately "
            "in effects with its own requirement_id and task_quote; evidence may be absent initially. "
            "Policy and lookup reads are observations, not automatic targets. Reuse these effect IDs "
            "with check_effect or record_goal_checks; bind outcomes to a selection source_requirement_id. "
            "When the output identity differs from the task anchor, source_identity_path locates "
            "the identity in the selected source and outcome.identity_path locates it in the output. "
            "Successful writes never automatically complete effects. Coverage is your responsibility.",
        )
