"""Structured task decisions and evidence checks, independent of benchmark workflows.

Not enabled in the current learning-curve baseline. The caller must supply known
mandatory requirements from the tool boundary: a model's omissions cannot waive
them. Passing this gate establishes only declared prerequisites, not task success
or completeness of the world's known constraints.
"""

import json
import re
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from .knowledge import Dependency, LiveState, OutputExpectation, Record, Scope, utc_now
from .store import KnowledgeStore


class UnresolvedRequirements(ValueError):
    """Controller could not resolve an action's known prerequisites."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


class LiveRequirement(Record):
    key: str = Field(min_length=1)
    expected: Any


class ActionRequirements(Record):
    claims: tuple[Dependency, ...] = ()
    facts: tuple[LiveRequirement, ...] = ()
    operation_id: str | None = None


class ProposedAction(Record):
    target_id: str | None = None
    tool: str = Field(default="", min_length=0)
    arguments: dict[str, Any]
    requirements: ActionRequirements = Field(default_factory=ActionRequirements)


class Explanation(Record):
    """A possible cause is never promoted just because the model proposes it."""

    hypothesis: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    distinguishing_query: ProposedAction | None = None
    expected_observations: tuple[str, ...] = Field(min_length=2)
    remedy: ProposedAction | None = None
    verification: ProposedAction | None = None


class Decision(Record):
    goal: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    action: ProposedAction
    alternatives: tuple[Explanation, ...] = ()
    # A proposal's evidence references are checked, not treated as live facts.
    evidence_ids: tuple[str, ...] = ()
    considered_claim_ids: tuple[str, ...] = Field(default=(), max_length=8)
    output_expectations: tuple[OutputExpectation, ...] = Field(default=(), max_length=8)


class CompactDecision(Record):
    """Acting surface omits unexecuted diagnostic trees; full Decision stays internal."""

    action: ProposedAction
    goal: str = Field(default="Next task step", min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=320)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=8)
    considered_claim_ids: tuple[str, ...] = Field(
        default=(), max_length=8,
        description="Retrieved claims or hypotheses considered for this decision; not prerequisites or proof.",
    )
    output_expectations: tuple[OutputExpectation, ...] = Field(default=(), max_length=4)

    @model_validator(mode="before")
    @classmethod
    def normalize_compact_decision(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if isinstance(data.get("goal"), str) and len(data["goal"]) > 240:
            data["goal"] = data["goal"][:240]
        if "arguments" in data and isinstance(data["arguments"], dict) and "tool" in data["arguments"] and not data.get("tool"):
            data["tool"] = data["arguments"].pop("tool")

        root_tool = data.pop("tool", None)
        if root_tool is not None:
            root_tool = str(root_tool).strip()

        action_val = data.get("action")
        # Check conflict with serialized string action if root_tool was provided
        if root_tool and isinstance(action_val, str):
            action_str = action_val.strip()
            if action_str.startswith("{"):
                try:
                    parsed = json.loads(action_str)
                    if isinstance(parsed, dict):
                        action_tool = parsed.get("tool")
                        if (not action_tool) and isinstance(parsed.get("arguments"), dict) and "tool" in parsed["arguments"]:
                            action_tool = parsed["arguments"].get("tool")
                        if action_tool and str(action_tool).strip() != root_tool:
                            raise ValueError(f"Conflicting tool specifications: root '{root_tool}' != action '{action_tool}'")
                except ValueError:
                    raise
                except Exception:
                    pass

        # Only normalize dictionary action payloads emitted by tool calls;
        # leave serialized strings to decode_serialized_action for strict validation.
        if isinstance(action_val, dict):
            action_dict = dict(action_val)
            action_tool = None
            if "tool" in action_dict and action_dict["tool"]:
                action_tool = str(action_dict["tool"]).strip()
            elif isinstance(action_dict.get("arguments"), dict) and "tool" in action_dict["arguments"]:
                action_tool = str(action_dict["arguments"].pop("tool")).strip()

            if root_tool and action_tool and root_tool != action_tool:
                raise ValueError(f"Conflicting tool specifications: root '{root_tool}' != action '{action_tool}'")

            effective_tool = action_tool or root_tool
            if effective_tool:
                action_dict["tool"] = effective_tool

            # 1. Hoist goal if nested inside action and not provided at root
            if "goal" in action_dict:
                nested_goal = action_dict.pop("goal")
                if ("goal" not in data or data["goal"] == "Next task step") and nested_goal:
                    data["goal"] = str(nested_goal)

            # 2. Hoist rationale if nested inside action and not provided at root
            if "rationale" in action_dict:
                nested_rationale = action_dict.pop("rationale")
                if ("rationale" not in data or not data["rationale"]) and nested_rationale:
                    data["rationale"] = str(nested_rationale)

            # Hoist decision-level annotations if nested inside action
            if "output_expectations" in action_dict:
                nested_oe = action_dict.pop("output_expectations")
                if ("output_expectations" not in data or not data["output_expectations"]) and nested_oe:
                    data["output_expectations"] = nested_oe

            if "evidence_ids" in action_dict:
                nested_ev = action_dict.pop("evidence_ids")
                if ("evidence_ids" not in data or not data["evidence_ids"]) and nested_ev:
                    data["evidence_ids"] = nested_ev

            if "considered_claim_ids" in action_dict:
                nested_cc = action_dict.pop("considered_claim_ids")
                if ("considered_claim_ids" not in data or not data["considered_claim_ids"]) and nested_cc:
                    data["considered_claim_ids"] = nested_cc

            # 3. Consolidate parameter fields into arguments
            known_action_fields = {"tool", "requirements"}
            extra_keys = set(action_dict.keys()) - known_action_fields - {"arguments"}
            if extra_keys:
                if "arguments" not in action_dict or not isinstance(action_dict["arguments"], dict):
                    action_dict["arguments"] = {}
                else:
                    action_dict["arguments"] = dict(action_dict["arguments"])
                for k in extra_keys:
                    action_dict["arguments"][k] = action_dict.pop(k)

            data["action"] = action_dict

        return data

    @field_validator("action", mode="before")
    @classmethod
    def decode_serialized_action(cls, value):
        # Some tool-call providers serialize a nested object a second time,
        # or emit bare tool aliases/names (e.g. "a1" or "api_search").
        # Decode this one boundary only; ProposedAction and dispatch still check
        # the full schema, arguments, permissions and prerequisites afterward.
        if isinstance(value, str):
            value_str = value.strip()
            # Preserve tool aliases/names (e.g. "a1", "api_search", "a1:api_search") until registry resolution
            if value_str.lower() not in ("null", "true", "false", "none") and re.fullmatch(
                r"(?:[a-zA-Z0-9_-]+:)?[a-zA-Z0-9_-]+", value_str
            ):
                return {"tool": value_str, "arguments": {}}
            try:
                parsed = json.loads(value_str)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("Serialized action must be a JSON object") from exc
            if isinstance(parsed, dict):
                return parsed
            if (
                isinstance(parsed, str)
                and parsed.strip().lower() not in ("null", "true", "false", "none")
                and re.fullmatch(r"(?:[a-zA-Z0-9_-]+:)?[a-zA-Z0-9_-]+", parsed.strip())
            ):
                return {"tool": parsed.strip(), "arguments": {}}
            raise ValueError("Serialized action must be a JSON object")
        return value


class GateResult(Record):
    allowed: bool
    gaps: tuple[str, ...]
    checked_claims: tuple[str, ...]
    checked_facts: tuple[str, ...]


def check_decision(
    decision: Decision,
    *,
    mandatory: ActionRequirements,
    store: KnowledgeStore,
    scope: Scope,
    live: LiveState,
    allowed_tools: set[str],
    now: datetime | None = None,
    evidence_store: KnowledgeStore | None = None,
) -> GateResult:
    """Recheck immediately before execution; neither writes nor spends tool calls.

    Missing, stale, contradictory, or cross-task evidence yields explicit gaps.
    Dependency closure is checked independently of the retrieved context budget.
    Monitoring a disputed proposition does not require assuming it true.
    """
    now = now or utc_now()
    evidence_store = evidence_store or store
    gaps: set[str] = set()
    checked_claims: set[str] = set()
    checked_facts: set[str] = set()
    if scope.task_id != live.task_id:
        gaps.add("scope:live_state")
    if decision.action.tool not in allowed_tools:
        gaps.add(f"tool:{decision.action.tool}")
    facts = {f.key: f for f in live.facts(now)}

    def check_fact(key, expected):
        checked_facts.add(key)
        fact = facts.get(key)
        if fact is None:
            gaps.add(f"refresh:{key}")
            return
        try:
            episode = evidence_store.episode(fact.episode_id)
        except KeyError:
            gaps.add(f"evidence:{fact.episode_id}")
            return
        if (
            episode.task_id != live.task_id
            or episode.scope != scope
            or episode.outcome != "success"
            or episode.timestamp > now
        ):
            gaps.add(f"untrusted_live:{key}")
        if type(fact.value) is not type(expected) or fact.value != expected:
            gaps.add(f"prerequisite:{key}")

    proposed = decision.action.requirements
    for requirement in (*mandatory.facts, *proposed.facts):
        check_fact(requirement.key, requirement.expected)
    pending = [*mandatory.claims, *proposed.claims]
    visited = set()
    while pending:
        dependency = pending.pop()
        if dependency.mode == "monitors":
            continue
        # Check every pin/field requirement even when several paths reach a claim.
        identity = (dependency.claim_id, dependency.version, dependency.fields)
        if identity in visited:
            continue
        visited.add(identity)
        checked_claims.add(dependency.claim_id)
        try:
            claim = store.claim(dependency.claim_id)
        except KeyError:
            gaps.add(f"missing_claim:{dependency.claim_id}")
            continue
        if not claim.scope.applies_to(scope):
            gaps.add(f"scope:{claim.claim_id}")
        if not store.dependency_compatible(dependency):
            gaps.add(f"dependency:{claim.claim_id}@{dependency.version}")
        if claim.valid_until is not None and claim.valid_until <= now:
            gaps.add(f"refresh_claim:{claim.claim_id}")
        for key, expected in claim.conditions.items():
            check_fact(key, expected)
        pending.extend(claim.dependencies)

    evidence_ids = set(decision.evidence_ids)
    for alternative in decision.alternatives:
        evidence_ids.update(alternative.evidence_ids)
    for evidence_id in evidence_ids:
        try:
            episode = evidence_store.episode(evidence_id)
        except KeyError:
            try:
                episode = store.episode(evidence_id)
            except KeyError:
                gaps.add(f"evidence:{evidence_id}")
                continue
        # Historical episodes may inform diagnosis, but not another tenant's data.
        if (
            episode.scope.environment != scope.environment
            or episode.scope.version != scope.version
            or episode.scope.role != scope.role
            or episode.scope.tenant != scope.tenant
            or episode.timestamp > now
        ):
            gaps.add(f"evidence_scope:{evidence_id}")

    # Observational contrasts, including legacy "supported" records, do not
    # establish necessity. Only explicit contracts and pinned requirements gate.

    return GateResult(
        allowed=not gaps,
        gaps=tuple(sorted(gaps)),
        checked_claims=tuple(sorted(checked_claims)),
        checked_facts=tuple(sorted(checked_facts)),
    )
