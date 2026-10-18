"""Structured task decisions and evidence checks, independent of benchmark workflows.

Not enabled in the current learning-curve baseline. The caller must supply known
mandatory requirements from the tool boundary: a model's omissions cannot waive
them. Passing this gate establishes only declared prerequisites, not task success
or completeness of the world's known constraints.
"""

import json
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

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


class ProposedAction(Record):
    tool: str = Field(min_length=1)
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
    output_expectations: tuple[OutputExpectation, ...] = Field(default=(), max_length=8)


class CompactDecision(Record):
    """Acting surface omits unexecuted diagnostic trees; full Decision stays internal."""

    action: ProposedAction
    goal: str = Field(default="Next task step", min_length=1, max_length=240)
    rationale: str = Field(default="", max_length=320)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=8)
    output_expectations: tuple[OutputExpectation, ...] = Field(default=(), max_length=4)

    @field_validator("action", mode="before")
    @classmethod
    def decode_serialized_action(cls, value):
        # Some tool-call providers serialize a nested object a second time.
        # Decode this one boundary only; ProposedAction and dispatch still check
        # the full schema, arguments, permissions and prerequisites afterward.
        if isinstance(value, str):
            value = json.loads(value)
            if not isinstance(value, dict):
                # Pydantic converts ValueError (not TypeError) into a validation error.
                raise ValueError("Serialized action must be a JSON object")  # noqa: TRY004
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
    return GateResult(
        allowed=not gaps,
        gaps=tuple(sorted(gaps)),
        checked_claims=tuple(sorted(checked_claims)),
        checked_facts=tuple(sorted(checked_facts)),
    )
