"""Budgeted retrieval with scope, dependency, and freshness checks."""

import re
from datetime import datetime
from urllib.parse import urlsplit

from .knowledge import Claim, LiveState, Scope, WorkingGraph, utc_now
from .store import KnowledgeStore


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def decision_focus(adapter):
    """Only explicit action identity/subgoal, never arbitrary response prose."""
    decision = getattr(adapter, "active_decision", None)
    if decision is None:
        return (), ""
    action = decision.action
    entities = [action.tool]
    for key in ("url", "endpoint_id", "table", "entity", "procedure_id"):
        value = action.arguments.get(key)
        if isinstance(value, str):
            entities.append(value)
    return tuple(entities), decision.goal


def entity_match(entity, participant):
    if entity == participant:
        return True
    if entity.startswith("https://") and participant.startswith("https://"):
        actual, template = urlsplit(entity), urlsplit(participant)
        pattern = re.escape(template.path)
        pattern = re.sub(r"\\\{[^{}]+\\\}", "[^/]+", pattern)
        return actual.netloc == template.netloc and re.fullmatch(pattern, actual.path) is not None
    return False


def retrieve(
    store: KnowledgeStore,
    scope: Scope,
    query: str,
    *,
    live: LiveState | None = None,
    max_chars: int = 12000,
    expand: bool = True,
    layer_aware: bool = True,
    now: datetime | None = None,
    focus_entities: tuple[str, ...] = (),
    subgoal: str = "",
) -> WorkingGraph:
    """Same claim content for flat and graph arms; only access/selection differs.

    An oversized mandatory dependency bundle is rejected intact, not truncated.
    Explicit gaps let execution seek evidence rather than silently skip checks.
    """
    if max_chars < 1:
        raise ValueError("Retrieval budget must be positive")
    now = now or utc_now()
    candidates = {c.claim_id: c for c in store.claims(scope)}
    query_terms = terms(query)
    facts = live.facts(now) if live else ()
    fact_values = {f.key: f.value for f in facts}
    selected: dict[str, Claim] = {}
    gaps: set[str] = set()

    def rank(claim: Claim):
        overlap = len(query_terms & terms(" ".join(claim.participants) + " " + claim.statement))
        focus = sum(entity_match(e, p) for e in focus_entities for p in claim.participants)
        local = len(terms(subgoal) & terms(claim.statement))
        base = focus * 100 + local * 4 + overlap
        actionable = (
            2
            if claim.relation in ("tested_response_rule", "endpoint_contract", "contrastive_precondition")
            or claim.payload.get("response_rule") is not None
            else (1 if claim.payload.get("workspace_fact") else 0)
        )
        layer_priority = {"interface": 2, "configuration": 1, "behavior": 0}
        return (
            base + (actionable * 3 if base > 0 else 0),
            actionable,
            layer_priority[claim.layer] if layer_aware else 0,
            claim.claim_id,
        )

    ranked = sorted((c for c in candidates.values() if rank(c)[0] > 0), key=rank, reverse=True)
    queued = {c.claim_id for c in ranked}
    # Lexical seeds may expand. Reusable contracts stay available when the next
    # task shares no words with the stored description; they do not pull a neighborhood.
    initial_seeds = set(queued)
    remembered = (
        []
        if focus_entities
        else [
            c
            for c in candidates.values()
            if c.claim_id not in queued
            and c.relation in ("endpoint_contract", "contrastive_precondition")
            and c.status == "supported"
            and not c.needs_review
            and (c.valid_until is None or c.valid_until > now)
        ][:2]
    )
    ranked.extend(
        sorted(
            remembered,
            key=lambda c: (c.last_verified_at or c.observed_at, c.claim_id),
            reverse=True,
        )
    )
    queued.update(c.claim_id for c in remembered)
    for seed in ranked:
        bundle: dict[str, Claim] = {}
        pending = [seed]
        local_gaps: set[str] = set()
        while pending:
            claim = pending.pop()
            if claim.claim_id in bundle:
                continue
            bundle[claim.claim_id] = claim
            if claim.status != "supported" or claim.needs_review:
                local_gaps.add(f"unsupported:{claim.claim_id}")
            if claim.valid_until is not None and claim.valid_until <= now:
                local_gaps.add(f"refresh:{claim.claim_id}")
            for key, value in claim.conditions.items():
                if key not in fact_values or fact_values[key] != value:
                    local_gaps.add(f"prerequisite:{key}")
            for dependency in claim.dependencies:
                if dependency.mode == "monitors":
                    continue
                other = candidates.get(dependency.claim_id)
                if other is None or not store.dependency_compatible(dependency):
                    local_gaps.add(f"dependency:{dependency.claim_id}@{dependency.version}")
                elif expand:
                    pending.append(other)
                elif other.claim_id not in selected:
                    local_gaps.add(f"dependency:{other.claim_id}@{other.version}")
        if local_gaps:
            gaps.update(local_gaps)
            continue
        combined = {**selected, **bundle}
        if sum(len(c.model_dump_json()) for c in combined.values()) > max_chars:
            if seed.claim_id in initial_seeds:
                gaps.add(f"budget:{seed.claim_id}")
            continue
        selected = combined
        if expand and seed.claim_id in initial_seeds:
            # One relational expansion; dependencies of these candidates still
            # go through the same mandatory validation loop on subsequent seeds.
            related = store.related(set(seed.participants), scope)
            for candidate in related:
                if candidate.claim_id not in queued:
                    queued.add(candidate.claim_id)
                    ranked.append(candidate)
    return WorkingGraph(
        scope=scope,
        claims=tuple(selected.values()),
        facts=facts,
        gaps=tuple(sorted(gaps)),
        blocked=bool(gaps) and not selected,
    )
