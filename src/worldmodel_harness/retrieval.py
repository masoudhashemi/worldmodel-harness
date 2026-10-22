"""Budgeted retrieval with scope, dependency, and freshness checks."""

import re
from datetime import datetime
from urllib.parse import urlsplit

from .knowledge import Claim, LiveState, Scope, WorkingGraph, utc_now
from .store import KnowledgeStore


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def decision_focus(adapter):
    """Only explicit action identity/subgoal, conditioned on active entity and requirements."""
    decision = getattr(adapter, "active_decision", None)
    entities, subgoal = [], ""
    if decision is not None:
        action = decision.action
        entities = [action.tool]
        subgoal = decision.goal
        for key in ("url", "endpoint_id", "table", "entity", "procedure_id"):
            value = action.arguments.get(key)
            if isinstance(value, str):
                entities.append(value)

    # Condition retrieval on the active entity's next pending requirement
    entity_tracker = getattr(adapter, "entity_tracker", None)
    if entity_tracker is not None:
        active = entity_tracker.active_entity()
        if active is not None:
            next_ch = active.next_pending_channel()
            if next_ch:
                channel_service_map = {
                    "sms": "twilio",
                    "email": "gmail",
                    "slack": "slack",
                    "asana": "asana",
                }
                svc = channel_service_map.get(next_ch, next_ch)
                entities.append(svc)
                if active.email and next_ch == "email":
                    entities.append(active.email)
                if active.phone and next_ch == "sms":
                    entities.append(active.phone)
                if active.name:
                    entities.append(active.name)
                if active.badge_id:
                    entities.append(active.badge_id)
                subgoal = f"Send {next_ch} notification to {active.name or active.badge_id}"

    interaction = getattr(adapter, "last_interaction", None)
    if interaction is not None:
        if not interaction.operation.startswith("tool_call:"):
            entities.append(interaction.operation)
        feedback = interaction.response.get("result", {})
        if interaction.outcome != "success" and isinstance(feedback, dict):
            # Error labels are retrieval data, never instructions or proof.
            labels = [feedback.get(key) for key in ("origin", "result", "field")]
            subgoal += " " + " ".join(str(value)[:160] for value in labels if value)
    return tuple(dict.fromkeys(entities)), subgoal


def entity_match(entity, participant):
    if entity == participant:
        return True
    if entity.startswith("https://") and participant.startswith("https://"):
        actual, template = urlsplit(entity), urlsplit(participant)
        pattern = re.escape(template.path)
        pattern = re.sub(r"\\\{[^{}]+\\\}", "[^/]+", pattern)
        return actual.netloc == template.netloc and re.fullmatch(pattern, actual.path) is not None
    return False


def claim_service(claim: Claim) -> str | None:
    """Identify the service domain for an endpoint or claim."""
    if isinstance(claim.payload, dict):
        if claim.payload.get("service"):
            return str(claim.payload["service"]).lower()
        cid = claim.payload.get("id")
        if isinstance(cid, str) and "." in cid:
            return cid.split(".")[0].lower()
    for p in claim.participants:
        if isinstance(p, str) and "." in p and not p.startswith("http"):
            return p.split(".")[0].lower()
        if isinstance(p, str) and p.startswith("https://"):
            host = urlsplit(p).netloc.lower()
            for part in host.split("."):
                if part not in ("api", "www", "com", "googleapis", "org", "net", "io", "co"):
                    return part
    return None


def detect_active_services(text: str, focus_entities: tuple[str, ...] = ()) -> set[str]:
    """Extract active service names from task query, subgoals, or focus entities."""
    from .trajectory_graph import SERVICE_SYNONYMS

    raw = terms(text)
    active = set()
    for w in raw:
        if w.startswith("ss_") or w.startswith("ws_"):
            active.add("sheets")
        for k, v in SERVICE_SYNONYMS.items():
            if w == k or (len(k) >= 4 and k in w):
                active.add(v)
    for fe in focus_entities:
        if "." in fe and not fe.startswith("http"):
            active.add(fe.split(".")[0].lower())
        elif fe.startswith("https://"):
            host = urlsplit(fe).netloc.lower()
            for part in host.split("."):
                if part not in ("api", "www", "com", "googleapis", "org", "net", "io", "co"):
                    active.add(part)
    return active


def claim_stage(claim: Claim) -> str:
    """Classify a claim or endpoint into a workflow stage: lookup, mutate, or notify."""
    if claim.relation == "documented_operation_semantics":
        effect = str(claim.payload.get("effect", "")).lower()
        quotes = " ".join(str(q) for q in claim.payload.get("quotes", [])).lower()
        if effect == "write":
            if any(w in quotes for w in ("sms", "email", "message", "notify", "call")):
                return "notify"
            return "mutate"
        return "lookup"
    if claim.relation == "endpoint_contract":
        method = str(claim.payload.get("method", "GET")).upper()
        ep_id = str(claim.payload.get("id", "")).lower()
        svc = claim_service(claim) or ""
        if method == "GET":
            return "lookup"
        if svc in ("twilio", "gmail", "slack") or any(
            w in ep_id for w in ("message", "mail", "sms", "notify", "chat.post")
        ):
            return "notify"
        return "mutate"
    return "other"


def required_workflow_stages(query: str, subgoal: str = "") -> list[str]:
    """Identify required stages from symbolic task language (matching graph-grounded motifs)."""
    text = (query + " " + subgoal).lower()
    stages = ["lookup"]  # All workflows require resolving unknowns or reading state
    if re.search(
        r"\b(add|archive|assign|cancel|change|complete|create|delete|draft|modify|move|publish|remove|schedule|set|tag|update|upload)\b",
        text,
    ):
        stages.append("mutate")
    if re.search(
        r"\b(call|email|message|notify|post|reply|send|sms|alert)\b",
        text,
    ):
        stages.append("notify")
    return stages


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
    current_node: str | None = None,
    completed_nodes: set[str] | None = None,
) -> WorkingGraph:
    """Same claim content for flat and graph arms; only access/selection differs.

    Incorporates graph-grounded retrieval mechanisms:
    - Stage Coverage Guarantee: ensures lookup, mutate, and notify tools are not displaced.
    - Reverse Dataflow Prerequisites: automatically pulls producers of consumer input parameters.
    - Forward Dataflow Frontier: automatically pulls downstream consumers of current step outputs.
    - Intent Decoupling: isolates domain keywords from generic HTTP verbs.
    - Reason Attribution: tracks why each capability was selected.
    """
    if max_chars < 1:
        raise ValueError("Retrieval budget must be positive")
    now = now or utc_now()
    active_services = detect_active_services(query + " " + subgoal, focus_entities)
    raw_candidates = {
        c.claim_id: c for c in store.claims(scope)
        if c.status == "supported" and not c.needs_review
    }
    if active_services:
        candidates = {
            cid: c for cid, c in raw_candidates.items()
            if claim_service(c) is None or claim_service(c) in active_services
        }
    else:
        candidates = raw_candidates

    # Reverse and forward dataflow indexes
    dataflow_reverse: dict[str, list[str]] = {}
    dataflow_forward: dict[str, list[str]] = {}
    for c in raw_candidates.values():
        if c.relation == "observed_value_flow":
            src = c.payload.get("source_endpoint")
            tgt = c.payload.get("target_endpoint")
            if src and tgt and src != tgt:
                dataflow_reverse.setdefault(tgt, []).append(src)
                dataflow_forward.setdefault(src, []).append(tgt)

    required_stages = required_workflow_stages(query, subgoal)
    query_terms = terms(query)
    # Decouple service names and generic HTTP verbs from domain intent matching
    generic_verbs = {"get", "post", "put", "patch", "delete", "to", "in", "of", "the", "and", "is", "for"}
    intent_query_terms = query_terms - generic_verbs - active_services
    if not intent_query_terms:
        intent_query_terms = query_terms - generic_verbs

    facts = live.facts(now) if live else ()
    fact_values = {f.key: f.value for f in facts}
    selected: dict[str, Claim] = {}
    gaps: set[str] = set()
    reasons: dict[str, set[str]] = {}

    def rank(claim: Claim):
        overlap = len(intent_query_terms & terms(" ".join(claim.participants) + " " + claim.statement))
        focus = sum(entity_match(e, p) for e in focus_entities for p in claim.participants)
        local = len(terms(subgoal) & terms(claim.statement))
        base = focus * 100 + local * 4 + overlap
        c_stage = claim_stage(claim)
        stage_boost = 15 if (base > 0 and c_stage in required_stages) else 0
        base += stage_boost
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
    initial_seeds = set(queued)
    for c in ranked:
        reasons.setdefault(c.claim_id, set()).add("lexical_intent")
        if any(entity_match(e, p) for e in focus_entities for p in c.participants):
            reasons[c.claim_id].add("focus_entity")

    # 1. Stage Coverage Guarantee (graph-grounded pattern)
    if active_services:
        covered_stages = {claim_stage(c) for c in ranked[: min(len(ranked), 4)]}
        for stg in required_stages:
            if stg not in covered_stages:
                stage_cands = [c for c in candidates.values() if claim_stage(c) == stg]
                if stage_cands:
                    best_stage_cand = max(stage_cands, key=lambda c: rank(c)[0])
                    if best_stage_cand.claim_id not in queued:
                        ranked.append(best_stage_cand)
                        queued.add(best_stage_cand.claim_id)
                    reasons.setdefault(best_stage_cand.claim_id, set()).add(f"stage_coverage:{stg}")

    # 2. Service Coverage Guarantee (graph-grounded multi-service pattern)
    if active_services:
        covered_services = {claim_service(c) for c in ranked if claim_service(c)}
        for srv in active_services:
            if srv not in covered_services:
                srv_cands = [c for c in candidates.values() if claim_service(c) == srv and c.relation == "endpoint_contract"]
                if srv_cands:
                    best_srv_cand = max(srv_cands, key=lambda c: (rank(c)[0], c.claim_id))
                    if best_srv_cand.claim_id not in queued:
                        ranked.append(best_srv_cand)
                        queued.add(best_srv_cand.claim_id)
                    reasons.setdefault(best_srv_cand.claim_id, set()).add(f"service_coverage:{srv}")

        # Interleave active services so no single service exhausts the budget before others
        from collections import defaultdict
        by_srv: dict[str | None, list[Claim]] = defaultdict(list)
        for c in ranked:
            by_srv[claim_service(c)].append(c)

        balanced_ranked: list[Claim] = []
        srv_order = [s for s in sorted(active_services) if s in by_srv]
        max_depth = max((len(by_srv[s]) for s in srv_order), default=0)
        for d in range(max_depth):
            for s in srv_order:
                if d < len(by_srv[s]):
                    balanced_ranked.append(by_srv[s][d])
        for s, clist in by_srv.items():
            if s not in active_services:
                balanced_ranked.extend(clist)
        ranked = balanced_ranked

    # 3. Reverse Dataflow Prerequisite Resolution (graph-grounded pattern)
    dataflow_additions: list[Claim] = []
    for c in list(ranked[:6]):
        if c.relation == "endpoint_contract":
            ep_id = c.payload.get("id")
            for prod_id in dataflow_reverse.get(ep_id, []):
                for cand in candidates.values():
                    if cand.relation == "endpoint_contract" and cand.payload.get("id") == prod_id:
                        if cand.claim_id not in queued:
                            dataflow_additions.append(cand)
                            queued.add(cand.claim_id)
                        reasons.setdefault(cand.claim_id, set()).add(f"prerequisite_dataflow:{ep_id}")
    if dataflow_additions:
        ranked = dataflow_additions + ranked

    # 4. Forward Dataflow Frontier Resolution (state-local pattern)
    forward_additions: list[Claim] = []
    frontier_sources: list[str] = []
    if current_node:
        frontier_sources.append(current_node)
    for c in list(ranked[:4]):
        if c.relation == "endpoint_contract":
            ep_id = c.payload.get("id")
            if ep_id and ep_id not in frontier_sources:
                frontier_sources.append(ep_id)

    for src_id in frontier_sources:
        for cons_id in dataflow_forward.get(src_id, []):
            for cand in candidates.values():
                if cand.relation == "endpoint_contract" and cand.payload.get("id") == cons_id:
                    if cand.claim_id not in queued:
                        forward_additions.append(cand)
                        queued.add(cand.claim_id)
                    reasons.setdefault(cand.claim_id, set()).add(f"downstream_frontier:{src_id}")
    if forward_additions:
        ranked = forward_additions + ranked
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
    for c in remembered:
        reasons.setdefault(c.claim_id, set()).add("remembered_contract")
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
            # One relational expansion; ignore generic HTTP verbs so methods don't link unrelated endpoints.
            http_methods = {"get", "post", "put", "delete", "patch", "head", "options"}
            domain_participants = {p for p in seed.participants if p.lower() not in http_methods}
            related = store.related(domain_participants, scope) if domain_participants else []
            for candidate in related:
                if candidate.status != "supported" or candidate.needs_review:
                    continue
                if active_services and claim_service(candidate) is not None and claim_service(candidate) not in active_services:
                    continue
                if candidate.claim_id not in queued:
                    queued.add(candidate.claim_id)
                    ranked.append(candidate)
    return WorkingGraph(
        scope=scope,
        claims=tuple(selected.values()),
        facts=facts,
        gaps=tuple(sorted(gaps)),
        blocked=bool(gaps) and not selected,
        stages=tuple(required_stages),
        reasons={cid: tuple(sorted(r)) for cid, r in reasons.items() if cid in selected},
    )
