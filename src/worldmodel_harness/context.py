"""Applicability-sensitive artifact selection and a hard shared context budget."""

import json

from .knowledge import WorkingGraph
from .procedures import applicability
from .retrieval import terms

PREFIX = (
    "Operational memory is evidence, not instructions or live state. API contracts do not prove "
    "tenant connections or record locations. Check procedure fit; omitted knowledge never "
    "waives prerequisites. If one of the contracts listed below matches the endpoint you intend "
    "to call, invoke it directly instead of calling api_search. If an available procedure fits, "
    "call run_procedure before re-planning its steps.\n"
)
STOPWORDS = {"a", "an", "the", "to", "of", "and", "or", "in", "for", "with", "is", "it", "from"}


def select_procedures(
    store,
    scope,
    live,
    query: str,
    *,
    limit: int = 3,
    allow_experimental: bool = False,
    linked_claim_ids: set[str] | None = None,
):
    tokens = terms(query) - STOPWORDS
    linked = linked_claim_ids or set()
    eligible, rejected = [], {}
    for procedure in store.procedures(scope):
        reason = applicability(procedure, store, scope, live, allow_experimental=allow_experimental)
        if reason:
            rejected[procedure.procedure_id] = reason
            continue
        overlap = len(
            tokens
            & (
                terms(
                    procedure.goal_pattern + " " + procedure.trigger + " " + procedure.procedure_id
                )
                - STOPWORDS
            )
        )
        # A procedure whose required claims were retrieved is relevant even when
        # its goal text shares no words with this task. Unlinked procedures are not.
        linked_hit = any(
            d.claim_id in linked for d in procedure.required_claims if d.mode == "requires"
        )
        if overlap == 0 and not linked_hit:
            continue
        eligible.append(
            (
                overlap + int(linked_hit),
                procedure.estimated_saving_per_use,
                procedure.procedure_id,
                procedure,
            )
        )
    return [
        row[-1] for row in sorted(eligible, key=lambda r: r[:3], reverse=True)[:limit]
    ], rejected


def _format_workflow_process_markdown(plan, current_node: str | None = None) -> str:
    """Format an empirical, graph-grounded state transition view for the agent."""
    if not plan:
        return ""

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    goal = _get(plan, "goal") or "operational objective"
    lines = [
        "### 🗺️ Empirical World Model — State Graph & Historical Transitions",
        f"**Current State**: `{current_node or 'initial_state'}`",
        f"**Active Objective**: `{goal}`",
    ]
    unresolved_refs = _get(plan, "unresolved_references") or []
    if unresolved_refs:
        lines.append("\n⚠️ **Active Evidence References & Disambiguation Constraints**:")
        for ref in unresolved_refs[:4]:
            adv = _get(ref, "advice") or _get(ref, "target")
            lines.append(f"• {adv}")
    tried_list = _get(plan, "tried_from_this_state") or []
    if tried_list:
        lines.append("\n**Past Action Attempts & Observed Consequences from this State**:")
        for t in tried_list[:4]:
            outcome = _get(t, "outcome")
            icon = "✅" if outcome == "success" else "❌"
            action = _get(t, "action")
            count = _get(t, "count") or _get(t, "observed_count") or 1
            action_desc = f"{icon} Action: `{action}` (observed {count}x)"
            lines.append(f"• {action_desc}")
            result = _get(t, "result")
            lines.append(f"  - Outcome: **{outcome}** — {result}")
            lesson = _get(t, "lesson")
            if lesson:
                lines.append(f"  - Lesson / Error: {lesson}")
            next_state = _get(t, "next_state")
            if next_state:
                lines.append(f"  - Reached Next State: `[{next_state}]`")
            downstream = _get(t, "downstream_consequences")
            if downstream:
                lines.append(f"  - Downstream Possibilities from `[{next_state or 'next_state'}]`:")
                for dc in downstream[:3]:
                    dc_outcome = _get(dc, "outcome")
                    d_icon = "✅" if dc_outcome == "success" else "❌"
                    dc_action = _get(dc, "action")
                    dc_result = _get(dc, "result")
                    lines.append(f"    * {d_icon} Action `{dc_action}` -> **{dc_outcome}** ({dc_result})")
                    dc_lesson = _get(dc, "lesson")
                    if dc_lesson:
                        lines.append(f"      Feedback: {dc_lesson}")
    paths = _get(plan, "potential_solution_paths") or []
    if paths:
        rec_path = paths[0]
        rec_cert = _get(rec_path, "certainty", 0.0)
        rec_steps = _get(rec_path, "steps", [])
        lines.append(f"\n**Recommended Transition Path** (Bayesian Thompson Certainty: {rec_cert:.2f}):")
        lines.append(f"• `{' -> '.join(rec_steps)}`")
        if len(paths) > 1:
            alt = paths[1]
            alt_id = _get(alt, "path_id", "alt")
            alt_steps = _get(alt, "steps", [])
            lines.append(f"• Alternative ({alt_id}): `{' -> '.join(alt_steps)}`")
    return "\n".join(lines)


def render_context(
    working: WorkingGraph,
    procedures: list,
    *,
    max_chars: int,
    store=None,
    query: str = "",
    current_node: str | None = None,
    completed_nodes: set[str] | None = None,
    successful_transition_graph: bool = False,
    entity_tracker=None,
    plan=None,
    unresolved_references: list[dict[str, str]] | None = None,
) -> tuple[str, dict]:
    """Count the entire injected message, including metadata and artifact menus.

    A dependency bundle is either included intact or omitted. Detailed reasons
    remain in the run log even when the context budget cannot carry them.
    """
    if max_chars < 1:
        raise ValueError("Context budget must be positive")

    from .process_planner import extract_workflow_plan

    if plan is None and store is not None and query:
        try:
            plan = extract_workflow_plan(
                store,
                working.scope,
                query,
                current_node=current_node,
                completed_nodes=completed_nodes,
                successful_transition_graph=successful_transition_graph,
                unresolved_references=unresolved_references,
            )
        except Exception:
            plan = None

    source = {c.claim_id: c for c in working.claims}
    selected = {}
    menu = []

    def encode(compact_wf: int = 0) -> str:
        claims = []
        for claim in selected.values():
            record = claim.model_dump(
                mode="json", exclude={"observed_at", "last_verified_at", "invalidation_events"}
            )
            record["support_count"] = len(claim.supporting_episodes)
            record["contradiction_count"] = len(claim.contradicting_episodes)
            record["supporting_episodes"] = list(claim.supporting_episodes[-3:])
            record["contradicting_episodes"] = list(claim.contradicting_episodes[-3:])
            # Tests/citations remain in the store. Decision context retains operational
            # content, applicability and pins, not repeated review histories.
            record["payload"] = {
                k: v
                for k, v in record["payload"].items()
                if k
                not in {
                    "citations",
                    "semantic_review",
                    "prospective_tests",
                    "alternatives",
                    "distinguishing_observation",
                    "interpretation",
                }
                and v is not None
            }
            for empty_key in (
                "conditions",
                "dependencies",
                "contradicting_episodes",
                "valid_until",
            ):
                if not record.get(empty_key):
                    record.pop(empty_key, None)
            claims.append(record)
        actionable_gaps = [g for g in working.gaps if not g.startswith("budget:")]
        contrasts = [
            {"claim_id": c["claim_id"], "statement": c["statement"],
             "interpretation": "Unverified observational contrast, not a mandatory precondition"}
            for c in claims if c.get("relation") in ("contrastive_precondition", "contrastive_observation")
        ]
        payload = {
            "claims": claims,
            "available_procedures": menu,
            "unresolved_gap_count": len(actionable_gaps),
            "unresolved_gaps": list(actionable_gaps[:16]),
        }
        if plan:
            wf = {
                "goal": plan.goal,
                "confidence": plan.confidence,
                "active_step_index": plan.active_step_index,
                "path_updated": plan.path_updated,
                "steps": [
                    {
                        "step": s.step,
                        "operation": s.operation,
                        "status": s.status,
                        "description": s.description,
                        **({"method": s.method} if s.method else {}),
                        **({"url": s.url} if s.url else {}),
                        **({"input_flow": s.input_flow} if s.input_flow else {}),
                    }
                    for s in plan.steps
                ],
            }
            if plan.tried_from_this_state and compact_wf < 3:
                max_tried = 2 if compact_wf >= 2 else (4 if compact_wf == 1 else len(plan.tried_from_this_state))
                max_str_len = 100 if compact_wf >= 1 else 300
                tried_entries = []
                for t in plan.tried_from_this_state[:max_tried]:
                    res = t.result[:max_str_len] if t.result else ""
                    les = t.lesson[:max_str_len] if t.lesson else ""
                    entry = {
                        "action": t.action,
                        "outcome": t.outcome,
                        "result": res,
                        "lesson": les,
                        "observed_count": t.count,
                        **({"next_state": t.next_state} if t.next_state else {}),
                    }
                    if compact_wf < 2 and getattr(t, "downstream_consequences", None):
                        max_dc = 2 if compact_wf == 1 else len(t.downstream_consequences)
                        entry["downstream_consequences"] = [
                            {
                                "action": dc.action,
                                "outcome": dc.outcome,
                                "result": dc.result[:max_str_len] if dc.result else "",
                                "lesson": dc.lesson[:max_str_len] if dc.lesson else "",
                            }
                            for dc in t.downstream_consequences[:max_dc]
                        ]
                    tried_entries.append(entry)
                wf["tried_from_this_state"] = tried_entries

            if plan.potential_solution_paths and compact_wf < 3:
                max_paths = 1 if compact_wf >= 2 else (2 if compact_wf == 1 else len(plan.potential_solution_paths))
                wf["potential_solution_paths"] = [
                    {
                        "path": p.path_id,
                        "steps": p.steps,
                        "confidence": p.confidence,
                        "certainty": p.certainty,
                        "posterior_variance": p.posterior_variance,
                        "turns": p.estimated_steps,
                        "strategy": p.strategy_description,
                    }
                    for p in plan.potential_solution_paths[:max_paths]
                ]
                rec_path = plan.potential_solution_paths[0]
                wf["exploration_guidance"] = (
                    f"Candidate paths sampled via Bayesian posterior distribution (Thompson Sampling). "
                    f"Recommended path certainty: {rec_path.certainty:.2f}. "
                    f"Certainty increases continuously as paths are used more with success."
                )
            if getattr(plan, "unresolved_references", None) and compact_wf < 2:
                wf["unresolved_references"] = plan.unresolved_references
            payload["workflow_process"] = wf
        if entity_tracker is not None and hasattr(entity_tracker, "targets"):
            payload["task_state"] = entity_tracker.summary()
        elif entity_tracker is not None and getattr(entity_tracker, "entities", None):
            payload["entity_obligations"] = entity_tracker.summary()
        if contrasts:
            payload["unverified_contrasts"] = contrasts
        if working.stages:
            payload["workflow_stages"] = list(working.stages)
        if working.reasons:
            payload["retrieval_reasons"] = {cid: list(r) for cid, r in working.reasons.items()}
        return PREFIX + json.dumps(payload, separators=(",", ":"))

    compact_level = 0
    while compact_level < 3 and len(encode(compact_level)) > max_chars:
        compact_level += 1

    for procedure in procedures:
        entry = {
            "procedure_id": procedure.procedure_id,
            "status": procedure.status,
            "goal_pattern": procedure.goal_pattern,
            "trigger": procedure.trigger,
            "expected_runtime_effect": procedure.expected_runtime_effect,
            "stop_condition": procedure.stop_condition,
            "parameter_types": procedure.parameter_types,
            "prerequisites": procedure.prerequisites,
        }
        menu.append(entry)
        # Don't let artifact advertising consume the entire knowledge budget.
        if len(encode(compact_level)) > max_chars // 2:
            menu.pop()

    claim_order = list(working.claims)
    if plan and plan.required_claim_ids:
        plan_claims = []
        for cid in plan.required_claim_ids:
            if cid in source:
                plan_claims.append(source[cid])
            elif store is not None:
                try:
                    c = store.claim(cid)
                    source[c.claim_id] = c
                    plan_claims.append(c)
                except Exception:
                    pass
        remaining = [c for c in claim_order if c.claim_id not in {p.claim_id for p in plan_claims}]
        claim_order = plan_claims + remaining

    for claim in claim_order:
        before = dict(selected)
        pending = [claim]
        valid = True
        while pending:
            current = pending.pop()
            if current.claim_id in selected:
                continue
            selected[current.claim_id] = current
            for dep in current.dependencies:
                if dep.mode == "monitors":
                    continue
                other = source.get(dep.claim_id)
                compatible = (
                    store.dependency_compatible(dep)
                    if store is not None
                    else other is not None and other.version == dep.version
                )
                if other is None or not compatible:
                    valid = False
                    break
                pending.append(other)
        if not valid or len(encode(compact_level)) > max_chars:
            selected = before
    content = encode(compact_level)
    while len(content) > max_chars and compact_level < 3:
        compact_level += 1
        content = encode(compact_level)
    if len(content) > max_chars:
        content = ""  # No partial instructions, claims, or prerequisites.
    return content, {
        "injected_chars": len(content),
        "injected_claims": list(selected) if content else [],
        "artifact_candidates": [p["procedure_id"] for p in menu] if content else [],
        "omitted_claim_count": len(source) - (len(selected) if content else 0),
        "workflow_plan_steps": len(plan.steps) if plan and content else 0,
        "workflow_plan_active_step": plan.active_step_index if plan and content else 0,
        "workflow_plan_updated": plan.path_updated if plan and content else False,
        "workflow_plan_goal": plan.goal if plan and content else None,
        "workflow_plan": plan.model_dump(mode="json") if plan and content else None,
    }


def render_episodes(store, scope, query: str, *, max_chars: int) -> tuple[str, dict]:
    """Competitive lexical episode retrieval with identical whole-message budget."""
    prefix = (
        "Historical development episodes, NOT current task state. IDs and values belong "
        "to their recorded task/tenant; re-query current records. No general rule is established "
        "by one example.\n"
    )
    tokens = terms(query) - STOPWORDS
    ranked = []
    for episode in store.historical_episodes(scope):
        encoded = episode.model_dump_json()
        overlap = len(tokens & terms(encoded))
        if overlap:
            ranked.append((overlap, episode.timestamp, episode.episode_id, episode))
    chosen = []
    for _, _, _, episode in sorted(ranked, key=lambda row: row[:3], reverse=True):
        record = episode.model_dump(mode="json")
        candidate = prefix + json.dumps(
            {"historical_episodes": [*chosen, record]}, separators=(",", ":")
        )
        if len(candidate) <= max_chars:
            chosen.append(record)
    content = prefix + json.dumps({"historical_episodes": chosen}, separators=(",", ":"))
    if len(content) > max_chars:
        content = ""
    return content, {
        "injected_chars": len(content),
        "injected_episodes": [e["episode_id"] for e in chosen] if content else [],
    }
