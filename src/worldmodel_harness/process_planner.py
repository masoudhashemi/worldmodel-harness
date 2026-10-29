"""Graph-theoretic process planner for multi-step workflow synthesis.

Extracts operational execution paths from the empirical trajectory graph based purely
on natural language task instructions. Strictly decoupled from benchmark metadata.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from .knowledge import Scope
from .retrieval import terms
from .store import KnowledgeStore
from .trajectory_graph import (
    GRAPH_STOPWORDS,
    SERVICE_SYNONYMS,
    OperationNode,
    TrajectoryGraph,
    compile_trajectory_graph,
)

MUTATION_VERBS = {
    "create", "update", "delete", "post", "put", "patch", "remove", "archive",
    "clean", "suppress", "alert", "notify", "send", "mutate", "assign", "resolve",
    "add", "close", "cancel", "submit", "publish", "blast", "modify"
}

OBSERVATION_VERBS = {
    "read", "get", "find", "list", "search", "check", "inspect", "view",
    "report", "lookup", "fetch", "query", "scan", "extract", "retrieve"
}


class WorkflowStep(BaseModel):
    """An individual sequential operational step in a workflow process plan."""
    step: int
    operation: str
    description: str
    method: str = ""
    url: str = ""
    claim_id: str | None = None
    input_flow: str | None = None
    expected_data: str | None = None
    status: str = "pending"  # "completed", "active", "pending"


class DownstreamConsequence(BaseModel):
    """Downstream consequence or error from reaching this state in previous trials."""
    action: str
    outcome: str
    result: str
    lesson: str
    next_state: str | None = None


class TriedActionSummary(BaseModel):
    """Summary of an action previously attempted from this state and its outcome."""
    action: str
    outcome: str  # "error", "failure", "success", "blocked"
    result: str
    lesson: str
    count: int = 1
    next_state: str | None = None
    downstream_consequences: list[DownstreamConsequence] = Field(default_factory=list)


class SolutionPath(BaseModel):
    """A candidate solution path covering the goal from the current state."""
    path_id: str
    steps: list[str]
    confidence: float
    certainty: float = 0.0
    posterior_variance: float = 0.0
    sample_score: float = 0.0
    support_count: int = 1
    estimated_steps: int
    strategy_description: str
    is_recommended: bool = False


class WorkflowProcess(BaseModel):
    """Declarative workflow plan providing step-by-step operational guidance."""
    goal: str
    steps: list[WorkflowStep]
    estimated_steps: int
    confidence: float = 1.0
    required_claim_ids: list[str] = Field(default_factory=list)
    active_step_index: int = 1
    path_updated: bool = False
    current_node: str | None = None
    tried_from_this_state: list[TriedActionSummary] = Field(default_factory=list)
    potential_solution_paths: list[SolutionPath] = Field(default_factory=list)
    unresolved_references: list[dict[str, str]] = Field(default_factory=list)


def _stage_rank(node: OperationNode) -> int:
    """Ontological stage rank: lookup (1) -> mutate (2) -> notify (3)."""
    if node.action_type == "read" or node.method == "GET":
        return 1
    if node.service in ("gmail", "slack", "twilio") and any(
        term in (node.endpoint_id or "").lower() for term in ("send", "post", "sms", "notify")
    ):
        return 3
    return 2


def _generate_step_description(node: OperationNode, next_node: OperationNode | None = None) -> str:
    """Generate a concise, human-readable operational instruction for a step."""
    if node.node_id == "base64_encode":
        return "Encode notification or payload into URL-safe base64 format."
    if node.node_id == "finish":
        return "Verify all mutations succeeded and return the exact required counts/verbatim values."

    action = "Fetch" if node.action_type == "read" else "Execute"
    if node.method == "GET":
        action = "Read"
    elif node.method == "POST":
        action = "Create / Post"
    elif node.method in ("PUT", "PATCH"):
        action = "Update"
    elif node.method == "DELETE":
        action = "Delete / Remove"

    service_name = node.service.replace("_", " ").title()
    resource_name = node.resource.replace("_", " ")

    if node.statement:
        # Use first sentence of verified claim statement if available
        first_sentence = node.statement.split(". ")[0].strip()
        if len(first_sentence) < 120 and "Observed" not in first_sentence:
            return f"{first_sentence}."

    return f"{action} {resource_name} via {service_name}."


def extract_workflow_plan(
    store: KnowledgeStore,
    scope: Scope,
    query: str,
    trajectory_graph: TrajectoryGraph | None = None,
    current_node: str | None = None,
    completed_nodes: set[str] | None = None,
    successful_transition_graph: bool = True,
    unresolved_references: list[dict[str, str]] | None = None,
) -> WorkflowProcess | None:
    """Extract an operational workflow process plan covering task intent from the trajectory graph."""
    if not query or not query.strip():
        return None

    graph = trajectory_graph or compile_trajectory_graph(
        store, scope, successful_transitions_only=successful_transition_graph
    )
    if not graph.nodes or not graph.edges:
        return None

    completed = set(completed_nodes or ())
    if current_node and current_node in graph.nodes:
        completed.add(current_node)

    # 1. Parse and expand query terms
    raw_terms = terms(query)
    q_terms = raw_terms - GRAPH_STOPWORDS

    # Also detect spreadsheet/worksheet prefixes (e.g. ss_bounce -> sheets, ws_bounces -> sheets)
    expanded_terms = set(q_terms)
    active_services: set[str] = set()
    for w in raw_terms:
        if w.startswith("ss_") or w.startswith("ws_"):
            expanded_terms.add("sheets")
            active_services.add("sheets")
        for k, v in SERVICE_SYNONYMS.items():
            if w == k or (len(k) >= 4 and k in w):
                expanded_terms.add(v)
                active_services.add(v)

    # Enterprise channel handles (e.g. #ops-updates, #security, #alerts) represent chat/slack notifications
    if re.search(r"#\w+", query):
        expanded_terms.update({"slack", "channel", "message", "chat", "notify", "post"})
        active_services.add("slack")

    # 2. Score candidate start nodes (readers) and target nodes (mutators)
    start_candidates: list[tuple[int, str, OperationNode]] = []
    target_candidates: list[tuple[int, str, OperationNode]] = []

    for nid, node in graph.nodes.items():
        if nid in ("base64_encode", "finish"):
            continue

        service_match = 3 if node.service in expanded_terms else 0
        overlap = len(expanded_terms & node.terms)
        score = service_match * 4 + overlap
        if score < 2:
            continue

        # Prioritize services that were explicitly matched
        if node.service in expanded_terms:
            score += 5

        if node.action_type == "read":
            start_candidates.append((score, nid, node))
        else:
            # Prefer active services if any were identified, but do not hard-exclude high-overlap mutations
            if active_services and node.service not in active_services and score < 4:
                continue
            target_candidates.append((score, nid, node))

    start_candidates.sort(reverse=True, key=lambda x: x[0])
    target_candidates.sort(reverse=True, key=lambda x: x[0])

    if not target_candidates and not start_candidates:
        return None

    # Keep only high-scoring targets (within 60% of top score and score >= 4)
    top_target_score = target_candidates[0][0] if target_candidates else 0
    min_target_score = max(4, int(top_target_score * 0.6))
    filtered_targets = [item for item in target_candidates if item[0] >= min_target_score]

    # Order targets by ontological lifecycle stage (mutations before notifications), then by score
    filtered_targets.sort(key=lambda item: (_stage_rank(item[2]), -item[0]))

    # Select best targets (allowing composite intra-service mutations!)
    # Allow up to 2 distinct operations per service, preserving sequential multi-step mutations
    service_target_counts: dict[str, int] = defaultdict(int)
    best_targets: list[str] = []
    for _, nid, node in filtered_targets:
        if service_target_counts[node.service] < 2:
            service_target_counts[node.service] += 1
            best_targets.append(nid)
            if len(best_targets) >= 6:
                break

    # Group start candidates by service
    top_start_score = start_candidates[0][0] if start_candidates else 0
    min_start_score = max(3, int(top_start_score * 0.5))
    filtered_starts = [item for item in start_candidates if item[0] >= min_start_score]

    seen_start_services: set[str] = set()
    top_starts: list[str] = []
    for _, nid, node in filtered_starts:
        if node.service not in seen_start_services:
            seen_start_services.add(node.service)
            top_starts.append(nid)

    if not top_starts and best_targets:
        min_stage = min(_stage_rank(graph.nodes[t]) for t in best_targets)
        top_starts = [t for t in best_targets if _stage_rank(graph.nodes[t]) == min_stage][:2]

    # Identify intermediate investigation waypoints from unresolved references (e.g. check email, check policy sheet)
    investigation_targets: list[str] = []
    if unresolved_references:
        for ref in unresolved_references:
            adv = ref.get("advice", "").lower()
            cat = ref.get("category", "")
            target_str = ref.get("target", "").lower()
            if any(k in adv or k in target_str for k in ("email", "gmail", "inbox", "monitoring")):
                for nid, n in graph.nodes.items():
                    if n.service == "gmail" and n.action_type == "read" and nid not in completed:
                        if nid not in investigation_targets:
                            investigation_targets.append(nid)
            if "policy" in adv or "policy" in cat:
                for nid, n in graph.nodes.items():
                    if n.service == "sheets" and n.action_type == "read" and nid not in completed:
                        if nid not in investigation_targets and nid != current_node:
                            investigation_targets.append(nid)

    search_targets = [t for t in investigation_targets if t not in completed] + [t for t in best_targets if t not in completed]
    if not search_targets:
        search_targets = best_targets

    # 3. Find candidate solution paths covering the start and target nodes
    path_updated = False
    candidate_paths: list[tuple[list[str], float, float, int]] = []

    auth_mutations = set(best_targets)
    if current_node and current_node in graph.nodes:
        remaining_targets = [t for t in search_targets if t != current_node]
        if remaining_targets:
            candidate_paths = graph.find_all_candidate_paths(
                [current_node], remaining_targets, authorized_mutations=auth_mutations, max_paths=3
            )
            path_updated = True
        else:
            candidate_paths = [([current_node], 0.0, 1.0, 1)]

    if not candidate_paths:
        candidate_paths = graph.find_all_candidate_paths(
            top_starts, search_targets, authorized_mutations=auth_mutations, max_paths=3
        )

    if not candidate_paths:
        # Fallback: check if any top start node can reach any valid target
        import math
        for s in top_starts:
            for _, t, _ in target_candidates:
                res = graph.shortest_path(s, t, authorized_mutations=auth_mutations)
                if res:
                    seg_path, seg_cost = res
                    conf = max(0.05, min(1.0, math.exp(-seg_cost)))
                    candidate_paths = [(seg_path, seg_cost, conf, 1)]
                    break
            if candidate_paths:
                break

    if not candidate_paths:
        return None

    # Construct candidate solution paths for multi-path exploration
    # Rank paths primarily by authorized target coverage (to prevent multi-objective truncation)
    # and secondarily via Bayesian posterior draws (Thompson Sampling)
    ranked_candidates = list(candidate_paths)
    if len(ranked_candidates) > 1:
        def path_ranking_key(cand):
            p_nodes = cand[0]
            covered_targets = sum(1 for t in best_targets if t in p_nodes)
            target_coverage = covered_targets / len(best_targets) if best_targets else 1.0
            score = getattr(cand, "sample_score", cand[2])
            return (target_coverage, score)

        ranked_candidates.sort(key=path_ranking_key, reverse=True)

    path_nodes, cost, _, _ = ranked_candidates[0][:4]
    if not path_nodes:
        return None

    potential_solution_paths: list[SolutionPath] = []
    for p_idx, cand in enumerate(ranked_candidates):
        p_nodes, p_cost, p_conf, p_supp = cand[:4]
        full_steps = list(p_nodes) + ["finish"]
        is_rec = (p_idx == 0)
        p_certainty = getattr(cand, "certainty", round(p_conf, 2))
        p_var = getattr(cand, "posterior_variance", 0.0)
        p_sample = getattr(cand, "sample_score", round(p_conf, 2))
        p_name = f"Path {chr(65 + p_idx)} (" + (
            f"Recommended - Sampled Score: {p_sample:.2f}, Certainty: {p_certainty:.2f}"
            if is_rec
            else f"Alternative Exploration Path - Sampled Score: {p_sample:.2f}, Certainty: {p_certainty:.2f}"
        ) + ")"
        strat = f"Execute: {' -> '.join(full_steps[:-1])} -> finish"
        potential_solution_paths.append(
            SolutionPath(
                path_id=p_name,
                steps=full_steps,
                confidence=round(p_conf, 2),
                certainty=p_certainty,
                posterior_variance=round(p_var, 4),
                sample_score=round(p_sample, 3),
                support_count=p_supp,
                estimated_steps=len(full_steps),
                strategy_description=strat,
                is_recommended=is_rec,
            )
        )

    # 4. Extract what has been tried from this state/node
    tried_summaries: list[TriedActionSummary] = []
    state_key = current_node or "root"
    attempts = graph.node_attempts.get(state_key, [])
    for att in attempts[:5]:
        act_label = att.action_type
        if att.method:
            act_label += f" {att.method}"
        if att.url:
            act_label += f" {att.url}"
        if att.parameters_summary:
            act_label += f" ({att.parameters_summary})"

        downstream: list[DownstreamConsequence] = []
        if att.target_node and att.outcome == "success":
            for d_att in graph.node_attempts.get(att.target_node, [])[:3]:
                d_label = d_att.action_type
                if d_att.method:
                    d_label += f" {d_att.method}"
                if d_att.url:
                    d_label += f" {d_att.url}"
                if d_att.parameters_summary:
                    d_label += f" ({d_att.parameters_summary})"
                downstream.append(
                    DownstreamConsequence(
                        action=d_label.strip(),
                        outcome=d_att.outcome,
                        result=d_att.result_summary,
                        lesson=d_att.lesson,
                        next_state=d_att.target_node,
                    )
                )

        tried_summaries.append(
            TriedActionSummary(
                action=act_label.strip(),
                outcome=att.outcome,
                result=att.result_summary,
                lesson=att.lesson,
                count=att.count,
                next_state=att.target_node if att.outcome == "success" else None,
                downstream_consequences=downstream,
            )
        )

    # 5. Construct sequential workflow steps for the recommended path
    steps: list[WorkflowStep] = []
    required_claims: list[str] = []

    for idx, nid in enumerate(path_nodes):
        node = graph.nodes.get(nid)
        if not node:
            continue

        if node.contract_claim_id and node.contract_claim_id not in required_claims:
            required_claims.append(node.contract_claim_id)

        next_node = graph.nodes.get(path_nodes[idx + 1]) if idx + 1 < len(path_nodes) else None
        desc = _generate_step_description(node, next_node)

        # Inspect edge value flows from previous step
        input_flow: str | None = None
        if idx > 0:
            prev_nid = path_nodes[idx - 1]
            edge = graph.edges.get(prev_nid, {}).get(nid)
            if edge and edge.value_flows:
                flow = edge.value_flows[0]
                input_flow = f"Reuse {flow.output_path} from prior step as {flow.input_path}"

        step = WorkflowStep(
            step=len(steps) + 1,
            operation=node.endpoint_id or node.node_id,
            description=desc,
            method=node.method,
            url=node.url_template,
            claim_id=node.contract_claim_id,
            input_flow=input_flow,
        )
        steps.append(step)

    # 6. Append final verification / finish step
    finish_step = WorkflowStep(
        step=len(steps) + 1,
        operation="finish",
        description="Verify all mutations and complete task with exact counts/values.",
    )
    steps.append(finish_step)

    # 7. Resolve step execution status: completed, active, pending
    active_step_index = 1
    found_active = False
    for s in steps:
        op = s.operation
        if op in completed or (current_node and op == current_node):
            s.status = "completed"
        elif not found_active:
            s.status = "active"
            active_step_index = s.step
            found_active = True
        else:
            s.status = "pending"

    if not found_active and steps:
        steps[-1].status = "active"
        active_step_index = steps[-1].step

    # Compute confidence based on path transition cost
    confidence = max(0.2, min(1.0, 1.0 / (1.0 + cost * 0.2)))

    # Synthesize concise goal statement
    if investigation_targets and any(t not in completed for t in investigation_targets):
        inv_str = ", ".join(investigation_targets)
        goal = f"Investigate referenced evidence ({inv_str}) before mutating actions"
    elif confidence >= 0.70:
        goal = f"Execute verified workflow: {' -> '.join(s.operation for s in steps[:-1])}"
    else:
        goal = f"Fulfill task requirements (Candidate lead: {' -> '.join(s.operation for s in steps[:-1])})"

    return WorkflowProcess(
        goal=goal,
        steps=steps,
        estimated_steps=len(steps),
        confidence=round(confidence, 2),
        required_claim_ids=required_claims,
        active_step_index=active_step_index,
        path_updated=path_updated,
        current_node=current_node,
        tried_from_this_state=tried_summaries,
        potential_solution_paths=potential_solution_paths,
        unresolved_references=unresolved_references or [],
    )
