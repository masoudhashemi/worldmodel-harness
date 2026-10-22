"""Graph-theoretic process planner for multi-step workflow synthesis.

Extracts operational execution paths from the empirical trajectory graph based purely
on natural language task instructions. Strictly decoupled from benchmark metadata.
"""

from __future__ import annotations

import re
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


class WorkflowProcess(BaseModel):
    """Declarative workflow plan providing step-by-step operational guidance."""
    goal: str
    steps: list[WorkflowStep]
    estimated_steps: int
    confidence: float = 1.0
    required_claim_ids: list[str] = Field(default_factory=list)
    active_step_index: int = 1
    path_updated: bool = False


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
) -> WorkflowProcess | None:
    """Extract an operational workflow process plan covering task intent from the trajectory graph."""
    if not query or not query.strip():
        return None

    graph = trajectory_graph or compile_trajectory_graph(store, scope)
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
            # Target mutations must belong to an active service if any were identified
            if active_services and node.service not in active_services:
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

    # Group target candidates by service to cover distinct required domain mutations
    seen_target_services: set[str] = set()
    best_targets: list[str] = []
    for _, nid, node in filtered_targets:
        if node.service not in seen_target_services:
            seen_target_services.add(node.service)
            best_targets.append(nid)
            if len(best_targets) >= 3:
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
        top_starts = [best_targets[0]]

    # 3. Find optimal path covering the start and target nodes
    path_updated = False
    covering_result = None

    if current_node and current_node in graph.nodes:
        remaining_targets = [t for t in best_targets if t not in completed and t != current_node]
        if remaining_targets:
            covering_result = graph.find_path_covering(
                [current_node], remaining_targets, allowed_services=active_services or None
            )
            path_updated = True
        else:
            covering_result = ([current_node], 0.0)

    if not covering_result:
        covering_result = graph.find_path_covering(
            top_starts, best_targets, allowed_services=active_services or None
        )

    if not covering_result:
        # Fallback: check if the top start node can reach any target within allowed services
        for s in top_starts:
            for t in [nid for _, nid, _ in target_candidates[:3]]:
                res = graph.shortest_path(s, t, allowed_services=active_services or None)
                if res:
                    covering_result = res
                    break
            if covering_result:
                break

    if not covering_result:
        return None

    path_nodes, cost = covering_result
    if not path_nodes:
        return None

    # 4. Construct sequential workflow steps
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

    # 5. Append final verification / finish step
    finish_step = WorkflowStep(
        step=len(steps) + 1,
        operation="finish",
        description="Verify all mutations and complete task with exact counts/values.",
    )
    steps.append(finish_step)

    # 6. Resolve step execution status: completed, active, pending
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
    goal = f"Execute multi-step workflow: {' -> '.join(s.operation for s in steps[:-1])}"

    return WorkflowProcess(
        goal=goal,
        steps=steps,
        estimated_steps=len(steps),
        confidence=round(confidence, 2),
        required_claim_ids=required_claims,
        active_step_index=active_step_index,
        path_updated=path_updated,
    )
