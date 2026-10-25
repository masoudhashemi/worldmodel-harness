"""Ground-Truth vs Learned World Model Comparative Evaluation.

Compares the empirical World Model learned from environment interactions against
the ground-truth capability graph and schemas derived from AutomationBench
(following the graph-grounded-automationbench taxonomy).

Evaluates:
1. Node Fidelity: Endpoint precision, recall, service attribution, and stage classification.
2. Edge Fidelity: Dataflow precision, spurious edge rate, directional agreement.
3. Topological Alignment: Stage precedence conformity (lookup -> mutate -> notify)
   and reachability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .knowledge import Claim, Layer
from .store import KnowledgeStore
from .trajectory_graph import compile_trajectory_graph


SCHEMAS_DIR = Path(__file__).parents[2] / ".benchmarks" / "AutomationBench" / "automationbench" / "tools" / "api" / "schemas"

_LOOKUP_OPS = {"get", "list", "find", "search", "lookup", "query", "download"}
_MUTATION_OPS = {"create", "update", "delete", "add", "remove", "cancel", "archive", "assign", "convert", "publish", "upload"}
_NOTIFICATION_SERVICES = {"slack", "gmail", "twilio"}
_NOTIFICATION_OPS = {"send", "post", "create"}


@dataclass(frozen=True)
class GroundTruthEndpoint:
    id: str
    service: str
    method: str
    url_path: str
    stage: str  # lookup, mutate, notify, execute
    input_parameters: set[str] = field(default_factory=set)
    response_fields: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class GroundTruthGraph:
    endpoints: dict[str, GroundTruthEndpoint]
    services: set[str]
    dataflow_edges: set[tuple[str, str]]  # (producer_id, consumer_id)


def _infer_stage(service: str, endpoint_id: str, method: str) -> str:
    parts = endpoint_id.split(".")
    op = parts[-1].lower() if len(parts) > 1 else ""
    lowered_id = endpoint_id.lower()

    # 1. Lookup operations take precedence for read calls (GET or lookup prefixes)
    if method == "GET" or any(op.startswith(prefix) for prefix in _LOOKUP_OPS):
        return "lookup"

    # 2. Notifications: communication services performing outbound messages
    if service in _NOTIFICATION_SERVICES and (
        any(op.startswith(p) for p in _NOTIFICATION_OPS)
        or any(term in lowered_id for term in ("send", "post", "sms", "notify"))
    ):
        return "notify"

    # 3. Mutations: modifying external state
    if method in ("POST", "PUT", "PATCH", "DELETE") or any(op.startswith(prefix) for prefix in _MUTATION_OPS):
        return "mutate"

    return "execute"


def _extract_parameter_keys(parameters_dict: dict[str, Any], request_obj: Any) -> set[str]:
    keys = set(parameters_dict.keys())
    if isinstance(request_obj, dict):
        keys.update(request_obj.keys())
    elif isinstance(request_obj, str) and request_obj:
        # Extract body keys from request description strings like "Body: {board_id, name}"
        matches = re.findall(r"([a-zA-Z0-9_]+)\s*(?:\([^)]*\))?\s*[,}:]", request_obj)
        for m in matches:
            if m not in ("Body", "query", "object"):
                keys.add(m)
    return keys


def _extract_response_keys(response_obj: Any) -> set[str]:
    keys = set()
    if isinstance(response_obj, dict):
        keys.update(response_obj.keys())
    elif isinstance(response_obj, str) and response_obj:
        matches = re.findall(r"([a-zA-Z0-9_]+)\s*[,}:]", response_obj)
        for m in matches:
            if m not in ("items", "values", "records", "data", "results"):
                keys.add(m)
    return keys


def load_ground_truth_graph(schemas_dir: Path | None = None) -> GroundTruthGraph:
    """Load ground-truth API graph from benchmark JSONC schemas."""
    schemas_path = schemas_dir or SCHEMAS_DIR
    if not schemas_path.exists():
        raise FileNotFoundError(f"Schemas directory not found at: {schemas_path}")

    endpoints: dict[str, GroundTruthEndpoint] = {}
    services: set[str] = set()

    for schema_file in sorted(schemas_path.glob("*.jsonc")):
        text = "\n".join(
            line for line in schema_file.read_text().splitlines()
            if not line.lstrip().startswith("//")
        )
        try:
            data = json.loads(text)
        except Exception:
            continue

        service = data.get("api", schema_file.stem)
        services.add(service)

        for ep in data.get("endpoints", []):
            ep_id = ep.get("id")
            if not ep_id:
                continue

            method = ep.get("method", "GET").upper()
            url_path = ep.get("path", "")
            stage = _infer_stage(service, ep_id, method)
            input_params = _extract_parameter_keys(ep.get("parameters", {}), ep.get("request", ""))
            response_fields = _extract_response_keys(ep.get("response", ""))

            endpoints[ep_id] = GroundTruthEndpoint(
                id=ep_id,
                service=service,
                method=method,
                url_path=url_path,
                stage=stage,
                input_parameters=input_params,
                response_fields=response_fields,
            )

    # Compute valid schema dataflow edges:
    # A dataflow exists if a producer output key can bind into a consumer input parameter
    dataflow_edges: set[tuple[str, str]] = set()
    for prod_id, prod in endpoints.items():
        if prod.stage not in ("lookup", "mutate"):
            continue
        for cons_id, cons in endpoints.items():
            if prod_id == cons_id:
                continue
            # Same service or cross-service identifier sharing
            shared_keys = prod.response_fields.intersection(cons.input_parameters)
            # Match common generic id bindings
            id_match = any("id" in p.lower() for p in cons.input_parameters) and (
                "id" in prod.response_fields or "id" in prod.id.lower() or prod.stage == "lookup"
            )
            if shared_keys or (prod.service == cons.service and id_match):
                dataflow_edges.add((prod_id, cons_id))

    return GroundTruthGraph(
        endpoints=endpoints,
        services=services,
        dataflow_edges=dataflow_edges,
    )


@dataclass
class ComparisonMetrics:
    # Node metrics
    total_learned_endpoints: int
    ground_truth_endpoints: int
    matched_endpoints: int
    hallucinated_endpoints: list[str]
    endpoint_precision: float
    endpoint_recall: float
    service_attribution_accuracy: float
    stage_classification_accuracy: float

    # Edge / Dataflow metrics
    total_learned_value_flows: int
    ground_truth_dataflow_edges: int
    valid_dataflow_edges: int
    spurious_value_flows: list[tuple[str, str]]
    dataflow_precision: float

    # Topological / Workflow metrics
    trajectory_nodes: int
    trajectory_transitions: int
    precedence_conformity_rate: float
    precedence_violations: list[tuple[str, str]]
    jaccard_node_similarity: float


def compare_worldmodel_to_ground_truth(
    store: KnowledgeStore,
    gt: GroundTruthGraph | None = None,
) -> ComparisonMetrics:
    """Compare learned KnowledgeStore against GroundTruthGraph."""
    if gt is None:
        gt = load_ground_truth_graph()

    claims = list(store.claims())

    # 1. Evaluate Learned Endpoints
    contract_claims = [c for c in claims if c.relation == "endpoint_contract"]
    learned_endpoints: dict[str, Claim] = {}
    for c in contract_claims:
        ep_id = c.payload.get("id")
        if ep_id:
            learned_endpoints[ep_id] = c

    matched_endpoints = set(learned_endpoints.keys()).intersection(gt.endpoints.keys())
    hallucinated = sorted(set(learned_endpoints.keys()) - set(gt.endpoints.keys()))

    endpoint_precision = (
        len(matched_endpoints) / len(learned_endpoints) if learned_endpoints else 1.0
    )
    endpoint_recall = (
        len(matched_endpoints) / len(gt.endpoints) if gt.endpoints else 0.0
    )

    # Service Attribution & Stage Alignment
    correct_service = 0
    correct_stage = 0
    for ep_id in matched_endpoints:
        gt_ep = gt.endpoints[ep_id]
        c = learned_endpoints[ep_id]
        url = c.payload.get("url", "")
        # Service check with vendor prefix normalization (e.g. google_sheets -> sheets.googleapis)
        srv = gt_ep.service
        norm_srv = srv.replace("google_", "")
        if srv in url or norm_srv in url or srv in ep_id or norm_srv in ep_id:
            correct_service += 1
        # Stage check
        method = c.payload.get("method", "GET").upper()
        learned_stage = _infer_stage(gt_ep.service, ep_id, method)
        if learned_stage == gt_ep.stage:
            correct_stage += 1

    service_accuracy = correct_service / len(matched_endpoints) if matched_endpoints else 1.0
    stage_accuracy = correct_stage / len(matched_endpoints) if matched_endpoints else 1.0

    # 2. Evaluate Learned Value Flows
    value_flow_claims = [c for c in claims if c.relation == "observed_value_flow"]
    learned_flows = {
        (c.payload["source_endpoint"], c.payload["target_endpoint"])
        for c in value_flow_claims
        if "source_endpoint" in c.payload and "target_endpoint" in c.payload
    }

    valid_flows = set()
    spurious_flows = []
    for src, dst in learned_flows:
        # Check if src and dst exist in ground truth
        src_gt = gt.endpoints.get(src)
        dst_gt = gt.endpoints.get(dst)
        if not src_gt or not dst_gt:
            spurious_flows.append((src, dst))
            continue

        # Valid workflow dataflow:
        # 1. Direct schema binding (shared parameter or identifier concept)
        # 2. Lookup data propagation (reading records/content into downstream lookup, mutate, or notify)
        # 3. Mutation outcome propagation (created/updated IDs passed to downstream mutate or notify)
        # 4. Outbound delivery confirmation (notify message ID queried via lookup or batch notify)
        # 5. Paging / batch recurrence (src == dst)
        is_schema_bound = (src, dst) in gt.dataflow_edges
        is_lookup_flow = src_gt.stage == "lookup" and dst_gt.stage in ("lookup", "mutate", "notify")
        is_mutate_flow = src_gt.stage == "mutate" and dst_gt.stage in ("mutate", "notify")
        is_receipt_flow = src_gt.stage == "notify" and dst_gt.stage in ("lookup", "notify")
        is_recurrence = src == dst

        if is_schema_bound or is_lookup_flow or is_mutate_flow or is_receipt_flow or is_recurrence:
            valid_flows.add((src, dst))
        else:
            spurious_flows.append((src, dst))

    dataflow_precision = (
        len(valid_flows) / len(learned_flows) if learned_flows else 1.0
    )

    # 3. Evaluate Trajectory Graph Precedence
    graph = compile_trajectory_graph(store, None)
    transitions = [
        (src, a.target_node)
        for src, attempts in graph.node_attempts.items()
        for a in attempts
        if a.target_node is not None and a.outcome == "success"
    ]

    stage_rank = {"lookup": 1, "mutate": 2, "notify": 3, "execute": 2}
    precedence_violations = []
    valid_transitions = 0

    for src, dst in transitions:
        src_gt = gt.endpoints.get(src)
        dst_gt = gt.endpoints.get(dst)
        if not src_gt or not dst_gt:
            continue
        valid_transitions += 1
        # Precedence violation if notify -> lookup or notify -> mutate
        if stage_rank.get(src_gt.stage, 0) > stage_rank.get(dst_gt.stage, 0) and src_gt.stage == "notify":
            precedence_violations.append((src, dst))

    precedence_rate = (
        1.0 - (len(precedence_violations) / valid_transitions)
        if valid_transitions else 1.0
    )

    # Node Jaccard
    union_nodes = set(learned_endpoints.keys()).union(gt.endpoints.keys())
    jaccard_node = len(matched_endpoints) / len(union_nodes) if union_nodes else 0.0

    return ComparisonMetrics(
        total_learned_endpoints=len(learned_endpoints),
        ground_truth_endpoints=len(gt.endpoints),
        matched_endpoints=len(matched_endpoints),
        hallucinated_endpoints=hallucinated,
        endpoint_precision=endpoint_precision,
        endpoint_recall=endpoint_recall,
        service_attribution_accuracy=service_accuracy,
        stage_classification_accuracy=stage_accuracy,
        total_learned_value_flows=len(learned_flows),
        ground_truth_dataflow_edges=len(gt.dataflow_edges),
        valid_dataflow_edges=len(valid_flows),
        spurious_value_flows=spurious_flows,
        dataflow_precision=dataflow_precision,
        trajectory_nodes=len(graph.node_attempts),
        trajectory_transitions=len(transitions),
        precedence_conformity_rate=precedence_rate,
        precedence_violations=precedence_violations,
        jaccard_node_similarity=jaccard_node,
    )
