"""Operational trajectory graph compiled from execution episodes and claims.

Connects canonical operations (API contracts and tools) via empirical transition
probabilities and observed value flows, enabling graph-theoretic workflow planning.
Strictly decoupled from benchmark task IDs and internal metadata.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .behavior import endpoint_contract, request_target
from .knowledge import Claim, Episode, Scope
from .retrieval import terms
from .store import KnowledgeStore

# Stopwords and URL/domain noise terms to filter out during intent matching
GRAPH_STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "in", "for", "with", "is", "it",
    "from", "that", "this", "be", "as", "at", "by", "on", "are", "was", "were",
    "we", "our", "you", "your", "they", "them", "their", "so", "if", "not", "but",
    "what", "let", "got", "up", "com", "org", "net", "io", "example", "api",
    "v1", "v2", "v3", "v4", "v19", "https", "http", "www", "company", "accordingly",
    "happened", "know", "make", "sure", "following", "current", "procedures",
    "before", "making", "changes", "when", "including", "values", "source", "data",
    "notifications", "records", "preserve", "verbatim", "dont", "paraphrase",
    "round", "include", "relevant", "counts", "message", "messages", "team",
    "just", "sent", "over", "info", "can", "project", "under", "section", "about",
    "details", "due", "date", "tag", "use", "actually", "some", "other", "matter",
    "policy", "rules", "should", "actioned", "once", "created"
}

# Domain service synonyms for semantic bridging
SERVICE_SYNONYMS = {
    "sheet": "sheets",
    "sheets": "sheets",
    "spreadsheet": "sheets",
    "spreadsheets": "sheets",
    "ss": "sheets",
    "ws": "sheets",
    "drive": "google_drive",
    "files": "google_drive",
    "file": "google_drive",
    "google_drive": "google_drive",
    "email": "gmail",
    "emails": "gmail",
    "mail": "gmail",
    "gmail": "gmail",
    "messages": "gmail",
    "message": "gmail",
    "mailchimp": "mailchimp",
    "subscriber": "mailchimp",
    "subscribers": "mailchimp",
    "bounce": "mailchimp",
    "bounces": "mailchimp",
    "asana": "asana",
    "task": "asana",
    "tasks": "asana",
    "slack": "slack",
    "channel": "slack",
    "chat": "slack",
    "salesforce": "salesforce",
    "lead": "salesforce",
    "leads": "salesforce",
    "contact": "salesforce",
    "contacts": "salesforce",
    "case": "salesforce",
    "cases": "salesforce",
    "zendesk": "zendesk",
    "ticket": "zendesk",
    "tickets": "zendesk",
    "twilio": "twilio",
    "sms": "twilio",
    "bamboohr": "bamboohr",
    "employee": "bamboohr",
    "employees": "bamboohr",
    "googleads": "google_ads",
    "google_ads": "google_ads",
    "ads": "google_ads",
    "campaign": "google_ads",
    "campaigns": "google_ads",
    "intercom": "intercom",
    "wave": "wave",
    "invoice": "wave",
    "invoices": "wave",
    "quickbooks": "quickbooks",
    "docusign": "docusign",
    "envelope": "docusign",
}


@dataclass
class OperationNode:
    """Canonical representation of an executable operation or endpoint."""
    node_id: str
    operation: str
    method: str = ""
    url_template: str = ""
    endpoint_id: str | None = None
    contract_claim_id: str | None = None
    statement: str = ""
    service: str = ""
    resource: str = ""
    action_type: str = "read"  # "read" vs "mutate"
    terms: set[str] = field(default_factory=set)


@dataclass
class ValueFlowInfo:
    """Observed data flow between two operations."""
    source_endpoint: str
    target_endpoint: str
    output_path: str
    input_path: str
    statement: str = ""


@dataclass
class TransitionEdge:
    """Directed transition between two operations with transition probability."""
    source: str
    target: str
    count: int = 1
    probability: float = 1.0
    cost: float = 0.0
    value_flows: list[ValueFlowInfo] = field(default_factory=list)


class TrajectoryGraph:
    """Directed operational graph with shortest-path traversal."""

    def __init__(self) -> None:
        self.nodes: dict[str, OperationNode] = {}
        self.edges: dict[str, dict[str, TransitionEdge]] = defaultdict(dict)
        self.reverse_edges: dict[str, dict[str, TransitionEdge]] = defaultdict(dict)

    def add_node(self, node: OperationNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: TransitionEdge) -> None:
        self.edges[edge.source][edge.target] = edge
        self.reverse_edges[edge.target][edge.source] = edge

    def neighbors(self, node_id: str) -> list[str]:
        return list(self.edges.get(node_id, {}).keys())

    def shortest_path(self, start: str, target: str) -> tuple[list[str], float] | None:
        """Dijkstra shortest path minimizing cumulative transition cost (-ln P)."""
        if start == target:
            return [start], 0.0
        if start not in self.nodes or target not in self.nodes:
            return None

        dist: dict[str, float] = {start: 0.0}
        prev: dict[str, str] = {}
        pq: list[tuple[float, str]] = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == target:
                path = []
                curr = u
                while curr in prev:
                    path.append(curr)
                    curr = prev[curr]
                path.append(start)
                return list(reversed(path)), d
            if d > dist.get(u, float("inf")):
                continue
            for v, edge in self.edges.get(u, {}).items():
                new_d = d + edge.cost
                if new_d < dist.get(v, float("inf")):
                    dist[v] = new_d
                    prev[v] = u
                    heapq.heappush(pq, (new_d, v))

        return None

    def find_path_covering(
        self, starts: list[str], targets: list[str]
    ) -> tuple[list[str], float] | None:
        """Find the lowest-cost path starting from one of `starts` covering `targets`."""
        if not targets:
            return None

        # Filter valid starts and targets
        valid_starts = [s for s in starts if s in self.nodes]
        valid_targets = [t for t in targets if t in self.nodes]

        if not valid_targets:
            return None

        if not valid_starts:
            valid_starts = [valid_targets[0]]

        import itertools

        # Limit permutations to avoid explosion if many targets
        target_permutations = (
            list(itertools.permutations(valid_targets))
            if len(valid_targets) <= 4
            else [valid_targets]
        )

        best_path: list[str] | None = None
        best_cost = float("inf")

        for ordered_targets in target_permutations:
            for s in valid_starts:
                curr = s
                curr_path = [s]
                curr_cost = 0.0
                possible = True

                for tgt in ordered_targets:
                    if curr == tgt:
                        continue
                    segment = self.shortest_path(curr, tgt)
                    if segment is None:
                        possible = False
                        break
                    seg_path, seg_cost = segment
                    curr_path.extend(seg_path[1:])
                    curr_cost += seg_cost
                    curr = tgt

                if possible and curr_cost < best_cost:
                    best_cost = curr_cost
                    best_path = curr_path

        if best_path is not None:
            return best_path, best_cost

        # Fallback: try individual shortest paths to any reachable target
        for s in valid_starts:
            for tgt in valid_targets:
                res = self.shortest_path(s, tgt)
                if res is not None:
                    return res

        return None


def _extract_service_and_resource(endpoint_id: str, url: str) -> tuple[str, str]:
    """Extract canonical service and resource name from endpoint ID or URL."""
    if endpoint_id and "." in endpoint_id:
        parts = endpoint_id.split(".")
        return parts[0], parts[-1]
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    parts = host.split(".")
    service = parts[-2] if len(parts) >= 2 else host
    path_segments = [p for p in parsed.path.split("/") if p and not p.startswith("{")]
    resource = path_segments[-1] if path_segments else ""
    return service, resource


def compile_trajectory_graph(store: KnowledgeStore, scope: Scope) -> TrajectoryGraph:
    """Compile an operational trajectory graph from stored claims and execution episodes."""
    graph = TrajectoryGraph()
    claims = store.claims(scope)

    # 1. Index all endpoint contracts
    contracts = [c for c in claims if c.relation == "endpoint_contract"]
    contract_nodes: dict[str, OperationNode] = {}
    compiled_matchers: list[tuple[str, re.Pattern, Claim]] = []

    for c in contracts:
        cid = c.payload.get("id") or c.claim_id
        method = c.payload.get("method", "GET").upper()
        url = c.payload.get("url", "")
        service, resource = _extract_service_and_resource(cid, url)
        action_type = "read" if method in ("GET", "HEAD") else "mutate"

        # Precompile pattern for episode URL matching
        parts = re.split(r"(\{[^{}]+\})", url)
        pat = re.compile(
            "".join("[^/?]+" if part.startswith("{") else re.escape(part) for part in parts)
        )
        compiled_matchers.append((method, pat, c))

        # Node terms for semantic retrieval
        node_text = f"{cid} {method} {url} {c.statement} {service} {resource}"
        node_terms = (terms(node_text) - GRAPH_STOPWORDS)

        node = OperationNode(
            node_id=cid,
            operation="api_fetch",
            method=method,
            url_template=url,
            endpoint_id=cid,
            contract_claim_id=c.claim_id,
            statement=c.statement,
            service=service,
            resource=resource,
            action_type=action_type,
            terms=node_terms,
        )
        graph.add_node(node)
        contract_nodes[cid] = node

    # 2. Add non-API tools that operate in the environment (e.g. base64_encode)
    base64_node = OperationNode(
        node_id="base64_encode",
        operation="base64_encode",
        method="",
        url_template="",
        statement="Encode email or string payload into URL-safe base64 format for transmission.",
        service="gmail",
        resource="messages",
        action_type="read",
        terms={"base64", "encode", "raw", "email", "message", "gmail", "encoding"},
    )
    graph.add_node(base64_node)

    # 3. Collect value flows
    value_flows_by_endpoints: dict[tuple[str, str], list[ValueFlowInfo]] = defaultdict(list)
    for c in claims:
        if c.relation == "observed_value_flow":
            src = c.payload.get("source_endpoint", "")
            tgt = c.payload.get("target_endpoint", "")
            if src and tgt:
                flow = ValueFlowInfo(
                    source_endpoint=src,
                    target_endpoint=tgt,
                    output_path=c.payload.get("output_path", ""),
                    input_path=c.payload.get("input_path", ""),
                    statement=c.statement,
                )
                value_flows_by_endpoints[(src, tgt)].append(flow)

    # 4. Map episodes to node IDs and count empirical transitions
    def match_episode_to_node_id(e: Episode) -> str:
        if e.operation == "api_fetch":
            m = e.arguments.get("method", "GET").upper()
            target = request_target(e.arguments.get("url", ""))
            for meth, pat, c in compiled_matchers:
                if meth == m and pat.fullmatch(target):
                    return c.payload.get("id") or c.claim_id
            return f"{m} {target}"
        return e.operation

    if scope is not None:
        episodes = store.historical_episodes(scope)
    else:
        episodes = [
            Episode.model_validate_json(row[0])
            for row in store.db.execute("SELECT body FROM episodes ORDER BY rowid")
        ]
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        # Filter discovery / meta tools
        if e.operation in ("api_search", "lookup_knowledge"):
            continue
        by_task[e.task_id].append(e)

    transitions: Counter[tuple[str, str]] = Counter()
    for task_eps in by_task.values():
        collapsed: list[str] = []
        for e in task_eps:
            nid = match_episode_to_node_id(e)
            if not collapsed or collapsed[-1] != nid:
                collapsed.append(nid)
        for i in range(len(collapsed) - 1):
            u, v = collapsed[i], collapsed[i + 1]
            transitions[(u, v)] += 1

    # 5. Reinforce with observed value flows and contrastive prerequisites
    for (src, tgt), flows in value_flows_by_endpoints.items():
        if src in graph.nodes and tgt in graph.nodes:
            transitions[(src, tgt)] += len(flows)

    # Contrastive preconditions: if a contract requires base64_encode or prior action
    for c in claims:
        if c.relation == "contrastive_precondition":
            # e.g., messages.send requires prior base64_encode
            if "base64_encode" in c.statement and "gmail.users.messages.send" in graph.nodes:
                transitions[("base64_encode", "gmail.users.messages.send")] += 5

    # 6. Build transition edges with normalized probabilities and costs
    out_degree: Counter[str] = Counter()
    for (u, v), cnt in transitions.items():
        if u in graph.nodes and v in graph.nodes:
            out_degree[u] += cnt

    for (u, v), cnt in transitions.items():
        if u not in graph.nodes or v not in graph.nodes:
            continue
        total = out_degree[u]
        p = cnt / total if total > 0 else 0.5
        cost = -math.log(max(p, 1e-4))
        flows = value_flows_by_endpoints.get((u, v), [])

        edge = TransitionEdge(
            source=u,
            target=v,
            count=cnt,
            probability=p,
            cost=cost,
            value_flows=flows,
        )
        graph.add_edge(edge)

    return graph
