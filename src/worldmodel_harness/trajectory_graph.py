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
    "drive": "google_drive",
    "files": "google_drive",
    "file": "google_drive",
    "google_drive": "google_drive",
    "email": "gmail",
    "emails": "gmail",
    "mail": "gmail",
    "gmail": "gmail",
    "draft": "gmail",
    "drafts": "gmail",
    "mailchimp": "mailchimp",
    "subscriber": "mailchimp",
    "subscribers": "mailchimp",
    "bounce": "mailchimp",
    "bounces": "mailchimp",
    "asana": "asana",
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
    "jira": "jira",
    "confluence": "confluence",
    "monday": "monday",
    "board": "monday",
    "boards": "monday",
    "notify": "slack",
    "notification": "slack",
    "notifications": "slack",
    "pipefy": "pipefy",
    "card": "pipefy",
    "cards": "pipefy",
    "phase": "pipefy",
    "airtable": "airtable",
    "base": "airtable",
    "trello": "trello",
    "hubspot": "hubspot",
    "notion": "notion",
    "zoom": "zoom",
    "xero": "xero",
    "calendly": "calendly",
    "canva": "canva",
    "freshdesk": "freshdesk",
    "helpscout": "helpscout",
    "gorgias": "gorgias",
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
    """Directed transition between two operations with transition probability and Beta posterior."""
    source: str
    target: str
    count: int = 1
    success_count: int = 1
    failure_count: int = 0
    alpha: float = field(default=0.0)
    beta: float = field(default=0.0)
    probability: float = 1.0
    cost: float = 0.0
    value_flows: list[ValueFlowInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            self.alpha = 1.0 + float(max(self.count, self.success_count))
        if self.beta <= 0.0:
            self.beta = 1.0 + float(self.failure_count)

    def posterior_mean(self) -> float:
        ab = self.alpha + self.beta
        return self.alpha / ab if ab > 0 else 0.5

    def posterior_variance(self) -> float:
        ab = self.alpha + self.beta
        if ab <= 0:
            return 0.25
        return (self.alpha * self.beta) / (ab * ab * (ab + 1.0))

    def sample_beta(self) -> float:
        import random
        return random.betavariate(max(self.alpha, 1e-3), max(self.beta, 1e-3))

    def certainty(self) -> float:
        """Continuous certainty score in [0.0, 1.0] that increases as the edge is used more with success.

        Defined by posterior concentration relative to an uninformative prior Beta(1, 1) (var = 1/12).
        As successes accumulate with few failures, variance drops to 0 and certainty approaches 1.0.
        """
        var_uninformed = 1.0 / 12.0  # Var of Beta(1, 1) = 1/12 = 0.0833
        var = self.posterior_variance()
        concentration = max(0.0, 1.0 - (var / var_uninformed))
        return round(self.posterior_mean() * concentration, 3)


class CandidatePath(tuple):
    """Candidate solution path with tuple backward compatibility and rich Bayesian properties."""

    def __new__(
        cls,
        path: list[str],
        cost: float,
        confidence: float,
        support_count: int,
        certainty: float = 0.0,
        posterior_variance: float = 0.0,
        sample_score: float = 0.0,
    ):
        obj = super().__new__(cls, (path, cost, confidence, support_count))
        obj.certainty = certainty
        obj.posterior_variance = posterior_variance
        obj.sample_score = sample_score
        return obj

    @property
    def path(self) -> list[str]:
        return self[0]

    @property
    def cost(self) -> float:
        return self[1]

    @property
    def confidence(self) -> float:
        return self[2]

    @property
    def support_count(self) -> int:
        return self[3]


@dataclass
class NodeAttempt:
    """An action attempted from a node/state and its observed outcome."""
    source_node: str
    target_node: str | None
    action_type: str  # "api_fetch", "api_search", etc.
    method: str = ""
    url: str = ""
    parameters_summary: str = ""
    outcome: str = "success"  # "success", "error", "blocked", "unproductive"
    result_summary: str = ""
    lesson: str = ""
    count: int = 1


class TrajectoryGraph:
    """Directed operational graph with shortest-path traversal."""

    def __init__(self) -> None:
        self.nodes: dict[str, OperationNode] = {}
        self.edges: dict[str, dict[str, TransitionEdge]] = defaultdict(dict)
        self.reverse_edges: dict[str, dict[str, TransitionEdge]] = defaultdict(dict)
        self.forward_value_flows: dict[str, list[ValueFlowInfo]] = defaultdict(list)
        self.reverse_value_flows: dict[str, list[ValueFlowInfo]] = defaultdict(list)
        self.node_attempts: dict[str, list[NodeAttempt]] = defaultdict(list)
        self.compiled_matchers: list[tuple[str, re.Pattern, Claim]] = []
        self.prerequisites: dict[str, set[str]] = defaultdict(set)

    def match_episode_to_node_id(self, episode: Episode | None) -> str | None:
        """Resolve an execution episode to a canonical node_id in this graph."""
        if not episode:
            return None
        if episode.operation == "api_fetch":
            m = episode.arguments.get("method", "GET").upper()
            target = request_target(episode.arguments.get("url", ""))
            for meth, pat, c in self.compiled_matchers:
                if meth == m and pat.fullmatch(target):
                    return c.payload.get("id") or c.claim_id
            return f"{m} {target}"
        if episode.operation in self.nodes:
            return episode.operation
        return None

    def add_node(self, node: OperationNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: TransitionEdge) -> None:
        self.edges[edge.source][edge.target] = edge
        self.reverse_edges[edge.target][edge.source] = edge

    def neighbors(self, node_id: str) -> list[str]:
        return list(self.edges.get(node_id, {}).keys())

    def shortest_path(
        self,
        start: str,
        target: str,
        allowed_services: set[str] | None = None,
        authorized_mutations: set[str] | None = None,
    ) -> tuple[list[str], float] | None:
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
                v_node = self.nodes.get(v)
                if v_node and allowed_services is not None:
                    if v_node.service not in allowed_services and v_node.node_id != "base64_encode":
                        continue
                if authorized_mutations is not None:
                    # Every mutation, including a fallback target, needs authorization
                    if v_node and v_node.action_type == "mutate":
                        is_auth = v in authorized_mutations
                        is_prereq = v in self.prerequisites.get(target, set())
                        if not (is_auth or is_prereq):
                            continue
                new_d = d + edge.cost
                if new_d < dist.get(v, float("inf")):
                    dist[v] = new_d
                    prev[v] = u
                    heapq.heappush(pq, (new_d, v))

        return None

    def find_all_candidate_paths(
        self,
        starts: list[str],
        targets: list[str],
        allowed_services: set[str] | None = None,
        authorized_mutations: set[str] | None = None,
        max_paths: int = 3,
    ) -> list[tuple[list[str], float, float, int]]:
        """Find up to `max_paths` distinct valid paths from starts covering targets.

        Returns list of tuples: (path_nodes, cost, confidence, support_count).
        """
        if not targets:
            return []

        # Filter valid starts and targets
        valid_starts = [s for s in starts if s in self.nodes]
        valid_targets = [t for t in targets if t in self.nodes]

        if not valid_targets:
            return []

        if not valid_starts:
            valid_starts = [valid_targets[0]]


        import itertools

        read_targets = [t for t in valid_targets if self.nodes[t].action_type == "read"]
        mutate_targets = [t for t in valid_targets if self.nodes[t].action_type != "read"]

        if read_targets and mutate_targets:
            r_perms = list(itertools.permutations(read_targets)) if len(read_targets) <= 2 else [read_targets]
            m_perms = list(itertools.permutations(mutate_targets)) if len(mutate_targets) <= 4 else [mutate_targets]
            target_permutations = [list(rp) + list(mp) for rp in r_perms for mp in m_perms]
        else:
            target_permutations = (
                list(itertools.permutations(valid_targets))
                if len(valid_targets) <= 4
                else [valid_targets]
            )

        candidates: list[tuple[list[str], float, float, int]] = []
        seen_paths: set[tuple[str, ...]] = set()

        for ordered_targets in target_permutations:
            for s in valid_starts:
                curr = s
                curr_path = [s]
                curr_cost = 0.0
                min_support = 999999
                possible = True

                for tgt in ordered_targets:
                    if curr == tgt:
                        continue
                    segment = self.shortest_path(
                        curr, tgt, allowed_services=allowed_services, authorized_mutations=authorized_mutations
                    )
                    if segment is None:
                        possible = False
                        break
                    seg_path, seg_cost = segment
                    curr_path.extend(seg_path[1:])
                    curr_cost += seg_cost
                    for i in range(len(seg_path) - 1):
                        e = self.edges.get(seg_path[i], {}).get(seg_path[i + 1])
                        cnt = e.count if e else 1
                        if cnt < min_support:
                            min_support = cnt
                    curr = tgt

                if possible:
                    t_path = tuple(curr_path)
                    if t_path not in seen_paths:
                        seen_paths.add(t_path)
                        conf = max(0.05, min(1.0, math.exp(-curr_cost)))
                        supp = min_support if min_support != 999999 else 1

                        path_edges = [
                            self.edges.get(curr_path[i], {}).get(curr_path[i + 1])
                            for i in range(len(curr_path) - 1)
                        ]
                        edge_means = [e.posterior_mean() if e else 0.67 for e in path_edges]
                        edge_vars = [e.posterior_variance() if e else 0.0556 for e in path_edges]
                        edge_samples = [e.sample_beta() if e else 0.67 for e in path_edges]
                        edge_certs = [e.certainty() if e else 0.222 for e in path_edges]

                        exp_conf = math.prod(edge_means) if edge_means else conf
                        path_certainty = math.prod(edge_certs) if edge_certs else 0.0
                        second_moments = [v + (m ** 2) for v, m in zip(edge_vars, edge_means)]
                        path_var = max(0.0, math.prod(second_moments) - (exp_conf ** 2)) if second_moments else 0.0
                        sample_score = math.prod(edge_samples) if edge_samples else exp_conf

                        candidates.append(
                            CandidatePath(
                                curr_path,
                                curr_cost,
                                conf,
                                supp,
                                certainty=round(path_certainty, 3),
                                posterior_variance=round(path_var, 4),
                                sample_score=round(sample_score, 3),
                            )
                        )

        # Sort candidates by cost ascending (lowest cost / highest probability first)
        candidates.sort(key=lambda x: x[1])

        if not candidates:
            # Fallback: try individual shortest paths to any reachable target
            for s in valid_starts:
                for tgt in valid_targets:
                    res = self.shortest_path(
                        s, tgt, allowed_services=allowed_services,
                        authorized_mutations=authorized_mutations,
                    )
                    if res is not None:
                        seg_path, seg_cost = res
                        t_path = tuple(seg_path)
                        if t_path not in seen_paths:
                            seen_paths.add(t_path)
                            conf = max(0.05, min(1.0, math.exp(-seg_cost)))
                            path_edges = [
                                self.edges.get(seg_path[i], {}).get(seg_path[i + 1])
                                for i in range(len(seg_path) - 1)
                            ]
                            edge_means = [e.posterior_mean() if e else 0.67 for e in path_edges]
                            edge_vars = [e.posterior_variance() if e else 0.0556 for e in path_edges]
                            edge_samples = [e.sample_beta() if e else 0.67 for e in path_edges]
                            edge_certs = [e.certainty() if e else 0.222 for e in path_edges]

                            exp_conf = math.prod(edge_means) if edge_means else conf
                            path_certainty = math.prod(edge_certs) if edge_certs else 0.0
                            second_moments = [v + (m ** 2) for v, m in zip(edge_vars, edge_means)]
                            path_var = max(0.0, math.prod(second_moments) - (exp_conf ** 2)) if second_moments else 0.0
                            sample_score = math.prod(edge_samples) if edge_samples else exp_conf

                            candidates.append(
                                CandidatePath(
                                    seg_path,
                                    seg_cost,
                                    conf,
                                    1,
                                    certainty=round(path_certainty, 3),
                                    posterior_variance=round(path_var, 4),
                                    sample_score=round(sample_score, 3),
                                )
                            )

        return candidates[:max_paths]

    def find_path_covering(
        self,
        starts: list[str],
        targets: list[str],
        allowed_services: set[str] | None = None,
    ) -> tuple[list[str], float] | None:
        """Find the lowest-cost path starting from one of `starts` covering `targets`."""
        candidates = self.find_all_candidate_paths(
            starts, targets, allowed_services=allowed_services, max_paths=1
        )
        if candidates:
            path_nodes, cost, _, _ = candidates[0]
            return path_nodes, cost
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


def compile_trajectory_graph(
    store: KnowledgeStore, scope: Scope, *, successful_transitions_only: bool = False
) -> TrajectoryGraph:
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

    graph.compiled_matchers = compiled_matchers

    # Structural evidence may guide routing, but value reuse is not causal authorization.
    from .knowledge import utc_now

    now = utc_now()

    def applicable_structure(claim):
        return (
            claim.status == "supported"
            and not claim.needs_review
            and (claim.valid_until is None or claim.valid_until > now)
        )

    # 3. Collect value flows
    value_flows_by_endpoints: dict[tuple[str, str], list[ValueFlowInfo]] = defaultdict(list)
    for c in claims:
        if c.relation == "observed_value_flow" and applicable_structure(c):
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
                graph.forward_value_flows[src].append(flow)
                graph.reverse_value_flows[tgt].append(flow)

    # 4. Map episodes to node IDs, track tried actions/outcomes, and count empirical transitions
    def match_episode_to_node_id(e: Episode) -> str:
        return graph.match_episode_to_node_id(e) or e.operation

    if scope is not None:
        raw_episodes = store.historical_episodes(scope)
    else:
        raw_episodes = [
            Episode.model_validate_json(row[0])
            for row in store.db.execute("SELECT body FROM episodes ORDER BY rowid")
        ]

    all_by_task: dict[str, list[Episode]] = defaultdict(list)
    for e in raw_episodes:
        all_by_task[e.task_id].append(e)

    # 4a. Track attempted actions and observed results per node/state
    for task_eps in all_by_task.values():
        current_state = "root"
        for ep in task_eps:
            if ep.origin != "environment":
                continue

            target_nid = match_episode_to_node_id(ep)
            action_type = ep.operation
            args = ep.arguments or {}
            method = args.get("method", "").upper() if isinstance(args, dict) else ""
            url = args.get("url", "") if isinstance(args, dict) else ""

            param_summary = ""
            if isinstance(args, dict):
                from .experience import Symbolizer
                sym = Symbolizer()
                for p_key in ("params", "body", "query", "ranges"):
                    if p_key in args and args[p_key]:
                        try:
                            val = args[p_key]
                            if isinstance(val, str) and (val.startswith("{") or val.startswith("[")):
                                import json
                                val = json.loads(val)
                            snap = sym.snapshot(val)
                            param_summary = json.dumps(snap.structure)[:120]
                        except Exception:
                            param_summary = str(args[p_key])[:100]
                        break

            outcome = ep.outcome
            res = ep.response
            result_summary = ""
            lesson = ""

            if outcome in ("error", "blocked"):
                err_type = getattr(ep, "error_type", None)
                if isinstance(res, dict) and "error" in res:
                    err = res["error"]
                    result_summary = err.get("message") if isinstance(err, dict) else str(err)
                elif err_type:
                    result_summary = f"{err_type}: {str(res)[:80]}"
                else:
                    result_summary = str(res)[:100]

                err_lower = result_summary.lower()
                if "base64" in err_lower:
                    lesson = "Encoding requirement: Gmail/email payloads require base64url encoding."
                elif "recipient" in err_lower or "no 'to'" in err_lower:
                    lesson = "Header requirement: Email must specify recipient 'To:' and 'From:' headers."
                elif "unable to parse range" in err_lower or "range" in err_lower:
                    lesson = "Parameter syntax error: verify range syntax (e.g. use clean 'A:Z')."
                elif "method" in err_lower or err_type == "malformed_http_method":
                    lesson = "Action syntax error: method must be GET/POST/PUT/DELETE, not tool name."
                elif "404" in err_lower or "not found" in err_lower:
                    lesson = "Resource not found: check resource identifier."
                elif "401" in err_lower or "403" in err_lower:
                    lesson = "Authentication or permission error."
                elif outcome == "blocked":
                    lesson = "Precondition gate blocked execution before dispatch."
                else:
                    lesson = f"Call failed: {result_summary[:80]}"
            else:
                from .decision_runtime import is_empty_query_result
                is_read = method == "GET" or any(s in url for s in (":find", "/find", ":search", "/search", ":query", "/query"))
                if is_read and is_empty_query_result(res):
                    outcome = "zero_matches"
                    result_summary = "HTTP 200: Request accepted, zero matches, identifier unresolved"
                    lesson = "Request accepted, zero matches, identifier unresolved: verify entity title, table ID, or query parameters."
                    target_nid = None
                elif isinstance(res, dict):
                    if "values" in res and isinstance(res["values"], list):
                        result_summary = f"HTTP 200: Successfully returned {len(res['values'])} rows"
                    elif "sid" in res or "id" in res:
                        item_id = res.get("sid") or res.get("id")
                        status_str = res.get("status", "created")
                        result_summary = f"HTTP 200: Mutation succeeded (id={item_id}, status={status_str})"
                    else:
                        keys = list(res.keys())[:3]
                        result_summary = f"HTTP 200: Succeeded (keys: {keys})"
                    lesson = "Valid verified operational transition."
                else:
                    result_summary = "Operation succeeded"
                    lesson = "Valid verified operational transition."

            attempt = NodeAttempt(
                source_node=current_state,
                target_node=target_nid,
                action_type=action_type,
                method=method,
                url=url,
                parameters_summary=param_summary,
                outcome=outcome,
                result_summary=result_summary,
                lesson=lesson,
                count=1,
            )

            existing = None
            for prev in graph.node_attempts[current_state]:
                if (
                    prev.action_type == attempt.action_type
                    and prev.method == attempt.method
                    and prev.url == attempt.url
                    and prev.parameters_summary == attempt.parameters_summary
                    and prev.outcome == attempt.outcome
                ):
                    existing = prev
                    break
            if existing:
                existing.count += 1
            else:
                graph.node_attempts[current_state].append(attempt)

            # Advance current_state on environment success with valid target and non-empty result
            if (
                outcome == "success"
                and target_nid is not None
                and ep.origin == "environment"
                and ep.operation not in ("api_search", "lookup_knowledge")
            ):
                current_state = target_nid

    # 4b. Map successful execution transitions into graph edges
    from .decision_runtime import is_empty_query_result
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for e in raw_episodes:
        # Harness records wrap environment events; they are not graph transitions.
        # Failed calls or zero-match queries supply recovery evidence, not successful execution paths.
        args = e.arguments if isinstance(e.arguments, dict) else {}
        is_read = args.get("method", "GET").upper() == "GET" or any(
            s in str(args.get("url", "")).lower() for s in (":find", "/find", ":search", "/search", ":query", "/query")
        )
        if successful_transitions_only and (
            e.origin != "environment"
            or e.outcome != "success"
            or (is_read and is_empty_query_result(e.response))
        ):
            continue
        if e.operation in ("api_search", "lookup_knowledge"):
            continue
        by_task[e.task_id].append(e)

    empirical_transitions: Counter[tuple[str, str]] = Counter()
    for task_eps in by_task.values():
        collapsed: list[str] = []
        for e in task_eps:
            if getattr(e, "outcome", "success") in ("success", None):
                nid = match_episode_to_node_id(e)
                if nid and (not collapsed or collapsed[-1] != nid):
                    collapsed.append(nid)
        for i in range(len(collapsed) - 1):
            u, v = collapsed[i], collapsed[i + 1]
            empirical_transitions[(u, v)] += 1

    # 5. Track structural prerequisite edges for reachability and causal dependencies
    structural_edges: set[tuple[str, str]] = set()
    for (src, tgt), flows in value_flows_by_endpoints.items():
        if src in graph.nodes and tgt in graph.nodes:
            structural_edges.add((src, tgt))

    # Generic contrastive preconditions: ONLY supported/verified claims establish structural prerequisite edges
    for c in claims:
        if c.relation == "contrastive_precondition" and applicable_structure(c):
            src = c.payload.get("source_operation") or (c.participants[0] if len(c.participants) >= 1 else None)
            tgt = c.payload.get("target_operation") or (c.participants[1] if len(c.participants) >= 2 else None)
            if src and tgt and src in graph.nodes and tgt in graph.nodes:
                structural_edges.add((src, tgt))
                graph.prerequisites[tgt].add(src)

    # All candidate graph edges: union of empirical observations and structural prerequisite edges
    all_edge_pairs = set(empirical_transitions.keys()) | structural_edges

    # 6. Build transition edges with normalized probabilities, costs, and pure empirical Beta posteriors
    out_degree: Counter[str] = Counter()
    for (u, v) in all_edge_pairs:
        if u in graph.nodes and v in graph.nodes:
            # Dijkstra routing weight: observed successes + structural connectivity prior (1)
            w = empirical_transitions.get((u, v), 0) + (1 if (u, v) in structural_edges else 0)
            out_degree[u] += w

    for (u, v) in all_edge_pairs:
        if u not in graph.nodes or v not in graph.nodes:
            continue
        succ_cnt = empirical_transitions.get((u, v), 0)
        failed_cnt = sum(
            att.count
            for att in graph.node_attempts.get(u, [])
            if att.target_node == v and att.outcome in ("error", "blocked")
        )
        w = succ_cnt + (1 if (u, v) in structural_edges else 0)
        total = out_degree[u]
        p = w / total if total > 0 else 0.5
        cost = -math.log(max(p, 1e-4))
        flows = value_flows_by_endpoints.get((u, v), [])

        # Empirical Beta-Binomial posterior:
        # success_count and failure_count reflect ONLY actual observed execution episodes.
        # Zero execution episodes = alpha=1.0, beta=1.0 (uniform prior, posterior_mean=0.5, variance=0.0833).
        alpha = 1.0 + float(succ_cnt)
        beta = 1.0 + float(failed_cnt)

        edge = TransitionEdge(
            source=u,
            target=v,
            count=succ_cnt,
            success_count=succ_cnt,
            failure_count=failed_cnt,
            alpha=alpha,
            beta=beta,
            probability=p,
            cost=cost,
            value_flows=flows,
        )
        graph.add_edge(edge)

    return graph
