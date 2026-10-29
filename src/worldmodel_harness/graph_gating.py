"""Graph Feasibility Gating, Entity Collision Prevention, and Dynamic Fallback.

Evaluates whether an active task should execute WITH_GRAPH, HYBRID_SANITIZE,
or WITHOUT_GRAPH (reactive fallback), preventing both the over-discovery search
trap and concrete literal memory poisoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .knowledge import Scope, WorkingGraph
from .store import KnowledgeStore

KNOWN_PLATFORMS = {
    "sheets": ["sheet", "spreadsheet", "google_sheets", "sheets.googleapis.com"],
    "drive": ["drive", "google_drive", "drive.googleapis.com"],
    "gmail": ["gmail", "email", "mail"],
    "slack": ["slack", "#", "channel"],
    "twilio": ["twilio", "sms", "text", "page", "phone"],
    "jira": ["jira", "issue", "ticket"],
    "confluence": ["confluence", "wiki"],
    "asana": ["asana", "project", "task"],
    "salesforce": ["salesforce", "lead", "contact", "opportunity", "soql"],
    "hubspot": ["hubspot", "deal", "company"],
    "monday": ["monday", "monday.com", "board"],
    "pipefy": ["pipefy", "pipe", "card"],
    "zoom": ["zoom", "meeting"],
}

COMMON_RESOURCE_PREFIXES = ["ss_", "ws_", "proj_", "sec_", "msg_", "thr_", "C0"]


@dataclass
class GatingDecision:
    action: Literal["with_graph", "sanitize", "without_graph"]
    reachability: float
    entity_grounding: float
    collision_hazard: bool
    required_services: list[str]
    missing_services: list[str]
    colliding_entities: list[str]
    reason: str


def detect_services_from_prompt(prompt: str) -> list[str]:
    """Detect required external services from task prompt text."""
    p_lower = prompt.lower()
    services = []
    for platform, terms in KNOWN_PLATFORMS.items():
        if any(term in p_lower for term in terms):
            services.append(platform)
    return services


def extract_resource_literals(text: str) -> set[str]:
    """Extract concrete resource identifiers (e.g. ss_badges, ws_queue, proj_facilities, tbl_ops)."""
    tokens = re.findall(r"\b(?:ss|ws|proj|sec|msg|thr|rec|tbl)_[a-zA-Z0-9_-]+\b", text)
    # Also find quoted spreadsheet names like "Purchase Requests"
    quoted = re.findall(r"['\"]([A-Z][a-zA-Z0-9\s_-]+)['\"]", text)
    return set(tokens) | {q.lower().strip() for q in quoted}


def evaluate_graph_feasibility(
    prompt: str,
    store: KnowledgeStore | None,
    scope: Scope | None = None,
) -> GatingDecision:
    """Pre-flight evaluation of task reachability and entity collision hazard."""
    if store is None:
        return GatingDecision(
            action="without_graph",
            reachability=0.0,
            entity_grounding=0.0,
            collision_hazard=False,
            required_services=[],
            missing_services=[],
            colliding_entities=[],
            reason="No knowledge store provided",
        )

    services = detect_services_from_prompt(prompt)
    if not services:
        return GatingDecision(
            action="with_graph",
            reachability=1.0,
            entity_grounding=1.0,
            collision_hazard=False,
            required_services=[],
            missing_services=[],
            colliding_entities=[],
            reason="No external service constraints detected",
        )

    # Inspect claims in store to compute per-service coverage respecting scope and status
    raw_claims = store.claims(scope) if scope else store.claims()
    all_claims = [c for c in raw_claims if getattr(c, "status", "supported") in ("supported", "active")]
    service_contract_counts: dict[str, int] = {}
    graph_endpoints: set[str] = set()

    for c in all_claims:
        if c.relation == "endpoint_contract":
            for p in c.participants:
                p_str = str(p).lower()
                graph_endpoints.add(p_str)
                for s in services:
                    if s in p_str:
                        service_contract_counts[s] = service_contract_counts.get(s, 0) + 1

    covered_services = [s for s in services if service_contract_counts.get(s, 0) > 0]
    missing_services = [s for s in services if service_contract_counts.get(s, 0) == 0]
    reachability = len(covered_services) / max(1, len(services))

    # Inspect entity grounding and collision hazard
    prompt_literals = extract_resource_literals(prompt)
    colliding: list[str] = []

    # Check if historical claims reference concrete entities not in prompt
    for c in all_claims:
        if c.relation in ("observed_value_flow", "endpoint_contract", "contrastive_observation"):
            stmt = c.statement.lower()
            for lit in extract_resource_literals(stmt):
                if lit.startswith("ss_") and lit not in prompt_literals:
                    # If prompt asks for a different spreadsheet (e.g. ss_maint_queue or "Purchase Requests"),
                    # this lit (e.g. ss_badges) is an anchor hazard.
                    if any(
                        p in prompt.lower()
                        for p in ["spreadsheet", "queue", "purchase", "requests", "maintenance", "sheet"]
                    ):
                        colliding.append(lit)

    colliding = sorted(list(set(colliding)))
    collision_hazard = len(colliding) > 0

    # Decision rule:
    # 1. If terminal platform has 0 coverage and reachability is low (< 0.5) -> WITHOUT_GRAPH (prevent search trap)
    if reachability < 0.5 and missing_services:
        return GatingDecision(
            action="without_graph",
            reachability=reachability,
            entity_grounding=0.0,
            collision_hazard=collision_hazard,
            required_services=services,
            missing_services=missing_services,
            colliding_entities=colliding,
            reason=f"Terminal platform unreached (missing: {missing_services}). Falling back to reactive execution.",
        )

    # 2. If collision hazard exists, sanitize concrete literals so memory doesn't poison the LLM
    if collision_hazard:
        return GatingDecision(
            action="sanitize",
            reachability=reachability,
            entity_grounding=0.5 if prompt_literals else 0.0,
            collision_hazard=True,
            required_services=services,
            missing_services=missing_services,
            colliding_entities=colliding,
            reason=f"Entity collision hazard detected ({colliding}). Sanitizing concrete memory literals.",
        )

    # 3. Otherwise, full graph mode
    return GatingDecision(
        action="with_graph",
        reachability=reachability,
        entity_grounding=1.0 if prompt_literals else 0.5,
        collision_hazard=False,
        required_services=services,
        missing_services=missing_services,
        colliding_entities=[],
        reason="Full graph coverage with no collision hazard.",
    )


def sanitize_action_literals(action_str: str, allowed_literals: set[str]) -> str:
    """Sanitize concrete resource IDs like 'ss_badges', 'tbl_ops' into abstract placeholders '{spreadsheet_id}', '{table_id}'."""
    def replacer(match):
        token = match.group(0)
        if token in allowed_literals:
            return token
        if token.startswith("ss_"):
            return "{spreadsheet_id}"
        if token.startswith("ws_"):
            return "{worksheet_name}"
        if token.startswith("proj_"):
            return "{project_id}"
        if token.startswith("sec_"):
            return "{section_id}"
        if token.startswith("tbl_"):
            return "{table_id}"
        if token.startswith("msg_"):
            return "{message_id}"
        if token.startswith("thr_"):
            return "{thread_id}"
        if token.startswith("rec_"):
            return "{record_id}"
        if token.startswith("base_"):
            return "{base_id}"
        if token.startswith("att_"):
            return "{attachment_id}"
        if token.startswith("usr_"):
            return "{user_id}"
        return token

    return re.sub(r"\b(?:ss|ws|proj|sec|tbl|msg|thr|rec|base|att|usr)_[a-zA-Z0-9_-]+\b", replacer, action_str)
