"""Lean advisory memory: a short, fixed note of routes that worked before.

The agent keeps the baseline tools, loop and finish. The world model adds one
small block, computed once per task: endpoints that succeeded in at least
MIN_TASKS earlier tasks for the services the task names, the value flows
observed between them, and the named services it has no confirmed route for.
The note says it is incomplete, so the agent still discovers what is missing.
"""

import json
import re
from collections import defaultdict

from .knowledge import Scope
from .store import KnowledgeStore

MIN_TASKS = 2
MAX_PER_SERVICE = 3
MAX_ENDPOINTS = 10
MAX_FLOWS = 4

HEADER = (
    "Hints from earlier runs in this environment. They are incomplete and may not fit "
    "this task: verify them, and use api_search for any action or data source not listed."
)

# Words in a task that name a service without using its API prefix.
SERVICE_TERMS = {
    "sheets": ("spreadsheet", "spreadsheets", "sheet", "sheets"),
    "google_drive": ("spreadsheet", "spreadsheets", "drive", "file", "files", "document"),
    "gmail": ("email", "emails", "gmail", "inbox", "draft", "mail"),
    "slack": ("slack", "channel"),
    "twilio": ("twilio", "sms", "text", "call", "phone"),
    "calendar": ("calendar",),
    "facebook_pages": ("facebook",),
    "facebook_lead_ads": ("facebook",),
    "zoho_desk": ("zoho",),
    "reamaze": ("reamaze", "re:amaze"),
}


def _service(endpoint_id: str) -> str:
    return endpoint_id.split(".", 1)[0]


# Literal shapes in a task that imply a service.
SERVICE_PATTERNS = {
    "sheets": r"\bss_\w+",
    "google_drive": r"\bss_\w+",
    "gmail": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "slack": r"#[a-z0-9][\w-]*",
}

# Endpoint-id segments that name an operation or API layout, not a resource.
NON_RESOURCE = {
    "users", "me", "list", "get", "create", "update", "delete", "send", "search", "query",
    "batch", "append", "patch", "post", "add", "remove", "set", "mark", "move", "copy",
    "v1", "v2", "v3", "api", "info", "by", "for", "all",
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9:]+", text.lower()))


def task_services(task_text: str, known: set[str]) -> list[str]:
    """Services from the store's endpoint prefixes that the task text names."""
    words, lower = _words(task_text), task_text.lower()
    found = []
    for service in sorted(known):
        terms = SERVICE_TERMS.get(service, (service.split("_")[0],)) + (service.replace("_", " "),)
        pattern = SERVICE_PATTERNS.get(service)
        if any(" " in t and t in lower or t in words for t in terms) or (
            pattern and re.search(pattern, lower)
        ):
            found.append(service)
    return found


def _resources(endpoint_id: str) -> set[str]:
    parts = (
        w.lower()
        for segment in endpoint_id.split(".")[1:]
        for w in re.findall(r"[A-Z]?[a-z0-9]+", segment)
    )
    return {w.rstrip("s") for w in parts if w not in NON_RESOURCE and len(w) > 2}


def resource_gaps(task_text: str, service: str, documented: set[str], confirmed: list[str]) -> list[str]:
    """Resources of `service` the task names that no confirmed endpoint covers."""
    named = {w.rstrip("s") for w in _words(task_text)}
    covered = set().union(*(_resources(e) for e in confirmed)) if confirmed else set()
    known = set().union(*(_resources(e) for e in documented)) if documented else set()
    return sorted((named & known) - covered - {service.rstrip("s")})


def build_advisory_block(store: KnowledgeStore, scope: Scope, task_text: str) -> tuple[str, dict]:
    claims = store.claims(scope)
    tenants: dict[str, str | None] = {}

    def tasks_of(episode_ids) -> set[str]:
        out = set()
        for eid in episode_ids:
            if eid not in tenants:
                try:
                    tenants[eid] = store.episode(eid).scope.tenant
                except KeyError:
                    tenants[eid] = None
            if tenants[eid]:
                out.add(tenants[eid])
        return out

    contracts = {
        c.payload["id"]: c
        for c in claims
        if c.relation == "endpoint_contract" and c.status == "supported" and c.payload.get("id")
    }
    succeeded = defaultdict(set)
    for c in claims:
        if c.relation == "observed_output_type" and c.payload.get("endpoint") in contracts:
            succeeded[c.payload["endpoint"]] |= tasks_of(c.supporting_episodes)

    services = task_services(task_text, {_service(e) for e in contracts})
    endpoints, gaps, uncovered = [], [], []
    for service in services:
        ranked = sorted(
            (e for e in succeeded if _service(e) == service and len(succeeded[e]) >= MIN_TASKS),
            key=lambda e: (-len(succeeded[e]), e),
        )[:MAX_PER_SERVICE]
        if not ranked:
            gaps.append(service)
            continue
        endpoints.extend(ranked)
        documented = {e for e in contracts if _service(e) == service}
        uncovered += [f"{service} {r}" for r in resource_gaps(task_text, service, documented, ranked)]
    endpoints = endpoints[:MAX_ENDPOINTS]

    chosen = set(endpoints)
    flows = {}
    for c in claims:
        if c.relation != "observed_value_flow":
            continue
        p = c.payload
        if p.get("target_endpoint") not in chosen or p.get("source_endpoint") == p.get("target_endpoint"):
            continue
        if c.status == "supported" or len(tasks_of(c.supporting_episodes)) >= MIN_TASKS:
            line = f"- {p['source_endpoint']} {p['output_path']} -> {p['target_endpoint']} {p['input_path']}"
            flows.setdefault(line, p)
    flows = dict(sorted(flows.items())[:MAX_FLOWS])

    lines = []
    if endpoints:
        lines.append("Endpoints that worked in earlier tasks:")
        for e in endpoints:
            p = contracts[e].payload
            lines.append(f"- {p.get('method', '').upper()} {p.get('url', '')}  ({e})")
    if flows:
        lines.append("Values passed between them:")
        lines.extend(flows)
    if uncovered:
        lines.append(
            "The task names " + ", ".join(uncovered)
            + ", which none of these endpoints handles. Discover the right endpoint with api_search."
        )
    if gaps:
        lines.append("No confirmed routes yet for: " + ", ".join(gaps) + ". Discover them with api_search.")
    block = HEADER + "\n" + "\n".join(lines) if lines else ""
    return block, {
        "services": services,
        "endpoints": endpoints,
        "flows": [f"{p['source_endpoint']}->{p['target_endpoint']}" for p in flows.values()],
        "uncovered": uncovered,
        "gaps": gaps,
        "chars": len(block),
    }


def recent_tool_rows(messages) -> list[dict]:
    """Tool calls from the conversation, in the decision-audit row shape detect_trap reads."""
    rows = []
    for message in messages:
        for block in getattr(message, "blocks", None) or ():
            name = getattr(block, "name", None)
            if name is None or not hasattr(block, "arguments") or name == "finish":
                continue
            try:
                arguments = json.loads(block.arguments or "{}")
            except (TypeError, ValueError):
                arguments = {"raw": block.arguments}
            rows.append({"decision": {"action": {"tool": name, "arguments": arguments}}, "gate": {}})
    return rows
