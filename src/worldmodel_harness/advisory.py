"""Advisory memory: a short, fixed note of routes that worked in earlier tasks.

The agent keeps the baseline tools, loop and finish. The world model adds one
small note, built once per task from the carried store: endpoints that succeeded
in at least MIN_TASKS earlier tasks for the services the task names (with the
documented request format of write endpoints), the value flows observed between
them, and the named services or resources it has no confirmed route for. The
note says it is incomplete, so the agent still discovers what is missing. If the
agent starts repeating itself, the note is withdrawn for the rest of the task.
"""

import json
import re
from collections import Counter, defaultdict

from .knowledge import Scope
from .store import KnowledgeStore

MIN_TASKS = 2
MAX_PER_SERVICE = 3
MAX_ENDPOINTS = 10
MAX_FLOWS = 4
MAX_REQUEST_CHARS = 160

# A loop: the same tool call this many times among the latest LOOP_WINDOW calls.
LOOP_WINDOW = 6
LOOP_REPEATS = 3

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

# Literal shapes in a task that imply a service.
SERVICE_PATTERNS = {
    "sheets": r"\bss_\w+",
    "google_drive": r"\bss_\w+",
    "gmail": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "slack": r"#[a-z0-9][\w-]*",
}

# Endpoint-id words that name an operation or API layout, not a resource.
NON_RESOURCE = {
    "users", "me", "list", "get", "create", "update", "delete", "send", "search", "query",
    "batch", "append", "patch", "post", "add", "remove", "set", "mark", "move", "copy",
    "v1", "v2", "v3", "api", "info", "by", "for", "all",
}


def _service(endpoint_id: str) -> str:
    return endpoint_id.split(".", 1)[0]


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
    covered = set().union(*(_resources(e) for e in confirmed))
    known = set().union(*(_resources(e) for e in documented))
    return sorted((named & known) - covered - {service.rstrip("s")})


def _endpoint_line(payload: dict) -> str:
    method = payload.get("method", "").upper()
    line = f"- {method} {payload.get('url', '')}  ({payload['id']})"
    request = " ".join(str(payload.get("request") or "").split())
    if method != "GET" and request:
        if len(request) > MAX_REQUEST_CHARS:
            request = request[: MAX_REQUEST_CHARS - 3] + "..."
        line += f"\n  request: {request}"
    return line


def build_note(store: KnowledgeStore, scope: Scope, task_text: str) -> tuple[str, dict]:
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
        p = c.payload
        if (
            c.relation != "observed_value_flow"
            or p.get("target_endpoint") not in chosen
            or p.get("source_endpoint") == p.get("target_endpoint")
        ):
            continue
        if c.status == "supported" or len(tasks_of(c.supporting_episodes)) >= MIN_TASKS:
            line = f"- {p['source_endpoint']} {p['output_path']} -> {p['target_endpoint']} {p['input_path']}"
            flows.setdefault(line, p)
    flows = dict(sorted(flows.items())[:MAX_FLOWS])

    lines = []
    if endpoints:
        lines.append("Endpoints that worked in earlier tasks:")
        lines.extend(_endpoint_line(contracts[e].payload) for e in endpoints)
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
    note = HEADER + "\n" + "\n".join(lines) if lines else ""
    return note, {
        "services": services,
        "endpoints": endpoints,
        "flows": [f"{p['source_endpoint']}->{p['target_endpoint']}" for p in flows.values()],
        "uncovered": uncovered,
        "gaps": gaps,
        "chars": len(note),
    }


def detect_loop(messages) -> dict | None:
    """The same tool call LOOP_REPEATS times among the latest LOOP_WINDOW calls."""
    calls = []
    for message in messages:
        for block in getattr(message, "blocks", None) or ():
            name = getattr(block, "name", None)
            if name is None or not hasattr(block, "arguments") or name == "finish":
                continue
            try:
                arguments = json.loads(block.arguments or "{}")
            except (TypeError, ValueError):
                arguments = {"raw": block.arguments}
            calls.append(json.dumps([name, arguments], sort_keys=True, default=str))
    if not calls:
        return None
    key, count = Counter(calls[-LOOP_WINDOW:]).most_common(1)[0]
    if count < LOOP_REPEATS:
        return None
    tool, arguments = json.loads(key)
    return {"reason": "repeated_action", "count": count, "tool": tool, "arguments": arguments}
