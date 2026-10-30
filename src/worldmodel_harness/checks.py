"""Check memory: which sources earlier runs read before a kind of write.

Learned from past trajectories only (tool calls, never grades). For each task we
record, per write endpoint, the read endpoints consulted before it and how many
distinct targets of each were read. Across tasks, a read that preceded the same
write in at least MIN_TASKS other tasks becomes a check. The note asks the agent to
consult those sources before writing: it adds exploration instead of replacing it.
"""

import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .advisory import _service, task_services

MIN_TASKS = 2
MAX_WRITES = 2
MAX_READS_PER_WRITE = 4

HEADER = (
    "Checks from earlier runs in this environment. Before writing, those runs also read "
    "the sources below; deciding details such as exclusions, policies and the right "
    "recipients were often there. Consult the ones that apply to this task before you "
    "write, even if you already have enough to act."
)

Resolve = Callable[[str, str], str | None]


def trajectory_calls(path: Path) -> list[dict]:
    """api_fetch arguments in call order from a saved Stirrup trajectory."""
    groups = json.loads(path.read_text())
    messages = groups[0] if groups and isinstance(groups[0], list) else groups
    calls = []
    for message in messages:
        for block in message.get("blocks") or ():
            if block.get("kind") != "tool_call" or block.get("name") != "api_fetch":
                continue
            try:
                arguments = json.loads(block.get("arguments") or "{}")
            except ValueError:
                continue
            if isinstance(arguments, dict) and arguments.get("url"):
                calls.append(arguments)
    return calls


def write_profile(calls: list[dict], resolve: Resolve) -> dict[str, dict[str, int]]:
    """For each write endpoint: read endpoints seen before it, with distinct targets read."""
    reads: dict[str, set[str]] = defaultdict(set)
    profile: dict[str, dict[str, int]] = defaultdict(dict)
    for call in calls:
        method = str(call.get("method") or "GET").upper()
        endpoint = resolve(method, str(call["url"]))
        if endpoint is None:
            continue
        if method == "GET":
            reads[endpoint].add(json.dumps([call["url"], call.get("params")], sort_keys=True))
            continue
        for read, targets in reads.items():
            if read != endpoint:
                profile[endpoint][read] = max(profile[endpoint].get(read, 0), len(targets))
    return profile


def learn_checks(trajectories: dict[str, list[Path]], resolve: Resolve) -> dict:
    """Per task, the union over its runs of read-before-write profiles."""
    table = {}
    for task, paths in sorted(trajectories.items()):
        merged: dict[str, dict[str, int]] = defaultdict(dict)
        for path in paths:
            for write, reads in write_profile(trajectory_calls(path), resolve).items():
                for read, targets in reads.items():
                    merged[write][read] = max(merged[write].get(read, 0), targets)
        table[task] = {w: dict(sorted(r.items())) for w, r in sorted(merged.items())}
    return {"tasks": table}


def build_checks_note(table: dict, task_text: str, task_name: str) -> tuple[str, dict]:
    """Checks for the services the task names, learned from other tasks only."""
    writers: dict[str, set[str]] = defaultdict(set)
    seen: dict[tuple[str, str], list[int]] = defaultdict(list)
    for task, profile in table["tasks"].items():
        if task == task_name:
            continue
        for write, reads in profile.items():
            writers[write].add(task)
            for read, targets in reads.items():
                seen[(write, read)].append(targets)
    known = {_service(e) for e in writers} | {_service(r) for _, r in seen}
    services = set(task_services(task_text, known))
    writes = sorted(
        (w for w in writers if _service(w) in services), key=lambda w: (-len(writers[w]), w)
    )
    lines, chosen = [], {}
    for write in writes:
        reads = sorted(
            (
                (read, len(targets), statistics.median(targets))
                for (w, read), targets in seen.items()
                if w == write and len(targets) >= MIN_TASKS and _service(read) in services
            ),
            key=lambda item: (-item[1], item[0]),
        )[:MAX_READS_PER_WRITE]
        if not reads:
            continue
        parts = [
            f"{read} ({count} tasks{', often several targets' if median >= 2 else ''})"
            for read, count, median in reads
        ]
        lines.append(f"- before {write}: " + "; ".join(parts))
        chosen[write] = [read for read, _, _ in reads]
        if len(chosen) == MAX_WRITES:
            break
    note = HEADER + "\n" + "\n".join(lines) if lines else ""
    return note, {"services": sorted(services), "checks": chosen, "chars": len(note)}


GENERIC_CHECKS = (
    "Before you write or send anything, look for details that change the answer: "
    "instruction or policy emails, announcements in other channels, other tabs or files "
    "about the same data, and full records (for example the account) rather than only "
    "a list. In this environment the deciding detail is often in a second source."
)
