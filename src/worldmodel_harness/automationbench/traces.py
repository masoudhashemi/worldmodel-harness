"""Episodes from saved Stirrup trajectories of AutomationBench runs."""

import json
from pathlib import Path

from ..core.trace import Episode, Step

# Text the harness added to tool results is not part of what the environment returned.
ADDED = ("\n\n[Experience]", "\n\n[Fix memory]")


def episode(path: Path, task: str) -> Episode | None:
    """Every tool call of a saved trajectory with its result, in order."""
    try:
        groups = json.loads(path.read_text())
    except ValueError:
        return None
    if not isinstance(groups, list) or not groups or not isinstance(groups[0], list):
        return None
    messages = [m for group in groups for m in group if isinstance(m, dict)]
    query = next((str(m.get("content")) for m in messages if m.get("role") == "user"), "")
    found, calls = Episode(task, query), {}
    for message in messages:
        if message.get("role") == "assistant":
            for block in message.get("blocks") or []:
                if block.get("kind") == "tool_call":
                    calls[block.get("tool_call_id")] = block
        elif message.get("role") == "tool":
            call = calls.get(message.get("tool_call_id"))
            try:
                arguments = json.loads(call["arguments"]) if call else None
            except ValueError:
                arguments = None
            if isinstance(arguments, dict):
                content = str(message.get("content"))
                for marker in ADDED:
                    content = content.split(marker, 1)[0]
                found.steps.append(Step(str(message.get("name")), arguments, content, str(message.get("success")) == "True"))
    return found


def task_of(path: Path, run: Path) -> str:
    parts = path.relative_to(run).parts
    return next((p for p in parts if "." in p and not p.endswith(".json")), "")


def load(runs: list[Path], keep=lambda task: True) -> list[Episode]:
    """Episodes of every saved trajectory under the given run folders."""
    found = []
    for run in runs:
        for path in sorted(run.rglob("trajectory.json")):
            task = task_of(path, run)
            if task and keep(task) and (e := episode(path, task)) and e.steps:
                found.append(e)
    return found
