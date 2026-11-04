"""Episodes from saved Stirrup trajectories of AutomationBench runs."""

import json
from pathlib import Path

from ..core.trace import Episode, Step

# Text the harness added to tool results is not part of what the environment returned.
ADDED = ("\n\n[Experience]", "\n\n[Fix memory]")
# Calls that never reach the environment: the agent's finish tool, and calls the harness
# blocked for not matching the tool schema (repairing_agent.py).
HARNESS_TOOLS = ("finish",)


def reached(content: str) -> bool:
    """Whether a tool result came from the environment rather than the harness."""
    try:
        data = json.loads(content)
    except ValueError:
        return True
    return not (isinstance(data, dict) and data.get("environment_called") is False)


def episode(path: Path, task: str) -> Episode | None:
    """Every tool call of a saved trajectory that reached the environment, with its result,
    in order: the calls the advisor saw, so its log's step numbers index these steps."""
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
            content = str(message.get("content"))
            if isinstance(arguments, dict) and message.get("name") not in HARNESS_TOOLS and reached(content):
                for marker in ADDED:
                    content = content.split(marker, 1)[0]
                found.steps.append(Step(str(message.get("name")), arguments, content, str(message.get("success")) == "True"))
    return found


def task_of(path: Path, run: Path) -> str:
    parts = path.relative_to(run).parts
    return next((p for p in parts if "." in p and not p.endswith(".json")), "")


def trajectories(runs: list[Path], keep=lambda task: True) -> list[tuple[str, str]]:
    """(path, task) of every saved trajectory under the given run folders."""
    found = []
    for run in runs:
        for path in sorted(run.rglob("trajectory.json")):
            task = task_of(path, run)
            if task and keep(task):
                found.append((str(path), task))
    return found


def load(runs: list[Path], keep=lambda task: True) -> list[Episode]:
    """Episodes of every saved trajectory under the given run folders."""
    return episodes(trajectories(runs, keep))


def episodes(listed: list[tuple[str, str]]) -> list[Episode]:
    return [e for path, task in listed if (e := episode(Path(path), task)) and e.steps]
