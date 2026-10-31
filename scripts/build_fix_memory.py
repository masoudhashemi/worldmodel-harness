"""Build the error -> fix memory from saved trajectories (no grades are read).

Usage: build_fix_memory.py <out.json> <exclude-tasks.json> <run-dir> [<run-dir> ...]
Trajectories of tasks listed in exclude-tasks.json are skipped, so the memory can be
evaluated on those tasks.
"""

import json
import sys
from pathlib import Path

from worldmodel_harness.fixes import MIN_DEAD_ENDS, MIN_SUPPORT, FixMemory


def api_fetches(path: Path) -> list[tuple[dict, str, bool]]:
    """(arguments, raw response, success) for every api_fetch call, in order."""
    trajectory = json.loads(path.read_text())
    if not isinstance(trajectory, list) or not trajectory or not isinstance(trajectory[0], list):
        return []
    calls, fetches = {}, []
    for message in (m for group in trajectory for m in group if isinstance(m, dict)):
        if message.get("role") == "assistant":
            for block in message.get("blocks") or []:
                if block.get("kind") == "tool_call":
                    calls[block.get("tool_call_id")] = block
        elif message.get("role") == "tool" and message.get("name") == "api_fetch":
            call = calls.get(message.get("tool_call_id"))
            try:
                arguments = json.loads(call["arguments"]) if call else None
            except ValueError:
                arguments = None
            if isinstance(arguments, dict):
                # Drop any fix-memory note so learned errors match the environment's text.
                content = str(message.get("content")).split("\n\n[Fix memory]", 1)[0]
                fetches.append((arguments, content, str(message.get("success")) == "True"))
    return fetches


def task_of(path: Path, run: Path) -> str:
    parts = path.relative_to(run).parts
    return next((p for p in parts if "." in p and not p.endswith(".json")), "")


def main():
    out, exclude = Path(sys.argv[1]), set(json.loads(Path(sys.argv[2]).read_text())["tasks"])
    memory, used = FixMemory(), 0
    for run in map(Path, sys.argv[3:]):
        for path in sorted(run.rglob("trajectory.json")):
            if task_of(path, run) in exclude:
                continue
            memory.learn(api_fetches(path))
            used += 1
    memory.save(out)
    fixes = sum(any(n >= MIN_SUPPORT for n in c.values()) for c in memory.fixes.values())
    routes = sum(
        (n := c.most_common(1)[0][1]) >= MIN_SUPPORT and 2 * n > sum(c.values()) for c in memory.routes.values()
    )
    dead = sum(n >= MIN_DEAD_ENDS and not r for (host, _), (n, r) in memory.outcomes.items() if host)
    print(f"{used} trajectories: {fixes} shape fixes, {routes} routes, {dead} dead ends")


if __name__ == "__main__":
    main()
