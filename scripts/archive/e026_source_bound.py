"""E026 offline go/no-go: would a learned source hint be available when a search loop starts?

For every trial with a streak of >= 3 word-for-word repeated search turns (E022 all arms,
E024 both arms), build the graph from earlier runs only (E022 trials: runs before E022;
E024 trials: everything but E024), replay the trial's fetches up to its first repeated
search turn, and ask ExperienceAdvisor.source_hint what it would add. Diagnosis only
(grades never reach the graph): is a value the grader expected in a failed check held by
the service the hint names, in the task's initial state?
"""

import json
import sys
from collections import Counter
from pathlib import Path

from worldmodel_harness.automationbench import task_row
from worldmodel_harness.experience import ExperienceAdvisor, Model, load
from worldmodel_harness.lifting import episode

sys.path.insert(0, str(Path(__file__).parent))
from e024_search_repeats import norm

RUNS = {"runs/e022-learning-stream": ["e022-learning-stream", "e024-search-repeats"],
        "runs/e024-search-repeats": ["e024-search-repeats"]}
SERVICES = {"gmail": "gmail", "sheets": "google_sheets", "drive": "google_drive", "slack": "slack"}


def first_repeat_calls(path: Path) -> int | None:
    """How many tool calls came before the first turn of a >= 3 repeated-search streak."""
    groups = json.loads(path.read_text())
    seen, calls, run, start = set(), 0, 0, None
    for m in (m for g in groups for m in g if isinstance(m, dict)):
        if m.get("role") != "assistant":
            continue
        cs = [b for b in m.get("blocks") or [] if b.get("kind") == "tool_call"]
        queries = []
        for b in cs:
            if b["name"] == "api_search":
                try:
                    queries.append(norm(json.loads(b["arguments"]).get("query")))
                except (TypeError, ValueError, AttributeError):
                    queries.append("")
        if cs and len(queries) == len(cs) and all(q in seen for q in queries):
            if run == 0:
                start = calls
            run += 1
            if run >= 3:
                return start
        elif cs:
            run = 0
        seen.update(queries)
        calls += len(cs)
    return None


def expected_values(result: dict) -> set[str]:
    out = set()
    for d in result.get("grading_details", []):
        if not d.get("passed"):
            for v in (d.get("params") or {}).values():
                if isinstance(v, str) and "@" in v:
                    out.add(v.lower())
    return out


def holds(task: str, read: str, values: set[str]) -> bool:
    state = json.loads(task_row(task)["info"])["initial_state"]
    service = next((s for k, s in SERVICES.items() if k in read.split(" ", 1)[-1].split("/", 1)[0]), None)
    blob = json.dumps(state.get(service, "")).lower() if service else ""
    return any(v in blob for v in values)


def main():
    stats = Counter()
    for run, exclude in RUNS.items():
        out = Path(run)
        others = [p for p in Path("runs").iterdir() if p.is_dir() and p.name not in exclude]
        graph = Model(load(others, lambda t: True))
        for row in json.loads((out / "summary.json").read_text())["results"]:
            trial = out / "trajectories" / row["task"] / f"{row['arm']}-r{row['round']}"
            n = first_repeat_calls(trial / "trajectory.json")
            if n is None:
                continue
            e = episode(trial / "trajectory.json", row["task"])
            advisor = ExperienceAdvisor(graph, e.query)
            for step in e.steps[:n]:
                if step.tool == "api_fetch":
                    advisor.observe(step.arguments, step.response, step.success)
            hint = advisor._source_hint()
            later = {s.action for s in e.steps[n:] if s.success}
            called = {s.action for s in advisor.steps if s.success}
            reads = [s[2] for s in graph.sources(advisor.template, called)] or [a for a, _, _ in graph.unmade(advisor.template, called)]
            kind = "source" if graph.sources(advisor.template, called) else "unmade" if reads else "-"
            values = expected_values(json.loads((trial / "result.json").read_text()))
            right = [r for r in reads if holds(row["task"], r, values)]
            stats["loops"] += 1
            stats["hint"] += bool(hint)
            stats[f"hint kind {kind}"] += 1
            stats["hint names a service holding an expected value"] += bool(right)
            stats["Gemma made a hinted read later anyway"] += any(r in later for r in reads)
            print(f"{run.split('/')[-1][:4]} {row['task']:42s} {row['arm']:9s} hint={kind:6s} "
                  f"reads={[r.split(' ', 1)[1][:55] for r in reads]} right={bool(right)}")
    print(dict(stats))


if __name__ == "__main__":
    main()
