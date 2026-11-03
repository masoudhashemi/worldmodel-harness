"""Same-day comparison of the baseline and the experience graph.

The arms run back to back per task, in an order rotated across tasks, for the given
number of rounds. The graph starts from every earlier run under runs/ (or, with
--unseen, from earlier runs outside these tasks' workflows) and learns from each of its
own trials as it finishes, from tool calls and results only. Results are appended to
<out>/summary.json, so an interrupted run resumes where it stopped.
Usage: run_arm_eval.py <out-dir> <tasks.json> [rounds] [arm,arm,...] [--unseen]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from worldmodel_harness.automationbench.domain import HttpDomain
from worldmodel_harness.automationbench.traces import episode, load
from worldmodel_harness.core.graph import Graph, workflow
from worldmodel_harness.experiment import TrialSettings, run_trial

ARMS = {
    "baseline": {},
    "graph": {"experience": True},
}


def experience_graph(out: Path, tasks: list[str], rows: list[dict], arm: str, unseen: bool) -> Graph:
    """The graph at the start of the run, then the arm's own trials so far, in order (so a
    resumed run is the same)."""
    flows = {workflow(t) for t in tasks}
    others = [p for p in Path("runs").iterdir() if p.is_dir() and p.resolve() != out.resolve()]
    graph = Graph(HttpDomain(), load(others, (lambda t: workflow(t) not in flows) if unseen else (lambda t: True)))
    for row in rows:
        if row["arm"] == arm:
            learn(graph, out, row["task"], row["round"], arm)
    return graph


def learn(graph: Graph, out: Path, task: str, rnd: int, arm: str):
    """Add one trial to the graph, from its tool calls and results only."""
    found = episode(out / "trajectories" / task / f"{arm}-r{rnd}" / "trajectory.json", task)
    if found and found.steps:
        graph.add(found)


def _row(task: str, arm: str, rnd: int, result: dict, seconds: float) -> dict:
    shown = [e for e in result.get("experience", []) if e["shown"]]
    return {
        "task": task,
        "arm": arm,
        "round": rnd,
        "strict_eval": result.get("task_completed_correctly") == 1,
        "partial_credit": result.get("partial_credit", 0.0),
        "failed_checks": [
            d["type"] + ":" + json.dumps(d.get("params"))[:120]
            for d in result.get("grading_details", [])
            if not d.get("passed")
        ],
        "environment_calls": (result.get("ledger") or {}).get("environment_calls", 0),
        "model_calls": len(result.get("model_call_costs", [])),
        "input_tokens": (result.get("token_usage") or {}).get("input"),
        "shown": {kind: sum(e["kind"] == kind for e in shown) for kind in dict.fromkeys(e["kind"] for e in shown)},
        "error_type": result.get("error_type"),
        "duration_seconds": round(seconds, 1),
    }


async def main():
    args = [a for a in sys.argv[1:] if a != "--unseen"]
    out, tasks = Path(args[0]), json.loads(Path(args[1]).read_text())["tasks"]
    rounds = int(args[2]) if len(args) > 2 else 1
    names = args[3].split(",") if len(args) > 3 else ["baseline", "graph"]
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    rows = json.loads(summary.read_text())["results"] if summary.exists() else []
    done = {(r["task"], r["arm"], r["round"]) for r in rows}
    learner = next((n for n in names if ARMS[n].get("experience")), None)
    graph = experience_graph(out, tasks, rows, learner, "--unseen" in sys.argv) if learner else None
    for rnd in range(rounds):
        for index, task in enumerate(tasks):
            shift = (index + rnd) % len(names)
            for arm in names[shift:] + names[:shift]:
                if (task, arm, rnd) in done:
                    continue
                print(f"[{arm} r{rnd}] {task}", flush=True)
                started = time.perf_counter()
                settings = TrialSettings(task_name=task, **ARMS[arm])
                try:
                    result = await run_trial(Path.cwd(), settings, out / "trajectories" / task / f"{arm}-r{rnd}", graph=graph)
                except Exception as exc:  # noqa: BLE001 -- record and continue with the next trial
                    result = {"error_type": type(exc).__name__}
                row = _row(task, arm, rnd, result, time.perf_counter() - started)
                rows.append(row)
                if arm == learner:
                    learn(graph, out, task, rnd, arm)
                print(
                    f"  -> strict={row['strict_eval']} partial={row['partial_credit']:.2f} "
                    f"env={row['environment_calls']} turns={row['model_calls']} shown={row['shown']} "
                    f"time={row['duration_seconds']}s error={row['error_type']}",
                    flush=True,
                )
                summary.write_text(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
