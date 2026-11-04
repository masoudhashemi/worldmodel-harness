"""Same-day comparison of the baseline and the experience graph.

The arms run back to back per task, in an order rotated across tasks, for the given
number of rounds. The graph starts from every earlier run under runs/ (or, with
--unseen, from earlier runs outside these tasks' workflows), listed in <out>/history.json
when the run starts so a resume starts from the same history, and learns from each of its
own trials once both arms of a task have run, from tool calls and results only. In the
graph arm a learned policy decides which kinds of message each trial is shown
(core/policy.py); it learns from the graph arm's own trials of this run, each measured
against the same task's baseline trial. Results are appended to
<out>/summary.json, so an interrupted run resumes where it stopped.
Usage: run_arm_eval.py <out-dir> <tasks.json> [rounds] [arm,arm,...] [--unseen]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from worldmodel_harness.automationbench.domain import HttpDomain
from worldmodel_harness.automationbench.traces import episode, episodes, trajectories
from worldmodel_harness.core.graph import Graph, workflow
from worldmodel_harness.core.lifting import lift_query
from worldmodel_harness.core.policy import LearnedPolicy
from worldmodel_harness.experiment import TrialSettings, run_trial

ARMS = {
    "baseline": {},
    "graph": {"experience": True},
}
CONTROL = "baseline"  # the arm the learned policy's outcomes are measured against


def experience_graph(out: Path, tasks: list[str], rows: list[dict], arm: str, paired: bool, unseen: bool) -> tuple[Graph, LearnedPolicy]:
    """The graph at the start of the run, then the arm's own trials so far, in order (so a
    resumed run is the same), and the policy learned from those trials."""
    manifest = out / "history.json"
    if manifest.exists():
        history = json.loads(manifest.read_text())["trajectories"]
    else:
        flows = {workflow(t) for t in tasks}
        others = [p for p in Path("runs").iterdir() if p.is_dir() and p.resolve() != out.resolve()]
        history = trajectories(others, (lambda t: workflow(t) not in flows) if unseen else (lambda t: True))
        manifest.write_text(json.dumps({"trajectories": history}, indent=1))
    graph = Graph(HttpDomain(), episodes(history))
    policy = LearnedPolicy(graph.domain)
    done = {(r["task"], r["arm"], r["round"]) for r in rows}
    for row in rows:
        if row["arm"] == arm and (not paired or (row["task"], CONTROL, row["round"]) in done):
            learn(graph, policy, out, row["task"], row["round"], arm)
    return graph, policy


def learn(graph: Graph, policy: LearnedPolicy, out: Path, task: str, rnd: int, arm: str):
    """Add one trial to the graph, and to the policy against the task's control trial,
    from tool calls and results only."""
    folder = out / "trajectories" / task / f"{arm}-r{rnd}"
    found = episode(folder / "trajectory.json", task)
    if not found or not found.steps:
        return
    graph.add(found)
    other = out / "trajectories" / task / f"{CONTROL}-r{rnd}" / "trajectory.json"
    control = episode(other, task) if other.exists() else None
    result = json.loads((folder / "result.json").read_text()) if (folder / "result.json").exists() else {}
    log = result.get("experience", [])
    if result.get("advisor_calls", len(found.steps)) != len(found.steps):
        print(f"  (policy: {task} r{rnd} skipped, its trace and the advisor's calls differ)", flush=True)
        return
    if control:
        policy.learn(lift_query(found.query)[0], log, found.steps, control.steps)


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
        "policy": {kind: (d["show"], d["reason"]) for kind, d in (result.get("policy") or {}).items()},
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
    paired = CONTROL in names
    graph, policy = experience_graph(out, tasks, rows, learner, paired, "--unseen" in sys.argv) if learner else (None, None)
    for rnd in range(rounds):
        for index, task in enumerate(tasks):
            shift = (index + rnd) % len(names)
            ran = False
            for arm in names[shift:] + names[:shift]:
                if (task, arm, rnd) in done:
                    continue
                ran = True
                print(f"[{arm} r{rnd}] {task}", flush=True)
                started = time.perf_counter()
                settings = TrialSettings(task_name=task, trial=f"r{rnd}", **ARMS[arm])
                try:
                    result = await run_trial(Path.cwd(), settings, out / "trajectories" / task / f"{arm}-r{rnd}", graph=graph, policy=policy)
                except Exception as exc:  # noqa: BLE001 -- record and continue with the next trial
                    result = {"error_type": type(exc).__name__}
                row = _row(task, arm, rnd, result, time.perf_counter() - started)
                rows.append(row)
                print(
                    f"  -> strict={row['strict_eval']} partial={row['partial_credit']:.2f} "
                    f"env={row['environment_calls']} turns={row['model_calls']} shown={row['shown']} "
                    f"time={row['duration_seconds']}s error={row['error_type']}",
                    flush=True,
                )
                summary.write_text(json.dumps({"results": rows}, indent=2))
            if learner and ran:  # both arms of this task have now run
                learn(graph, policy, out, task, rnd, learner)


if __name__ == "__main__":
    asyncio.run(main())
