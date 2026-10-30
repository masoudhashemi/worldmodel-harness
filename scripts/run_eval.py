"""Baseline vs. advisory memory on AutomationBench.

1. Baseline: every task `--baseline-repeats` times with no memory. Agent runs are
   stochastic, so one run is not a stable reference. `--baseline-from` reuses the
   baseline trials of an earlier summary.json instead.
2. Advisory: every task `--iterations` times in order, with one store carried
   across all tasks and iterations (seeded from `--seed-store`). Each trial reads
   a short note from the store and adds what it observes back to it.

Results go to <output-dir>/summary.json as {"results": [...], "aggregates": {...}}.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from worldmodel_harness.checkpoint import copy_checkpoint
from worldmodel_harness.experiment import TrialSettings, run_trial
from worldmodel_harness.store import KnowledgeStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--task-file", type=Path, help='JSON {"tasks": [...]} or a list of task names')
    tasks.add_argument("--tasks", nargs="+", help="Task names, e.g. sales.zoom_calendar_conflict")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--baseline-from", type=Path, help="Reuse baseline trials from this summary.json")
    parser.add_argument("--iterations", type=int, default=3, help="Advisory passes over the task list")
    parser.add_argument("--seed-store", type=Path, help="Store to start the advisory memory from")
    parser.add_argument("--skip-advisory", action="store_true", help="Run only the baseline")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-env-calls", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1200, help="Seconds per trial")
    parser.add_argument("--resume", action="store_true", help="Keep completed trials and the store")
    return parser.parse_args()


def load_tasks(args) -> list[str]:
    if args.tasks:
        return list(args.tasks)
    raw = json.loads(args.task_file.read_text())
    return list(raw["tasks"] if isinstance(raw, dict) else raw)


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(rows: list[dict]) -> dict:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(bool(r.get("strict_eval")))
    return {
        "trials": len(rows),
        "tasks": len(by_task),
        "strict_rate": _mean(float(bool(r.get("strict_eval"))) for r in rows),
        "mean_partial_credit": _mean(r.get("partial_credit", 0.0) for r in rows),
        "mean_environment_calls": _mean(r.get("environment_calls") for r in rows),
        "mean_discovery_calls": _mean(r.get("discovery_calls") for r in rows),
        "mean_model_calls": _mean(r.get("model_calls") for r in rows),
        "mean_input_tokens": _mean((r.get("token_usage") or {}).get("input") for r in rows),
        "mean_duration_seconds": _mean(r.get("duration_seconds") for r in rows),
        # pass^k: share of tasks solved on every one of their trials.
        "tasks_passing_every_trial": _mean(float(all(v)) for v in by_task.values()),
        "tasks_passing_any_trial": _mean(float(any(v)) for v in by_task.values()),
        "note_withdrawn_rate": _mean(float(bool(r.get("note_withdrawn"))) for r in rows if r["arm"] == "advisory"),
    }


def aggregate(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[f"{r['arm']}#{r['iteration']}"].append(r)
        if r["arm"] == "baseline":
            groups["baseline#pooled"].append(r)  # baseline repeats are replicates
    return {key: summarize(rows) for key, rows in sorted(groups.items())}


def print_aggregates(aggregates: dict) -> None:
    def fmt(value, spec):
        return format(value, spec) if value is not None else "-"

    print(f"\n{'group':<18} {'n':>4} {'strict':>7} {'pass^k':>7} {'partial':>8} "
          f"{'env':>6} {'turns':>6} {'in_tok':>8} {'secs':>6}")
    for key, a in aggregates.items():
        tokens = a["mean_input_tokens"] / 1000 if a["mean_input_tokens"] is not None else None
        print(f"{key:<18} {a['trials']:>4} {fmt(a['strict_rate'], '7.1%')} "
              f"{fmt(a['tasks_passing_every_trial'], '7.1%')} {fmt(a['mean_partial_credit'], '8.3f')} "
              f"{fmt(a['mean_environment_calls'], '6.1f')} {fmt(a['mean_model_calls'], '6.1f')} "
              f"{fmt(tokens, '7.0f')}k {fmt(a['mean_duration_seconds'], '6.0f')}")


async def trial(args, task: str, arm: str, iteration: int, store_path: Path | None) -> dict:
    settings = TrialSettings(
        task_name=task,
        advisory=arm == "advisory",
        max_turns=args.max_turns,
        max_environment_calls=args.max_env_calls,
        timeout_seconds=args.timeout,
    )
    output = args.output_dir / "trajectories" / task / f"{arm}-iter{iteration}"
    started = time.perf_counter()
    try:
        res = await run_trial(Path.cwd(), settings, output, store_path)
    except Exception as exc:  # noqa: BLE001 -- one failed trial must not end the run
        res = {"error_type": type(exc).__name__}
    ledger = res.get("ledger") or {}
    advisory = res.get("advisory") or {}
    return {
        "task": task,
        "arm": arm,
        "iteration": iteration,
        "strict_eval": res.get("task_completed_correctly") == 1,
        "partial_credit": res.get("partial_credit", 0.0),
        "grading_details": res.get("grading_details", []),
        "token_usage": res.get("token_usage", {}),
        "model_calls": len(res.get("model_call_costs", [])),
        "environment_calls": ledger.get("environment_calls", 0),
        "discovery_calls": (ledger.get("by_purpose") or {}).get("discovery", 0),
        "note_chars": advisory.get("chars"),
        "note_withdrawn": advisory.get("withdrawn"),
        "error_type": res.get("error_type"),
        "duration_seconds": round(time.perf_counter() - started, 1),
    }


async def main():
    args = parse_args()
    task_list = load_tasks(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / "summary.json"
    results: list[dict] = []
    if args.resume and summary_file.exists():
        results = json.loads(summary_file.read_text())["results"]
    if args.baseline_from is not None:
        reused = [
            r for r in json.loads(args.baseline_from.read_text())["results"]
            if r["arm"] == "baseline" and r["task"] in task_list
        ]
        have = {(r["arm"], r["iteration"], r["task"]) for r in results}
        results += [r for r in reused if ("baseline", r["iteration"], r["task"]) not in have]
        print(f"Reusing {len(reused)} baseline trials from {args.baseline_from}")
    done = {(r["arm"], r["iteration"], r["task"]) for r in results}

    async def run(task, arm, iteration, store_path=None):
        if (arm, iteration, task) in done:
            return
        print(f"[{arm} #{iteration}] {task}", flush=True)
        row = await trial(args, task, arm, iteration, store_path)
        results.append(row)
        done.add((arm, iteration, task))
        print(f"  -> strict={row['strict_eval']} partial={row['partial_credit']:.2f} "
              f"env={row['environment_calls']} turns={row['model_calls']} "
              f"time={row['duration_seconds']}s note={row['note_chars']} "
              f"withdrawn={bool(row['note_withdrawn'])} error={row['error_type']}", flush=True)
        summary_file.write_text(json.dumps({"results": results, "aggregates": aggregate(results)}, indent=2))

    for repeat in range(args.baseline_repeats):
        for task in task_list:
            await run(task, "baseline", repeat)

    if not args.skip_advisory:
        store_path = args.output_dir / "advisory.sqlite"
        if not (args.resume and store_path.exists()):
            store_path.unlink(missing_ok=True)
            if args.seed_store is not None:
                copy_checkpoint(args.seed_store, store_path)
            else:
                KnowledgeStore(store_path).close()
        for iteration in range(1, args.iterations + 1):
            for task in task_list:
                await run(task, "advisory", iteration, store_path)

    aggregates = aggregate(results)
    summary_file.write_text(json.dumps({"results": results, "aggregates": aggregates}, indent=2))
    print_aggregates(aggregates)
    print(f"\nSaved {summary_file}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
