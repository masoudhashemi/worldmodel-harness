"""Same-day comparison of arms: baseline, checklist, ledger review, fix memory.

The chosen arms run back to back per task, in an order rotated across tasks, for the
given number of rounds. Results are appended to <out>/summary.json, so an interrupted
run resumes where it stopped.
Usage: run_arm_eval.py <out-dir> <tasks.json> [rounds] [arm,arm,...]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from worldmodel_harness.checks import GENERIC_CHECKS
from worldmodel_harness.experiment import TrialSettings, run_trial

FIX_MEMORY = "runs/fix-memory.json"
ARMS = {
    "baseline": {"note": None},
    "checklist": {"note": GENERIC_CHECKS},
    "checklist_review": {"note": GENERIC_CHECKS, "review": True},
    "fixes": {"note": None, "fixes": True, "fix_memory": FIX_MEMORY},
    # Adds route and dead-end edges to the shape fixes (the recovery graph).
    "graph": {"note": None, "fixes": True, "fix_memory": "runs/fix-memory-graph.json"},
    # E020's fixes arm: the old output limit and no unavailable-service note.
    "fixes_e020": {"note": None, "fixes": True, "fix_memory": FIX_MEMORY, "fix_unavailable": False, "max_tokens": 12000},
    # Shape fixes plus dead ends only: routes added nothing in E019.
    "dead_end": {"note": None, "fixes": True, "fix_memory": "runs/fix-memory-dead-end.json"},
}


def _row(task: str, arm: str, rnd: int, result: dict, seconds: float) -> dict:
    review = result.get("review") or {}
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
        "review_shown": bool(review.get("text")),
        "review_chars": len(review.get("text") or ""),
        "fix_notes": sum(bool(set(e["kinds"]) - {"loop"}) for e in result.get("fixes", [])),
        "loop_notes": sum("loop" in e["kinds"] for e in result.get("fixes", [])),
        "error_type": result.get("error_type"),
        "duration_seconds": round(seconds, 1),
    }


async def main():
    out, tasks = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["tasks"]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    rows = json.loads(summary.read_text())["results"] if summary.exists() else []
    done = {(r["task"], r["arm"], r["round"]) for r in rows}
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["baseline", "checklist", "checklist_review"]
    for rnd in range(rounds):
        for index, task in enumerate(tasks):
            shift = (index + rnd) % len(names)
            for arm in names[shift:] + names[:shift]:
                if (task, arm, rnd) in done:
                    continue
                print(f"[{arm} r{rnd}] {task}", flush=True)
                started = time.perf_counter()
                options = {k: v for k, v in ARMS[arm].items() if k != "note"}
                settings = TrialSettings(task_name=task, **options)
                try:
                    result = await run_trial(
                        Path.cwd(), settings, out / "trajectories" / task / f"{arm}-r{rnd}", note=ARMS[arm]["note"]
                    )
                except Exception as exc:  # noqa: BLE001 -- record and continue with the next trial
                    result = {"error_type": type(exc).__name__}
                row = _row(task, arm, rnd, result, time.perf_counter() - started)
                rows.append(row)
                print(
                    f"  -> strict={row['strict_eval']} partial={row['partial_credit']:.2f} "
                    f"env={row['environment_calls']} turns={row['model_calls']} review={row['review_shown']} "
                    f"fix_notes={row['fix_notes']} loop_notes={row['loop_notes']} "
                    f"time={row['duration_seconds']}s error={row['error_type']}",
                    flush=True,
                )
                summary.write_text(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
