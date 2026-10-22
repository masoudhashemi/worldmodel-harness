"""Compare agent performance on AutomationBench tasks with vs. without the learned Environment World Model.

Arms:
1. without_worldmodel: Fresh empty store (cold start; no pre-learned environment knowledge).
2. with_worldmodel: Pre-loaded store with verified environment claims (endpoints, schemas, flows).
"""

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from worldmodel_harness.checkpoint import copy_checkpoint
from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.store import KnowledgeStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/worldmodel-comparison"),
        help="Directory to save run results and databases",
    )
    parser.add_argument(
        "--worldmodel-store",
        type=Path,
        default=Path("runs/focused-env-audit/store.sqlite"),
        help="Path to pre-learned world model store",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "operations.contractor_badge_expiration",
            "operations.asana_fire_drill",
        ],
        help="List of task names to evaluate",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum turns per task trial",
    )
    parser.add_argument(
        "--max-env-calls",
        type=int,
        default=50,
        help="Maximum environment calls per task trial",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-trial timeout in seconds",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["without_worldmodel", "with_worldmodel"],
        help="Arms to evaluate: without_worldmodel, with_worldmodel",
    )
    return parser.parse_args()


async def run_trial(
    root: Path,
    task_name: str,
    arm: str,
    store_path: Path,
    max_turns: int,
    max_env_calls: int,
    timeout: int,
) -> dict:
    domain = task_name.split(".", 1)[0]
    settings = ExperimentSettings(
        domain=domain,
        task_names=(task_name,),
        limit=1,
        max_turns=max_turns,
        max_environment_calls=max_env_calls,
        timeout_seconds=timeout,
        memory="graph",
        operational=True,
        semantic_learning=False,  # Isolate static environment retrieval from online reflection
        adaptive=True,
    )
    start_time = time.perf_counter()
    report = await run_automationbench(root, settings, store_path)
    duration = time.perf_counter() - start_time

    res = report["results"][0] if report.get("results") else {}
    ledger = res.get("ledger") or {}
    purposes = ledger.get("by_purpose") or {}

    return {
        "task": task_name,
        "arm": arm,
        "run_id": report.get("run_id"),
        "strict_eval": bool(res.get("task_completed_correctly") == 1),
        "partial_credit": res.get("partial_credit", 0.0),
        "validation_available": bool(res.get("validation_available")),
        "turns_used": res.get("turns_taken") or len(res.get("messages", [])) // 2,
        "environment_calls": ledger.get("environment_calls", 0),
        "discovery_calls": purposes.get("discovery", 0),
        "error_type": res.get("error_type"),
        "duration_seconds": round(duration, 1),
    }


async def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / "summary.json"

    if not args.worldmodel_store.exists():
        raise FileNotFoundError(f"World model store not found at {args.worldmodel_store}")

    arms = args.arms
    results = []

    print("=" * 75)
    print("WORLD MODEL COMPARISON EVALUATION")
    print(f"Tasks: {args.tasks}")
    print(f"Model Profile: {os.environ.get('WORLDMODEL_MODEL_CONFIG', 'configs/model.toml')}")
    print(f"World Model Store: {args.worldmodel_store}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 75)

    for task_name in args.tasks:
        for arm in arms:
            print(f"\n[RUNNING] Task: {task_name} | Arm: {arm}")
            trial_store = args.output_dir / f"{task_name.replace('.', '_')}_{arm}.sqlite"

            if trial_store.exists():
                trial_store.unlink()

            if arm == "with_worldmodel":
                copy_checkpoint(args.worldmodel_store, trial_store)
            else:
                KnowledgeStore(trial_store).close()

            try:
                metrics = await run_trial(
                    Path.cwd(),
                    task_name,
                    arm,
                    trial_store,
                    args.max_turns,
                    args.max_env_calls,
                    args.timeout,
                )
                results.append(metrics)
                print(
                    f"  -> Result: strict_eval={metrics['strict_eval']} | "
                    f"partial={metrics['partial_credit']} | "
                    f"env_calls={metrics['environment_calls']} | "
                    f"discovery_calls={metrics['discovery_calls']} | "
                    f"time={metrics['duration_seconds']}s"
                )
            except Exception as e:
                print(f"  -> ERROR during trial: {e}")
                results.append({
                    "task": task_name,
                    "arm": arm,
                    "error": str(e),
                    "strict_eval": False,
                    "partial_credit": 0.0,
                })

            summary_file.write_text(json.dumps({"results": results}, indent=2))

    # Print summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY SCORECARD")
    print("=" * 80)
    print(f"{'Task':<40} | {'Arm':<20} | {'Strict':<7} | {'Partial':<8} | {'Disc':<5} | {'Env':<5}")
    print("-" * 80)
    for r in results:
        task_short = r["task"].split(".")[-1][:38]
        arm_short = "WITH_WM" if r["arm"] == "with_worldmodel" else "WITHOUT_WM"
        strict = "PASS" if r.get("strict_eval") else "FAIL"
        partial = f"{r.get('partial_credit', 0.0):.2f}"
        disc = str(r.get("discovery_calls", "-"))
        env = str(r.get("environment_calls", "-"))
        print(f"{task_short:<40} | {arm_short:<20} | {strict:<7} | {partial:<8} | {disc:<5} | {env:<5}")
    print("=" * 80)
    print(f"Full report saved to {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
