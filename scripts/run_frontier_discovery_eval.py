#!/usr/bin/env python3
"""Frontier Discovery Cold-Start Benchmark (Task 4 / Experiment E010).

Compares:
1. Cold-Start Baseline: Agent begins with empty store; must discover endpoints via api_search.
2. Frontier Discovered: Agent begins with pre-indexed catalog (full_frontier.sqlite);
   eliminates api_search discovery overhead (N -> 0) and directly queries endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from worldmodel_harness.automationbench import UPSTREAM_COMMIT
from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.frontier_discovery import discover_and_index_frontier
from worldmodel_harness.knowledge import Scope
from worldmodel_harness.store import KnowledgeStore


async def run_trial(
    root: Path,
    task_name: str,
    arm: str,
    store_path: Path,
    max_turns: int = 35,
    max_env_calls: int = 35,
    timeout: int = 450,
    gated_graph: bool = True,
) -> dict[str, Any]:
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
        resolve_execution_frontier=True,
        successful_transition_graph=True,
        phase="test",
        semantic_learning=False,
        adaptive=True,
        retrieval_chars=12000,
        gated_graph=gated_graph,
    )
    start_time = time.perf_counter()
    report = await run_automationbench(root, settings, store_path)
    duration = time.perf_counter() - start_time

    res = report["results"][0] if report.get("results") else {}
    ledger = res.get("ledger") or {}
    purposes = ledger.get("by_purpose") or {}

    store_claims = 0
    if store_path.exists():
        try:
            ks = KnowledgeStore(store_path)
            store_claims = len(ks.claims(None))
            ks.close()
        except Exception:
            pass

    return {
        "task": task_name,
        "arm": arm,
        "run_id": report.get("run_id"),
        "strict_eval": bool(res.get("task_completed_correctly") == 1),
        "partial_credit": res.get("partial_credit", 0.0),
        "turns_used": res.get("turns_taken") or len(res.get("messages", [])) // 2,
        "environment_calls": ledger.get("environment_calls", 0),
        "discovery_calls": purposes.get("discovery", 0),
        "store_claims": store_claims,
        "error_type": res.get("error_type"),
        "duration_seconds": round(duration, 1),
    }


def print_scorecard(results: list[dict[str, Any]]):
    print("\n" + "=" * 105)
    print("FRONTIER DISCOVERY COLD-START EVALUATION SCORECARD")
    print("=" * 105)
    print(
        f"{'Task':<40} | {'Arm':<24} | {'Score':<7} | {'Strict':<6} | "
        f"{'Disc Calls':<10} | {'Env Calls':<10} | {'Time':<7}"
    )
    print("-" * 105)
    for r in results:
        t_short = r["task"].replace("operations.", "")
        strict_str = "PASS" if r.get("strict_eval") else "FAIL"
        print(
            f"{t_short:<40} | {r['arm']:<24} | {r['partial_credit']:<7.2f} | "
            f"{strict_str:<6} | {r['discovery_calls']:<10} | {r['environment_calls']:<10} | "
            f"{r['duration_seconds']:<6.1f}s"
        )
    print("=" * 105)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="operations.pipefy_slack_purchase_request",
        help="Cold-start task to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/eval-task4-frontier-benchmark"),
        help="Output directory for evaluations",
    )
    parser.add_argument("--timeout", type=int, default=450, help="Per-trial timeout in seconds")
    parser.add_argument("--max-turns", type=int, default=35, help="Maximum turns per trial")
    parser.add_argument("--resume", action="store_true", help="Resume completed trials")
    args = parser.parse_args()

    output_dir = args.output_dir
    stores_dir = output_dir / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"

    # Step 1: Pre-index Full Frontier Store
    frontier_seed = stores_dir / "full_frontier.sqlite"
    if not frontier_seed.exists():
        print(f"Indexing full frontier into {frontier_seed}...")
        store = KnowledgeStore(frontier_seed)
        scope = Scope(environment="automationbench", version=UPSTREAM_COMMIT)
        discover_and_index_frontier(store, scope)
        store.close()
        print(f"Frontier indexed successfully.")

    all_results: list[dict[str, Any]] = []
    completed_trials = set()
    if args.resume and summary_file.exists():
        try:
            cached = json.loads(summary_file.read_text())
            all_results = cached.get("results", [])
            for r in all_results:
                completed_trials.add(r["arm"])
        except Exception:
            pass

    arms = [
        ("cold_start_baseline", None),
        ("frontier_discovered", frontier_seed),
    ]

    for arm_name, seed_path in arms:
        if arm_name in completed_trials:
            print(f"[RESUME] Skipping completed arm: {arm_name}")
            continue

        print(f"\n{'='*75}")
        print(f">>> EVALUATING: {arm_name.upper()} on {args.task} <<<")
        print(f"{'='*75}")

        trial_store_path = stores_dir / f"store_{arm_name}.sqlite"
        if trial_store_path.exists():
            trial_store_path.unlink()

        if seed_path and seed_path.exists():
            print(f"Seeding store from {seed_path}...")
            shutil.copy2(seed_path, trial_store_path)
        else:
            print("Starting from empty store...")
            KnowledgeStore(trial_store_path).close()

        metrics = await run_trial(
            root=Path.cwd(),
            task_name=args.task,
            arm=arm_name,
            store_path=trial_store_path,
            max_turns=args.max_turns,
            max_env_calls=35,
            timeout=args.timeout,
            gated_graph=True,
        )
        all_results.append(metrics)
        completed_trials.add(arm_name)
        summary_file.write_text(json.dumps({"results": all_results}, indent=2))
        print_scorecard(all_results)

    print(f"\nFinal evaluation saved to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
