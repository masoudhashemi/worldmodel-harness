"""Iterative World Model Evaluation across multiple tasks per domain.

Protocol:
1. Baseline (WITHOUT_WM): Evaluates each task once from a fresh, empty knowledge store.
2. World Model (WITH_WM): Evaluates each task over N iterations (default: 3), carrying
   and reusing the persistent knowledge store across tasks and iterations to measure
   whether the learned World Model accumulates knowledge and improves execution.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from worldmodel_harness.checkpoint import copy_checkpoint
from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.store import KnowledgeStore

DEFAULT_TASKS = {
    "operations": [
        "operations.contractor_badge_expiration",
        "operations.asana_fire_drill",
        "operations.sheets_monday_maintenance_queue",
        "operations.pipefy_slack_purchase_request",
        "operations.twilio_afterhours_incident",
    ],
    "sales": [
        "sales.multi_hop_lookup",
        "sales.negative_selection",
        "sales.recency_selection",
        "sales.priority_selection",
        "sales.format_ambiguity",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["operations"],
        help="Domains to evaluate (e.g. operations, sales)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit list of task names (overrides domain defaults)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of learning iterations for the WITH_WM arm",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=35,
        help="Maximum turns per task trial",
    )
    parser.add_argument(
        "--max-env-calls",
        type=int,
        default=35,
        help="Maximum environment calls per task trial",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=450,
        help="Per-trial timeout in seconds",
    )
    parser.add_argument(
        "--seed-store",
        type=Path,
        default=Path("runs/focused-env-audit/store.sqlite"),
        help="Initial seed world model store (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/iterative-worldmodel-eval"),
        help="Output directory for reports and persistent databases",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline cold-start run and only execute WM iterations",
    )
    parser.add_argument(
        "--only-baseline",
        action="store_true",
        help="Run only the cold-start baseline and skip WM iterations",
    )
    parser.add_argument(
        "--no-semantic-learning",
        dest="semantic_learning",
        action="store_false",
        default=True,
        help="Disable semantic learning consolidation between episodes",
    )
    parser.add_argument(
        "--retrieval-chars",
        type=int,
        default=12000,
        help="Maximum character budget for injected operational memory",
    )
    parser.add_argument(
        "--gated-graph",
        action="store_true",
        help="Enable pre-flight feasibility and entity collision gating with discovery plateau cutoff",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run, reusing completed trials and the carried store",
    )
    return parser.parse_args()


async def run_trial(
    root: Path,
    task_name: str,
    arm: str,
    iteration: int,
    store_path: Path,
    max_turns: int,
    max_env_calls: int,
    timeout: int,
    semantic_learning: bool = False,
    retrieval_chars: int = 12000,
    gated_graph: bool = False,
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
        resolve_execution_frontier=True,
        successful_transition_graph=True,
        phase="test",  # Enables learning from trajectories into the store
        semantic_learning=semantic_learning,
        adaptive=True,
        retrieval_chars=retrieval_chars,
        gated_graph=gated_graph,
    )
    start_time = time.perf_counter()
    report = await run_automationbench(root, settings, store_path)
    duration = time.perf_counter() - start_time

    res = report["results"][0] if report.get("results") else {}
    ledger = res.get("ledger") or {}
    purposes = ledger.get("by_purpose") or {}

    # Inspect store statistics
    store_claims = 0
    store_episodes = 0
    if store_path.exists():
        try:
            ks = KnowledgeStore(store_path)
            store_claims = len(ks.claims(None))
            store_episodes = len(ks.historical_episodes(None))
            ks.close()
        except Exception:
            pass

    return {
        "task": task_name,
        "arm": arm,
        "iteration": iteration,
        "run_id": report.get("run_id"),
        "strict_eval": bool(res.get("task_completed_correctly") == 1),
        "partial_credit": res.get("partial_credit", 0.0),
        "validation_available": bool(res.get("validation_available")),
        "turns_used": res.get("turns_taken") or len(res.get("messages", [])) // 2,
        "environment_calls": ledger.get("environment_calls", 0),
        "discovery_calls": purposes.get("discovery", 0),
        "store_claims": store_claims,
        "store_episodes": store_episodes,
        "error_type": res.get("error_type"),
        "duration_seconds": round(duration, 1),
    }


def print_scorecard(results: list[dict]):
    print("\n" + "=" * 90)
    print("ITERATIVE WORLD MODEL EVALUATION SCORECARD")
    print("=" * 90)
    print(f"{'Task':<35} | {'Phase':<15} | {'Strict':<7} | {'Partial':<8} | {'Disc':<5} | {'Env':<5} | {'Claims':<6} | {'Time'}")
    print("-" * 90)
    for r in results:
        t_short = r["task"].split(".")[-1][:33]
        phase = "BASELINE" if r["arm"] == "baseline" else f"WM Iter {r['iteration']}"
        strict = "PASS" if r.get("strict_eval") else "FAIL"
        partial = f"{r.get('partial_credit', 0.0):.2f}"
        disc = str(r.get("discovery_calls", "-"))
        env = str(r.get("environment_calls", "-"))
        claims = str(r.get("store_claims", "-"))
        dur = f"{r.get('duration_seconds', 0.0)}s"
        print(f"{t_short:<35} | {phase:<15} | {strict:<7} | {partial:<8} | {disc:<5} | {env:<5} | {claims:<6} | {dur}")
    print("=" * 90)


async def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stores_dir = args.output_dir / "stores"
    stores_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / "summary.json"

    # Resolve task list
    if args.tasks:
        task_list = list(args.tasks)
    else:
        task_list = []
        for d in args.domains:
            task_list.extend(DEFAULT_TASKS.get(d, []))

    if not task_list:
        print("No tasks selected for evaluation.", file=sys.stderr)
        return

    print("=" * 75)
    print("ITERATIVE WORLD MODEL LEARNING EVALUATION")
    print(f"Tasks ({len(task_list)}): {task_list}")
    print(f"Model Profile: {os.environ.get('WORLDMODEL_MODEL_CONFIG', 'configs/model.toml')}")
    print(f"Iterations: {args.iterations}")
    print(f"Seed Store: {args.seed_store}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 75)

    all_results: list[dict] = []
    completed_trials: set[tuple[str, int, str]] = set()

    if args.resume and summary_file.exists():
        try:
            prior_data = json.loads(summary_file.read_text())
            for r in prior_data.get("results", []):
                # Keep trials that completed cleanly or weren't interrupted by cluster crashes
                err = r.get("error_type") or r.get("error")
                if not err or err in ("TimeoutError",):
                    all_results.append(r)
                    completed_trials.add((r["arm"], r["iteration"], r["task"]))
            print(f"\n>>> Resuming run. Found {len(completed_trials)} already completed trials. <<<")
        except Exception as e:
            print(f"Warning: could not parse existing summary file: {e}")

    # -------------------------------------------------------------
    # PHASE 1: Baseline (WITHOUT_WM) - Cold start once per task
    # -------------------------------------------------------------
    if not args.skip_baseline:
        print("\n>>> PHASE 1: COLD-START BASELINE (WITHOUT_WM) <<<")
        for task_name in task_list:
            if ("baseline", 0, task_name) in completed_trials:
                print(f"\n[BASELINE] Task: {task_name} (Already completed, skipping)")
                continue

            print(f"\n[BASELINE] Task: {task_name}")
            baseline_store = stores_dir / f"{task_name.replace('.', '_')}_baseline.sqlite"
            if baseline_store.exists():
                baseline_store.unlink()
            KnowledgeStore(baseline_store).close()

            try:
                metrics = await run_trial(
                    Path.cwd(),
                    task_name,
                    arm="baseline",
                    iteration=0,
                    store_path=baseline_store,
                    max_turns=args.max_turns,
                    max_env_calls=args.max_env_calls,
                    timeout=args.timeout,
                    semantic_learning=False,
                    retrieval_chars=args.retrieval_chars,
                    gated_graph=args.gated_graph,
                )
                all_results.append(metrics)
                completed_trials.add(("baseline", 0, task_name))
                print(
                    f"  -> Result: strict={metrics['strict_eval']} | "
                    f"partial={metrics['partial_credit']:.2f} | "
                    f"disc={metrics['discovery_calls']} | env={metrics['environment_calls']} | "
                    f"time={metrics['duration_seconds']}s"
                )
            except Exception as e:
                print(f"  -> ERROR in {task_name} baseline: {e}")
                all_results.append({
                    "task": task_name,
                    "arm": "baseline",
                    "iteration": 0,
                    "error": str(e),
                    "strict_eval": False,
                    "partial_credit": 0.0,
                })

            summary_file.write_text(json.dumps({"results": all_results}, indent=2))

        if args.only_baseline:
            print_scorecard(all_results)
            print(f"\nBaseline results saved to: {summary_file}")
            return

    # -------------------------------------------------------------
    # PHASE 2: Iterative World Model (WITH_WM) - Persistent Memory
    # -------------------------------------------------------------
    carried_store = stores_dir / "worldmodel_carried.sqlite"
    if not (args.resume and carried_store.exists()):
        if carried_store.exists():
            carried_store.unlink()
        if args.seed_store and args.seed_store.exists():
            print(f"\nSeeding initial World Model from {args.seed_store}...")
            copy_checkpoint(args.seed_store, carried_store)
        else:
            print("\nStarting World Model from fresh empty store...")
            KnowledgeStore(carried_store).close()
    else:
        print(f"\nReusing existing carried World Model from {carried_store}...")

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*75}")
        print(f">>> PHASE 2: WORLD MODEL ITERATION {iteration}/{args.iterations} <<<")
        print(f"{'='*75}")

        for task_name in task_list:
            if ("with_worldmodel", iteration, task_name) in completed_trials:
                print(f"\n[WM ITERATION {iteration}] Task: {task_name} (Already completed, skipping)")
                continue

            print(f"\n[WM ITERATION {iteration}] Task: {task_name}")
            try:
                metrics = await run_trial(
                    Path.cwd(),
                    task_name,
                    arm="with_worldmodel",
                    iteration=iteration,
                    store_path=carried_store,
                    max_turns=args.max_turns,
                    max_env_calls=args.max_env_calls,
                    timeout=args.timeout,
                    semantic_learning=args.semantic_learning,
                    retrieval_chars=args.retrieval_chars,
                    gated_graph=args.gated_graph,
                )
                all_results.append(metrics)
                completed_trials.add(("with_worldmodel", iteration, task_name))
                print(
                    f"  -> Result: strict={metrics['strict_eval']} | "
                    f"partial={metrics['partial_credit']:.2f} | "
                    f"disc={metrics['discovery_calls']} | env={metrics['environment_calls']} | "
                    f"claims_in_store={metrics['store_claims']} | "
                    f"time={metrics['duration_seconds']}s"
                )
            except Exception as e:
                print(f"  -> ERROR in {task_name} WM Iter {iteration}: {e}")
                all_results.append({
                    "task": task_name,
                    "arm": "with_worldmodel",
                    "iteration": iteration,
                    "error": str(e),
                    "strict_eval": False,
                    "partial_credit": 0.0,
                })

            summary_file.write_text(json.dumps({"results": all_results}, indent=2))

    # Final summary scorecard
    print_scorecard(all_results)
    print(f"\nComplete results and learning trajectory saved to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
