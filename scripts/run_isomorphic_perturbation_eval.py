#!/usr/bin/env python3
"""Isomorphic Task Transfer & Surface Perturbation Evaluation Harness.

Evaluates how empirical World Model arms handle isomorphic workflows where the
underlying solving process is structurally identical, but surface entities,
requester personas, resource identifiers, and filter thresholds have shifted.

Measures:
1. Task Completion / Partial Credit Score (out-of-distribution transfer success)
2. Memory Anchor Hazard Rate (propensity to hallucinate stale identifiers from prior tasks)
3. Discovery Budget Footprint (api_search calls)
4. Environment Interaction Footprint (api_fetch calls)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.knowledge import Scope
from worldmodel_harness.store import KnowledgeStore


# Poison literal signatures from original tasks that indicate memory poisoning
POISON_ANCHOR_PATTERNS = {
    "operations.contractor_badge_expiration_isomorphic": [
        "ss_badges",
        "ws_contractors",
        "bdg-",
        "wayne ellis",
        "fiona hart",
        "luis moreno",
        "\"security\"",
        "'security'",
    ],
    "operations.twilio_afterhours_incident_isomorphic": [
        "ss_incident_queue",
        "ss_oncall_schedule",
        "ws_queue",
        "ws_schedule",
        "alex rivera",
        "jordan smith",
        "sp_incidents",
        "\"key\": \"inc\"",
        "'inc'",
    ],
    "operations.asana_fire_drill_isomorphic": [
        "ss_ops_policy",
        "ws_email_rules",
        "proj_facilities",
        "sec_feb",
        "compliance",
        "facilities@company.example.com",
        "monthly fire drill checklist - february",
    ],
    "operations.monday_slack_inventory_isomorphic": [
        "ss_inventory",
        "ws_badges",
        "ws_badge_policy",
        "brd_ops",
        "itm_411",
        "badge access",
    ],
}


def audit_anchor_hazards(run_id: str | None, task_name: str) -> dict[str, Any]:
    """Scan the execution trace for stale literal anchors carried over from source tasks."""
    if not run_id:
        return {"anchor_hazard_hits": 0, "hazard_details": []}

    run_dir = Path("runs") / run_id
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"anchor_hazard_hits": 0, "hazard_details": []}

    try:
        with open(summary_path) as f:
            data = json.load(f)
    except Exception:
        return {"anchor_hazard_hits": 0, "hazard_details": []}

    results = data.get("results") or []
    if not results:
        return {"anchor_hazard_hits": 0, "hazard_details": []}

    patterns = POISON_ANCHOR_PATTERNS.get(task_name, [])
    hits = 0
    details = []

    for item in results[0].get("decision_audit", []):
        action = item.get("decision", {}).get("action", {})
        args_str = json.dumps(action.get("arguments", {})).lower()
        matched = [p for p in patterns if p in args_str]
        if matched:
            hits += 1
            details.append({
                "tool": action.get("tool"),
                "matched_anchors": matched,
                "url": action.get("arguments", {}).get("url") if isinstance(action.get("arguments"), dict) else None,
            })

    return {"anchor_hazard_hits": hits, "hazard_details": details}


async def run_trial(
    root: Path,
    task_name: str,
    arm: str,
    store_path: Path,
    max_turns: int = 35,
    max_env_calls: int = 35,
    timeout: int = 600,
    gated_graph: bool = False,
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

    run_id = report.get("run_id")
    hazard_info = audit_anchor_hazards(run_id, task_name)

    return {
        "task": task_name,
        "arm": arm,
        "run_id": run_id,
        "strict_eval": bool(res.get("task_completed_correctly") == 1),
        "partial_credit": res.get("partial_credit", 0.0),
        "turns_used": res.get("turns_taken") or len(res.get("messages", [])) // 2,
        "environment_calls": ledger.get("environment_calls", 0),
        "discovery_calls": purposes.get("discovery", 0),
        "anchor_hazard_hits": hazard_info["anchor_hazard_hits"],
        "hazard_details": hazard_info["hazard_details"],
        "error_type": res.get("error_type"),
        "duration_seconds": round(duration, 1),
    }


def print_scorecard(results: list[dict[str, Any]]):
    print("\n" + "=" * 105)
    print("ISOMORPHIC TASK PERTURBATION EVALUATION SCORECARD")
    print("=" * 105)
    print(
        f"{'Task':<42} | {'Arm':<15} | {'Score':<7} | {'Strict':<6} | "
        f"{'Disc':<5} | {'Env':<5} | {'Anchors':<8} | {'Time':<7}"
    )
    print("-" * 105)
    for r in results:
        t_short = r["task"].replace("operations.", "")
        strict_str = "PASS" if r.get("strict_eval") else "FAIL"
        print(
            f"{t_short:<42} | {r['arm']:<15} | {r['partial_credit']:<7.2f} | "
            f"{strict_str:<6} | {r['discovery_calls']:<5} | {r['environment_calls']:<5} | "
            f"{r['anchor_hazard_hits']:<8} | {r['duration_seconds']:<6.1f}s"
        )
    print("=" * 105)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "operations.contractor_badge_expiration_isomorphic",
            "operations.twilio_afterhours_incident_isomorphic",
            "operations.asana_fire_drill_isomorphic",
            "operations.monday_slack_inventory_isomorphic",
        ],
        help="Isomorphic tasks to evaluate",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["baseline", "with_wm_raw", "with_wm_gated"],
        choices=["baseline", "with_wm_raw", "with_wm_gated"],
        help="Evaluation arms to compare",
    )
    parser.add_argument(
        "--seed-store",
        type=Path,
        default=Path("runs/eval-isomorphic-transfer/stores/worldmodel_carried.sqlite"),
        help="Path to learned world model sqlite checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/eval-isomorphic-perturbations"),
        help="Output directory for evaluations",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds per trial")
    parser.add_argument("--max-turns", type=int, default=35, help="Maximum turns per trial")
    parser.add_argument("--resume", action="store_true", help="Resume completed trials")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"

    all_results: list[dict[str, Any]] = []
    completed_trials = set()
    if args.resume and summary_file.exists():
        try:
            cached = json.loads(summary_file.read_text())
            all_results = cached.get("results", [])
            for r in all_results:
                completed_trials.add((r["arm"], r["task"]))
        except Exception:
            pass

    for task_name in args.tasks:
        for arm in args.arms:
            if (arm, task_name) in completed_trials:
                print(f"[RESUME] Skipping completed trial: {arm} on {task_name}")
                continue

            print(f"\n{'='*75}")
            print(f">>> EVALUATING: {arm.upper()} on {task_name} <<<")
            print(f"{'='*75}")

            # Configure ephemeral store
            trial_store_path = output_dir / f"store_{arm}_{task_name}.sqlite"
            if trial_store_path.exists():
                trial_store_path.unlink()

            gated_graph = False
            if arm == "baseline":
                KnowledgeStore(trial_store_path).close()
            elif arm == "with_wm_raw":
                if args.seed_store.exists():
                    shutil.copy2(args.seed_store, trial_store_path)
                else:
                    KnowledgeStore(trial_store_path).close()
            elif arm == "with_wm_gated":
                gated_graph = True
                if args.seed_store.exists():
                    shutil.copy2(args.seed_store, trial_store_path)
                else:
                    KnowledgeStore(trial_store_path).close()

            metrics = await run_trial(
                root=Path.cwd(),
                task_name=task_name,
                arm=arm,
                store_path=trial_store_path,
                max_turns=args.max_turns,
                max_env_calls=35,
                timeout=args.timeout,
                gated_graph=gated_graph,
            )
            all_results.append(metrics)
            completed_trials.add((arm, task_name))
            summary_file.write_text(json.dumps({"results": all_results}, indent=2))
            print_scorecard(all_results)

    print(f"\nFinal results saved to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
