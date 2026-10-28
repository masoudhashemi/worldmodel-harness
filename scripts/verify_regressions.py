#!/usr/bin/env python3
"""Run verification trials on specific regressed tasks after harness fixes."""

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from worldmodel_harness.experiment import ExperimentSettings, run_automationbench


async def test_task(
    root: Path,
    task_name: str,
    store_path: Path,
    output_dir: Path,
    max_turns: int = 50,
    timeout: int = 1200,
) -> dict:
    domain = task_name.split(".", 1)[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    task_out = output_dir / "trajectories" / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    settings = ExperimentSettings(
        domain=domain,
        task_names=(task_name,),
        limit=1,
        max_turns=max_turns,
        timeout_seconds=timeout,
        memory="graph",
        operational=True,
        resolve_execution_frontier=True,
        successful_transition_graph=True,
        phase="test",
        semantic_learning=False,
        adaptive=True,
        retrieval_chars=12000,
        gated_graph=False,
    )

    print(f"\n{'='*70}\n[VERIFY] Testing task: {task_name}\n{'='*70}")
    t0 = time.perf_counter()
    report = await run_automationbench(root, settings, store_path, output_dir=output_dir)
    elapsed = time.perf_counter() - t0

    res = report["results"][0] if report.get("results") else {}
    strict = res.get("task_completed_correctly", False)
    partial = res.get("partial_credit", 0.0)
    grading = res.get("grading_details", [])
    err = res.get("error_type")

    print(f"\n[RESULT] {task_name}: strict={strict} | partial={partial:.3f} | err={err} | time={elapsed:.1f}s")
    for g in grading:
        status = "EXCLUDED" if g.get("excluded") else ("PASS" if g.get("passed") else "FAIL")
        print(f"  [{status}] {g.get('type')}: {g.get('params')}")

    out_record = {
        "task": task_name,
        "strict_eval": bool(strict),
        "partial_credit": float(partial) if partial is not None else 0.0,
        "grading_details": grading,
        "error_type": err,
        "duration_seconds": elapsed,
    }
    return out_record


async def main():
    root = Path.cwd()
    target_tasks = sys.argv[1:] or [
        "marketing.content_localization",
        "operations.airtable_gmail_visitor_followup",
        "finance.invoice_file_organization",
    ]

    out_dir = root / "runs" / "regression-verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a copy of the learned carried store
    source_store = root / "runs" / "iterative-worldmodel-mix-eval" / "stores" / "worldmodel_carried.sqlite"
    test_store = out_dir / "store.sqlite"
    if source_store.exists():
        print(f"Copying learned store {source_store} -> {test_store}")
        shutil.copy2(source_store, test_store)
    else:
        print("Carried store not found, using empty store")

    results = []
    for task in target_tasks:
        rec = await test_task(root, task, test_store, out_dir)
        results.append(rec)
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    print("\n" + "="*70)
    print("REGRESSION VERIFICATION SUMMARY")
    print("="*70)
    for r in results:
        print(f"  {r['task']}: strict={r['strict_eval']} | partial={r['partial_credit']:.3f} | err={r['error_type']} | {r['duration_seconds']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
