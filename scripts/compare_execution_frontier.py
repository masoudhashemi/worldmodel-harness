"""Compare two immutable source snapshots on identical starting knowledge.

Each attempt gets a disposable checkpoint copy. No learning carries between trials.
Order is counterbalanced across repetitions; model randomness is not seed-controlled.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def source_hash(path):
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*.py")):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--fixed-source", type=Path)
    parser.add_argument(
        "--factorial-source",
        type=Path,
        help="Run baseline, frontier-only, graph-only, and combined from one snapshot",
    )
    parser.add_argument("--resolve-frontier", action="store_true")
    parser.add_argument("--filter-transitions", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["operations.asana_fire_drill", "operations.contractor_badge_expiration"],
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--worker-task")
    args = parser.parse_args()
    if args.worker_task:
        from worldmodel_harness.experiment import ExperimentSettings, run_automationbench

        experimental = {}
        if args.resolve_frontier:
            experimental["resolve_execution_frontier"] = True
        if args.filter_transitions:
            experimental["successful_transition_graph"] = True
        settings = ExperimentSettings(
            **experimental,
            domain="operations",
            task_names=(args.worker_task,),
            limit=1,
            phase="checkpoint",
            memory="graph",
            operational=True,
            semantic_learning=False,
            adaptive=False,
            max_turns=30,
            max_environment_calls=30,
            timeout_seconds=450,
        )
        report = asyncio.run(run_automationbench(Path.cwd(), settings, args.checkpoint.resolve()))
        r = report["results"][0]
        retrievals = r.get("retrievals", [])
        signatures = []
        for audit in r.get("decision_audit", []):
            action = audit["decision"]["action"]
            signatures.append(json.dumps(action, sort_keys=True))
        from collections import Counter

        result = {
            "run_id": report["run_id"],
            "partial_credit": r.get("partial_credit"),
            "strict": r.get("task_completed_correctly"),
            "error_type": r.get("error_type"),
            "decisions": len(retrievals),
            "plans": sum(bool(x.get("workflow_plan_steps")) for x in retrievals),
            "wrapper_frontiers": sum(
                str(x.get("current_node", "")).startswith("tool_call:") for x in retrievals
            ),
            "endpoint_frontiers": sum(
                x.get("current_node") not in (None, "api_search")
                and not str(x.get("current_node")).startswith("tool_call:")
                for x in retrievals
            ),
            "repeated_actions": sum(n - 1 for n in Counter(signatures).values()),
            "blocked_actions": sum(not x.get("invoked") for x in r.get("decision_audit", [])),
            "ledger": r.get("ledger"),
            "model_settings": report.get("model_settings"),
            "token_usage": r.get("token_usage"),
        }
        args.output.write_text(json.dumps(result, indent=2))
        return
    if not args.model_config or not (
        args.factorial_source or (args.baseline_source and args.fixed_source)
    ):
        parser.error("Require --model-config and either --factorial-source or both source arms")
    profile = args.model_config.read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    profile_path = args.output / "model.toml"
    profile_path.write_bytes(profile)
    switches = {
        "baseline": [],
        "frontier_only": ["--resolve-frontier"],
        "graph_only": ["--filter-transitions"],
        "combined": ["--resolve-frontier", "--filter-transitions"],
    }
    if args.factorial_source:
        sources = {arm: args.factorial_source.resolve() for arm in switches}
    else:
        sources = {"baseline": args.baseline_source.resolve(), "fixed": args.fixed_source.resolve()}
        switches = {arm: [] for arm in sources}
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest = {
        "checkpoint_sha256": checkpoint_hash,
        "model_config_sha256": hashlib.sha256(profile).hexdigest(),
        "valid_comparison": False,
        "source_hashes": {k: source_hash(v) for k, v in sources.items()},
        "arm_switches": switches,
        "tasks": args.tasks,
        "repeats": args.repeats,
        "within_task_updates": "disposable, same starting store for every attempt",
        "results": [],
        "complete": False,
    }
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(manifest, indent=2))
    for repeat in range(args.repeats):
        for task in args.tasks:
            for arm in list(sources) if repeat % 2 == 0 else list(reversed(sources)):
                key = f"{repeat + 1}-{task}-{arm}"
                output = args.output / f"{key}.json"
                print(f"START {key}", flush=True)
                env = dict(
                    os.environ,
                    PYTHONPATH=str(sources[arm]),
                    WORLDMODEL_MODEL_CONFIG=str(profile_path.resolve()),
                )
                with (args.output / f"{key}.log").open("w") as log:
                    try:
                        result = subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker-task",
                                task,
                                "--checkpoint",
                                str(args.checkpoint.resolve()),
                                "--output",
                                str(output.resolve()),
                                *switches[arm],
                            ],
                            check=False,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=540,
                        )
                        record = (
                            json.loads(output.read_text())
                            if output.exists()
                            else {"error_type": "WorkerFailed", "returncode": result.returncode}
                        )
                    except subprocess.TimeoutExpired:
                        record = {"error_type": "WorkerTimeout"}
                record.update(arm=arm, repeat=repeat + 1, task=task)
                manifest["results"].append(record)
                assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
                assert source_hash(sources[arm]) == manifest["source_hashes"][arm]
                summary.write_text(json.dumps(manifest, indent=2))
                print(
                    f"DONE {key}: partial={record.get('partial_credit')} "
                    f"plans={record.get('plans')}/{record.get('decisions')} "
                    f"error={record.get('error_type')}",
                    flush=True,
                )
                if record.get("error_type") and not (record.get("ledger") or {}).get(
                    "environment_calls"
                ):
                    manifest["invalid_reason"] = "Trial failed before any environment action"
                    summary.write_text(json.dumps(manifest, indent=2))
                    return
    manifest["complete"] = True
    manifest["valid_comparison"] = all(not r.get("error_type") for r in manifest["results"])
    summary.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
