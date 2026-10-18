"""Twelve sequential, isolated trials from a predeclared randomized schedule."""

import argparse
import asyncio
import json
import random
from pathlib import Path
from uuid import uuid4

from worldmodel_harness.checkpoint import checkpoint_hash, copy_checkpoint
from worldmodel_harness.config import load_settings
from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.reliability import reliability
from worldmodel_harness.store import KnowledgeStore


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--plan", default="docs/checkpoint-reliability-plan.md")
    parser.add_argument("--replicates", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Only the two learned-arm first-replicate trials; usability, not reliability",
    )
    args = parser.parse_args()
    root = Path.cwd()
    model = load_settings(root)
    if model.name != "openrouter/nvidia/nemotron-3-super-120b-a12b:free":
        raise ValueError("This predeclared experiment requires the free model")
    output = root / "runs" / ("checkpoint-reliability-" + uuid4().hex)
    output.mkdir()
    sources = {"empty": output / "empty.sqlite", "learned": output / "learned.sqlite"}
    baseline = json.loads(args.baseline_manifest.read_text()) if args.baseline_manifest else None
    original = (
        root / "runs/small-sample-bd095ff2395c485bbe451ecfe3be3881/E-learned-repeat-after.sqlite"
    )
    if baseline:
        if not baseline["complete"] or baseline["model"] != model.model_dump():
            raise ValueError("Baseline must be complete and use identical model settings")
        for arm in sources:
            source = root / baseline["sources"][arm]
            if checkpoint_hash(source) != baseline["source_hashes"][arm]:
                raise ValueError("Baseline source hash changed")
            # Use the exact immutable input; run_automationbench makes each
            # disposable backup. SQLite backups need not be byte-identical.
            sources[arm] = source
    else:
        KnowledgeStore(sources["empty"]).close()
        copy_checkpoint(original, sources["learned"])
    names = ["simple.email_sf_contact_phone_update", "simple.email_sf_contact_email_update"]
    rng = random.Random(1729)
    schedule = []
    for replicate in range(1, args.replicates + 1):
        task_order = list(names)
        rng.shuffle(task_order)
        for task in task_order:
            arms = list(sources)
            rng.shuffle(arms)
            schedule.extend({"arm": arm, "task": task, "replicate": replicate} for arm in arms)
    if args.probe:
        schedule = [
            item for item in schedule if item["arm"] == "learned" and item["replicate"] == 1
        ]
    manifest = {
        "plan": args.plan,
        "baseline_manifest": str(args.baseline_manifest) if baseline else None,
        "complete": False,
        "protocol": "in-sample-verification-usability-probe"
        if args.probe
        else "in-sample-fixed-checkpoint-reliability",
        "model": model.model_dump(),
        "task_names": names,
        "schedule": schedule,
        "results": [],
        "source_hashes": {arm: checkpoint_hash(path) for arm, path in sources.items()},
        "sources": {arm: str(path.relative_to(root)) for arm, path in sources.items()},
        "prior_learning_source": str(original.relative_to(root)),
        "semantic_learning": False,
        "procedures": False,
    }
    expected_schedule = [
        item for item in baseline["schedule"] if item["replicate"] <= args.replicates
    ] if baseline else schedule
    if args.probe:
        expected_schedule = [
            item
            for item in expected_schedule
            if item["arm"] == "learned" and item["replicate"] == 1
        ]
    if schedule != expected_schedule:
        raise ValueError("Trial schedule differs from baseline")

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    print(json.dumps({"experiment": str(output)}), flush=True)
    no_interactions = 0
    for item in schedule:
        manifest["current"] = item
        save()
        print(json.dumps({"starting": item}), flush=True)
        report = await run_automationbench(
            root,
            ExperimentSettings(
                domain="simple",
                task_names=(item["task"],),
                phase="checkpoint",
                operational=True,
                max_turns=20,
                max_environment_calls=24,
                timeout_seconds=240,
            ),
            sources[item["arm"]],
        )
        result = report["results"][0]
        row = {
            **item,
            "run_id": report["run_id"],
            "harness_revision": report["harness_revision"],
            "strict_success": result["task_completed_correctly"] == 1,
            "clean_finish": result["finished"] and "error_type" not in result,
            "checked": result.get("completion_evidence_status") == "checked",
            "error": result.get("error_type"),
            "environment_calls": result["ledger"]["environment_calls"],
            "model_calls": len(result["model_call_costs"]),
            "returned_tokens": sum(
                sum(c.get("token_usage", {}).values()) for c in result["model_call_costs"]
            ),
            "calls_without_usage": sum(
                not c["usage_available"] for c in result["model_call_costs"]
            ),
            "model_seconds": sum(c["elapsed_seconds"] for c in result["model_call_costs"]),
            "discovery_calls": result["ledger"].get("by_purpose", {}).get("discovery", 0),
            "blocked_calls": sum(not a["invoked"] for a in result.get("decision_audit", [])),
            "outer_schema_rejections": len(result.get("outer_validation_audit", [])),
            "goal_checks": result.get("goal_checks", []),
            "checkpoint_sha256": report["checkpoint"]["sha256"],
        }
        manifest["results"].append(row)
        for arm, source in sources.items():
            if checkpoint_hash(source) != manifest["source_hashes"][arm]:
                raise ValueError("Source checkpoint changed between trials")
        save()
        print(
            json.dumps({"finished": {k: v for k, v in row.items() if k != "goal_checks"}}),
            flush=True,
        )
        no_interactions = no_interactions + 1 if not row["environment_calls"] else 0
        if no_interactions >= 2:
            manifest["stop_reason"] = "two_consecutive_trials_without_environment_interaction"
            save()
            return
    manifest["metrics"] = {
        arm: {
            metric: reliability(
                {
                    name: [
                        r[metric]
                        for r in manifest["results"]
                        if r["arm"] == arm and r["task"] == name
                    ]
                    for name in names
                },
                trials=1 if args.probe else args.replicates,
            )
            for metric in ("strict_success", "clean_finish", "checked")
        }
        for arm in ({item["arm"] for item in schedule})
    }
    manifest["complete"] = True
    manifest.pop("current", None)
    save()
    print(json.dumps({"complete": True, "metrics": manifest["metrics"]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
