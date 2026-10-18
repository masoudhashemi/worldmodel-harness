"""Bounded diagnostic from docs/small-sample-experiment-plan.md; no score inputs to learning."""

import asyncio
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from worldmodel_harness.automationbench import dataset
from worldmodel_harness.config import load_settings
from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.learning_curve import LearningCurveSettings, learn_from_round


async def main():
    root = Path.cwd()
    model = load_settings(root)
    if model.name != "openrouter/nvidia/nemotron-3-super-120b-a12b:free":
        raise ValueError("This diagnostic requires the explicit free-model profile")
    output = root / "runs" / ("small-sample-" + uuid4().hex)
    output.mkdir()
    names = (
        "simple.email_sf_contact_phone_update",
        "simple.email_sf_contact_email_update",
    )
    manifest = {
        "plan": "docs/small-sample-experiment-plan.md",
        "model": model.model_dump(),
        "task_names": names,
        "feedback_protocol": "trajectory-only",
        "complete": False,
        "arms": [],
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    print(json.dumps({"experiment": str(output)}), flush=True)
    common = {
        "domain": "simple",
        "task_names": names,
        "limit": 2,
        "phase": "online",
        "max_turns": 20,
        "max_environment_calls": 24,
        "timeout_seconds": 240,
        "semantic_timeout_seconds": 120,
    }
    empty_arms = 0
    for label, changes in (
        ("A-basic", {"memory": "none"}),
        ("B-operational", {"operational": True}),
        (
            "C-semantic",
            {
                "operational": True,
                "semantic_learning": True,
                "adaptive": True,
                "allow_experimental": True,
            },
        ),
        (
            "D-fresh-repeat",
            {
                "operational": True,
                "semantic_learning": True,
                "adaptive": True,
                "allow_experimental": True,
            },
        ),
        (
            "E-learned-repeat",
            {
                "operational": True,
                "semantic_learning": True,
                "adaptive": True,
                "allow_experimental": True,
            },
        ),
    ):
        knowledge = output / ("C-semantic.sqlite" if label.startswith("E-") else label + ".sqlite")
        if label.startswith("E-"):
            prototype = next(
                r for r in dataset("simple") if json.loads(r["info"])["task_name"] == names[0]
            )
            try:
                async with asyncio.timeout(120):
                    manifest["boundary_update"] = await learn_from_round(
                        root,
                        knowledge,
                        prototype,
                        LearningCurveSettings(
                            task_count=2, domain="simple", operational=True, semantic_learning=True
                        ),
                        horizon=2,
                    )
            except TimeoutError:
                manifest["boundary_update"] = {
                    "error_type": "TimeoutError",
                    "deployment": "inspect_store",
                    "timeout_seconds": 120,
                }
            save()
        manifest["current_arm"] = label
        save()
        print(json.dumps({"starting": label}), flush=True)
        report = await run_automationbench(root, ExperimentSettings(**common, **changes), knowledge)
        snapshot = output / (label + "-after.sqlite")
        with sqlite3.connect(knowledge) as source, sqlite3.connect(snapshot) as destination:
            source.backup(destination)
        manifest["arms"].append(
            {
                "label": label,
                "run_id": report["run_id"],
                "knowledge": str(knowledge.relative_to(root)),
                "snapshot": str(snapshot.relative_to(root)),
            }
        )
        save()
        print(
            json.dumps(
                {
                    "finished": label,
                    "run_id": report["run_id"],
                    "grade": report["mean_strict_success"],
                    "errors": [r.get("error_type") for r in report["results"]],
                }
            ),
            flush=True,
        )
        empty_arms = (
            empty_arms + 1
            if not any(r["ledger"]["environment_calls"] for r in report["results"])
            else 0
        )
        if empty_arms >= 2:
            manifest["stop_reason"] = "two_consecutive_arms_without_environment_interaction"
            save()
            return
    manifest["complete"] = True
    manifest.pop("current_arm", None)
    save()


if __name__ == "__main__":
    asyncio.run(main())
