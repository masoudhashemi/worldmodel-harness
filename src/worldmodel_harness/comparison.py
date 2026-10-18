"""Matched frozen-memory Q4 pilot runner; not a significance claim from one stream."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .experiment import ExperimentSettings, run_automationbench


def paired_report(arms: dict[str, dict]) -> dict:
    expected = {"flat_fixed", "graph_fixed", "flat_adaptive", "graph_adaptive"}
    if set(arms) != expected:
        raise ValueError("Q4 requires all four representation/adaptation arms")
    reference = arms["flat_fixed"]
    for report in arms.values():
        for key in (
            "model_settings",
            "upstream_commit",
            "versions",
            "harness_revision",
            "feedback_protocol",
        ):
            if report.get(key) != reference.get(key):
                raise ValueError("Model, runtime, or benchmark version differs between arms")
        controls = {
            k: v for k, v in report.get("settings", {}).items() if k not in ("memory", "adaptive")
        }
        expected_controls = {
            k: v
            for k, v in reference.get("settings", {}).items()
            if k not in ("memory", "adaptive")
        }
        if controls != expected_controls:
            raise ValueError("Task protocol or resource budgets differ between arms")
    rows = {}
    for name, report in arms.items():
        tasks = [r["task_name"] for r in report["results"]]
        if len(set(tasks)) != len(tasks):
            raise ValueError("Duplicate task identities cannot be paired")
        rows[name] = {r["task_name"]: r for r in report["results"]}
    task_names = set(rows["flat_fixed"])
    if not task_names or any(set(arm) != task_names for arm in rows.values()):
        raise ValueError("All arms must complete the same task selection")
    # Each delta is descriptive. Tasks within one learned stream are not independent replicas.
    paired = []
    for task in sorted(task_names):
        values = {name: data[task] for name, data in rows.items()}

        def value(name, field, values=values):
            result = values[name]
            if "error_type" in result:
                raise ValueError("Execution errors make this comparison incomplete")
            if field == "success":
                return result["task_completed_correctly"]
            if field == "environment_calls":
                return result["ledger"][field]
            return sum(result["token_usage"].values())

        contrasts = {}
        for field in ("success", "environment_calls", "tokens"):
            flat = value("flat_adaptive", field) - value("flat_fixed", field)
            graph = value("graph_adaptive", field) - value("graph_fixed", field)
            contrasts[field] = {
                "flat_adaptation_delta": flat,
                "graph_adaptation_delta": graph,
                "interaction": graph - flat,
            }
        paired.append({"task_name": task, "contrasts": contrasts})
    return {
        "paired_tasks": paired,
        "artifact_invocations": {
            arm: sum(row.get("artifact_invocations", 0) for row in report["results"])
            for arm, report in arms.items()
        },
        "artifact_invocation_reporting_complete": all(
            "artifact_invocations" in row for report in arms.values() for row in report["results"]
        ),
        "inference": "descriptive_one_stream_no_significance_claim",
        "utility_comparison_complete": False,
        "remaining_costs": "Bootstrap, failed-provider billing, and adaptation maintenance must be included before claiming net value.",
    }


async def compare_automationbench(
    root: Path, settings: ExperimentSettings, knowledge_path: Path
) -> dict:
    if settings.phase != "frozen":
        raise ValueError("Matched Q4 pilot uses frozen shared knowledge")
    if not knowledge_path.is_file():
        raise ValueError("A shared development snapshot is required")
    output = root / "runs" / uuid4().hex
    output.mkdir(parents=True)
    before = hashlib.sha256(knowledge_path.read_bytes()).hexdigest()
    arms = {}
    report = {
        "kind": "q4-four-arm-pilot",
        "run_id": output.name,
        "knowledge_sha256": before,
        "arm_run_ids": {},
        "complete": False,
    }
    try:
        for name, memory, adaptive in (
            ("flat_fixed", "flat", False),
            ("graph_fixed", "graph", False),
            ("flat_adaptive", "flat", True),
            ("graph_adaptive", "graph", True),
        ):
            arm_settings = settings.model_copy(update={"memory": memory, "adaptive": adaptive})
            arms[name] = await run_automationbench(root, arm_settings, knowledge_path)
            report["arm_run_ids"][name] = arms[name]["run_id"]
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            if hashlib.sha256(knowledge_path.read_bytes()).hexdigest() != before:
                raise RuntimeError("Shared frozen knowledge changed between arms")
        report.update(paired_report(arms))
        report["complete"] = True
        return report
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
