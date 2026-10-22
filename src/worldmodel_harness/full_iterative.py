"""Repeated full-harness tasks with independently validated procedure updates.

Both arms learn from interactions; only store lifetime differs. One reset pass is
an economical reference, not an estimate of stochastic baseline variability.
"""
import json
from pathlib import Path

from .automationbench import dataset
from .checkpoint import copy_checkpoint
from .controlled_learning import validated_update
from .experiment import ExperimentSettings, run_automationbench
from .learning_curve import summarize_round
from .opportunities import execution_costs
from .store import KnowledgeStore


def schedule(tasks, iterations, reset_passes=1):
    if iterations < 1 or not 1 <= reset_passes <= iterations:
        raise ValueError("Require 1 <= reset passes <= learning iterations")
    return [(arm, iteration, index, task)
            for iteration in range(1, iterations + 1)
            for arm in (("reset", "learn") if iteration <= reset_passes else ("learn",))
            for index, task in enumerate(tasks)]


def update_cost(update):
    calls = update.get("model_call_costs", [])
    return {"reported_tokens": sum(sum(c.get("token_usage", {}).values()) for c in calls),
            "token_usage_complete": all(c.get("usage_available", False) for c in calls),
            "environment_calls": update.get("ledger", {}).get("environment_calls", 0)}


def curve(rows):
    rounds = []
    for arm, iteration in dict.fromkeys((r["arm"], r["iteration"]) for r in rows):
        group = [r for r in rows if (r["arm"], r["iteration"]) == (arm, iteration)]
        rounds.append({"arm": arm, "iteration": iteration, "tasks": len(group),
            "strict_accuracy": sum(r.get("metrics", {}).get("strict_success", 0) for r in group) / len(group),
            "reported_tokens": sum(r.get("metrics", {}).get("reported_tokens", 0)
                                   + update_cost(r.get("learning_update", {}))["reported_tokens"] for r in group),
            "token_usage_complete": all(r.get("metrics", {}).get("token_usage_complete", False)
                                        and update_cost(r.get("learning_update", {}))["token_usage_complete"] for r in group),
            "task_environment_calls": sum(r.get("metrics", {}).get("environment_calls", 0) for r in group),
            "validation_environment_calls": sum(update_cost(r.get("learning_update", {}))["environment_calls"] for r in group),
            "artifact_invocations": sum(r.get("metrics", {}).get("artifact_invocations", 0) for r in group),
            "adopted_updates": sum(r.get("learning_update", {}).get("deployment") == "active" for r in group),
            "execution_errors": sum(r.get("metrics", {}).get("execution_errors", 0) + bool(r.get("error_type")) for r in group),
            "update_errors": sum(bool(r.get("learning_update", {}).get("error_type")) for r in group)})
    return rounds


async def run_full_iterative(root: Path, output: Path, *, tasks, validation_names,
                             iterations=3, reset_passes=1, base: ExperimentSettings):
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Require distinct task names")
    if len(set(validation_names)) < 2 or len(validation_names) != len(set(validation_names)) or set(tasks) & set(validation_names):
        raise ValueError("Require at least two unique procedure-validation tasks disjoint from repeated tasks")
    runs = schedule(tasks, iterations, reset_passes)
    available = {json.loads(row["info"])["task_name"] for row in dataset(base.domain)}
    if not {*tasks, *validation_names} <= available:
        raise ValueError("Task or validation name is outside the selected domain")
    enabled = base.model_copy(update={"memory": "graph", "operational": True,
        "adaptive": True, "semantic_learning": True, "allow_experimental": False, "phase": "test"})
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    report = {"protocol": "full-iterative-v1", "complete": False, "tasks": tasks,
        "validation_tasks": validation_names, "iterations": iterations, "reset_passes": reset_passes,
        "settings": enabled.model_dump(mode="json"), "expected_trials": len(runs), "results": [],
        "interpretation": "Repeated in-sample tasks; grades are report-only. One reset pass does not quantify model variability. "
            "Both arms enable identical learning and validated adaptation; only knowledge persistence differs."}
    def save():
        report["curve"] = curve(report["results"])
        temporary = output / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "summary.json")
    save()
    carried = output / "stores/learn.sqlite"
    KnowledgeStore(carried).close()
    for arm, iteration, index, name in runs:
        path = carried if arm == "learn" else output / f"stores/reset-{iteration}-{index}.sqlite"
        if arm == "reset":
            KnowledgeStore(path).close()
        report["running"] = {"arm": arm, "iteration": iteration, "task": name}
        save()
        print(json.dumps({"starting": report["running"]}), flush=True)
        row = {"arm": arm, "iteration": iteration, "task": name, "knowledge_path": str(path)}
        try:
            settings = enabled.model_copy(update={"domain": name.split(".", 1)[0], "task_names": (name,), "limit": 1, "offset": 0})
            run = await run_automationbench(root, settings, path)
            row.update(run_id=run["run_id"], metrics=summarize_round(run))
            # Persist task results before potentially slow artifact proposal/validation.
            report["results"].append(row)
            save()
            row["learning_update"] = await validated_update(root, path, validation_names,
                domain=base.domain, horizon=len(tasks) * iterations, costs=execution_costs(run), max_calls=20)
        except Exception as exc:  # noqa: BLE001 -- retain failed calls, do not replay mutated worlds
            row["error_type"] = type(exc).__name__
            if row not in report["results"]:
                report["results"].append(row)
        store = KnowledgeStore(path)
        try:
            row["knowledge_after"] = {"claims": len(store.claims()), "procedures": len(store.procedures()),
                "active_procedures": sum(p.status == "active" for p in store.procedures())}
        finally:
            store.close()
        row["complete"] = True
        report.pop("running", None)
        if arm == "learn" and index == len(tasks) - 1:
            copy_checkpoint(path, output / f"checkpoints/iteration-{iteration}.sqlite")
        save()
        print(json.dumps({"finished": {"arm": arm, "iteration": iteration, "task": name,
              "metrics": row.get("metrics"), "update": row.get("learning_update", {}).get("deployment"),
              "error_type": row.get("error_type")}}), flush=True)
    report["complete"] = True
    save()
    return report
