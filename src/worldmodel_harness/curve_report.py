"""Read-only paired reporting for repeated in-sample learning curves."""

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path


def runtime_diagnostics(tasks: list[dict]) -> dict:
    """Keep runtime failures in the denominator; returned usage is a lower bound."""
    calls = [c for t in tasks for c in t.get("model_call_costs", [])]
    return {
        "returned_call_tokens_lower_bound": sum(
            sum(c.get("token_usage", {}).values()) for c in calls
        ),
        "recorded_model_calls": len(calls),
        "model_calls_without_usage": sum(not c.get("usage_available", False) for c in calls),
        "call_telemetry_complete": all("model_call_costs" in t for t in tasks),
        "outcomes_by_execution_error": {
            error: {
                "tasks": len(group),
                "strict_passes": sum(t["task_completed_correctly"] for t in group),
            }
            for error in sorted({t.get("error_type", "none") for t in tasks})
            for group in [[t for t in tasks if t.get("error_type", "none") == error]]
        },
    }


def execution_diagnostics(tasks: list[dict]) -> dict:
    """Describe recorded execution, without equating retrieval or reuse with value."""
    purposes = Counter()
    for task in tasks:
        purposes.update(task["ledger"].get("by_purpose", {}))
    retrievals = [r for task in tasks for r in task.get("retrievals", [])]
    return {
        "environment_calls_by_purpose": dict(purposes),
        "purpose_accounting_complete": all(
            sum(t["ledger"].get("by_purpose", {}).values()) == t["ledger"]["environment_calls"]
            for t in tasks
        ),
        "recorded_retrieval_decisions": len(retrievals),
        "decisions_with_injected_claims": sum(bool(r.get("injected_claims")) for r in retrievals),
        "recorded_injected_chars": sum(r.get("injected_chars", 0) for r in retrievals),
        "artifact_invocations": sum(t.get("artifact_invocations", 0) for t in tasks),
        "information_invocations": sum(t.get("information_invocations", 0) for t in tasks),
        "invocation_accounting_complete": all(
            "artifact_invocations" in t and "information_invocations" in t for t in tasks
        ),
    }


def compare_completed_rounds(curve: dict, runs: dict[str, dict]) -> dict:
    names = tuple(curve["task_names"])
    if len(names) != len(set(names)) or len(names) > 50:
        raise ValueError("Expected at most 50 distinct tasks")
    rows = curve["rounds"]
    result = {
        "protocol": curve["protocol"],
        "unique_tasks": len(names),
        "completed_rounds": len(rows),
        "experiment_complete": curve["complete"],
        "rounds": [],
        "paired_changes": [],
        "generalization_evaluated": False,
        "causal_learning_benefit_established": False,
        "billing_complete": False,
    }
    indexed = []
    reference = None
    for row in rows:
        run = runs[row["run_id"]]
        tasks = {r["task_name"]: r for r in run["results"]}
        if len(tasks) != len(run["results"]) or set(tasks) != set(names):
            raise ValueError("Rounds must contain the same complete task set without duplicates")
        controls = {
            k: run.get(k)
            for k in (
                "settings",
                "model_settings",
                "upstream_commit",
                "versions",
                "harness_revision",
                "feedback_protocol",
            )
        }
        if reference is None:
            reference = controls
        elif controls != reference:
            raise ValueError("Run controls changed between rounds")
        indexed.append(tasks)
        usage_complete = all("token_usage" in t for t in tasks.values())
        update = row.get("learning_update", {})
        result["rounds"].append(
            {
                "round": row["round"],
                "run_id": row["run_id"],
                "strict_passes": sum(t["task_completed_correctly"] for t in tasks.values()),
                "environment_calls": sum(t["ledger"]["environment_calls"] for t in tasks.values()),
                "reported_task_tokens": sum(
                    sum(t.get("token_usage", {}).values()) for t in tasks.values()
                ),
                "returned_task_usage_complete": usage_complete,
                "reported_model_seconds": sum(
                    c["elapsed_seconds"]
                    for t in tasks.values()
                    for c in t.get("model_call_costs", [])
                ),
                "model_timing_complete": all("model_call_costs" in t for t in tasks.values()),
                "errors": dict(
                    Counter(t["error_type"] for t in tasks.values() if "error_type" in t)
                ),
                "adaptation_seconds_after_round": update.get("adaptation_seconds", 0),
                "adaptation_deployment": update.get("deployment", "none"),
                "adaptation_error": update.get("error_type"),
                "task_adaptation_seconds": sum(
                    t["ledger"].get("adaptation_seconds", 0) for t in tasks.values()
                ),
                "reported_semantic_learning_tokens": sum(
                    sum(t.get("semantic_learning", {}).get("token_usage", {}).values())
                    + sum(
                        t.get("semantic_learning", {})
                        .get("review", {})
                        .get("token_usage", {})
                        .values()
                    )
                    for t in tasks.values()
                ),
                "execution_diagnostics": execution_diagnostics(list(tasks.values())),
                "runtime_diagnostics": runtime_diagnostics(list(tasks.values())),
            }
        )
    for i in range(1, len(indexed)):
        before, after = indexed[i - 1], indexed[i]
        transitions = {
            "new_passes": [],
            "regressions": [],
            "still_failing": [],
            "still_passing": [],
        }
        per_task = []
        for name in names:
            a, b = before[name], after[name]
            passed_a = a["task_completed_correctly"] == 1
            passed_b = b["task_completed_correctly"] == 1
            category = (
                ("still_passing" if passed_a else "new_passes")
                if passed_b
                else ("regressions" if passed_a else "still_failing")
            )
            transitions[category].append(name)
            per_task.append(
                {
                    "task_name": name,
                    "success_change": b["task_completed_correctly"] - a["task_completed_correctly"],
                    "before_error": a.get("error_type"),
                    "after_error": b.get("error_type"),
                    "environment_call_change": b["ledger"]["environment_calls"]
                    - a["ledger"]["environment_calls"],
                    "reported_token_change": sum(b["token_usage"].values())
                    - sum(a["token_usage"].values())
                    if "token_usage" in a and "token_usage" in b
                    else None,
                }
            )
        result["paired_changes"].append(
            {
                "from_round": rows[i - 1]["round"],
                "to_round": rows[i]["round"],
                **transitions,
                "per_task": per_task,
            }
        )
    scores = [r["strict_passes"] for r in result["rounds"]]
    result["strict_passes_nondecreasing"] = (
        all(b >= a for a, b in pairwise(scores)) if len(scores) >= 2 else None
    )
    result["strict_passes_increased"] = scores[-1] > scores[0] if len(scores) >= 2 else None
    result["interpretation"] = (
        "Descriptive in-sample comparisons. Fewer calls with more failures are not an efficiency gain. "
        "Missing usage is unknown, not zero. Construction/maintenance and unreported provider billing "
        "must be considered before claiming net improvement; repeated rounds do not establish causality."
    )
    return result


def load_curve_report(path: Path) -> dict:
    curve = json.loads(path.read_text())
    runs_root = path.parent.parent
    runs = {
        row["run_id"]: json.loads((runs_root / row["run_id"] / "summary.json").read_text())
        for row in curve["rounds"]
    }
    return compare_completed_rounds(curve, runs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curve", type=Path)
    args = parser.parse_args()
    print(json.dumps(load_curve_report(args.curve), indent=2))
