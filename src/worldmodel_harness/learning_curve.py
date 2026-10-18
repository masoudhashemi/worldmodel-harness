"""Repeated in-sample benchmark rounds with persistent G and experimental H.

All selected tasks are both learning and test tasks, as explicitly requested.
Scores are report-only; neither graders nor scores enter artifact proposal inputs.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import Field

from .adaptation import propose
from .automationbench import UPSTREAM_COMMIT, AutomationBenchAdapter, dataset
from .budget import Ledger
from .config import load_api_key, load_settings
from .curve_report import runtime_diagnostics
from .experiment import ExperimentSettings, run_automationbench
from .knowledge import Record, Scope
from .opportunities import execution_costs, plan_from_runs
from .runtime import build_client
from .store import KnowledgeStore


class LearningCurveSettings(Record):
    rounds: int = Field(default=3, ge=2)
    task_count: int = Field(default=50, ge=1, le=50)
    seed: int = 1729
    domain: str = "public"
    max_environment_calls: int = Field(default=40, ge=1)
    max_turns: int = Field(default=20, ge=1)
    timeout_seconds: int = Field(default=300, ge=1)
    proposal_max_tokens: int = Field(default=16384, ge=1)
    operational: bool = False
    semantic_learning: bool = False


def select_task_names(rows, *, count: int, seed: int) -> tuple[str, ...]:
    """Seeded, domain-balanced sampling based on names only, never scores/solutions."""
    if not 1 <= count <= 50:
        raise ValueError("This experiment is capped at 50 unique tasks")
    buckets = defaultdict(list)
    seen = set()
    for row in rows:
        name = json.loads(row["info"])["task_name"]
        if name not in seen:
            buckets[name.split(".")[0]].append(name)
            seen.add(name)
    if len(seen) < count:
        raise ValueError("Not enough tasks for the requested subset")
    rng = random.Random(seed)
    domains = sorted(buckets)
    rng.shuffle(domains)
    for names in buckets.values():
        names.sort()
        rng.shuffle(names)
    selected = []
    while len(selected) < count:
        for domain in domains:
            if buckets[domain] and len(selected) < count:
                selected.append(buckets[domain].pop())
    return tuple(selected)


def summarize_round(report: dict) -> dict:
    results = report["results"]
    return {
        "task_count": len(results),
        "strict_success": sum(r["task_completed_correctly"] for r in results) / len(results),
        "environment_calls": sum(r["ledger"]["environment_calls"] for r in results),
        "reported_tokens": sum(sum(r.get("token_usage", {}).values()) for r in results),
        "token_usage_complete": all("token_usage" in r for r in results),
        "execution_errors": sum("error_type" in r for r in results),
        "runtime_diagnostics": runtime_diagnostics(results),
        "artifact_invocations": sum(r.get("artifact_invocations", 0) for r in results),
        "artifact_completed_subgoals": sum(
            a.get("completed", False) for r in results for a in r.get("artifact_reuse", [])
        ),
        "artifact_execution_seconds": sum(
            a["elapsed_seconds"] for r in results for a in r.get("artifact_reuse", [])
        ),
        "artifact_measured_savings": None,
        "compute_seconds": sum(r["ledger"]["compute_seconds"] for r in results),
        "reported_model_seconds": sum(
            c["elapsed_seconds"] for r in results for c in r.get("model_call_costs", [])
        ),
        "model_timing_complete": all("model_call_costs" in r for r in results),
        "semantic_learning_tokens": sum(
            sum(r.get("semantic_learning", {}).get("token_usage", {}).values())
            + sum(r.get("semantic_learning", {}).get("review", {}).get("token_usage", {}).values())
            for r in results
        ),
        "task_adaptation_seconds": sum(r["ledger"].get("adaptation_seconds", 0) for r in results),
    }


async def learn_from_round(root, knowledge_path, prototype_row, settings, horizon, *, costs=None):
    """At most one selective, trace-driven artifact proposal between rounds."""
    store = KnowledgeStore(knowledge_path)
    started = perf_counter()
    accounting = {}
    result = {"deployment": "none", "independent_validation": False}
    try:
        model = load_settings(root).model_copy(update={"max_tokens": settings.proposal_max_tokens})
        client = build_client(model, load_api_key(root, model))
        scope = Scope(environment="automationbench", version=UPSTREAM_COMMIT)
        if settings.operational:
            from .automatic_repair import repair_candidates, repair_from_runs

            ready, deferred = repair_candidates(store, scope)
            result["deferred_repairs"] = deferred
            if ready:
                prototype = AutomationBenchAdapter(prototype_row, settings.domain, store, Ledger())
                repair = await repair_from_runs(
                    client,
                    store,
                    scope,
                    {t.name: t for t in prototype.tools()},
                    horizon=horizon,
                    accounting=accounting,
                )
                result.update(repair)
                return result
        plan = await plan_from_runs(
            client,
            store,
            scope,
            [],
            horizon=horizon,
            accounting=accounting,
            require_validation=False,
            costs=costs,
        )
        result["plan"] = plan.model_dump(mode="json")
        if not plan.propose_artifact:
            return result
        prototype = AutomationBenchAdapter(prototype_row, settings.domain, store, Ledger())
        tools = {t.name: t for t in prototype.tools()}
        proposal_accounting = {}
        accounting["proposal"] = proposal_accounting
        candidate = await propose(
            client,
            store,
            scope,
            tools,
            tuple(sorted(store.learning_tasks())),
            goal=plan.goal,
            accounting=proposal_accounting,
            opportunity_id=plan.opportunity_id,
            plan_context={
                "trigger": plan.trigger,
                "expected_runtime_effect": plan.expected_runtime_effect,
                "stop_condition": plan.stop_condition,
            },
        )
        if candidate.parameter_types != plan.parameter_types:
            raise ValueError("Proposed artifact changed the learned parameter interface")
        # The controller assigns identity/provenance; names are not workflow templates.
        cost = perf_counter() - started
        candidate = candidate.model_copy(
            update={
                "procedure_id": "learned-" + uuid4().hex[:16],
                "learned_opportunity_id": plan.opportunity_id,
                "trigger": plan.trigger,
                "expected_runtime_effect": plan.expected_runtime_effect,
                "stop_condition": plan.stop_condition,
                "estimated_reuses": plan.estimated_reuses,
                "estimated_saving_per_use": plan.estimated_saving_seconds_per_use,
                "build_and_validation_cost": cost,
            }
        )
        # One construction per boundary and at most two invocations per task are
        # the experiment budget. Estimated ROI is not evidence required to try it.
        # No independently unvalidated candidate becomes active here.
        candidate = candidate.model_copy(update={"status": "experimental"})
        store.put_procedure(candidate, tools=tools)
        result.update(
            deployment=candidate.status,
            artifact=candidate.model_dump(mode="json"),
            benefit_is_estimated_not_measured=True,
            experiment_policy="one_candidate_per_boundary_two_invocations_per_task",
        )
        return result
    except Exception as exc:  # noqa: BLE001 -- do not expose SDK secrets or abort task learning
        # A failed proposal must not abort ordinary task learning or expose SDK details.
        result["error_type"] = type(exc).__name__
        return result
    finally:
        result["accounting"] = accounting
        result["adaptation_seconds"] = perf_counter() - started
        store.close()


async def run_learning_curve(root: Path, settings: LearningCurveSettings) -> dict:
    rows = list(dataset(settings.domain))
    names = select_task_names(rows, count=settings.task_count, seed=settings.seed)
    output = root / "runs" / uuid4().hex
    output.mkdir(parents=True)
    knowledge_path = output / "knowledge.sqlite"
    KnowledgeStore(knowledge_path).close()  # Start empty; no hand-written artifacts or prior runs.
    report = {
        "run_id": output.name,
        "harness_revision": "operational-learning-v3.4",
        "feedback_protocol": "trajectory-only",
        "protocol": "repeated-in-sample-online-learning",
        "settings": settings.model_dump(),
        "task_names": names,
        "training_and_evaluation_are_same_tasks": True,
        "held_out_tasks": [],
        "rounds": [],
        "complete": False,
        "improvement_demonstrated": False,
    }
    path = output / "learning_curve.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    prototype = next(r for r in rows if json.loads(r["info"])["task_name"] == names[0])
    for index in range(settings.rounds):
        round_settings = ExperimentSettings(
            domain=settings.domain,
            task_names=names,
            limit=len(names),
            phase="online",
            memory="graph",
            adaptive=True,
            operational=settings.operational,
            semantic_learning=settings.semantic_learning,
            allow_experimental=True,
            information_policy="future",
            max_environment_calls=settings.max_environment_calls,
            max_turns=settings.max_turns,
            timeout_seconds=settings.timeout_seconds,
        )
        run = await run_automationbench(root, round_settings, knowledge_path)
        metrics = summarize_round(run)
        if metrics["task_count"] != len(names):
            raise ValueError("Incomplete round cannot enter learning curve")
        row = {"round": index + 1, "run_id": run["run_id"], "metrics": metrics}
        report["rounds"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        if index + 1 < settings.rounds:
            row["learning_update"] = await learn_from_round(
                root,
                knowledge_path,
                prototype,
                settings,
                horizon=(settings.rounds - index - 1) * len(names),
                costs=execution_costs(run),
            )
        path.write_text(json.dumps(report, indent=2) + "\n")
    first, last = report["rounds"][0]["metrics"], report["rounds"][-1]["metrics"]
    report["observed_success_change"] = last["strict_success"] - first["strict_success"]
    report["observed_environment_call_change"] = (
        last["environment_calls"] - first["environment_calls"]
    )
    report["observed_token_change"] = last["reported_tokens"] - first["reported_tokens"]
    report["complete"] = True
    report["interpretation"] = (
        "Descriptive in-sample curve only; improvement requires inspecting success and all costs, not merely completed rounds. No claim of generalization or causal significance."
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
