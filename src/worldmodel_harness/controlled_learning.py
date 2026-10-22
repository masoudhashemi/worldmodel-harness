"""Disjoint development, artifact validation and matched checkpoint evaluation.

Scores are report-only. Each evaluation task/arm receives a disposable copy of
the same checkpoint; no evaluation trace flows back into development or siblings.
"""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .adaptation import ValidationEnvironment, propose, validate_and_adopt
from .automationbench import UPSTREAM_COMMIT, AutomationBenchAdapter, dataset
from .budget import Ledger
from .checkpoint import checkpoint_hash, copy_checkpoint
from .config import load_api_key, load_settings
from .experiment import ExperimentSettings, run_automationbench
from .knowledge import Scope
from .learning_curve import summarize_round
from .opportunities import execution_costs, plan_from_runs
from .runtime import MeteredClient, build_client
from .store import KnowledgeStore

ARMS = ("reset", "carried", "frozen", "no_reflection", "no_procedures", "no_semantics")


async def validated_update(root, knowledge_path, validation_names, *, domain, horizon,
                           costs, max_calls=20):
    """Propose at most one artifact from traces, then use existing isolated gates."""
    store = KnowledgeStore(knowledge_path)
    ledger = Ledger(max_environment_calls=max_calls)
    result = {"deployment": "none", "accounting": {}}
    started = perf_counter()
    client = None
    try:
        model = load_settings(root)
        client = MeteredClient(build_client(model, load_api_key(root, model)))
        rows = {json.loads(r["info"])["task_name"]: r for r in dataset(domain)}
        training = tuple(sorted(store.learning_tasks()))
        if len(set(validation_names)) < 2 or set(validation_names) & set(training):
            raise ValueError("Require two distinct validation tasks disjoint from training")
        scope = Scope(environment="automationbench", version=UPSTREAM_COMMIT)
        public = [{"task_name": name, "prompt": rows[name]["prompt"]}
                  for name in validation_names]
        plan = await plan_from_runs(client, store, scope, public, horizon=horizon,
                                   accounting=result["accounting"], costs=costs)
        result["plan"] = plan.model_dump(mode="json")
        if not plan.propose_artifact:
            return result

        def factory(trial, shared_ledger, evidence):
            adapter = AutomationBenchAdapter(rows[trial.task_name], domain, evidence,
                shared_ledger, knowledge=store, purpose_override="validation")
            return ValidationEnvironment(task_id=adapter.scope.task_id, scope=adapter.scope,
                live=adapter.live, tools={t.name: t for t in adapter.tools()})

        prototype = factory(plan.trials[0], ledger, store)
        proposal_accounting = {}
        result["accounting"]["proposal"] = proposal_accounting
        candidate = await propose(client, store, scope, prototype.tools, training,
            goal=plan.goal, accounting=proposal_accounting, opportunity_id=plan.opportunity_id,
            plan_context={"trigger": plan.trigger, "stop_condition": plan.stop_condition,
                          "expected_runtime_effect": plan.expected_runtime_effect})
        if candidate.parameter_types != plan.parameter_types:
            raise ValueError("Candidate changed the proposed parameter interface")
        candidate = candidate.model_copy(update={"procedure_id": "learned-" + uuid4().hex[:16],
            "learned_opportunity_id": plan.opportunity_id, "trigger": plan.trigger,
            "stop_condition": plan.stop_condition, "expected_runtime_effect": plan.expected_runtime_effect})
        checked = await validate_and_adopt(candidate, plan.trials, factory, store, ledger=ledger,
            estimated_reuses=plan.estimated_reuses,
            estimated_saving_per_use=plan.estimated_saving_seconds_per_use,
            build_cost=perf_counter() - started)
        result.update(deployment="active" if checked.adopted else "candidate",
                      validation=checked.model_dump(mode="json"))
        return result
    except Exception as exc:  # noqa: BLE001 -- preserve costs without provider payloads
        result["error_type"] = type(exc).__name__
        return result
    finally:
        result["model_call_costs"] = client.calls if client else []
        result["elapsed_seconds"] = perf_counter() - started
        result["ledger"] = asdict(ledger)
        store.close()


def arm_settings(base, arm, task):
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    return ExperimentSettings.model_validate({**base.model_dump(), "task_names": (task,),
        "limit": 1, "offset": 0, "phase": "frozen" if arm == "frozen" else "test",
        "adaptive": arm != "no_procedures", "allow_experimental": False,
        "semantic_learning": arm != "no_semantics",
        "max_online_reflections": 0 if arm == "no_reflection" else base.max_online_reflections,
        "information_policy": "off"})


async def run_controlled(root: Path, output: Path, *, training_names, validation_names,
                         evaluation_names, base: ExperimentSettings):
    groups = [tuple(training_names), tuple(validation_names), tuple(evaluation_names)]
    all_names = [name for group in groups for name in group]
    if (not all(groups) or len(validation_names) < 2
        or len(all_names) != len(set(all_names))):
        raise ValueError("Training, validation and evaluation must be nonempty disjoint task lists")
    if not base.operational or base.memory != "graph":
        raise ValueError("Controlled learning requires operational graph memory")
    output.mkdir(parents=True, exist_ok=False)
    knowledge = output / "development.sqlite"
    KnowledgeStore(knowledge).close()
    report = {"protocol": "disjoint-matched-checkpoint-learning-v1",
        "training_tasks": training_names, "validation_tasks": validation_names,
        "evaluation_tasks": evaluation_names, "settings": base.model_dump(mode="json"),
        "arms": list(ARMS), "validation_scores_used_for_learning": False,
        "results": [], "complete": False, "improvement_demonstrated": False,
        "interpretation": "Descriptive paired trials; no significance or causal benefit claim. "
            "Ablations disable runtime use/update of a layer, not its contribution during training."}

    def save():
        path = output / "summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    save()
    training_settings = ExperimentSettings.model_validate({**base.model_dump(),
        "task_names": training_names, "limit": len(training_names), "offset": 0,
        "phase": "train", "adaptive": True, "semantic_learning": True,
        "allow_experimental": False, "information_policy": "off"})
    training = await run_automationbench(root, training_settings, knowledge)
    report["training"] = {"run_id": training["run_id"], "metrics": summarize_round(training)}
    report["learning_update"] = await validated_update(root, knowledge, validation_names,
        domain=base.domain, horizon=len(evaluation_names), costs=execution_costs(training))
    checkpoint = output / "checkpoint.sqlite"
    copy_checkpoint(knowledge, checkpoint)
    digest = checkpoint_hash(checkpoint)
    report["checkpoint_sha256"] = digest
    save()
    for index, name in enumerate(evaluation_names):
        # Rotate order across tasks to reduce a fixed arm-order/provider-time confound.
        order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
        for arm in order:
            path = output / f"trial-{index}-{arm}.sqlite"
            if arm == "reset":
                KnowledgeStore(path).close()
            else:
                copy_checkpoint(checkpoint, path)
            run = await run_automationbench(root, arm_settings(base, arm, name), path)
            report["results"].append({"arm": arm, "task_name": name,
                "run_id": run["run_id"], "metrics": summarize_round(run)})
            if checkpoint_hash(checkpoint) != digest:
                raise ValueError("Evaluation modified the shared checkpoint")
            save()
    report["complete"] = True
    save()
    return report
