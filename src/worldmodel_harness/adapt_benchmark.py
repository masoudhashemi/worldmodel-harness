"""Model proposal and isolated AutomationBench validation, without grader feedback."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import Field

from .adaptation import (
    RepairRequest,
    ValidationEnvironment,
    ValidationTrial,
    propose,
    validate_and_adopt,
)
from .automationbench import UPSTREAM_COMMIT, AutomationBenchAdapter, dataset
from .budget import Ledger
from .config import load_api_key, load_settings
from .knowledge import Record, Scope
from .runtime import build_client
from .store import KnowledgeStore


class AdaptationSpec(Record):
    domain: str = "simple"
    goal: str
    parameter_types: dict[str, str]
    trials: tuple[ValidationTrial, ...]
    estimated_reuses: int = Field(ge=0)
    estimated_saving_seconds_per_use: float = Field(ge=0)
    max_validation_calls: int = Field(default=20, ge=1)
    proposal_max_tokens: int = Field(default=8192, ge=1)
    repair: RepairRequest | None = None


async def adapt_automationbench(root: Path, knowledge_path: Path, spec: AdaptationSpec) -> dict:
    settings = load_settings(root)
    proposal_settings = settings.model_copy(update={"max_tokens": spec.proposal_max_tokens})
    client = build_client(proposal_settings, load_api_key(root, settings))
    if not knowledge_path.is_file():
        raise ValueError("Adaptation requires an existing development knowledge store")
    store = KnowledgeStore(knowledge_path)
    output = root / "runs" / uuid4().hex
    output.mkdir(parents=True)
    ledger = Ledger(max_environment_calls=spec.max_validation_calls)
    proposal_started = None
    accounting = {}
    try:
        training = tuple(sorted(store.learning_tasks(exclude_validation=True)))
        names = [t.task_name for t in spec.trials]
        if len(set(names)) < 2 or len(set(names)) != len(names) or set(names) & set(training):
            raise ValueError(
                "Use distinct validation tasks not used to construct this knowledge store"
            )
        rows = {json.loads(row["info"])["task_name"]: row for row in dataset(spec.domain)}
        if any(name not in rows for name in names):
            raise ValueError("Unknown validation task")

        def factory(trial, shared_ledger, evidence):
            adapter = AutomationBenchAdapter(
                rows[trial.task_name],
                spec.domain,
                evidence,
                shared_ledger,
                knowledge=store,
                purpose_override="validation",
            )
            return ValidationEnvironment(
                task_id=adapter.scope.task_id,
                scope=adapter.scope,
                live=adapter.live,
                tools={t.name: t for t in adapter.tools()},
            )

        prototype = factory(spec.trials[0], ledger, store)
        scope = Scope(environment="automationbench", version=UPSTREAM_COMMIT)
        repairing = None
        if spec.repair:
            repairing = next(
                (p for p in store.procedures(scope) if p.procedure_id == spec.repair.procedure_id),
                None,
            )
            if (
                repairing is None
                or repairing.status != "suspended"
                or repairing.version != spec.repair.suspended_version
            ):
                raise ValueError("Repair target is not the requested suspended version")
        proposal_started = perf_counter()
        candidate = await propose(
            client,
            store,
            scope,
            prototype.tools,
            training,
            goal=spec.goal + " Required parameter_types: " + json.dumps(spec.parameter_types),
            accounting=accounting,
            repairing=repairing,
        )
        build_cost = perf_counter() - proposal_started
        ledger.adaptation_seconds += build_cost
        proposal_started = None
        if candidate.parameter_types != spec.parameter_types:
            raise ValueError("Proposal parameter interface does not match the declared trials")
        (output / "candidate.json").write_text(candidate.model_dump_json(indent=2) + "\n")
        result = await validate_and_adopt(
            candidate,
            spec.trials,
            factory,
            store,
            ledger=ledger,
            estimated_reuses=spec.estimated_reuses,
            estimated_saving_per_use=spec.estimated_saving_seconds_per_use,
            build_cost=build_cost,
            repair=spec.repair,
        )
        report = {
            "kind": "isolated-artifact-development-validation",
            "run_id": output.name,
            "model": settings.name,
            "proposal_model_settings": proposal_settings.model_dump(),
            "proposal_accounting": accounting,
            "spec": spec.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "ledger": asdict(ledger),
            "benefit_is_estimated_not_measured": True,
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as exc:
        if proposal_started is not None:
            ledger.adaptation_seconds += perf_counter() - proposal_started
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "kind": "artifact-development-failure",
                    "error_type": type(exc).__name__,
                    "spec": spec.model_dump(mode="json"),
                    "ledger": asdict(ledger),
                    "proposal_accounting": accounting,
                    "failed_provider_token_usage": "unavailable"
                    if not accounting
                    else "see proposal_accounting",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        store.close()
