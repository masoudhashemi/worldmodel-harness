"""Real-model blocker decisions on paired synthetic environment fixtures.

Measures reflection decisions and recovery execution, not actor retry savings or
transfer. Fixture responses are evaluation data, never production recovery rules.
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from stirrup.core.models import Tool, ToolMessage, ToolResult

from worldmodel_harness.budget import Ledger
from worldmodel_harness.config import load_api_key, load_settings
from worldmodel_harness.interactions import CURRENT_INTERACTION, InteractionRecorder
from worldmodel_harness.knowledge import Episode, Record, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.online_learning import OnlineLearner
from worldmodel_harness.runtime import MeteredClient, build_client
from worldmodel_harness.store import KnowledgeStore


class Read(Record):
    cursor: str


async def case(root, output, settings, blocked):
    name = "blocked" if blocked else "recoverable"
    store = KnowledgeStore(output / f"{name}.sqlite")
    client = MeteredClient(build_client(settings, load_api_key(root, settings)))
    scope = Scope(environment="blocker-fixture", version="1", tenant=name, task_id=name)
    adapter = SimpleNamespace(knowledge=store, evidence=store, scope=scope, ledger=Ledger(),
        learner=Learner(store), retain_interactions=False,
        task=SimpleNamespace(prompt=[{"role": "user", "content": "Read the current record."}]))
    actual = []

    async def inspect(params):
        if blocked:
            body = {"ok": False, "error": "account_suspended",
                "detail": "Account reactivation requires an external administrator. This tool cannot change account state."}
        elif params.cursor != "current-token-9":
            body = {"ok": False, "error": "invalid_cursor", "current_cursor": "current-token-9"}
        else:
            body = {"ok": True, "record": {"id": "record-1"}}
        actual.append({"arguments": params.model_dump(), "response": body})
        store.append_episode(Episode(task_id=name, scope=scope, origin="environment", operation="inspect",
            arguments=params.model_dump(), response=body, coverage=tuple(body),
            outcome="success" if body["ok"] else "error", environment_called=True,
            interaction_id=CURRENT_INTERACTION.get()))
        return ToolResult(content=json.dumps(body), success=body["ok"])

    tool = Tool(name="inspect", description="Read the current record using a cursor. This read cannot change account state.",
                parameters=Read, executor=inspect)
    recorder = InteractionRecorder(adapter)
    recorder.register_tools([tool])

    async def call(arguments):
        identity, token = recorder.begin()
        result = await inspect(Read.model_validate(arguments))
        return recorder.end(identity, token,
            SimpleNamespace(name="inspect", arguments=json.dumps(arguments)),
            result=ToolMessage(tool_call_id=identity, name="inspect", content=result.content,
                success=result.success, args_was_valid=True))

    for _ in range(3):
        await call({"cursor": "obsolete"})
    learner = OnlineLearner(adapter, client, max_reflections=1, timeout_seconds=120)
    try:
        await learner.before_decision()
        audit = learner.audit[0]
        if blocked:
            pause = audit.get("pause", {})
            passed = bool(pause and pause["blocker"]["origin"] == "environment"
                and any(store.episode(eid).origin == "environment" for eid in pause["evidence_ids"]))
        else:
            if learner.pending_experiment is not None:
                await call(learner.pending_experiment.arguments)
                await learner.before_decision()
            outcome = audit.get("experiment_outcome", {})
            passed = bool(outcome.get("result") == "executed" and actual[-1]["response"]["ok"]
                and any(p["result"] == "matched" for p in outcome.get("predictions", [])))
        return {"passed": passed, "actual_requests": actual, "reflection": learner.audit,
                "model_call_costs": client.calls}
    finally:
        store.close()


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    settings = load_settings(args.root).model_copy(update={"max_tokens": 4096, "timeout_seconds": 60})
    report = {"complete": False, "model": settings.model_dump(), "fixture_seeded_failures_per_case": 3}
    path = args.output / "summary.json"
    for blocked in (True, False):
        report["blocked" if blocked else "recoverable"] = await case(args.root, args.output, settings, blocked)
        path.write_text(json.dumps(report, indent=2) + "\n")
    report.update(complete=True, passed=report["blocked"]["passed"] and report["recoverable"]["passed"])
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    asyncio.run(main(parser.parse_args()))
