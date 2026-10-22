"""Matched actor evaluation of learned recovery, scoped stopping, and resumption.

Synthetic fixture, one trial per scenario/arm: diagnostic evidence, not a broad
benchmark or a causal estimate for all learning layers. Grades never enter prompts.
"""
import argparse
import asyncio
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from stirrup.core.models import SystemMessage, Tool, ToolMessage, ToolResult, UserMessage

from worldmodel_harness.budget import Ledger
from worldmodel_harness.config import load_api_key, load_settings
from worldmodel_harness.interactions import CURRENT_INTERACTION, InteractionRecorder
from worldmodel_harness.knowledge import Episode, Record, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.online_learning import OnlineLearner
from worldmodel_harness.repairing_agent import RepairingAgent
from worldmodel_harness.runtime import MeteredClient, build_client
from worldmodel_harness.semantic_learning import hypothesis_context
from worldmodel_harness.store import KnowledgeStore

ARMS = ("no_knowledge", "retained", "active")
SCENARIOS = ("recoverable", "blocked", "resumption")
PROMPT = ("Read the current record using inspect. Start with cursor 'obsolete' if no current cursor is known. "
          "Use observed feedback to decide what to do next. When you finish, report whether you read the "
          "record or are blocked, and explain the evidence. Do not claim to have read a record you did not read.")


class Read(Record):
    cursor: str


class Finish(Record):
    status: Literal["read", "blocked"]
    explanation: str


class Fixture:
    def __init__(self, store, task, blocked, cursor, learning):
        self.store, self.blocked, self.cursor = store, blocked, cursor
        self.phase = "initial"
        self.requests, self.finishes = [], []
        self.scope = Scope(environment="behavior-fixture", version="1", tenant="workspace", task_id=task)
        self.adapter = SimpleNamespace(knowledge=store, evidence=store, scope=self.scope,
            ledger=Ledger(), learner=Learner(store) if learning else None, retain_interactions=True,
            task=SimpleNamespace(prompt=[{"role": "user", "content": PROMPT}]))
        self.read_tool = Tool(name="inspect", description="Read the current record using a cursor. This read cannot change account state.",
                              parameters=Read, executor=self.inspect)
        self.finish_tool = Tool(name="finish", description="Report the outcome of this attempt and its evidence.",
                                parameters=Finish, executor=self.finish)
        self.recorder = InteractionRecorder(self.adapter)
        self.recorder.register_tools([self.read_tool, self.finish_tool])

    async def inspect(self, params):
        if self.blocked:
            body = {"ok": False, "error": "account_suspended",
                    "detail": "Account reactivation requires an external administrator. This tool cannot change account state."}
        elif params.cursor != self.cursor:
            body = {"ok": False, "error": "invalid_cursor", "current_cursor": self.cursor}
        else:
            body = {"ok": True, "record": {"id": self.scope.task_id, "state": "ready"}}
        self.requests.append({"phase": self.phase, "arguments": params.model_dump(), "response": body})
        self.store.append_episode(Episode(task_id=self.scope.task_id, scope=self.scope,
            origin="environment", operation="inspect", arguments=params.model_dump(), response=body,
            coverage=tuple(body), outcome="success" if body["ok"] else "error", environment_called=True,
            interaction_id=CURRENT_INTERACTION.get()))
        return ToolResult(content=json.dumps(body), success=body["ok"])

    async def finish(self, params):
        self.finishes.append({"phase": self.phase, **params.model_dump()})
        # Completion of an attempt is not task success. Only private metrics grade it.
        return ToolResult(content="Outcome recorded.", success=True)

    async def seed_failure(self):
        identity, token = self.recorder.begin()
        args = Read(cursor="obsolete")
        result = await self.inspect(args)
        self.recorder.end(identity, token,
            SimpleNamespace(name="inspect", arguments=args.model_dump_json()),
            result=ToolMessage(tool_call_id=identity, name="inspect", content=result.content,
                               success=result.success, args_was_valid=True))


class BehavioralClient:
    """Use production hypothesis retrieval/reflection with identical acting tools."""
    def __init__(self, actor, fixture, arm, semantic):
        self.actor, self.fixture, self.arm = actor, fixture, arm
        self.online = OnlineLearner(fixture.adapter, semantic, max_reflections=2, timeout_seconds=90) if arm == "active" else None
        self.hints = []

    def __getattr__(self, name):
        return getattr(self.actor, name)

    async def generate(self, messages, tools):
        if self.online:
            await self.online.before_decision()
        content = ""
        if self.arm != "no_knowledge":
            query = PROMPT + " " + " ".join(str(m.content) for m in messages[-3:] if m.role in {"user", "tool"})
            content = hypothesis_context(self.fixture.store, self.fixture.scope, query, max_chars=3000)
            if self.online:
                content += "\n" + self.online.context(max_chars=2400)
        self.hints.append(content)
        if content.strip():
            messages = [SystemMessage(content=content), *messages]
        return await self.actor.generate(messages, tools)


def costs(calls):
    return {"model_calls": len(calls), "reported_tokens": sum(
        sum(c.get("token_usage", {}).values()) for c in calls),
        "missing_usage_calls": sum(not c.get("usage_available", False) for c in calls),
        "model_seconds": sum(c.get("elapsed_seconds", 0) for c in calls)}


def metrics(fixture, scenario):
    initial = [r for r in fixture.requests if r["phase"] == "initial"]
    resumed = [r for r in fixture.requests if r["phase"] == "resumed"]
    read = any(r["response"]["ok"] for r in initial)
    resumed_read = any(r["response"]["ok"] for r in resumed)
    reports_blocked = any(f["phase"] == "initial" and f["status"] == "blocked" for f in fixture.finishes)
    # Observable duplicate failed attempts, not a claim that all distinct attempts were useful.
    failures = Counter((r["phase"], json.dumps(r["arguments"], sort_keys=True),
                        json.dumps(r["response"], sort_keys=True))
                       for r in fixture.requests if not r["response"]["ok"])
    stopped = reports_blocked and not read and any(not r["response"]["ok"] for r in initial)
    return {"environment_calls": len(fixture.requests),
        "duplicate_failed_calls": sum(max(0, n - 1) for n in failures.values()),
        "blocked_phase_calls": len(initial) if scenario != "recoverable" else 0,
        "read_initial": read, "reported_blocker": reports_blocked,
        "premature_abandonment": scenario == "recoverable" and reports_blocked and not read,
        "resumed_successfully": resumed_read,
        "false_read_reports": sum(f["status"] == "read" and not any(
            r["phase"] == f["phase"] and r["response"]["ok"] for r in fixture.requests) for f in fixture.finishes),
        "scenario_success": read if scenario == "recoverable" else stopped if scenario == "blocked" else stopped and resumed_read}


async def session(client, fixture, path, prompt):
    agent = RepairingAgent(client=client, name="behavior-eval", tools=[fixture.read_tool],
        finish_tool=fixture.finish_tool, max_turns=8, interaction_recorder=fixture.recorder,
        repair_schemas={t.name: t.parameters for t in (fixture.read_tool, fixture.finish_tool)})
    async with asyncio.timeout(180):
        async with agent.session(output_dir=path) as current:
            await current.run([UserMessage(content=prompt)])


async def train(root, output, settings):
    store = KnowledgeStore(output / "checkpoint.sqlite")
    client = MeteredClient(build_client(settings, load_api_key(root, settings)))
    records = []
    try:
        for blocked in (True, False):
            fixture = Fixture(store, "train-blocked" if blocked else "train-recoverable",
                              blocked, "training-token-1", True)
            for _ in range(3):
                await fixture.seed_failure()
            learner = OnlineLearner(fixture.adapter, client, max_reflections=1, timeout_seconds=90)
            await learner.before_decision()
            records.append({"task": fixture.scope.task_id, "seeded_requests": fixture.requests, "audit": learner.audit})
        return {"tasks": records, "cost": costs(client.calls), "model_call_costs": client.calls,
                "claims": [c.model_dump(mode="json") for c in store.claims()]}
    finally:
        store.close()


async def trial(root, output, settings, scenario, arm, checkpoint):
    folder = output / f"{scenario}-{arm}"
    folder.mkdir()
    path = folder / "knowledge.sqlite"
    if arm != "no_knowledge":
        shutil.copy2(checkpoint, path)
    store = KnowledgeStore(path)
    before = [c.model_dump(mode="json") for c in store.claims()]
    fixture = Fixture(store, "eval-" + scenario, scenario != "recoverable", "heldout-token-8", arm == "active")
    actor = MeteredClient(build_client(settings, load_api_key(root, settings)))
    semantic = MeteredClient(build_client(settings, load_api_key(root, settings)))
    client = BehavioralClient(actor, fixture, arm, semantic)
    errors = []
    try:
        try:
            await session(client, fixture, folder / "initial", PROMPT)
        except Exception as exc:  # noqa: BLE001 -- preserve failed trials, never silently retry
            errors.append({"phase": "initial", "type": type(exc).__name__})
        if scenario == "resumption":
            fixture.blocked = False
            fixture.phase = "resumed"
            notice = "An external administrator has now confirmed that this account is reactivated. Continue the original task and read the current record."
            fixture.adapter.task.prompt.append({"role": "user", "content": notice})
            try:
                await session(client, fixture, folder / "resumed", PROMPT + "\n" + notice)
            except Exception as exc:  # noqa: BLE001
                errors.append({"phase": "resumed", "type": type(exc).__name__})
        result = {"scenario": scenario, "arm": arm, "metrics": metrics(fixture, scenario),
            "errors": errors, "requests": fixture.requests, "finishes": fixture.finishes,
            "hints": client.hints, "reflection": client.online.audit if client.online else [],
            "cost": costs([*actor.calls, *semantic.calls]),
            "model_call_costs": [*actor.calls, *semantic.calls],
            "claims_unchanged": before == [c.model_dump(mode="json") for c in store.claims()]}
        (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        store.close()


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    settings = load_settings(args.root).model_copy(update={"max_tokens": 4096, "timeout_seconds": 60})
    checkpoint = args.output / "checkpoint.sqlite"
    report = {"complete": False, "settings": settings.model_dump(), "arms": ARMS,
        "scenarios": SCENARIOS, "trials": [], "limits": {"actor_turns_per_phase": 8,
        "seconds_per_phase": 180, "reflections_per_trial": 2},
        "scope": "Synthetic semantic-knowledge ablation; actor behavior, not all harness layers"}
    target = args.output / "summary.json"
    report["training"] = await train(args.root, args.output, settings)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["checkpoint_sha256"] = digest
    target.write_text(json.dumps(report, indent=2) + "\n")
    for index, scenario in enumerate(SCENARIOS):
        for arm in (*ARMS[index:], *ARMS[:index]):
            result = await trial(args.root, args.output, settings, scenario, arm, checkpoint)
            report["trials"].append(result)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
            target.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"scenario": scenario, "arm": arm, **result["metrics"]}), flush=True)
    report["complete"] = True
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    asyncio.run(main(parser.parse_args()))
