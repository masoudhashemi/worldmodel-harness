"""Predeclared opaque-error incident replay with held-out actor trials.

Three actual failed requests seed every trial, and all arms receive the identical
public history. Seeds deliberately trigger active reflection and are reported
separately. Production recovery logic is unchanged; fixtures own service behavior.
"""
import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from blocker_behavior_eval import ARMS, BehavioralClient, Fixture, costs
from stirrup.core.models import Tool, ToolResult, UserMessage

from worldmodel_harness.config import load_api_key, load_settings
from worldmodel_harness.interactions import CURRENT_INTERACTION
from worldmodel_harness.knowledge import Episode, Record
from worldmodel_harness.repairing_agent import RepairingAgent
from worldmodel_harness.runtime import MeteredClient, build_client
from worldmodel_harness.semantic_learning import learn_semantics
from worldmodel_harness.semantic_review import review_semantics
from worldmodel_harness.store import KnowledgeStore

PROMPT = "Read the current record using the available tools. Investigate the observed failures. Finish with read only after reading it, or blocked with the observed reason if progress requires an external change."
SCENARIOS = ("recoverable", "blocked", "resumption")


class Lookup(Record):
    reference: str


class Incident(Fixture):
    def __init__(self, store, task, blocked, cursor, learning):
        self.receipt = hashlib.sha256((task + "receipt").encode()).hexdigest()[:12]
        self.state_ref = hashlib.sha256((task + "state").encode()).hexdigest()[:12]
        super().__init__(store, task, blocked, cursor, learning)
        self.adapter.task.prompt = [{"role": "user", "content": PROMPT}]
        self.tools = [self.read_tool,
            Tool(name="lookup", description="Retrieve a diagnostic entry using a reference.", parameters=Lookup, executor=self.lookup),
            Tool(name="state", description="Read current service state using its reference.", parameters=Lookup, executor=self.state)]
        self.recorder.register_tools([*self.tools, self.finish_tool])

    def record(self, operation, arguments, body):
        self.requests.append({"phase": self.phase, "operation": operation, "arguments": arguments, "response": body})
        self.store.append_episode(Episode(task_id=self.scope.task_id, scope=self.scope,
            origin="environment", operation=operation, arguments=arguments, response=body,
            coverage=tuple(body), outcome="success" if body["ok"] else "error", environment_called=True,
            interaction_id=CURRENT_INTERACTION.get()))
        return ToolResult(content=json.dumps(body), success=body["ok"])

    async def inspect(self, params):
        body = {"ok": False, "error": "request_unavailable", "reference": self.receipt}
        if not self.blocked and params.cursor == self.cursor:
            body = {"ok": True, "record": {"id": "current", "state": "ready"}}
        return self.record("inspect", params.model_dump(), body)

    async def lookup(self, params):
        body = {"ok": True, "state_reference": self.state_ref} if params.reference == self.receipt else {
            "ok": False, "error": "reference_unknown"}
        return self.record("lookup", params.model_dump(), body)

    async def state(self, params):
        if params.reference != self.state_ref:
            body = {"ok": False, "error": "reference_unknown"}
        elif self.blocked:
            body = {"ok": True, "authorization": "held", "change_owner": "external_administrator"}
        else:
            body = {"ok": True, "authorization": "ready", "current_cursor": self.cursor}
        return self.record("state", params.model_dump(), body)


async def seed(fixture):
    fixture.phase = "seed"
    for _ in range(3):
        await fixture.seed_failure()
    fixture.phase = "initial"
    return PROMPT + "\nEarlier requests in this incident:\n" + json.dumps(fixture.requests)


async def act(client, fixture, folder, prompt):
    agent = RepairingAgent(client=client, name="incident-eval", tools=fixture.tools,
        finish_tool=fixture.finish_tool, max_turns=12, interaction_recorder=fixture.recorder,
        repair_schemas={t.name: t.parameters for t in [*fixture.tools, fixture.finish_tool]})
    async with asyncio.timeout(240):
        async with agent.session(output_dir=folder) as session:
            await session.run([UserMessage(content=prompt)])


def grade(fixture, scenario):
    actor = [r for r in fixture.requests if r["phase"] != "seed"]
    def read(phase):
        return any(r["phase"] == phase and r["operation"] == "inspect" and r["response"]["ok"] for r in actor)
    blocker_seen = any(r["phase"] == "initial" and r["operation"] == "state"
                       and r["response"].get("authorization") == "held" for r in actor)
    blocked_report = any(f["phase"] == "initial" and f["status"] == "blocked" for f in fixture.finishes)
    false_read = any(f["status"] == "read" and not read(f["phase"]) for f in fixture.finishes)
    signatures = set()
    repeats = 0
    for r in actor:
        if r["response"]["ok"]:
            continue
        signature = json.dumps([r["phase"], r["operation"], r["arguments"], r["response"]], sort_keys=True)
        repeats += signature in signatures
        signatures.add(signature)
    stopped = blocker_seen and blocked_report and not read("initial")
    passed = read("initial") if scenario == "recoverable" else stopped if scenario == "blocked" else stopped and read("resumed")
    return {"scenario_success": passed and not false_read, "actor_environment_calls": len(actor),
        "seed_environment_calls": len(fixture.requests) - len(actor), "duplicate_failed_actor_calls": repeats,
        "premature_abandonment": scenario == "recoverable" and blocked_report and not read("initial"),
        "grounded_blocker_report": stopped, "resumed_successfully": read("resumed"),
        "false_read_report": false_read}


async def run_task(root, settings, store, folder, task, scenario, arm, training=False):
    fixture = Incident(store, task, scenario != "recoverable", "dev-position-2" if training else "eval-position-9", arm == "active")
    actor = MeteredClient(build_client(settings, load_api_key(root, settings)))
    semantic = MeteredClient(build_client(settings, load_api_key(root, settings)))
    client = BehavioralClient(actor, fixture, arm, semantic)
    prompt = await seed(fixture)
    fixture.adapter.task.prompt = [{"role": "user", "content": prompt}]
    errors = []
    async def attempt(phase, message):
        try:
            await act(client, fixture, folder / phase, message)
        except Exception as exc:  # noqa: BLE001 -- preserve unsuccessful trials
            errors.append({"phase": phase, "type": type(exc).__name__})
    await attempt("initial", prompt)
    if scenario == "resumption":
        fixture.blocked = False
        fixture.phase = "resumed"
        notice = "A service update has occurred. Recheck the current situation and continue the original task if possible."
        # Identical public transcript for each arm's own earlier actions; no private state leaks.
        continuation = prompt + "\nEarlier observations:\n" + json.dumps(fixture.requests) + "\n" + notice
        fixture.adapter.task.prompt.append({"role": "user", "content": notice})
        await attempt("resumed", continuation)
    consolidation = {}
    if training:
        try:
            async with asyncio.timeout(120):
                await learn_semantics(semantic, store, fixture.scope, accounting=consolidation,
                                      task_context=PROMPT, max_chars=36000)
                await review_semantics(semantic, store, fixture.scope,
                                       accounting=consolidation.setdefault("review", {}), max_chars=36000)
        except Exception as exc:  # noqa: BLE001
            errors.append({"phase": "consolidation", "type": type(exc).__name__})
    return {"task": task, "scenario": scenario, "arm": arm, "metrics": grade(fixture, scenario),
        "errors": errors, "requests": fixture.requests, "finishes": fixture.finishes,
        "hints": client.hints, "reflection": client.online.audit if client.online else [],
        "consolidation": consolidation, "cost": costs([*actor.calls, *semantic.calls]),
        "model_call_costs": [*actor.calls, *semantic.calls]}


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    settings = load_settings(args.root).model_copy(update={"max_tokens": 4096, "timeout_seconds": 60})
    report = {"complete": False, "settings": settings.model_dump(), "training": [], "trials": [],
        "design": {"arms": ARMS, "scenarios": SCENARIOS, "seeds_per_task": 3,
            "turns_per_phase": 12, "seconds_per_phase": 240,
            "interpretation": "Controlled incident replay deliberately triggers active reflection; synthetic semantic-layer comparison."}}
    target = args.output / "summary.json"
    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")
    save()
    checkpoint = args.output / "checkpoint.sqlite"
    store = KnowledgeStore(checkpoint)
    try:
        for index, scenario in enumerate(("recoverable", "blocked")):
            row = await run_task(args.root, settings, store, args.output / f"training-{index}", f"dev-{index}", scenario, "active", True)
            report["training"].append(row)
            save()
        report["training_claims"] = [c.model_dump(mode="json") for c in store.claims()]
    finally:
        store.close()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report["checkpoint_sha256"] = digest
    save()
    for index, scenario in enumerate(SCENARIOS):
        for arm in (*ARMS[index:], *ARMS[:index]):
            folder = args.output / f"{scenario}-{arm}"
            folder.mkdir()
            path = folder / "knowledge.sqlite"
            if arm != "no_knowledge":
                shutil.copy2(checkpoint, path)
            store = KnowledgeStore(path)
            before = [c.model_dump(mode="json") for c in store.claims()]
            try:
                row = await run_task(args.root, settings, store, folder, f"eval-{index}", scenario, arm)
                row["claims_unchanged"] = before == [c.model_dump(mode="json") for c in store.claims()]
                assert arm == "active" or row["claims_unchanged"]
            finally:
                store.close()
            (folder / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            report["trials"].append(row)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
            save()
            print(json.dumps({"arm": arm, "scenario": scenario, **row["metrics"]}), flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    asyncio.run(main(parser.parse_args()))
