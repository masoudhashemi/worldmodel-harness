"""Small real-model acceptance checks before any broader benchmark run.

Fixtures define an environment, not production routing or recovery rules. The
recovery fixture seeds three actual rejected requests, then measures model actions.
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from stirrup.core.models import Tool, ToolResult, UserMessage

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.config import load_api_key, load_settings
from worldmodel_harness.experiment import ExperimentSettings, Finish, KnowledgeClient
from worldmodel_harness.goal_checks import GoalTracker
from worldmodel_harness.interactions import InteractionRecorder
from worldmodel_harness.knowledge import Episode, LiveState, Record, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.online_learning import OnlineLearner
from worldmodel_harness.repairing_agent import RepairingAgent
from worldmodel_harness.runtime import MeteredClient, build_client
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.verification_tools import simple_check_tool


class CursorInput(Record):
    cursor: str


class FetchInput(Record):
    method: str
    url: str
    body: dict | None = None


async def run_session(client, adapter, tools, finish, output, prompt, turns):
    recorder = InteractionRecorder(adapter)
    recorder.register_tools([*tools, finish])
    agent = RepairingAgent(client=client, name="acceptance", tools=tools, finish_tool=finish,
        max_turns=turns, interaction_recorder=recorder,
        repair_schemas={t.name: t.parameters for t in [*tools, finish]})
    async with agent.session(output_dir=output) as session:
        return await session.run([UserMessage(content=prompt)])


async def recovery(root, output, settings):
    store = KnowledgeStore(output / "recovery.sqlite")
    actor = MeteredClient(build_client(settings, load_api_key(root, settings)))
    semantic = MeteredClient(build_client(settings, load_api_key(root, settings)))
    scope = Scope(environment="cursor-fixture", version="1", tenant="fixture", task_id="recovery")
    prompt = "Read the current record. Prior requests using an obsolete cursor failed; inspect the observed feedback. Finish only after reading it."
    adapter = SimpleNamespace(scope=scope, knowledge=store, evidence=store, learner=Learner(store),
        retain_interactions=False, ledger=Ledger(), live=LiveState(scope.task_id),
        task=SimpleNamespace(prompt=[{"role": "user", "content": prompt}]), prediction_assessments=[])
    actual = []

    async def inspect(params):
        ok = params.cursor == "observed-token-7"
        body = {"ok": True, "record": {"id": "r7", "state": "ready"}} if ok else {
            "ok": False, "error": "invalid_cursor", "current_cursor": "observed-token-7"}
        actual.append({"arguments": params.model_dump(), "response": body})
        return ToolResult(content=json.dumps(body), success=ok)

    tool = Tool(name="inspect", description="Read a record using a current cursor. A rejected request may return the current cursor.",
                parameters=CursorInput, executor=inspect)
    for index in range(3):
        args = CursorInput(cursor="obsolete")
        response = await inspect(args)
        store.append_episode(Episode(task_id=scope.task_id, scope=scope, origin="harness",
            operation="tool_call:inspect", interaction_id=str(index), arguments=args.model_dump(),
            response={"result": json.loads(response.content)}, coverage=("result",), outcome="error"))
    wrapper = KnowledgeClient(actor, adapter, store,
        ExperimentSettings(operational=True, semantic_learning=True))
    wrapper.online_learner = OnlineLearner(adapter, semantic, max_reflections=1, timeout_seconds=120)
    finish = Tool(name="finish", description="Finish after reading the requested record.",
                  parameters=Finish, executor=lambda _: ToolResult(content="Finished"))
    result = {"fixture_seeded_failures": 3}
    try:
        async with asyncio.timeout(240):
            await run_session(wrapper, adapter, [tool], finish, output / "recovery-session", prompt, 5)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    audit = wrapper.online_learner.audit
    result.update(actual_requests=actual, reflection=audit,
        model_call_costs=[*actor.calls, *semantic.calls],
        passed=bool(audit and audit[0].get("experiment_outcome", {}).get("result") == "executed"
            and any(a["response"].get("ok") for a in actual[3:])
            and any(p["result"] == "matched" for p in audit[0].get("experiment_outcome", {}).get("predictions", []))))
    store.close()
    return result


async def verification(root, output, settings):
    store = KnowledgeStore(output / "verification.sqlite")
    adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", store, Ledger())
    prompt = "Set record-7 status to done at https://fixture.test/records/record-7. Read the current record first, update it, then read it back and verify the effect before finishing."
    adapter.task = SimpleNamespace(prompt=[{"role": "user", "content": prompt}])
    adapter.goal_tracker = GoalTracker(adapter)
    actor = MeteredClient(build_client(settings, load_api_key(root, settings)))
    state = {"id": "record-7", "status": "pending"}
    methods = []

    def api_fetch(method, url, body=None):
        methods.append(method)
        if url != "https://fixture.test/records/record-7":
            return json.dumps({"error": "unknown_resource"})
        if method == "PATCH":
            state.update(body or {})
        elif method != "GET":
            return json.dumps({"error": "unsupported_method"})
        return json.dumps(state)

    async def fetch(params):
        result = await adapter.execute(api_fetch, params.model_dump())
        event = [e for e in store.episodes(adapter.scope) if e.operation == "api_fetch"][-1]
        return ToolResult(content=result.content + "\nTask evidence references: " + event.episode_id,
                          success=result.success)

    tools = [Tool(name="api_fetch", description="GET reads the resource. PATCH updates fields supplied in body. The response includes its evidence ID.",
                  parameters=FetchInput, executor=fetch), simple_check_tool(adapter)]
    result = {}
    try:
        async with asyncio.timeout(180):
            await run_session(actor, adapter, tools, adapter.goal_tracker.finish_tool(),
                              output / "verification-session", prompt, 9)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    result.update(methods=methods, state=state, checks=adapter.goal_tracker.report(),
        verification=adapter.goal_tracker.verification_audit, finish=adapter.goal_tracker.finish_audit,
        model_call_costs=actor.calls,
        passed=(state["status"] == "done" and "PATCH" in methods
            and "GET" in methods[:methods.index("PATCH")] and "GET" in methods[methods.index("PATCH") + 1:]
            and getattr(adapter.goal_tracker, "completion_status", None) == "checked"))
    store.close()
    return result


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    settings = load_settings(args.root).model_copy(update={"max_tokens": 4096, "timeout_seconds": 60})
    report = {"model": settings.model_dump(), "complete": False}
    path = args.output / "summary.json"
    for name, case in (("recovery", recovery), ("verification", verification)):
        report[name] = await case(args.root, args.output, settings)
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    report["passed"] = report["recovery"]["passed"] and report["verification"]["passed"]
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    asyncio.run(main(parser.parse_args()))
