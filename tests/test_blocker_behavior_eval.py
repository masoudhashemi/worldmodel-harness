import asyncio
import importlib.util
import json
from pathlib import Path

from stirrup.core.models import AssistantMessage, ToolCall

from worldmodel_harness.runtime import MeteredClient
from worldmodel_harness.store import KnowledgeStore

spec = importlib.util.spec_from_file_location("blocker_behavior_eval", Path(__file__).resolve().parents[1] / "scripts/blocker_behavior_eval.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_actor_sessions_record_stopping_and_resumption_without_grades_in_prompts(tmp_path):
    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    fixture = module.Fixture(store, "eval-resume", True, "new-token", False)

    class Client:
        model_slug = "scripted"
        context_window_tokens = 32768

        async def generate(self, messages, tools):
            assert "scenario_success" not in str(messages)
            requests = [r for r in fixture.requests if r["phase"] == fixture.phase]
            if not requests:
                call = ToolCall(name="inspect", arguments=json.dumps({"cursor": "new-token"}))
            else:
                call = ToolCall(name="finish", arguments=json.dumps({
                    "status": "blocked" if fixture.blocked else "read", "explanation": "Observed response"}))
            return AssistantMessage(blocks=[call])

    client = module.BehavioralClient(MeteredClient(Client()), fixture, "retained", None)

    async def run():
        await module.session(client, fixture, tmp_path / "first", module.PROMPT)
        fixture.blocked = False
        fixture.phase = "resumed"
        await module.session(client, fixture, tmp_path / "second", "Account reactivated; continue.")

    asyncio.run(run())
    result = module.metrics(fixture, "resumption")
    assert result["scenario_success"] and result["resumed_successfully"]
    assert result["environment_calls"] == 2
    assert result["duplicate_failed_calls"] == result["false_read_reports"] == 0
    assert store.claims() == []
    wrappers = [e for e in store.episodes(fixture.scope) if e.operation == "tool_call:inspect"]
    assert len(wrappers) == 2 and all(e.related_episode_ids for e in wrappers)
    store.close()


def test_grading_penalizes_unobserved_blockers_and_premature_abandonment(tmp_path):
    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    fixture = module.Fixture(store, "eval-case", False, "new-token", False)
    fixture.finishes.append({"phase": "initial", "status": "blocked", "explanation": "Guess"})
    assert not module.metrics(fixture, "blocked")["scenario_success"]
    assert module.metrics(fixture, "recoverable")["premature_abandonment"]
    store.close()
