import asyncio
import importlib
from pathlib import Path

import pytest

from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("ambiguous_blocker_eval")


def test_same_failure_requires_new_observation_and_seeds_are_separate(evaluation, tmp_path):
    from blocker_behavior_eval import Read

    stores = [KnowledgeStore(tmp_path / f"{n}.sqlite") for n in range(2)]
    fixtures = [evaluation.Incident(store, "eval-opaque", blocked, "new-token", False)
                for store, blocked in zip(stores, (False, True), strict=True)]
    try:
        prompts = [asyncio.run(evaluation.seed(f)) for f in fixtures]
        assert prompts[0] == prompts[1]
        for fixture in fixtures:
            wrappers = [e for e in fixture.store.episodes(fixture.scope) if e.operation == "tool_call:inspect"]
            assert len(wrappers) == 3 and all(e.related_episode_ids for e in wrappers)
            asyncio.run(fixture.lookup(evaluation.Lookup(reference=fixture.receipt)))
            asyncio.run(fixture.state(evaluation.Lookup(reference=fixture.state_ref)))
            assert not evaluation.grade(fixture, "recoverable")["scenario_success"]  # diagnostic success is not a read
        asyncio.run(fixtures[0].inspect(Read(cursor="new-token")))
        assert evaluation.grade(fixtures[0], "recoverable")["scenario_success"]
        fixtures[1].finishes.append({"phase": "initial", "status": "blocked"})
        metrics = evaluation.grade(fixtures[1], "blocked")
        assert metrics["scenario_success"] and metrics["seed_environment_calls"] == 3
        assert metrics["actor_environment_calls"] == 2
        fixtures[1].blocked = False
        fixtures[1].phase = "resumed"
        asyncio.run(fixtures[1].state(evaluation.Lookup(reference=fixtures[1].state_ref)))
        assert not evaluation.grade(fixtures[1], "resumption")["scenario_success"]
        asyncio.run(fixtures[1].inspect(Read(cursor="new-token")))
        assert evaluation.grade(fixtures[1], "resumption")["scenario_success"]
    finally:
        for store in stores:
            store.close()


def test_guessing_blocker_without_state_evidence_fails(evaluation, tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite")
    try:
        fixture = evaluation.Incident(store, "eval-opaque", True, "token", False)
        asyncio.run(evaluation.seed(fixture))
        fixture.finishes.append({"phase": "initial", "status": "blocked"})
        assert not evaluation.grade(fixture, "blocked")["scenario_success"]
        assert evaluation.grade(fixture, "recoverable")["premature_abandonment"]
    finally:
        store.close()


def test_training_consolidation_calls_real_review_contract(evaluation, tmp_path, monkeypatch):
    from blocker_behavior_eval import Read
    from stirrup.core.models import AssistantMessage, TextBlock

    class Client:
        async def generate(self, messages, tools):
            return AssistantMessage(blocks=[TextBlock(text='{"hypotheses": []}')])

    async def act(client, fixture, folder, prompt):
        await fixture.inspect(Read(cursor=fixture.cursor))

    monkeypatch.setattr(evaluation, "act", act)
    monkeypatch.setattr(evaluation, "build_client", lambda *args: Client())
    monkeypatch.setattr(evaluation, "load_api_key", lambda *args: "test")
    store = KnowledgeStore(tmp_path / "checkpoint.sqlite")
    try:
        result = asyncio.run(evaluation.run_task(tmp_path, None, store, tmp_path,
                             "dev-0", "recoverable", "active", training=True))
        assert result["errors"] == []
        assert "review" in result["consolidation"]
    finally:
        store.close()
