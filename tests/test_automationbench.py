import asyncio
import json

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import BudgetExceeded, Ledger
from worldmodel_harness.experiment import (
    ExperimentSettings,
    attach_validation_score,
    learns_from_trajectories,
)
from worldmodel_harness.learning import Learner
from worldmodel_harness.retrieval import retrieve
from worldmodel_harness.store import KnowledgeStore


def test_upstream_discovery_becomes_documented_claim_without_grader_access(tmp_path):
    async def scenario():
        from automationbench.tools.api import api_search

        store = KnowledgeStore(tmp_path / "evidence.sqlite")
        row = dataset("simple")[0]
        adapter = AutomationBenchAdapter(row, "simple", store, Ledger(), Learner(store))
        tools = adapter.tools()
        assert {t.name for t in tools} == {"api_search", "api_fetch", "base64_encode"}
        assert all("world" not in t.parameters.model_fields for t in tools)
        assert not hasattr(adapter.task, "assertions")
        assert not hasattr(adapter.task, "initial_state")
        result = await adapter.execute(api_search, {"query": "salesforce contacts", "top_k": 2})
        assert json.loads(result.content) == json.loads(api_search("salesforce contacts", 2))
        claims = store.claims(adapter.scope)
        assert len(claims) == 2
        assert all(c.origin == "documented" for c in claims)
        assert all(c.scope.task_id is None for c in claims)
        assert adapter.ledger.environment_calls == 1
        assert set(adapter.score()) == {"partial_credit", "task_completed_correctly"}
        # Task initialization never mutates the dataset's state.
        assert row == dataset("simple")[0]
        store.close()

    asyncio.run(scenario())


def test_budget_applies_before_environment_call(tmp_path):
    async def scenario():
        from automationbench.tools.api import api_search

        store = KnowledgeStore(tmp_path / "evidence.sqlite")
        adapter = AutomationBenchAdapter(
            dataset("simple")[0], "simple", store, Ledger(max_environment_calls=0)
        )
        with pytest.raises(BudgetExceeded):
            await adapter.execute(api_search, {"query": "salesforce contacts"})
        assert adapter.ledger.environment_calls == 0
        assert store.episodes(adapter.scope) == []
        store.close()

    asyncio.run(scenario())


def test_episode_arm_retains_raw_evidence_without_consolidating(tmp_path):
    async def scenario():
        from automationbench.tools.api import api_search

        evidence = KnowledgeStore(tmp_path / "events.sqlite")
        knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite")
        adapter = AutomationBenchAdapter(
            dataset("simple")[0],
            "simple",
            evidence,
            Ledger(),
            knowledge=knowledge,
            retain_interactions=True,
        )
        await adapter.execute(api_search, {"query": "salesforce contacts", "top_k": 1})
        assert len(knowledge.historical_episodes(adapter.scope)) == 1
        assert knowledge.claims() == []
        knowledge.close()
        evidence.close()

    asyncio.run(scenario())


def test_actual_message_list_to_get_learns_structural_connection(tmp_path):
    async def scenario():
        from automationbench.tools.api import api_fetch, api_search

        store = KnowledgeStore(tmp_path / "evidence.sqlite")
        adapter = AutomationBenchAdapter(
            dataset("simple")[0], "simple", store, Ledger(), Learner(store)
        )
        for query in ("gmail list messages", "gmail get message"):
            await adapter.execute(api_search, {"query": query, "top_k": 5})
        listed = await adapter.execute(
            api_fetch,
            {
                "method": "GET",
                "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                "params": '{"q":"in:inbox"}',
            },
        )
        identifier = json.loads(listed.content)["messages"][0]["id"]
        fetched = await adapter.execute(
            api_fetch,
            {
                "method": "GET",
                "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/" + identifier,
                "params": '{"format":"full"}',
            },
        )
        assert fetched.success
        flows = [c for c in store.claims() if c.relation == "observed_value_flow"]
        assert len(flows) == 1
        assert flows[0].payload["output_path"] == "messages.*.id"
        assert flows[0].payload["input_path"] == "path.id"
        assert identifier not in flows[0].model_dump_json()
        assert adapter.ledger.environment_calls == 4
        store.close()

    asyncio.run(scenario())


def test_train_and_test_learn_from_trajectories_without_validation_scores(tmp_path):
    async def scenario():
        from automationbench.tools.api import api_search

        assert learns_from_trajectories("train")
        assert learns_from_trajectories("test")
        assert not learns_from_trajectories("frozen")
        ExperimentSettings(phase="train")
        ExperimentSettings(phase="test")
        store = KnowledgeStore(tmp_path / "knowledge.sqlite")
        training = json.loads(json.dumps(dataset("simple")[0]))
        training_info = json.loads(training["info"])
        training_info["assertions"] = []
        training["info"] = json.dumps(training_info)
        trained = AutomationBenchAdapter(
            training, "simple", store, Ledger(), Learner(store), store
        )
        await trained.execute(api_search, {"query": "salesforce contacts", "top_k": 2})
        train_report = {}
        attach_validation_score(train_report, trained)
        assert train_report == {
            "validation_scores_used_for_learning": False,
            "validation_available": False,
        }
        learned = store.claims()
        assert learned
        assert "partial_credit" not in store.db.execute(
            "SELECT group_concat(body) FROM claims"
        ).fetchone()[0]

        later = AutomationBenchAdapter(
            dataset("simple")[1], "simple", store, Ledger(), Learner(store), store
        )
        carried = store.claims(later.scope)
        assert {c.claim_id for c in learned} <= {c.claim_id for c in carried}
        retrieve(store, later.scope, later.task.prompt[0]["content"], live=later.live)
        scored = {}
        attach_validation_score(scored, later)
        assert scored["validation_available"] is True
        assert "partial_credit" in scored
        assert "partial_credit" not in store.db.execute(
            "SELECT group_concat(body) FROM claims"
        ).fetchone()[0]
        store.close()

    asyncio.run(scenario())
