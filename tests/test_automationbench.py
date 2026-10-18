import asyncio
import json

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import BudgetExceeded, Ledger
from worldmodel_harness.learning import Learner
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
