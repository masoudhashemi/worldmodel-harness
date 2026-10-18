import asyncio
import json

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import BudgetExceeded, Ledger
from worldmodel_harness.information import CURRENT_INFORMATION, InspectContract, information_tool
from worldmodel_harness.learning import Learner
from worldmodel_harness.store import KnowledgeStore


def request(**changes):
    return InspectContract(
        **{
            "query": "salesforce contacts",
            "top_k": 2,
            "uncertainty": "Which contact operations are available?",
            "distinguishing_outcomes": (
                "A contact query contract is documented",
                "No matching contract is returned",
            ),
            **changes,
        }
    )


def adapter(tmp_path, *, max_calls=5):
    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    result = AutomationBenchAdapter(
        dataset("simple")[0],
        "simple",
        store,
        Ledger(max_environment_calls=max_calls),
        Learner(store),
    )
    return result, store


def test_real_discovery_logs_intent_and_charges_shared_budget(tmp_path):
    async def scenario():
        env, store = adapter(tmp_path)
        tool = information_tool(env, policy="task", max_calls=2)
        result = await tool.executor(request())
        assert result.success
        assert len(json.loads(result.content)["results"]) == 2
        event = store.episodes(env.scope)[0]
        assert event.operation == "api_search"
        assert event.purpose == "probe"
        assert event.information_request.uncertainty == request().uncertainty
        assert env.ledger.by_purpose == {"probe": 1}
        assert len(store.claims(env.scope)) == 2
        assert CURRENT_INFORMATION.get() is None
        target = store.claims(env.scope)[0]
        result = await tool.executor(request(target_claim_id=target.claim_id))
        assert result.success
        assert env.ledger.by_purpose == {"probe": 1, "refresh": 1}
        assert not (await tool.executor(request())).success
        assert env.ledger.environment_calls == 2
        store.close()

    asyncio.run(scenario())


def test_task_policy_keeps_diagnosis_but_rejects_future_only_probe(tmp_path):
    async def scenario():
        env, store = adapter(tmp_path)
        tool = information_tool(env, policy="task", max_calls=2)
        assert not (await tool.executor(request(intended_use="future_tasks"))).success
        assert env.ledger.environment_calls == 0
        assert (await tool.executor(request())).success
        future = information_tool(env, policy="future", max_calls=1)
        assert (await future.executor(request(intended_use="future_tasks"))).success
        assert store.episodes(env.scope)[-1].information_request.intended_use == "future_tasks"
        assert env.ledger.environment_calls == 2
        store.close()

    asyncio.run(scenario())


def test_information_cannot_exceed_environment_budget_or_leak_context(tmp_path):
    async def scenario():
        env, store = adapter(tmp_path, max_calls=0)
        tool = information_tool(env, policy="task", max_calls=2)
        with pytest.raises(BudgetExceeded):
            await tool.executor(request())
        assert CURRENT_INFORMATION.get() is None
        assert store.episodes(env.scope) == []
        store.close()

    asyncio.run(scenario())


def test_invalid_target_and_non_distinguishing_request_do_not_execute(tmp_path):
    async def scenario():
        env, store = adapter(tmp_path)
        tool = information_tool(env, policy="task", max_calls=2)
        assert not (await tool.executor(request(target_claim_id="invented"))).success
        assert not (await tool.executor(request(distinguishing_outcomes=("same", "same")))).success
        assert env.ledger.environment_calls == 0
        store.close()

    asyncio.run(scenario())
