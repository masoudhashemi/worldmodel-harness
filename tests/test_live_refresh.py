import asyncio
from datetime import timedelta

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import BudgetExceeded, Ledger
from worldmodel_harness.decisions import Decision, ProposedAction, check_decision
from worldmodel_harness.knowledge import utc_now
from worldmodel_harness.learning import Learner
from worldmodel_harness.live_refresh import RefreshEstimate, refresh_live_fact
from worldmodel_harness.operational_requirements import api_requirements
from worldmodel_harness.store import KnowledgeStore

URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
KEY = "api_fetch:" + URL + ":messages"


@pytest.fixture
def adapter(tmp_path):
    from automationbench.tools.api import api_fetch, api_search

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    result = AutomationBenchAdapter(dataset("simple")[0], "simple", store, Ledger(), Learner(store))
    asyncio.run(result.execute(api_search, {"query": "gmail list messages", "top_k": 5}))
    asyncio.run(
        result.execute(api_fetch, {"url": URL, "method": "GET", "params": '{"q":"in:inbox"}'})
    )
    yield result
    store.close()


def estimate(**changes):
    return RefreshEstimate(expected_stale_decision_loss=10, check_cost=1, **changes)


def test_refresh_replays_observed_read_and_charges_shared_budget(adapter):
    before = adapter.ledger.environment_calls
    adapter.live.invalidate({KEY}, "write")
    result = asyncio.run(refresh_live_fact(adapter, KEY, estimate()))
    assert result.attempted and result.refreshed
    assert result.source_episode_id != result.observation_episode_id
    assert adapter.ledger.environment_calls == before + 1
    event = adapter.evidence.episode(result.observation_episode_id)
    assert event.purpose == "refresh"
    assert event.arguments == adapter.evidence.episode(result.source_episode_id).arguments


def test_optional_cost_gate_and_mandatory_override(adapter):
    before = adapter.ledger.environment_calls
    result = asyncio.run(
        refresh_live_fact(
            adapter, KEY, RefreshEstimate(expected_stale_decision_loss=0, check_cost=1)
        )
    )
    assert not result.attempted
    assert adapter.ledger.environment_calls == before
    result = asyncio.run(
        refresh_live_fact(
            adapter,
            KEY,
            RefreshEstimate(expected_stale_decision_loss=0, check_cost=1, mandatory=True),
        )
    )
    assert result.refreshed


def test_mandatory_refresh_cannot_bypass_budget(adapter):
    before = adapter.live.facts()
    adapter.ledger.max_environment_calls = adapter.ledger.environment_calls
    with pytest.raises(BudgetExceeded):
        asyncio.run(refresh_live_fact(adapter, KEY, estimate(mandatory=True)))
    assert adapter.live.facts() == before


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_pre_observation_failure_preserves_existing_fact(adapter, monkeypatch, failure):
    before = adapter.live.facts()

    async def fail(*args, **kwargs):
        raise failure()

    monkeypatch.setattr(adapter, "execute", fail)
    with pytest.raises(failure):
        asyncio.run(refresh_live_fact(adapter, KEY, estimate()))
    assert adapter.live.facts() == before


def test_unknown_field_does_not_invent_refresh_route(adapter):
    before = adapter.ledger.environment_calls
    result = asyncio.run(refresh_live_fact(adapter, KEY + "unknown", estimate(mandatory=True)))
    assert not result.attempted
    assert result.reason == "no_task_local_read_provenance"
    assert adapter.ledger.environment_calls == before


def test_failed_refresh_does_not_restore_old_value(adapter, monkeypatch):
    from stirrup.core.models import ToolResult

    async def fail(*args, **kwargs):
        from worldmodel_harness.knowledge import Episode

        episode = Episode(
            task_id=adapter.scope.task_id,
            scope=adapter.scope,
            operation="api_fetch",
            arguments=args[1],
            response={"error": "failed read"},
            outcome="error",
            coverage=("error",),
            purpose="refresh",
        )
        adapter.evidence.append_episode(episode)
        adapter.live.observe(episode)
        return ToolResult(content="unknown", success=False)

    monkeypatch.setattr(adapter, "execute", fail)
    result = asyncio.run(refresh_live_fact(adapter, KEY, estimate()))
    assert result.attempted and not result.refreshed
    assert KEY not in {f.key for f in adapter.live.facts()}


def test_pre_observation_failure_does_not_restore_already_invalidated_fact(adapter, monkeypatch):
    adapter.live.invalidate({KEY}, "earlier-write")

    async def fail(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(adapter, "execute", fail)
    with pytest.raises(RuntimeError):
        asyncio.run(refresh_live_fact(adapter, KEY, estimate()))
    assert KEY not in {f.key for f in adapter.live.facts()}


def test_stale_contract_is_required_not_silently_ignored(adapter):
    action = ProposedAction(tool="api_fetch", arguments={"url": URL, "method": "GET"})
    requirements = api_requirements(adapter.knowledge, adapter.scope, action)
    claim = adapter.knowledge.claim(requirements.claims[0].claim_id)
    adapter.knowledge.put_claim(
        claim.model_copy(
            update={"version": claim.version + 1, "valid_until": utc_now() - timedelta(seconds=1)}
        )
    )
    requirements = api_requirements(adapter.knowledge, adapter.scope, action)
    gate = check_decision(
        Decision(goal="read", rationale="refresh", action=action),
        mandatory=requirements,
        store=adapter.knowledge,
        scope=adapter.scope,
        live=adapter.live,
        allowed_tools={"api_fetch"},
    )
    assert not gate.allowed
    assert "refresh_claim:" + claim.claim_id in gate.gaps
    before = adapter.ledger.environment_calls
    result = asyncio.run(refresh_live_fact(adapter, KEY, estimate()))
    assert not result.attempted
    assert adapter.ledger.environment_calls == before
