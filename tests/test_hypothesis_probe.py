import asyncio

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import BudgetExceeded, Ledger
from worldmodel_harness.decisions import ProposedAction
from worldmodel_harness.hypothesis_probe import HypothesisProbe
from worldmodel_harness.information import CURRENT_INFORMATION
from worldmodel_harness.knowledge import Claim
from worldmodel_harness.learning import Learner
from worldmodel_harness.operational_tools import operational_tools
from worldmodel_harness.semantic_learning import investigation_context
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", store, Ledger(), Learner(store)
    )
    store.put_claim(
        Claim(
            claim_id="question",
            layer="behavior",
            relation="possible_routing",
            participants=("mail",),
            statement="mail list may return message identifiers",
            scope=adapter.scope,
            needs_review=False,
            payload={
                "alternatives": ["identifiers absent"],
                "distinguishing_observation": "read list response",
            },
        )
    )
    audit = []
    adapter.information_audit = []
    tools = operational_tools(
        adapter,
        adapter.tools(),
        adaptive=False,
        allow_experimental=False,
        audit=audit,
        refresh_audit=[],
        hypothesis_policy="task",
        max_hypothesis_probes=2,
        information_audit=adapter.information_audit,
    )
    probe = next(t for t in tools if t.name == "investigate_hypothesis")
    yield store, adapter, probe, audit
    store.close()


def request(tool="api_search", arguments=None, **updates):
    return HypothesisProbe(
        uncertainty="Does mail listing expose identifiers?",
        target_claim_id="question",
        distinguishing_outcomes=("identifiers present", "identifiers absent"),
        action=ProposedAction(
            tool=tool, arguments=arguments or {"query": "gmail list messages", "top_k": 5}
        ),
        **updates,
    )


def test_probe_checks_contract_and_logs_actual_observation_without_promotion(world):
    store, adapter, probe, audit = world
    fetch = request(
        "api_fetch",
        {"method": "GET", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages"},
    )
    assert not asyncio.run(probe.executor(fetch)).success
    assert adapter.ledger.environment_calls == 0
    assert not store.claim("question").needs_review
    assert asyncio.run(probe.executor(request())).success
    assert asyncio.run(probe.executor(fetch)).success
    assert adapter.ledger.environment_calls == 2
    assert len(adapter.information_audit) == 3
    assert all(r["tool"] == "investigate_hypothesis" for r in adapter.information_audit)
    assert adapter.ledger.by_purpose == {"probe": 2}
    claim = store.claim("question")
    assert claim.status == "hypothesis"
    assert claim.needs_review
    event = adapter.evidence.episodes(adapter.scope)[-1]
    assert event.information_request.target_claim_id == "question"
    assert event.information_request.distinguishing_outcomes == (
        "identifiers present",
        "identifiers absent",
    )
    assert CURRENT_INFORMATION.get() is None
    assert audit[-1]["invoked"]
    assert not asyncio.run(probe.executor(request())).success
    assert adapter.ledger.environment_calls == 2


@pytest.mark.parametrize("method", ["POST", "PATCH", "DELETE", None])
def test_probe_rejects_mutations_and_invalid_methods_without_calls(world, method):
    _store, adapter, probe, _audit = world
    assert not asyncio.run(
        probe.executor(request("api_fetch", {"method": method, "url": "https://example.invalid"}))
    ).success
    assert adapter.ledger.environment_calls == 0


def test_future_only_probe_is_blocked_by_task_policy(world):
    _store, adapter, probe, _audit = world
    assert not asyncio.run(probe.executor(request(intended_use="future_tasks"))).success
    assert adapter.ledger.environment_calls == 0


def test_probe_cannot_bypass_shared_budget_and_resets_context(world):
    _store, adapter, probe, _audit = world
    adapter.ledger.max_environment_calls = 0
    with pytest.raises(BudgetExceeded):
        asyncio.run(probe.executor(request()))
    assert adapter.ledger.environment_calls == 0
    assert CURRENT_INFORMATION.get() is None


def test_cross_scope_target_is_rejected(world):
    store, adapter, probe, _audit = world
    claim = store.claim("question")
    store.put_claim(
        claim.model_copy(
            update={
                "claim_id": "other-question",
                "scope": claim.scope.model_copy(update={"tenant": "other"}),
            }
        )
    )
    assert not asyncio.run(
        probe.executor(request().model_copy(update={"target_claim_id": "other-question"}))
    ).success
    assert adapter.ledger.environment_calls == 0


def test_pending_questions_are_not_operational_advice_or_rejected_claims(world):
    store, adapter, _probe, _audit = world
    c = store.claim("question")
    store.put_claim(c.model_copy(update={"version": 2, "needs_review": True}))
    hint = investigation_context(store, adapter.scope, "mail", max_chars=1200)
    assert "question" in hint
    assert "do not assume" in hint
    assert len(hint) <= 1200
    c = store.claim("question")
    store.put_claim(
        c.model_copy(
            update={
                "version": 3,
                "payload": {**c.payload, "semantic_review": {"verdict": "counterevidence"}},
            }
        )
    )
    assert not investigation_context(store, adapter.scope, "mail")
