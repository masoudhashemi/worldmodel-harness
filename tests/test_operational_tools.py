import asyncio
import json

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.decisions import Decision, ProposedAction, UnresolvedRequirements
from worldmodel_harness.learning import Learner
from worldmodel_harness.operational_requirements import api_requirements
from worldmodel_harness.operational_tools import RefreshInput, operational_tools
from worldmodel_harness.store import KnowledgeStore


def test_compact_aliases_resolve_to_checked_actions_and_real_evidence(tmp_path):
    from worldmodel_harness.decisions import CompactDecision

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", store, Ledger(), Learner(store)
    )
    audit = []
    tools = operational_tools(
        adapter,
        adapter.tools(),
        adaptive=False,
        allow_experimental=False,
        audit=audit,
        refresh_audit=[],
    )
    actor = tools[0]
    catalog = json.loads(
        actor.description.split("Available action tools (names and parameter schemas): ")[1]
    )
    alias = next(key for key, value in catalog.items() if value["name"] == "api_search")
    result = asyncio.run(
        actor.executor(
            CompactDecision(
                action=ProposedAction(
                    tool=alias, arguments={"query": "gmail list messages", "top_k": 1}
                )
            )
        )
    )
    assert result.success
    assert audit[-1]["decision"]["action"]["tool"] == "api_search"
    real_id = adapter.evidence_aliases["e1"]
    assert store.episode(real_id).operation == "api_search"
    followup = CompactDecision(
        action=ProposedAction(tool="lookup_knowledge", arguments={"query": "gmail"}),
        evidence_ids=("e1",),
    )
    assert asyncio.run(actor.executor(followup)).success
    assert audit[-1]["decision"]["evidence_ids"] == [real_id]
    assert adapter.ledger.environment_calls == 1
    from worldmodel_harness.knowledge import Episode

    source = Episode(
        task_id=adapter.scope.task_id,
        scope=adapter.scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://source.test/message"},
        response={"text": "Use new-value"},
        coverage=("text",),
        outcome="success",
    )
    target = Episode(
        task_id=adapter.scope.task_id,
        scope=adapter.scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://target.test/item"},
        response={"value": "new-value"},
        coverage=("value",),
        outcome="success",
    )
    for event in (source, target):
        store.append_episode(event)
        adapter.live.observe(event)
    adapter.evidence_aliases.update(e2=source.episode_id, e3=target.episode_id)
    observation = {"path": ["value"], "predicate": "equals", "expected": "new-value"}
    result = asyncio.run(
        actor.executor(
            CompactDecision(
                action=ProposedAction(
                    tool="record_goal_checks",
                    arguments={
                        "checks": [
                            {
                                "requirement_id": "synthetic-contract",
                                "task_quote": adapter.task.prompt[0]["content"],
                                "episode_id": "e3",
                                "observation": observation,
                                "outcome": {
                                    "target": observation,
                                    "resource_url": "https://target.test/item",
                                    "source_episode_id": "e2",
                                    "source_path": ["text"],
                                },
                            }
                        ]
                    },
                )
            )
        )
    )
    assert result.success
    check = adapter.goal_tracker.checks["synthetic-contract"]
    assert check.episode_id == target.episode_id
    assert check.outcome.source_episode_id == source.episode_id
    assert adapter.goal_tracker.assess(check) == "matched"
    store.close()


def test_operational_surface_discovers_before_fetch_and_refreshes_with_same_ledger(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", store, Ledger(), Learner(store)
    )
    audit, refresh_audit = [], []
    tools = {
        t.name: t
        for t in operational_tools(
            adapter,
            adapter.tools(),
            adaptive=True,
            allow_experimental=True,
            audit=audit,
            refresh_audit=refresh_audit,
        )
    }
    assert set(tools) == {"execute_decision", "refresh_live_fact", "check_effect"}
    url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"

    async def act(tool, args):
        return await tools["execute_decision"].executor(
            Decision(
                goal="read current inbox",
                rationale="observe current state",
                action=ProposedAction(tool=tool, arguments=args),
            )
        )

    fetch = {"method": "GET", "url": url, "params": '{"q":"in:inbox"}'}
    blocked = asyncio.run(act("api_fetch", fetch))
    assert not blocked.success
    assert not json.loads(blocked.content)["environment_called"]
    assert "contract" in json.loads(blocked.content)["next_step"]
    assert adapter.ledger.environment_calls == 0
    assert audit[-1]["gate"]["gaps"] == ("requirements:unresolved",)
    assert asyncio.run(act("api_search", {"query": "gmail list messages", "top_k": 5})).success
    ordinary = ProposedAction(tool="api_fetch", arguments={"method": "GET", "url": url})
    queried = ordinary.model_copy(
        update={"arguments": {"method": "GET", "url": url + "?maxResults=1#ignored"}}
    )
    assert api_requirements(store, adapter.scope, ordinary) == api_requirements(
        store, adapter.scope, queried
    )
    assert queried.arguments["url"].endswith("?maxResults=1#ignored")
    with pytest.raises(UnresolvedRequirements):
        api_requirements(
            store,
            adapter.scope,
            ordinary.model_copy(
                update={
                    "arguments": {
                        "method": "GET",
                        "url": url.replace("gmail.googleapis.com", "unrelated.invalid"),
                    }
                }
            ),
        )
    assert asyncio.run(act("lookup_knowledge", {"query": "gmail list messages"})).success
    assert adapter.ledger.environment_calls == 1
    assert adapter.ledger.knowledge_queries == 1
    assert asyncio.run(act("api_fetch", fetch)).success
    assert adapter.ledger.environment_calls == 2
    result = asyncio.run(
        tools["refresh_live_fact"].executor(
            RefreshInput(
                key="api_fetch:" + url + ":messages",
                expected_stale_decision_loss=10,
                check_cost=1,
            )
        )
    )
    assert json.loads(result.content)["refreshed"]
    assert adapter.ledger.environment_calls == 3
    assert adapter.ledger.by_purpose["refresh"] == 1
    assert len(refresh_audit) == 1
    assert adapter.ledger.compute_seconds >= sum(row["gate_seconds"] for row in audit)
    store.close()


def test_frozen_operational_followup_can_cite_current_task_episode(tmp_path):
    import hashlib

    from automationbench.tools.api import api_search

    path = tmp_path / "knowledge.sqlite"
    seed = KnowledgeStore(path)
    row = dataset("simple")[0]
    bootstrap = AutomationBenchAdapter(row, "simple", seed, Ledger(), Learner(seed))
    asyncio.run(bootstrap.execute(api_search, {"query": "gmail list messages", "top_k": 5}))
    seed.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    knowledge = KnowledgeStore(path, frozen=True)
    evidence = KnowledgeStore(tmp_path / "current-evidence.sqlite")
    adapter = AutomationBenchAdapter(row, "simple", evidence, Ledger(), knowledge=knowledge)
    audit = []
    tools = operational_tools(
        adapter,
        adapter.tools(),
        adaptive=False,
        allow_experimental=False,
        audit=audit,
        refresh_audit=[],
    )
    decision = Decision(
        goal="read current inbox",
        rationale="inspect current records",
        action=ProposedAction(
            tool="api_fetch",
            arguments={
                "method": "GET",
                "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            },
        ),
    )
    assert asyncio.run(tools[0].executor(decision)).success
    current = evidence.episodes(adapter.scope)[-1]
    with pytest.raises(KeyError):
        knowledge.episode(current.episode_id)
    assert asyncio.run(
        tools[0].executor(decision.model_copy(update={"evidence_ids": (current.episode_id,)}))
    ).success
    assert audit[-1]["invoked"]
    assert adapter.ledger.environment_calls == 2
    knowledge.close()
    evidence.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_compact_decision_bare_alias_uses_goal_query(tmp_path):
    from worldmodel_harness.decisions import CompactDecision

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", store, Ledger(), Learner(store)
    )
    tools = operational_tools(
        adapter,
        adapter.tools(),
        adaptive=False,
        allow_experimental=False,
        audit=[],
        refresh_audit=[],
    )
    actor = tools[0]
    
    # 1. Single-parameter generic inference for api_search
    cd_search = CompactDecision(action="a1:api_search", goal="Find spreadsheets endpoints")
    res_search = asyncio.run(actor.executor(cd_search))
    assert res_search.success
    assert "results" in res_search.content

    # 2. Single-parameter generic inference for lookup_knowledge
    cd_lookup = CompactDecision(action="lookup_knowledge", goal="spreadsheet permissions")
    res_lookup = asyncio.run(actor.executor(cd_lookup))
    assert res_lookup.success

    # 3. Multi-parameter tool (api_fetch) without arguments produces actionable format guidance
    cd_fetch = CompactDecision(action="api_fetch", goal="fetch data")
    res_fetch = asyncio.run(actor.executor(cd_fetch))
    assert not res_fetch.success
    assert "expected_format" in res_fetch.content
    assert "arguments" in res_fetch.content

    store.close()

