import asyncio
from datetime import timedelta
from types import SimpleNamespace

from worldmodel_harness.budget import Ledger
from worldmodel_harness.knowledge import Claim, Dependency, Episode, LiveState, Scope, utc_now
from worldmodel_harness.knowledge_lookup import LookupKnowledge, knowledge_lookup_tool
from worldmodel_harness.store import KnowledgeStore


def test_targeted_local_lookup_respects_scope_freshness_and_budget(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="unfamiliar", version="1", tenant="one", task_id="task")
    observation = Episode(
        scope=scope,
        task_id="task",
        operation="docs",
        arguments={},
        response={"contract": "custom lookup"},
        outcome="success",
    )
    store.append_episode(observation)
    base = Claim(
        claim_id="contract",
        layer="interface",
        relation="endpoint_contract",
        participants=("custom_lookup",),
        statement="Custom lookup capability",
        scope=scope.model_copy(update={"task_id": None}),
        status="supported",
        supporting_episodes=(observation.episode_id,),
    )
    store.put_claim(base)
    other = Episode(
        scope=scope.model_copy(update={"tenant": "other"}),
        task_id="task",
        operation="docs",
        arguments={},
        response={},
        outcome="success",
    )
    store.append_episode(other)
    store.put_claim(
        base.model_copy(
            update={
                "claim_id": "secret-other-tenant",
                "scope": scope.model_copy(update={"tenant": "other"}),
                "supporting_episodes": (other.episode_id,),
            }
        )
    )
    store.put_claim(
        base.model_copy(
            update={"claim_id": "stale", "valid_until": utc_now() - timedelta(seconds=1)}
        )
    )
    store.put_claim(
        base.model_copy(
            update={
                "claim_id": "dependent",
                "dependencies": (Dependency(claim_id="absent", version=1),),
            }
        )
    )
    adapter = SimpleNamespace(knowledge=store, scope=scope, live=LiveState("task"), ledger=Ledger())
    audit = []
    tool = knowledge_lookup_tool(
        adapter,
        max_chars=4000,
        expand=True,
        adaptive=False,
        allow_experimental=False,
        audit=audit,
        max_calls=2,
    )
    result = asyncio.run(tool.executor(LookupKnowledge(query="custom lookup", max_chars=12000)))
    assert result.success
    assert "secret-other-tenant" not in result.content
    assert audit[-1]["injected_claims"] == ["contract"]
    assert "refresh:stale" in audit[-1]["gaps"]
    assert len(result.content) <= 4000
    assert adapter.ledger.environment_calls == 0
    assert adapter.ledger.knowledge_queries == 1
    assert adapter.ledger.compute_seconds > 0
    # Re-query current versions, not an indefinitely cached earlier result.
    store.put_claim(base.model_copy(update={"version": 2, "status": "disputed"}))
    asyncio.run(tool.executor(LookupKnowledge(query="custom lookup")))
    assert audit[-1]["injected_claims"] == []
    assert not asyncio.run(tool.executor(LookupKnowledge(query="custom lookup"))).success
    assert adapter.ledger.knowledge_queries == 2
    store.close()
