import asyncio
import json

import pytest
from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.knowledge import Claim, Episode, Scope
from worldmodel_harness.semantic_learning import evidence_catalog, hypothesis_context
from worldmodel_harness.semantic_review import (
    HypothesisReview,
    ReviewBatch,
    retain_reviews,
    review_candidates,
    review_semantics,
)
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="new", version="1", tenant="tenant", task_id="task")
    episode = Episode(
        task_id="task",
        scope=scope,
        operation="read",
        arguments={},
        response={"header": "still present"},
        outcome="success",
    )
    store.append_episode(episode)
    claim = Claim(
        claim_id="hypothesis",
        layer="behavior",
        relation="possible_effect",
        participants=("write",),
        statement="write deletes the header",
        scope=scope.model_copy(update={"task_id": None}),
        needs_review=True,
        supporting_episodes=(episode.episode_id,),
        payload={
            "alternatives": ["write preserves header"],
            "distinguishing_observation": "read after write",
        },
    )
    store.put_claim(claim)
    ref = next(ref for ref, c in evidence_catalog([episode]).items() if c.source == "response")
    yield store, scope, episode, claim, ref
    store.close()


@pytest.mark.parametrize(
    "verdict", ["plausible", "counterevidence", "internally_inconsistent", "unresolved"]
)
def test_review_controls_hint_eligibility_but_never_promotes(world, verdict):
    store, scope, episode, claim, ref = world
    review = HypothesisReview(
        claim_id=claim.claim_id,
        version=1,
        verdict=verdict,
        reason="Inspect observed header",
        citations=(ref,),
    )
    retain_reviews(ReviewBatch(reviews=(review,)), [claim], [episode], store, scope)
    updated = store.claim(claim.claim_id)
    assert updated.status == "hypothesis"
    assert updated.needs_review == (verdict != "plausible")
    assert bool(hypothesis_context(store, scope, "header")) == (verdict == "plausible")
    if verdict == "plausible":
        hint = hypothesis_context(store, scope, "header")
        assert "review_caveat" in hint
        assert review.reason in hint
    assert updated.payload["semantic_review"]["citations"][0]["episode_id"] == episode.episode_id


@pytest.mark.parametrize("attack", ["citation", "version", "duplicate"])
def test_invalid_review_does_not_change_knowledge(world, attack):
    store, scope, episode, claim, ref = world
    review = HypothesisReview(
        claim_id=claim.claim_id,
        version=2 if attack == "version" else 1,
        verdict="counterevidence",
        reason="Observed header",
        citations=("invented" if attack == "citation" else ref,),
    )
    batch = ReviewBatch(reviews=(review, review) if attack == "duplicate" else (review,))
    with pytest.raises(ValueError):
        retain_reviews(batch, [claim], [episode], store, scope)
    assert store.claim(claim.claim_id) == claim


def test_model_review_has_observations_and_retains_cost(world):
    store, scope, _episode, claim, ref = world

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            assert set(prompt) == {"schema", "claims", "episodes", "evidence_catalog"}
            assert ref in prompt["evidence_catalog"]
            citation_schema = prompt["schema"]["$defs"]["HypothesisReview"]["properties"][
                "citations"
            ]["items"]
            assert set(citation_schema["enum"]) == set(prompt["evidence_catalog"])
            assert not set(citation_schema["enum"]) & {e["episode_id"] for e in prompt["episodes"]}
            review = HypothesisReview(
                claim_id=claim.claim_id,
                version=1,
                verdict="counterevidence",
                reason="Header is still present",
                citations=(ref,),
            )
            return AssistantMessage(
                blocks=[TextBlock(text=ReviewBatch(reviews=(review,)).model_dump_json())]
            )

    accounting = {}
    assert asyncio.run(review_semantics(Client(), store, scope, accounting=accounting)) == [
        claim.claim_id
    ]
    assert accounting["review_seconds"] >= 0
    assert "token_usage" in accounting
    assert store.claim(claim.claim_id).needs_review


def test_review_reserves_source_evidence_before_recent_observations(world):
    store, scope, source, _claim, _ref = world
    for i in range(14):
        store.append_episode(
            Episode(
                task_id=scope.task_id,
                scope=scope,
                operation=f"unrelated_read_{i}",
                arguments={},
                response={"value": i},
                outcome="success",
            )
        )

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            assert source.episode_id in {e["episode_id"] for e in prompt["episodes"]}
            assert len(prompt["episodes"]) <= 12
            assert len(json.dumps(prompt)) <= 24000
            return AssistantMessage(blocks=[TextBlock(text='{"reviews": []}')])

    accounting = {}
    asyncio.run(review_semantics(Client(), store, scope, accounting=accounting))
    assert accounting["omitted_source_evidence_ids"] == []


def test_favorable_review_without_source_observations_is_rejected(world):
    store, scope, _episode, claim, ref = world
    review = HypothesisReview(
        claim_id=claim.claim_id,
        version=1,
        verdict="plausible",
        reason="Looks plausible but source was omitted",
        citations=(ref,),
    )
    with pytest.raises(ValueError, match="source observations"):
        retain_reviews(ReviewBatch(reviews=(review,)), [claim], [], store, scope)
    assert store.claim(claim.claim_id) == claim


def test_overlarge_review_batch_defers_claims_instead_of_silently_skipping(world):
    store, scope, _episode, claim, _ref = world
    # Keep each individual claim manageable, but not the aggregate bundle.
    for i in range(5):
        store.put_claim(
            claim.model_copy(
                update={
                    "claim_id": f"other-{i}",
                    "statement": "question " * 500,
                }
            )
        )

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            assert len(prompt["claims"]) < 6
            assert prompt["episodes"]
            assert {eid for c in prompt["claims"] for eid in c["supporting_episodes"]} <= {
                e["episode_id"] for e in prompt["episodes"]
            }
            assert len(json.dumps(prompt)) <= 10000
            return AssistantMessage(blocks=[TextBlock(text='{"reviews": []}')])

    accounting = {}
    asyncio.run(review_semantics(Client(), store, scope, accounting=accounting, max_chars=10000))
    assert accounting["selected_claim_ids"]
    assert accounting["deferred_claim_ids"]


def test_impossible_review_budget_is_explicit_and_does_not_call_model(world):
    store, scope, _episode, _claim, _ref = world
    accounting = {}
    assert (
        asyncio.run(review_semantics(None, store, scope, accounting=accounting, max_chars=1)) == []
    )
    assert accounting["review_skipped_reason"]
    assert accounting["deferred_claim_ids"]


def test_review_scheduler_skips_unchanged_evidence_and_prioritizes_unreviewed(world):
    store, scope, episode, claim, ref = world
    retain_reviews(
        ReviewBatch(
            reviews=(
                HypothesisReview(
                    claim_id=claim.claim_id,
                    version=1,
                    verdict="unresolved",
                    reason="Need more evidence",
                    citations=(ref,),
                ),
            )
        ),
        [claim],
        [episode],
        store,
        scope,
    )
    assert store.claim(claim.claim_id).needs_review
    assert review_candidates(store, scope) == []
    store.put_claim(claim.model_copy(update={"claim_id": "z-new-question"}))
    assert [c.claim_id for c in review_candidates(store, scope)] == ["z-new-question"]
    store.append_episode(
        Episode(
            task_id=scope.task_id,
            scope=scope,
            operation="new_read",
            arguments={},
            response={"new": True},
            outcome="success",
        )
    )
    assert [c.claim_id for c in review_candidates(store, scope)] == [
        "z-new-question",
        claim.claim_id,
    ]


def test_other_tenant_evidence_does_not_trigger_repeat_review(world):
    store, scope, episode, claim, ref = world
    retain_reviews(
        ReviewBatch(
            reviews=(
                HypothesisReview(
                    claim_id=claim.claim_id,
                    version=1,
                    verdict="unresolved",
                    reason="Need more evidence",
                    citations=(ref,),
                ),
            )
        ),
        [claim],
        [episode],
        store,
        scope,
    )
    store.append_episode(
        Episode(
            task_id="other",
            scope=scope.model_copy(update={"tenant": "other", "task_id": "other"}),
            operation="read",
            arguments={},
            response={},
            outcome="success",
        )
    )
    assert review_candidates(store, scope) == []
