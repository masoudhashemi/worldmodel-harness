import asyncio
import json

import pytest
from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.semantic_learning import (
    SemanticProposal,
    evidence_catalog,
    hypothesis_context,
    learn_semantics,
    retain_hypotheses,
)
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="unseen", version="1", tenant="one", task_id="task")
    episode = Episode(
        scope=scope,
        task_id="task",
        operation="arbitrary_tool",
        arguments={},
        response={"error": "Permission denied for this resource"},
        outcome="error",
        coverage=("error",),
    )
    store.append_episode(episode)
    data = {
        "hypotheses": [
            {
                "layer": "configuration",
                "relation": "possible_precondition",
                "participants": ["arbitrary_tool"],
                "statement": "Resource access may require a permission",
                "citations": [
                    next(
                        ref
                        for ref, c in evidence_catalog([episode]).items()
                        if c.source == "response" and c.path == ("error",)
                    )
                ],
                "alternatives": ["The resource may be outside the permitted tenant"],
                "distinguishing_observation": "Read permitted resource and identity metadata to distinguish scope from role restrictions",
            }
        ]
    }
    yield store, scope, episode, data
    store.close()


def test_model_can_write_configuration_hypothesis_but_not_supported_rule(world):
    store, scope, episode, data = world
    ids = retain_hypotheses(SemanticProposal.model_validate(data), [episode], store, scope)
    claim = store.claim(ids[0])
    assert claim.layer == "configuration" and claim.status == "hypothesis"
    assert claim.needs_review
    assert claim.origin == "inferred"
    assert claim.scope.tenant == scope.tenant and claim.scope.task_id is None
    assert claim.supporting_episodes == (episode.episode_id,)
    retain_hypotheses(SemanticProposal.model_validate(data), [episode], store, scope)
    assert store.claim(ids[0]).status == "hypothesis"
    assert len(store.claim(ids[0]).supporting_episodes) == 1


@pytest.mark.parametrize("attack", ["id", "path", "quote"])
def test_fabricated_citations_rejected_before_any_write(world, attack):
    store, scope, episode, data = world
    if attack == "id":
        data["hypotheses"][0]["citations"] = ["invented"]
    elif attack == "path":
        data["hypotheses"][0]["citations"] = [{"path": ["not_observed"]}]
    else:
        data["hypotheses"][0]["citations"] = [{"excerpt": "User has administrator permission"}]
    with pytest.raises(ValueError):
        retain_hypotheses(SemanticProposal.model_validate(data), [episode], store, scope)
    assert store.claims() == []


def test_hypotheses_are_scoped_labeled_and_bounded(world):
    store, scope, episode, data = world
    ids = retain_hypotheses(SemanticProposal.model_validate(data), [episode], store, scope)
    assert hypothesis_context(store, scope, "resource permission", max_chars=1500) == ""
    # Simulate the distinct screening stage; the claim remains a hypothesis.
    claim = store.claim(ids[0])
    store.put_claim(claim.model_copy(update={"version": 2, "needs_review": False}))
    content = hypothesis_context(store, scope, "resource permission", max_chars=1500)
    assert content.startswith("Unverified diagnostic hypotheses")
    assert "distinguishing_observation" in content
    assert len(content) <= 1500
    assert hypothesis_context(store, scope, "resource permission", max_chars=1) == ""
    assert (
        hypothesis_context(
            store, scope.model_copy(update={"tenant": "other"}), "resource permission"
        )
        == ""
    )


def test_model_receives_observations_not_grades_and_cost_is_recorded(world):
    store, scope, episode, data = world

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            assert tools == {}
            assert set(prompt) == {"schema", "episodes", "evidence_catalog"}
            assert prompt["episodes"][0]["episode_id"] == episode.episode_id
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(data))])

    accounting = {}
    assert asyncio.run(learn_semantics(Client(), store, scope, accounting=accounting))
    assert "token_usage" in accounting
    assert accounting["semantic_learning_seconds"] >= 0


def test_disputed_hypothesis_is_not_resurrected(world):
    store, scope, episode, data = world
    proposal = SemanticProposal.model_validate(data)
    claim = store.claim(retain_hypotheses(proposal, [episode], store, scope)[0])
    store.put_claim(claim.model_copy(update={"version": 2, "status": "disputed"}))
    assert retain_hypotheses(proposal, [episode], store, scope) == []
    assert store.claim(claim.claim_id).status == "disputed"


def test_repeated_evidence_cannot_erase_screening_or_add_versions(world):
    store, scope, episode, data = world
    proposal = SemanticProposal.model_validate(data)
    claim = store.claim(retain_hypotheses(proposal, [episode], store, scope)[0])
    reviewed = claim.model_copy(
        update={
            "version": 2,
            "needs_review": True,
            "payload": {
                **claim.payload,
                "semantic_review": {
                    "verdict": "counterevidence",
                    "reason": "Contradicted by observed response",
                },
            },
        }
    )
    store.put_claim(reviewed)
    assert retain_hypotheses(proposal, [episode], store, scope) == []
    assert store.claim(claim.claim_id) == reviewed


def test_new_evidence_preserves_old_citations_and_review_until_rescreening(world):
    store, scope, episode, data = world
    claim = store.claim(
        retain_hypotheses(SemanticProposal.model_validate(data), [episode], store, scope)[0]
    )
    review = {"verdict": "unresolved", "reason": "More observations needed"}
    store.put_claim(
        claim.model_copy(
            update={"version": 2, "payload": {**claim.payload, "semantic_review": review}}
        )
    )
    later = Episode(
        scope=scope,
        task_id=scope.task_id,
        operation=episode.operation,
        arguments={},
        response=episode.response,
        outcome="error",
    )
    store.append_episode(later)
    data["hypotheses"][0]["citations"] = [
        next(
            ref
            for ref, c in evidence_catalog([later]).items()
            if c.source == "response" and c.path == ("error",)
        )
    ]
    retain_hypotheses(SemanticProposal.model_validate(data), [later], store, scope)
    updated = store.claim(claim.claim_id)
    assert updated.version == 3
    assert updated.needs_review
    assert updated.payload["semantic_review"] == review
    assert {c["episode_id"] for c in updated.payload["citations"]} == {
        episode.episode_id,
        later.episode_id,
    }


def test_malformed_answer_is_preserved_without_committing_claims(world):
    store, scope, _, _ = world

    class Client:
        async def generate(self, messages, tools):
            return AssistantMessage(blocks=[TextBlock(text='{"hypotheses": [')])

    accounting = {}
    with pytest.raises(ValueError):
        asyncio.run(learn_semantics(Client(), store, scope, accounting=accounting))
    assert accounting["rejected_proposal_text"] == '{"hypotheses": ['
    assert "token_usage" in accounting
    assert store.claims() == []


def test_failed_citation_keeps_proposal_for_offline_diagnosis(world):
    store, scope, _, data = world
    data["hypotheses"][0]["citations"] = ["Not observed anywhere"]

    class Client:
        async def generate(self, messages, tools):
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(data))])

    accounting = {}
    with pytest.raises(ValueError, match="Citation"):
        asyncio.run(learn_semantics(Client(), store, scope, accounting=accounting))
    assert accounting["proposed_hypotheses"] == data
    assert json.loads(accounting["rejected_proposal_text"]) == data
    assert store.claims() == []


def test_catalog_keeps_request_and_response_provenance_distinct(world):
    store, scope, episode, data = world
    catalog = evidence_catalog([episode])
    assert {c.source for c in catalog.values()} == {"arguments", "response"}
    ref = data["hypotheses"][0]["citations"][0]
    assert catalog[ref].excerpt == episode.response["error"]
    forged = episode.model_copy(update={"response": {"error": "Invented successful response"}})
    with pytest.raises(ValueError, match="retained task observation"):
        retain_hypotheses(SemanticProposal.model_validate(data), [forged], store, scope)
    assert store.claims() == []
