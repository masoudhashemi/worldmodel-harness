import pytest

from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.semantic_learning import (
    SemanticHypothesis,
    SemanticProposal,
    evidence_catalog,
    hypothesis_context,
    retain_hypotheses,
)
from worldmodel_harness.store import KnowledgeStore


def test_learned_blocker_retains_reopening_conditions_without_promoting_impossibility(tmp_path):
    scope = Scope(environment="test", version="1", tenant="account", task_id="t")
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    event = Episode(task_id="t", scope=scope, operation="read", origin="environment",
        arguments={}, response={"error": "suspended"}, outcome="error", coverage=("error",))
    store.append_episode(event)
    data = dict(layer="behavior", relation="possible_blocker", participants=("read",),
        statement="Reading may be blocked while this account is suspended",
        citations=(next(iter(evidence_catalog([event]))),),
        alternatives=("The account may already have been reactivated",),
        distinguishing_observation="An account status observation",
        blocker=dict(origin="environment", applies_when="The account remains suspended",
            stop_when="No available action changes suspension", reopen_when="Account reactivated"))
    hypothesis = SemanticHypothesis(**data)
    ids = retain_hypotheses(SemanticProposal(hypotheses=(hypothesis,)), [event], store, scope)
    assert store.claim(ids[0]).status == "hypothesis"
    from worldmodel_harness.semantic_review import HypothesisReview, ReviewBatch, retain_reviews
    claim = store.claim(ids[0])
    retain_reviews(ReviewBatch(reviews=(HypothesisReview(claim_id=claim.claim_id,
        version=claim.version, verdict="plausible", reason="Consistent but not proven",
        citations=data["citations"]),)), [claim], [event], store, scope)
    context = hypothesis_context(store, scope.model_copy(update={"task_id": "next"}), "read")
    assert "Account reactivated" in context
    assert not hypothesis_context(store, scope.model_copy(update={"tenant": "other"}), "read")
    with pytest.raises(ValueError, match="supply blocker conditions"):
        SemanticHypothesis(**{**data, "blocker": None})
    local = event.model_copy(update={"episode_id": "local", "event_id": "local-event", "origin": "harness", "environment_called": False})
    store.append_episode(local)
    fake = hypothesis.model_copy(update={"citations": (next(iter(evidence_catalog([local]))),)})
    with pytest.raises(ValueError, match="cited environment failure"):
        retain_hypotheses(SemanticProposal(hypotheses=(fake,)), [local], store, scope)
    store.close()


def test_blocker_schema_and_repair_feedback_expose_conditional_contract():
    import asyncio
    import json

    from stirrup.core.models import AssistantMessage, TextBlock

    from worldmodel_harness.structured_output import generate_checked

    schema = SemanticProposal.model_json_schema()
    contract = schema["$defs"]["SemanticHypothesis"]["allOf"][0]
    assert contract["then"]["properties"]["response_rule"] == {"type": "null"}
    assert "PAUSE RETRYING" in schema["$defs"]["BlockerConditions"]["properties"]["stop_when"]["description"]
    invalid = {"hypotheses": [{"layer": "behavior", "relation": "possible_blocker",
        "participants": ["read"], "statement": "May be blocked", "citations": ["e"],
        "alternatives": ["May be temporary"], "distinguishing_observation": "Check status"}]}

    class Client:
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 2:
                feedback = json.loads(messages[-1].content)["validation_feedback"]
                assert any("response_rule=null" in row and "pause retries" in row for row in feedback)
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(
                invalid if self.calls == 1 else {"hypotheses": []}))])

    accounting = {}
    result = asyncio.run(generate_checked(Client(), [], SemanticProposal.model_validate, accounting))
    assert not result.hypotheses
    assert accounting["returned_model_responses"] == 2
