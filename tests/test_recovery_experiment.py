import asyncio
import json
from types import SimpleNamespace

from stirrup.core.models import AssistantMessage, TextBlock, Tool, ToolResult

from worldmodel_harness.knowledge import Episode, OutputExpectation, Record, Scope
from worldmodel_harness.recovery_experiment import (
    RecoveryExperiment,
    assess_experiment,
    propose_experiment,
)


class Read(Record):
    cursor: str


def test_model_action_is_schema_checked_and_observation_distinguishes_outcomes():
    scope = Scope(environment="novel", version="1", task_id="t")
    event = Episode(task_id="t", scope=scope, origin="harness", operation="tool_call:inspect",
        arguments={"cursor": "old"}, response={"result": {"error": "invalid", "next": "opaque"}},
        coverage=("result",), outcome="error")
    tool = Tool(name="inspect", description="Read a record using its cursor", parameters=Read,
                executor=lambda _: ToolResult(content="{}"))
    adapter = SimpleNamespace(public_tools={"inspect": tool}, task=SimpleNamespace(prompt=[]))
    proposal = RecoveryExperiment(explanation="Test whether the observed token is accepted",
        tool="inspect", arguments={"cursor": "opaque"}, evidence_ids=(event.episode_id,),
        alternatives=("cursor invalid", "record accessible"), predictions=(
            OutputExpectation(path=("result", "ok"), expected=False),
            OutputExpectation(path=("result", "ok"), expected=True)))

    class Client:
        async def generate(self, messages, tools):
            assert "opaque" in messages[1].content
            prompt = json.loads(messages[1].content)
            assert event.event_id not in messages[1].content
            assert event.episode_id not in messages[1].content
            public = proposal.model_copy(update={"evidence_ids": (prompt["observations"][0]["evidence_ref"],)})
            return AssistantMessage(blocks=[TextBlock(text=public.model_dump_json())])

    checked = asyncio.run(propose_experiment(adapter, Client(), [event], [], {}))
    assert checked.evidence_ids == (event.episode_id,)
    observed = event.model_copy(update={"arguments": {"cursor": "opaque"},
        "response": {"result": {"ok": True}}, "outcome": "success"})
    result = assess_experiment(adapter, checked, observed)
    assert result["result"] == "executed"
    assert [p["result"] for p in result["predictions"]] == ["contradicted", "matched"]
    assert assess_experiment(adapter, checked, event)["result"] == "different_arguments"


def test_pause_requires_actual_environment_evidence_and_preserves_provenance():
    from worldmodel_harness.recovery_experiment import RecoveryPause

    scope = Scope(environment="novel", version="1", task_id="t")
    event = Episode(task_id="t", scope=scope, origin="environment", operation="inspect",
        arguments={}, response={"error": "account suspended; administrator must reactivate"},
        coverage=("error",), outcome="error", environment_called=True)
    adapter = SimpleNamespace(public_tools={}, task=SimpleNamespace(prompt=[]))

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            body = {"kind": "pause", "explanation": "Observed account restriction needs an external change",
                "evidence_ids": [prompt["observations"][0]["evidence_ref"]],
                "blocker": {"origin": "environment", "applies_when": "This account is suspended",
                    "stop_when": "No available operation can reactivate the account",
                    "reopen_when": "Administrator confirms reactivation"},
                "remaining_alternatives": ["The account state may have changed since the response"]}
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(body))])

    decision = asyncio.run(propose_experiment(adapter, Client(), [event], [], {}))
    assert isinstance(decision, RecoveryPause)
    assert decision.evidence_ids == (event.episode_id,)
    import pytest

    with pytest.raises(ValueError, match="Environment blockers require"):
        asyncio.run(propose_experiment(adapter, Client(), [event.model_copy(update={
            "origin": "harness", "environment_called": False})], [], {}))


def test_explicit_prediction_target_maps_tool_content_without_guessing():
    scope = Scope(environment="test", version="1", task_id="t")
    tool = Tool(name="inspect", description="Read", parameters=Read,
                executor=lambda _: ToolResult(content="{}"))
    adapter = SimpleNamespace(public_tools={"inspect": tool})
    proposal = RecoveryExperiment(explanation="Test the next cursor", tool="inspect",
        arguments={"cursor": "next"}, evidence_ids=("e",), alternatives=("works", "fails"),
        prediction_target="tool_result", predictions=(
            OutputExpectation(path=("ok",), expected=True),
            OutputExpectation(path=("ok",), expected=False)))
    event = Episode(task_id="t", scope=scope, origin="harness", operation="tool_call:inspect",
        arguments={"cursor": "next"}, response={"result": {"ok": True}},
        coverage=("result",), outcome="success")
    result = assess_experiment(adapter, proposal, event)
    assert [p["result"] for p in result["predictions"]] == ["matched", "contradicted"]
    assert result["predictions"][0]["prediction"]["path"] == ["result", "ok"]
    # The same path with a different explicit root must not be silently repaired.
    wrong_root = proposal.model_copy(update={"prediction_target": "harness_response"})
    assert all(p["result"] == "unobserved" for p in assess_experiment(adapter, wrong_root, event)["predictions"])
