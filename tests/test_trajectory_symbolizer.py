"""Unit tests for LLM-based trajectory symbolization."""

import asyncio
import json
import pytest
from pydantic import ValidationError
from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.structured_output import ContractError
from worldmodel_harness.trajectory_symbolizer import (
    SymbolizedStep,
    TrajectoryAbstraction,
    format_episodes_for_abstraction,
    store_trajectory_abstraction,
    symbolize_trajectory,
)


@pytest.fixture
def mock_store(tmp_path):
    store = KnowledgeStore(tmp_path / "test_store.sqlite")
    yield store
    store.close()


@pytest.fixture
def sample_scope():
    return Scope(environment="automationbench", version="1", tenant="corp", task_id="test_task")


def test_symbolized_step_validation():
    # Valid parameterized template
    step = SymbolizedStep(
        step_index=1,
        operation="api_fetch",
        action_template="POST https://api.airtable.com/v0/{base_id}/Visitors/{record_id}/comments",
        input_bindings={"record_id": "extracted from gmail message body"},
        abstract_intent="Add confirmation comment to visitor record in Airtable",
    )
    assert step.step_index == 1
    assert "{record_id}" in step.action_template

    # Invalid: leaks concrete Airtable record literal
    with pytest.raises((ContractError, ValidationError), match="leaks concrete instance literals"):
        SymbolizedStep(
            step_index=2,
            operation="api_fetch",
            action_template="POST https://api.airtable.com/v0/base_ops/Visitors/rec_10492837/comments",
            input_bindings={},
            abstract_intent="Add comment",
        )


def test_format_episodes_for_abstraction(sample_scope):
    ep1 = Episode(
        scope=sample_scope,
        task_id="test_task",
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://gmail.googleapis.com/v1/users/me/messages?q=Visitor"},
        response={"messages": [{"id": "msg_001"}]},
        outcome="success",
        origin="environment",
    )
    ep_harness = Episode(
        scope=sample_scope,
        task_id="test_task",
        operation="finish",
        arguments={},
        response={"success": True},
        outcome="success",
        origin="harness",
    )
    text, ep_ids = format_episodes_for_abstraction([ep1, ep_harness])
    assert len(ep_ids) == 1
    assert ep_ids[0] == ep1.episode_id
    assert "https://gmail.googleapis.com" in text
    assert "finish" not in text


def test_symbolize_trajectory(sample_scope):
    ep1 = Episode(
        scope=sample_scope,
        task_id="test_task",
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://gmail.googleapis.com/v1/users/me/messages/msg_001"},
        response={"body": "Visitor Nguyen signed NDA. Record: rec_104"},
        outcome="success",
        origin="environment",
    )
    ep2 = Episode(
        scope=sample_scope,
        task_id="test_task",
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://api.airtable.com/v0/base_ops/Visitors/rec_104/comments",
            "body": '{"text": "Confirmed"}',
        },
        response={"id": "com_001"},
        outcome="success",
        origin="environment",
    )

    mock_llm_response = {
        "goal_intent": "Process visitor NDA confirmation and record comments",
        "steps": [
            {
                "step_index": 1,
                "operation": "api_fetch",
                "action_template": "GET https://gmail.googleapis.com/v1/users/me/messages/{message_id}",
                "input_bindings": {"message_id": "Target email identifier"},
                "abstract_intent": "Retrieve visitor NDA notification email",
            },
            {
                "step_index": 2,
                "operation": "api_fetch",
                "action_template": "POST https://api.airtable.com/v0/{base_id}/Visitors/{record_id}/comments",
                "input_bindings": {"record_id": "Extracted from visitor NDA email body in step 1"},
                "abstract_intent": "Append confirmation comment to visitor record",
            },
        ],
        "invariants": ["Must extract record_id from email before commenting"],
    }

    class MockClient:
        async def generate(self, messages, tools):
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(mock_llm_response))])

    accounting = {}
    abstraction = asyncio.run(symbolize_trajectory(MockClient(), [ep1, ep2], accounting=accounting))
    assert abstraction is not None
    assert abstraction.goal_intent == "Process visitor NDA confirmation and record comments"
    assert len(abstraction.steps) == 2
    assert "{record_id}" in abstraction.steps[1].action_template
    assert "rec_104" not in abstraction.steps[1].action_template


def test_store_trajectory_abstraction(mock_store, sample_scope):
    abstraction = TrajectoryAbstraction(
        goal_intent="Sample abstracted workflow",
        steps=(
            SymbolizedStep(
                step_index=1,
                operation="api_fetch",
                action_template="GET https://api.example.com/{resource_id}",
                input_bindings={"resource_id": "Task input"},
                abstract_intent="Fetch resource",
            ),
        ),
        invariants=("Check status before mutation",),
    )

    ep = Episode(
        episode_id="ep_123",
        scope=sample_scope,
        task_id="test_task",
        operation="api_fetch",
        arguments={},
        response={},
        outcome="success",
        origin="environment",
    )
    mock_store.append_episode(ep)
    claim = store_trajectory_abstraction(mock_store, sample_scope, abstraction, ["ep_123"])
    assert claim.relation == "symbolized_workflow_template"
    assert claim.status == "supported"

    claims = mock_store.claims(sample_scope)
    found = [c for c in claims if c.relation == "symbolized_workflow_template"]
    assert len(found) == 1
    assert found[0].payload["goal_intent"] == "Sample abstracted workflow"
