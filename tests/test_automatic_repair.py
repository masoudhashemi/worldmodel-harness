import asyncio
import json

import pytest
from pydantic import BaseModel
from stirrup.core.models import AssistantMessage, TextBlock, Tool, ToolResult

from worldmodel_harness.automatic_repair import repair_candidates, repair_from_runs
from worldmodel_harness.knowledge import Claim, Dependency, Episode, Scope
from worldmodel_harness.procedures import Procedure, Step
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="new-system", version="1")
    episode = Episode(
        task_id="observed",
        scope=scope,
        operation="tool_schema",
        arguments={},
        response={"revision": 1},
        outcome="success",
    )
    store.append_episode(episode)
    claim = Claim(
        claim_id="contract",
        layer="interface",
        relation="contract",
        participants=("lookup",),
        statement="lookup revision",
        scope=scope,
        status="supported",
        supporting_episodes=(episode.episode_id,),
        payload={"revision": 1},
    )
    store.put_claim(claim)
    artifact = Procedure(
        procedure_id="learned",
        goal_pattern="lookup",
        scope=scope,
        parameter_types={},
        required_claims=(Dependency(claim_id="contract", version=1),),
        status="suspended",
        steps=(
            Step(name="read", kind="query", tool="lookup", next_step="verify"),
            Step(
                name="verify",
                kind="verify",
                value={"$ref": "steps.read.data"},
                expected="observed",
                next_step="stop",
            ),
            Step(name="stop", kind="stop"),
        ),
    )
    store.put_procedure(artifact)
    store.put_claim(claim.model_copy(update={"version": 2, "payload": {"revision": 2}}))

    class Inputs(BaseModel):
        pass

    tools = {
        "lookup": Tool(
            name="lookup",
            description="lookup",
            parameters=Inputs,
            executor=lambda _: ToolResult(content="observed"),
        )
    }
    yield store, scope, artifact, tools
    store.close()


class Client:
    def __init__(self, candidate, attack=None):
        self.candidate, self.attack = candidate, attack
        self.calls = 0

    async def generate(self, messages, tools):
        self.calls += 1
        prompt = json.loads(messages[1].content)
        if "requested_goal" in prompt:
            data = {"supported": True, "step_names": [self.candidate.steps[0].name],
                    "missing_requirements": [], "reason": "The lookup steps cover the goal."}
        elif self.calls == 1:
            assert "candidates" in prompt
            data = {
                "procedure_id": "learned",
                "reason": "Observed contract revision",
                "estimated_reuses": 10,
                "estimated_saving_seconds": 100,
            }
        else:
            assert prompt["repair_context"]["previous_artifact"]["status"] == "suspended"
            data = self.candidate.model_dump(mode="json")
            data["required_claims"] = [{"claim_id": "contract", "version": 2}]
            if self.attack == "weaken":
                data["required_claims"][0]["mode"] = "monitors"
            if self.attack == "interface":
                data["parameter_types"] = {"extra": "string"}
        return AssistantMessage(blocks=[TextBlock(text=json.dumps(data))])


def test_selects_changed_evidence_without_manual_repair_spec(world):
    store, scope, artifact, tools = world
    ready, deferred = repair_candidates(store, scope)
    assert len(ready) == 1 and deferred == {}
    accounting = {}
    result = asyncio.run(
        repair_from_runs(Client(artifact), store, scope, tools, horizon=10, accounting=accounting)
    )
    assert result["deployment"] == "experimental"
    repaired = store.procedures()[0]
    assert repaired.version == 2 and repaired.repair_from_version == 1
    assert repaired.validation == ()
    assert accounting["repair_seconds"] >= 0
    assert not result["independent_validation"]


@pytest.mark.parametrize("attack", ["weaken", "interface"])
def test_rejected_repair_leaves_suspended_artifact_unchanged(world, attack):
    store, scope, artifact, tools = world
    with pytest.raises(ValueError):
        asyncio.run(
            repair_from_runs(
                Client(artifact, attack), store, scope, tools, horizon=10, accounting={}
            )
        )
    assert store.procedures()[0] == artifact


def test_disputed_knowledge_is_deferred_without_model_call(world):
    store, scope, artifact, tools = world
    claim = store.claim("contract")
    store.put_claim(claim.model_copy(update={"version": 3, "status": "disputed"}))
    result = asyncio.run(repair_from_runs(None, store, scope, tools, horizon=10, accounting={}))
    assert result["deployment"] == "none"
    assert result["deferred"] == {"learned": ["refresh:contract"]}
    assert store.procedures()[0] == artifact


def test_evidence_only_revision_does_not_trigger_repair(world):
    store, scope, artifact, _ = world
    # Pin to the current semantics, then add an evidence-only revision.
    current = store.claim("contract")
    store.put_procedure(
        artifact.model_copy(
            update={"version": 2, "required_claims": (Dependency(claim_id="contract", version=2),)}
        )
    )
    store.put_claim(current.model_copy(update={"version": 3}))
    assert repair_candidates(store, scope) == ([], {})


def test_normalize_assistant_tool_calls_repairs_split_url_and_body():
    from stirrup.core.models import AssistantMessage, ToolCall
    from worldmodel_harness.repairing_agent import normalize_assistant_tool_calls

    args = {
        "action": {
            "arguments": {
                "body": '{"raw": "RnJvbTog..."},method:'
            }
        },
        "https": "//gmail.googleapis.com/gmail/v1/users/me/messages/send"
    }
    tc = ToolCall(tool_call_id="c1", name="execute_decision", arguments=json.dumps(args))
    msg = AssistantMessage(blocks=[tc])
    normalized = normalize_assistant_tool_calls(msg)

    repaired_tc = normalized.blocks[0]
    parsed = json.loads(repaired_tc.arguments)
    action = parsed["action"]
    assert action["tool"] == "api_fetch"
    assert action["arguments"]["method"] == "POST"
    assert action["arguments"]["url"] == "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    assert action["arguments"]["body"] == '{"raw": "RnJvbTog..."}'
    assert "https" not in parsed
