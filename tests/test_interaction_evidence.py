import asyncio
import json

from stirrup.core.models import AssistantMessage, Tool, ToolCall, ToolResult, UserMessage

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.interactions import CURRENT_INTERACTION, InteractionRecorder
from worldmodel_harness.knowledge import Record
from worldmodel_harness.learning import Learner
from worldmodel_harness.repairing_agent import RepairingAgent
from worldmodel_harness.store import KnowledgeStore


def test_schema_failure_and_success_form_learnable_chain(tmp_path):
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite")
    evidence = KnowledgeStore(tmp_path / "evidence.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", evidence, Ledger(), Learner(knowledge), knowledge
    )
    search = next(tool for tool in adapter.tools() if tool.name == "api_search")
    finish = Tool(name="finish", description="End interaction", parameters=Record,
                  executor=lambda _: ToolResult(content="Finished"))
    recorder = InteractionRecorder(adapter)
    recorder.register_tools([search, finish])
    assert knowledge.claims(adapter.scope) == []

    class Client:
        model_slug = "scripted"
        context_window_tokens = 32768
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                call = ToolCall(name="api_search", arguments='{"unexpected":true}')
            elif self.calls == 2:
                call = ToolCall(name="api_search", arguments=json.dumps({
                    "query": "gmail list messages", "top_k": 1,
                }))
            else:
                call = ToolCall(name="finish", arguments="{}")
            return AssistantMessage(blocks=[call])

    async def run():
        agent = RepairingAgent(
            client=Client(), name="interaction-evidence", tools=[search],
            finish_tool=finish, max_turns=4, interaction_recorder=recorder,
            repair_schemas={search.name: search.parameters, finish.name: finish.parameters},
        )
        async with agent.session(output_dir=tmp_path / "session") as session:
            finished, _, _ = await session.run([UserMessage(content="Inspect public tools")])
        assert finished is not None

    asyncio.run(run())
    episodes = evidence.episodes(adapter.scope)
    calls = [event for event in episodes if event.operation.startswith("tool_call:")]
    assert len(calls) == 3
    rejected, corrected, finished = calls
    assert rejected.origin == "harness" and rejected.outcome == "error"
    assert rejected.environment_called is False
    assert not rejected.related_episode_ids
    assert rejected.response["result"]["origin"] == "outer_tool_schema"
    assert corrected.previous_interaction_id == rejected.interaction_id
    assert corrected.environment_called is True
    assert len(corrected.related_episode_ids) == 1
    actual = evidence.episode(corrected.related_episode_ids[0])
    assert actual.origin == "environment" and actual.operation == "api_search"
    assert actual.interaction_id == corrected.interaction_id
    assert finished.previous_interaction_id == corrected.interaction_id
    assert adapter.ledger.environment_calls == 1
    assert CURRENT_INTERACTION.get() is None
    for event in episodes:
        assert knowledge.episode(event.episode_id) == event
        assert "grading_details" not in event.model_dump_json()
    assert not any(claim.relation == "tool_contract" for claim in knowledge.claims(adapter.scope))
    knowledge.close()
    evidence.close()


def test_frozen_interactions_only_write_separate_evidence(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    KnowledgeStore(path).close()
    before = path.read_bytes()
    knowledge = KnowledgeStore(path, frozen=True)
    evidence = KnowledgeStore(tmp_path / "evidence.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", evidence, Ledger(), knowledge=knowledge
    )
    recorder = InteractionRecorder(adapter)
    recorder.register_tools(adapter.tools())
    assert evidence.episodes(adapter.scope)
    assert not knowledge.episodes(adapter.scope)
    assert path.read_bytes() == before
    knowledge.close()
    evidence.close()
