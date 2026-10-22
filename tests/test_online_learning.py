import asyncio
import json
from types import SimpleNamespace

from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.budget import Ledger
from worldmodel_harness.experiment import ExperimentSettings, KnowledgeClient
from worldmodel_harness.knowledge import Episode, LiveState, Scope
from worldmodel_harness.online_learning import OnlineLearner
from worldmodel_harness.runtime import MeteredClient
from worldmodel_harness.semantic_learning import hypothesis_context
from worldmodel_harness.store import KnowledgeStore


class ProposalClient:
    def __init__(self):
        self.prompts = []

    async def generate(self, messages, tools):
        prompt = json.loads(messages[1].content)
        self.prompts.append(prompt)
        assert "grading_details" not in json.dumps(prompt)
        citation = next(iter(prompt["evidence_catalog"]))
        if "claims" in prompt:
            body = {"reviews": [{
                "claim_id": claim["claim_id"], "version": claim["version"],
                "verdict": "plausible", "reason": "Consistent with the observed rejection.",
                "citations": [citation],
            } for claim in prompt["claims"]]}
        else:
            body = {"hypotheses": [{
                "layer": "behavior", "relation": "possible_recovery",
                "participants": ["tool_call:inspect_record"],
                "statement": "The cursor may require the observed opaque token instead of a row number.",
                "citations": [citation],
                "alternatives": ["The cursor encoding differs", "The record is no longer available"],
                "distinguishing_observation": "Compare a call using the returned cursor with the rejected call.",
            }]}
        return AssistantMessage(blocks=[TextBlock(text=json.dumps(body))])


def test_stagnation_uses_model_hypotheses_in_next_decision_and_records_cost(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="test", version="1", tenant="tenant", task_id="task")
    adapter = SimpleNamespace(
        knowledge=store, scope=scope, live=LiveState("task"), ledger=Ledger(),
        task=SimpleNamespace(prompt=[{"role": "user", "content": "Inspect the record"}]),
        prediction_assessments=[],
    )
    for index in range(3):
        store.append_episode(Episode(
            task_id="task", scope=scope, origin="harness", operation="tool_call:inspect_record",
            interaction_id=str(index), arguments={"cursor": 3},
            response={"result": {"error": "invalid_cursor"}}, coverage=("result",), outcome="error",
        ))
    proposal_client = ProposalClient()
    learner = OnlineLearner(adapter, MeteredClient(proposal_client), max_reflections=1)

    class ActingClient:
        async def generate(self, messages, tools):
            assert "opaque token" in messages[0].content
            assert "not verified facts" in messages[0].content
            return AssistantMessage(blocks=[TextBlock(text="Inspect the returned cursor.")])

    wrapper = KnowledgeClient(ActingClient(), adapter, store,
                              ExperimentSettings(operational=True, semantic_learning=True))
    wrapper.online_learner = learner
    asyncio.run(wrapper.generate([], {}))
    assert len(learner.audit) == 1
    assert learner.audit[0]["trigger"] == "repeated_observation"
    assert len(learner.audit[0]["model_call_costs"]) == 2
    assert learner.audit[0]["learning"]["selected_evidence_ids"]
    assert adapter.ledger.adaptation_seconds > 0
    assert adapter.ledger.environment_calls == 0
    assert all(claim.status == "hypothesis" for claim in store.claims(scope))
    asyncio.run(learner.before_decision())
    assert len(proposal_client.prompts) == 2
    later_scope = scope.model_copy(update={"task_id": "later"})
    assert "opaque token" in hypothesis_context(store, later_scope, "cursor")
    assert hypothesis_context(store, later_scope.model_copy(update={"tenant": "other"}), "cursor") == ""
    store.close()


def test_different_successes_do_not_trigger_reflection(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="test", version="1", task_id="task")
    adapter = SimpleNamespace(knowledge=store, scope=scope)
    for index in range(4):
        store.append_episode(Episode(task_id="task", scope=scope, origin="harness",
            operation="tool_call:read", arguments={"page": index}, response={"result": index},
            coverage=("result",), outcome="success"))
    client = MeteredClient(ProposalClient())
    learner = OnlineLearner(adapter, client)
    asyncio.run(learner.before_decision())
    assert not learner.audit and not client.calls
    store.close()


def test_recovery_receives_linked_environment_evidence_and_can_pause_or_experiment(tmp_path):
    from stirrup.core.models import Tool, ToolResult

    from worldmodel_harness.knowledge import Record

    class Read(Record):
        cursor: str

    for blocked in (True, False):
        store = KnowledgeStore(tmp_path / f"linked-{blocked}.sqlite")
        scope = Scope(environment="test", version="1", tenant="tenant", task_id="task")
        tool = Tool(name="inspect_record", description="Read a record using a cursor", parameters=Read,
                    executor=lambda _: ToolResult(content="{}"))
        adapter = SimpleNamespace(knowledge=store, scope=scope, ledger=Ledger(),
            public_tools={tool.name: tool}, task=SimpleNamespace(prompt=[{
                "role": "user", "content": "Read the record"}]))
        store.append_episode(Episode(task_id="task", scope=scope, operation="unrelated",
            arguments={}, response={"error": "unrelated failure"}, outcome="error"))
        env_ids = set()
        for index in range(3):
            body = {"error": "suspended; external administrator must reactivate"} if blocked else {
                "error": "invalid_cursor", "current_cursor": "opaque"}
            env = Episode(task_id="task", scope=scope, origin="environment", operation="inspect_record",
                arguments={"cursor": "old"}, response=body, outcome="error", coverage=tuple(body),
                interaction_id=str(index), environment_called=True)
            env_ids.add(env.episode_id)
            store.append_episode(env)
            store.append_episode(Episode(task_id="task", scope=scope, origin="harness",
                operation="tool_call:inspect_record", arguments={"cursor": "old"},
                response={"result": body}, coverage=("result",), outcome="error",
                related_episode_ids=(env.episode_id,), interaction_id=str(index), environment_called=True))

        class Client(ProposalClient):
            async def generate(self, messages, tools, blocked=blocked):
                prompt = json.loads(messages[1].content)
                if "observations" not in prompt:
                    return await super().generate(messages, tools)
                observations = prompt["observations"]
                assert len(observations) == 6
                assert all(e["operation"] != "unrelated" for e in observations)
                environment = [e for e in observations if e["origin"] == "environment"]
                assert len(environment) == 3
                ref = environment[-1]["evidence_ref"]
                if blocked:
                    body = {"kind": "pause", "explanation": "External reactivation required",
                        "evidence_ids": [ref], "remaining_alternatives": ["Account state changed"],
                        "blocker": {"origin": "environment", "applies_when": "Account suspended",
                            "stop_when": "Reactivation is external", "reopen_when": "Account reactivated"}}
                else:
                    body = {"explanation": "Try the observed current cursor", "tool": "inspect_record",
                        "arguments": {"cursor": "opaque"}, "evidence_ids": [ref],
                        "alternatives": ["Current cursor works", "Current cursor also expired"],
                        "predictions": [{"path": ["result", "ok"], "expected": value} for value in (True, False)]}
                return AssistantMessage(blocks=[TextBlock(text=json.dumps(body))])

        learner = OnlineLearner(adapter, MeteredClient(Client()), max_reflections=1)
        asyncio.run(learner.before_decision())
        audit = learner.audit[0]
        assert "error_type" not in audit
        decision = audit["pause" if blocked else "experiment"]
        assert set(decision["evidence_ids"]) <= env_ids
        if blocked:
            assert learner.pending_experiment is None
            assert "Account reactivated" in learner.context()
        else:
            assert learner.pending_experiment.arguments == {"cursor": "opaque"}
        store.close()
