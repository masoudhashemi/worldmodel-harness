import asyncio
import json

from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.knowledge import Claim, Episode, Layer, Scope
from worldmodel_harness.opportunities import ExecutionCost, opportunities, plan_from_runs
from worldmodel_harness.store import KnowledgeStore


def test_opportunities_require_runs_and_generate_no_benchmark_workflow(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="unseen-service", version="1")
    assert opportunities(store, scope) == []
    result = asyncio.run(
        plan_from_runs(None, store, scope, [], horizon=50, require_validation=False)
    )
    assert not result.propose_artifact
    evidence = []
    for task in ("one", "two"):
        event = Episode(
            task_id=task,
            scope=scope.model_copy(update={"tenant": task, "task_id": task}),
            operation="a-tool-never-named-in-source",
            arguments={},
            response={"error": 42},
            outcome="error",
        )
        store.append_episode(event)
        evidence.append(event.episode_id)
    store.put_claim(
        Claim(
            claim_id="learned-error",
            layer=Layer.BEHAVIOR,
            relation="error_observed",
            participants=("new-tool",),
            statement="Observed error",
            scope=scope,
            supporting_episodes=tuple(evidence),
        )
    )
    choice = opportunities(store, scope)[0]
    assert choice.independent_tasks == 2
    assert choice.kind == "unresolved_failure"

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            assert prompt["validation_task_prompts_only"] == []
            assert "a-tool-never-named-in-source" in messages[1].content
            assert prompt["execution_cost_context"][0]["model_elapsed_seconds"] == 20
            assert "not measured savings" in prompt["cost_interpretation"]
            return AssistantMessage(
                blocks=[
                    TextBlock(
                        text=json.dumps(
                            {
                                "propose_artifact": False,
                                "rationale": "The error cause is unresolved; gather evidence first.",
                            }
                        )
                    )
                ]
            )

    plan = asyncio.run(
        plan_from_runs(
            Client(),
            store,
            scope,
            [],
            horizon=50,
            require_validation=False,
            costs=[ExecutionCost(task_name="one", environment_calls=2, model_elapsed_seconds=20)],
        )
    )
    assert not plan.propose_artifact
    store.close()


def test_linked_local_recoveries_become_candidates_without_becoming_rules(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="novel", version="1")
    for task in ("first", "second"):
        task_scope = scope.model_copy(update={"task_id": task})
        failure = Episode(task_id=task, scope=task_scope, origin="harness",
            operation="tool_call:unknown_tool", arguments={}, response={"rejected": True},
            outcome="error", interaction_id=task + "-failure", environment_called=False)
        success = Episode(task_id=task, scope=task_scope, origin="harness",
            operation="tool_call:discovered_tool", arguments={}, response={"accepted": True},
            outcome="success", interaction_id=task + "-success",
            previous_interaction_id=failure.interaction_id, environment_called=False)
        store.append_episode(failure)
        store.append_episode(success)
        if task == "first":
            assert opportunities(store, scope) == []
    candidates = opportunities(store, scope)
    assert len(candidates) == 1
    assert candidates[0].kind == "interaction_recovery_candidate"
    assert candidates[0].independent_tasks == 2
    assert len(candidates[0].evidence_ids) == 4
    assert not store.claims() and not store.procedures()
    store.close()


def test_repeated_attempts_of_one_tenant_task_are_not_independent(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="novel", version="1")
    for attempt in ("attempt1", "attempt2"):
        current = scope.model_copy(update={"task_id": attempt, "tenant": "same-task"})
        for index, outcome in enumerate(("error", "success")):
            store.append_episode(Episode(task_id=attempt, scope=current, origin="harness",
                operation="tool_call:action", arguments={}, response={}, outcome=outcome,
                interaction_id=f"{attempt}-{index}",
                previous_interaction_id=f"{attempt}-0" if index else None))
    assert opportunities(store, scope) == []
    store.close()
