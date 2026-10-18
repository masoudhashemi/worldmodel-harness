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
