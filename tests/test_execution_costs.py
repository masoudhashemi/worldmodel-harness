import asyncio
from types import SimpleNamespace

import pytest
from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.experiment import KnowledgeClient
from worldmodel_harness.opportunities import execution_costs


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_model_costs_survive_failure_and_cancellation(failure):
    class Client:
        async def generate(self, messages, tools):
            if failure:
                raise failure("sensitive SDK detail")
            return AssistantMessage(blocks=[TextBlock(text="done")])

    wrapper = KnowledgeClient(Client(), None, None, SimpleNamespace(memory="none", operational=False, advisory=False))
    if failure:
        with pytest.raises(failure):
            asyncio.run(wrapper.generate([], {}))
    else:
        asyncio.run(wrapper.generate([], {}))
    assert len(wrapper.model_call_costs) == 1
    cost = wrapper.model_call_costs[0]
    assert cost["elapsed_seconds"] >= 0
    assert cost["usage_available"] == (failure is None)
    assert "sensitive" not in str(cost)
    if failure:
        assert "token_usage" not in cost


def test_cost_context_allowlists_observations_and_preserves_unknowns():
    report = {
        "results": [
            {
                "task_name": "one",
                "task_completed_correctly": 1,
                "grading_details": {"secret_solution": "never pass this"},
                "ledger": {"environment_calls": 3},
                "model_call_costs": [
                    {
                        "elapsed_seconds": 5,
                        "usage_available": True,
                        "token_usage": {"input": 100, "answer": 10},
                    },
                    {"elapsed_seconds": 2, "usage_available": False},
                ],
            },
            {"task_name": "old", "ledger": {"environment_calls": 1}},
        ]
    }
    first, old = execution_costs(report)
    assert first.model_elapsed_seconds == 7
    assert first.model_calls == 2
    assert first.reported_tokens == 110
    assert not first.model_usage_complete
    assert old.model_elapsed_seconds is None
    assert old.model_calls is None
    assert old.reported_tokens is None
    assert not old.model_usage_complete
    assert "grading" not in first.model_dump_json()
    assert "secret_solution" not in first.model_dump_json()
