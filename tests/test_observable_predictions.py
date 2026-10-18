import asyncio

import pytest

from worldmodel_harness.knowledge import Episode, OutputExpectation, Scope
from worldmodel_harness.observable_predictions import (
    CURRENT_EXPECTATIONS,
    compare_expectations,
    prediction_feedback,
)


@pytest.mark.parametrize(
    "response,coverage,outcome,prediction,expected",
    [
        (
            {"value": 2},
            ("value",),
            "success",
            OutputExpectation(path=("value",), expected=2),
            "matched",
        ),
        (
            {"value": True},
            ("value",),
            "success",
            OutputExpectation(path=("value",), expected=1),
            "contradicted",
        ),
        ({}, (), "success", OutputExpectation(path=("value",), predicate="present"), "unobserved"),
        ({"value": 2}, (), "success", OutputExpectation(path=("value",), expected=2), "unobserved"),
        (
            {"value": 2},
            ("value",),
            "unknown",
            OutputExpectation(path=("value",), expected=2),
            "unobserved",
        ),
        (
            {"error": {"code": 401}},
            ("error",),
            "error",
            OutputExpectation(path=("error", "code"), expected=401),
            "matched",
        ),
        (
            {"value": 2},
            ("value",),
            "success",
            OutputExpectation(path=("value",), predicate="type", expected="integer"),
            "matched",
        ),
    ],
)
def test_predictions_only_compare_covered_observations(
    response, coverage, outcome, prediction, expected
):
    episode = Episode(
        task_id="task",
        scope=Scope(environment="test", version="1", task_id="task"),
        operation="read",
        arguments={},
        response=response,
        coverage=coverage,
        outcome=outcome,
        output_expectations=(prediction,),
    )
    assert compare_expectations(episode)[0]["result"] == expected


def test_operational_prediction_is_logged_before_comparison_and_context_resets(tmp_path):
    pytest.importorskip("automationbench")
    from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
    from worldmodel_harness.budget import Ledger
    from worldmodel_harness.decisions import Decision, ProposedAction
    from worldmodel_harness.operational_tools import operational_tools
    from worldmodel_harness.store import KnowledgeStore

    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", store, Ledger())
    tools = operational_tools(
        adapter,
        adapter.tools(),
        adaptive=False,
        allow_experimental=False,
        audit=[],
        refresh_audit=[],
    )
    decision = Decision(
        goal="Inspect encoded value",
        rationale="Predict public utility output",
        action=ProposedAction(tool="base64_encode", arguments={"text": "abc"}),
        output_expectations=(OutputExpectation(expected="YWJj"),),
    )
    result = asyncio.run(tools[0].executor(decision))
    assert result.success
    episode = store.episodes(adapter.scope)[-1]
    assert episode.output_expectations == decision.output_expectations
    assert (
        adapter.prediction_assessments[-1]["proposed_output_assessments"][0]["result"] == "matched"
    )
    assert CURRENT_EXPECTATIONS.get() is None
    assert store.claims() == []
    from stirrup.core.models import AssistantMessage, TextBlock

    from worldmodel_harness.experiment import ExperimentSettings, KnowledgeClient

    class Client:
        async def generate(self, messages, tools):
            assert "Observed checks of your prior output predictions" in messages[-1].content
            assert episode.episode_id in messages[-1].content
            assert '"matched"' in messages[-1].content
            assert len(messages[-1].content) <= 12000
            return AssistantMessage(blocks=[TextBlock(text="Observed prediction result")])

    wrapper = KnowledgeClient(Client(), adapter, store, ExperimentSettings(operational=True))
    asyncio.run(wrapper.generate([], {}))
    assert wrapper.retrievals[-1]["prediction_feedback_chars"] > 0
    store.close()


def test_feedback_is_bounded_and_does_not_duplicate_values():
    assessment = {
        "episode_id": "e1",
        "proposed_output_assessments": [
            {
                "episode_id": "e1",
                "prediction": {"path": ["status"], "predicate": "equals", "expected": "SECRET"},
                "observed": "PRIVATE",
                "result": "contradicted",
            }
        ],
    }
    result = prediction_feedback([assessment], max_chars=1000)
    assert "contradicted" in result
    assert "SECRET" not in result and "PRIVATE" not in result
    assert len(result) <= 1000
    from worldmodel_harness.context_compression import MandatoryContextOverflow

    with pytest.raises(MandatoryContextOverflow):
        prediction_feedback([assessment], max_chars=1)
