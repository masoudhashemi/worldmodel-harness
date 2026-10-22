import asyncio

import pytest

from worldmodel_harness import controlled_learning as controlled
from worldmodel_harness.experiment import ExperimentSettings
from worldmodel_harness.store import KnowledgeStore


def test_controlled_arms_use_same_checkpoint_and_isolate_evaluation(tmp_path, monkeypatch):
    calls = []

    async def run(root, settings, path):
        store = KnowledgeStore(path, frozen=settings.phase == "frozen")
        prior = store.learning_tasks()
        if settings.phase == "train" or path.name.endswith("reset.sqlite"):
            assert not prior
        else:
            assert prior == {"train1", "train2"}
        if settings.phase != "frozen":
            for name in settings.task_names:
                store.register_learning_task(name, settings.phase)
        store.close()
        calls.append(settings)
        return {"run_id": str(len(calls)), "results": [
            {"task_name": name, "task_completed_correctly": 0,
             "ledger": {"environment_calls": 1, "compute_seconds": 0},
             "model_call_costs": [], "all_model_call_costs": []}
            for name in settings.task_names]}

    async def update(root, path, names, **kwargs):
        assert names == ("validate1", "validate2")
        # The learner gets allowlisted execution costs, never grades.
        assert all(not hasattr(c, "task_completed_correctly") for c in kwargs["costs"])
        return {"deployment": "none"}

    monkeypatch.setattr(controlled, "run_automationbench", run)
    monkeypatch.setattr(controlled, "validated_update", update)
    report = asyncio.run(controlled.run_controlled(tmp_path, tmp_path / "evaluation",
        training_names=("train1", "train2"), validation_names=("validate1", "validate2"),
        evaluation_names=("eval1", "eval2"), base=ExperimentSettings(operational=True)))
    assert report["complete"] and not report["improvement_demonstrated"]
    assert len(calls) == 13
    assert {r["arm"] for r in report["results"]} == set(controlled.ARMS)
    assert all(not s.allow_experimental for s in calls)
    assert sum(s.phase == "frozen" for s in calls) == 2
    assert sum(s.max_online_reflections == 0 for s in calls) == 2


def test_overlapping_split_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        asyncio.run(controlled.run_controlled(tmp_path, tmp_path / "bad",
            training_names=("one",), validation_names=("one", "two"),
            evaluation_names=("three",), base=ExperimentSettings(operational=True)))
    assert not (tmp_path / "bad").exists()
