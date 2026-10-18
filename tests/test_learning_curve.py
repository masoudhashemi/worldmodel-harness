import asyncio
import json

import pytest
from pydantic import ValidationError

from worldmodel_harness import learning_curve
from worldmodel_harness.learning_curve import LearningCurveSettings, select_task_names
from worldmodel_harness.store import KnowledgeStore


def rows():
    return [
        {"info": json.dumps({"task_name": f"domain{d}.task{i}"})}
        for d in range(6)
        for i in range(100)
    ]


def test_fixed_50_subset_balanced_without_solutions_or_scores():
    selected = select_task_names(rows(), count=50, seed=1729)
    assert len(selected) == len(set(selected)) == 50
    assert selected == select_task_names(rows(), count=50, seed=1729)
    assert selected != select_task_names(rows(), count=50, seed=99)
    counts = [sum(n.startswith(f"domain{d}.") for n in selected) for d in range(6)]
    assert max(counts) - min(counts) <= 1
    with pytest.raises(ValidationError):
        LearningCurveSettings(task_count=600)


@pytest.mark.parametrize("operational", [False, True])
def test_rounds_reuse_same_tasks_and_memory_without_holdout_or_validation(
    tmp_path, monkeypatch, operational
):
    monkeypatch.setattr(learning_curve, "dataset", lambda _: rows())
    calls, updates = [], []

    async def run(root, settings, knowledge_path):
        store = KnowledgeStore(knowledge_path)
        if calls:
            assert store.learning_tasks() == set(settings.task_names)
        else:
            assert store.learning_tasks() == set()
            assert store.procedures() == []
        for name in settings.task_names:
            store.register_learning_task(name, "online")
        store.close()
        calls.append(settings)
        return {
            "run_id": str(len(calls)),
            "results": [
                {
                    "task_name": name,
                    "task_completed_correctly": 1 if len(calls) > 1 else 0,
                    "ledger": {"environment_calls": 2, "compute_seconds": 0.1},
                    "token_usage": {"input": 100, "answer": 10},
                    "artifact_invocations": 0,
                }
                for name in settings.task_names
            ],
        }

    async def learn(root, knowledge, prototype, settings, horizon, *, costs):
        assert len(costs) == 50
        assert all(c.reported_tokens == 110 for c in costs)
        updates.append(horizon)
        return {"deployment": "none", "adaptation_seconds": 0.0}

    monkeypatch.setattr(learning_curve, "run_automationbench", run)
    monkeypatch.setattr(learning_curve, "learn_from_round", learn)
    report = asyncio.run(
        learning_curve.run_learning_curve(
            tmp_path, LearningCurveSettings(rounds=3, task_count=50, operational=operational)
        )
    )
    assert len(calls) == 3
    assert all(c.phase == "online" and c.allow_experimental for c in calls)
    assert all(c.operational == operational for c in calls)
    assert all(c.task_names == calls[0].task_names for c in calls)
    assert updates == [100, 50]
    assert report["held_out_tasks"] == []
    assert report["observed_success_change"] == 1
    assert not report["improvement_demonstrated"]  # Not a statistical conclusion.
    assert report["complete"]
