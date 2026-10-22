import asyncio
import json

import pytest

from worldmodel_harness import full_iterative as runner
from worldmodel_harness.experiment import ExperimentSettings
from worldmodel_harness.store import KnowledgeStore


def test_full_schedule_persistence_validation_and_costs(tmp_path, monkeypatch):
    tasks, validation = ("simple.a", "simple.b"), ("simple.v1", "simple.v2")
    monkeypatch.setattr(runner, "dataset", lambda _: [{"info": json.dumps({"task_name": n})} for n in (*tasks, *validation)])
    observations, updates = [], []

    async def run(root, settings, path):
        assert settings.adaptive and settings.semantic_learning and settings.operational
        assert settings.memory == "graph" and not settings.allow_experimental
        store = KnowledgeStore(path)
        count = len(store.learning_tasks())
        observations.append(count)
        store.register_learning_task(f"visit-{count}", "test")
        store.close()
        return {"run_id": f"run-{len(observations)}", "private_grade": 123}

    async def update(root, path, names, **kwargs):
        assert names == validation
        assert kwargs["costs"] == ["sanitized execution costs"]
        updates.append(path)
        return {"deployment": "none", "model_call_costs": [{"token_usage": {"input": 3}, "usage_available": True}],
                "ledger": {"environment_calls": 2}}

    monkeypatch.setattr(runner, "run_automationbench", run)
    monkeypatch.setattr(runner, "validated_update", update)
    monkeypatch.setattr(runner, "execution_costs", lambda _: ["sanitized execution costs"])
    monkeypatch.setattr(runner, "summarize_round", lambda _: {"strict_success": 1, "reported_tokens": 10,
                        "token_usage_complete": True, "environment_calls": 4})
    report = asyncio.run(runner.run_full_iterative(tmp_path, tmp_path / "run", tasks=tasks,
        validation_names=validation, iterations=3, reset_passes=1, base=ExperimentSettings()))
    assert report["complete"] and len(report["results"]) == 8
    assert observations == [0, 0, 0, 1, 2, 3, 4, 5]
    assert len(updates) == 8
    assert len({r["knowledge_path"] for r in report["results"] if r["arm"] == "learn"}) == 1
    assert len(list((tmp_path / "run/checkpoints").glob("*.sqlite"))) == 3
    assert all(r["reported_tokens"] == 26 and r["validation_environment_calls"] == 4 for r in report["curve"])
    assert json.loads((tmp_path / "run/summary.json").read_text())["complete"]


def test_validation_tasks_cannot_overlap_learning_tasks(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        asyncio.run(runner.run_full_iterative(tmp_path, tmp_path / "run", tasks=("simple.a",),
            validation_names=("simple.a", "simple.v"), base=ExperimentSettings()))
    assert not (tmp_path / "run").exists()


def test_missing_usage_stays_incomplete_and_errors_are_visible():
    rows = [{"arm": "learn", "iteration": 1, "metrics": {"strict_success": 0,
        "reported_tokens": 10, "token_usage_complete": True},
        "learning_update": {"model_call_costs": [{"usage_available": False}], "error_type": "TimeoutError"}}]
    result = runner.curve(rows)[0]
    assert result["reported_tokens"] == 10
    assert not result["token_usage_complete"] and result["update_errors"] == 1
