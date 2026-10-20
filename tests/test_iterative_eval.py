"""Offline checks for the iterative curve. No model and no benchmark answers."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "iterative_eval", Path(__file__).parents[1] / "scripts/iterative_eval.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_task_budget_is_fifty_turns_without_added_caps():
    from worldmodel_harness.budget import Ledger
    from worldmodel_harness.experiment import ExperimentSettings

    settings = ExperimentSettings(
        max_turns=runner.DEFAULT_TURNS,
        max_environment_calls=None,
        timeout_seconds=None,
        operational=True,
        phase="test",
    )
    assert settings.max_turns == 50
    assert settings.max_environment_calls is None
    assert settings.timeout_seconds is None
    ledger = Ledger(max_environment_calls=None)
    ledger.charge_environment("task")
    ledger.charge_environment("discovery")
    assert ledger.environment_calls == 2


def test_public_selection_takes_five_from_each_domain_and_shares_one_order():
    pools = {domain: [f"{domain}.task{i}" for i in range(8)] for domain in ("sales", "hr", "finance")}
    chosen = runner.select_per_domain(pools, count=runner.DEFAULT_PER_DOMAIN, seed=1729)
    assert len(chosen) == 15
    assert all(sum(task.startswith(domain + ".") for task in chosen) == 5 for domain in pools)
    assert chosen == runner.select_per_domain(
        {domain: list(reversed(pools[domain])) for domain in reversed(pools)},
        count=5,
        seed=1729,
    )
    assert chosen != runner.select_per_domain(pools, count=5, seed=1730)
    assert runner.DEFAULT_ITERATIONS == 5


def test_rate_limit_wait_grows_and_includes_the_daily_free_cap():
    class RateLimitError(Exception):
        pass

    assert (
        runner.error_name(RateLimitError("Rate limit exceeded: free-models-per-day"))
        == "DailyRateLimit"
    )
    assert "DailyRateLimit" in runner.RATE_LIMIT_ERRORS
    assert [runner.retry_delay(attempt) for attempt in (1, 2, 3, 4, 8)] == [
        60,
        120,
        240,
        480,
        900,
    ]


def test_selection_is_seeded_and_ignores_input_order():
    names = [f"simple.task{i}" for i in range(5)]
    assert runner.select_names(names, limit=2, seed=1729) == runner.select_names(
        list(reversed(names)), limit=2, seed=1729
    )
    assert runner.select_names(names, limit=2, seed=1729) != runner.select_names(
        names, limit=2, seed=1730
    )


def test_reset_store_is_per_task_and_learn_store_is_shared(tmp_path):
    assert runner.store_path(tmp_path, "learn", 1, 0) == runner.store_path(tmp_path, "learn", 3, 1)
    assert runner.store_path(tmp_path, "reset", 1, 0) != runner.store_path(tmp_path, "reset", 1, 1)
    assert runner.store_path(tmp_path, "reset", 1, 0) != runner.store_path(tmp_path, "reset", 2, 0)


def test_missing_grade_is_not_a_pass():
    metrics = runner.task_metrics({"task_name": "simple.a", "finished": False})
    assert metrics["strict_success"] is False
    assert metrics["partial_credit"] is None
    assert metrics["validation_available"] is False


def _row(iteration, arm, task, passed, discovery, environment=10, partial=None):
    return {
        "iteration": iteration,
        "arm": arm,
        "task": task,
        "strict_success": passed,
        "partial_credit": 1 if passed else partial,
        "validation_available": True,
        "finished": True,
        "checked": passed,
        "error_type": None,
        "environment_calls": environment,
        "discovery_calls": discovery,
    }


def test_accuracy_drop_is_not_an_efficiency_win():
    records = []
    for iteration in (1, 2):
        records.append(_row(iteration, "reset", "simple.a", True, 4))
        records.append(_row(iteration, "learn", "simple.a", False, 1, partial=0))
    summary = runner.verdict(records)
    assert summary["accuracy_holds"] is False
    assert summary["efficiency_win"] is False
    assert summary["goal_pattern"] is False


def test_rising_accuracy_and_fewer_searches_match_the_goal():
    records = [
        _row(1, "reset", "simple.a", False, 6, partial=0),
        _row(1, "reset", "simple.b", True, 5),
        _row(1, "learn", "simple.a", False, 6, partial=0),
        _row(1, "learn", "simple.b", False, 5, partial=0),
        _row(2, "reset", "simple.a", False, 6, partial=0),
        _row(2, "reset", "simple.b", False, 5, partial=0),
        _row(2, "learn", "simple.a", True, 1),
        _row(2, "learn", "simple.b", True, 1),
    ]
    summary = runner.verdict(records)
    assert summary["accuracy_holds"] is True
    assert summary["accuracy_rose"] is True
    assert summary["fewer_searches_than_start"] is True
    assert summary["efficiency_win"] is True
    assert summary["goal_pattern"] is True
    assert summary["consistency"]["learn"]["always_passed"] == 0
    assert summary["iterations"][1]["learn"]["strict_accuracy"] == 1


def test_flat_accuracy_with_fewer_searches_is_an_efficiency_win_only():
    records = [
        _row(1, "reset", "simple.a", True, 8),
        _row(1, "learn", "simple.a", True, 8),
        _row(2, "reset", "simple.a", True, 8),
        _row(2, "learn", "simple.a", True, 2),
    ]
    summary = runner.verdict(records)
    assert summary["accuracy_holds"] is True
    assert summary["accuracy_rose"] is False
    assert summary["efficiency_win"] is True
    assert summary["goal_pattern"] is False
    assert summary["consistency"]["learn"]["stable_outcome_tasks"] == 1


def test_partial_progress_keeps_the_started_arm(tmp_path):
    runner.write_json(tmp_path / "manifest.json", {"tasks": ["simple.a"], "iterations": 1})
    (tmp_path / "results.jsonl").write_text(json.dumps(_row(1, "reset", "simple.a", False, 0)) + "\n")
    summary = runner.summarize_directory(tmp_path)
    assert summary["complete"] is False
    assert summary["by_arm"]["reset"][0]["tasks"] == 1
    assert "learn" not in summary["by_arm"]


def test_directory_summary_requires_every_arm_and_iteration(tmp_path):
    runner.write_json(
        tmp_path / "manifest.json",
        {"tasks": ["simple.a"], "iterations": 1},
    )
    (tmp_path / "results.jsonl").write_text(
        "\n".join(
            json.dumps(_row(1, arm, "simple.a", True, 1)) for arm in ("reset", "learn")
        )
        + "\n"
    )
    summary = runner.summarize_directory(tmp_path)
    assert summary["complete"] is True
    assert summary["iterations"][0]["reset"]["discovery_calls"] == 1
