import copy

import pytest

from worldmodel_harness.curve_report import (
    compare_completed_rounds,
    execution_diagnostics,
    runtime_diagnostics,
)


def test_returned_usage_on_failed_tasks_is_counted_without_hiding_failures():
    tasks = [{
        "task_completed_correctly": 0,
        "error_type": "TimeoutError",
        "model_call_costs": [
            {"usage_available": True, "token_usage": {"input": 100, "answer": 20}},
            {"usage_available": False, "error_type": "CancelledError"},
        ],
    }]
    report = runtime_diagnostics(tasks)
    assert report["returned_call_tokens_lower_bound"] == 120
    assert report["model_calls_without_usage"] == 1
    assert report["outcomes_by_execution_error"]["TimeoutError"] == {
        "tasks": 1, "strict_passes": 0,
    }
    assert report["call_telemetry_complete"]
    assert not runtime_diagnostics([{"task_completed_correctly": 0}])["call_telemetry_complete"]


def fixture():
    curve = {
        "task_names": ["a", "b"],
        "protocol": "repeated-in-sample-online-learning",
        "complete": True,
        "rounds": [{"round": 1, "run_id": "one"}, {"round": 2, "run_id": "two"}],
    }
    runs = {
        name: {
            "settings": {"budget": 40},
            "results": [
                {
                    "task_name": "a",
                    "task_completed_correctly": passed,
                    "ledger": {"environment_calls": 5},
                    "token_usage": {"input": 100},
                },
                {
                    "task_name": "b",
                    "task_completed_correctly": 1 - passed,
                    "ledger": {"environment_calls": 3},
                    "error_type": "OutputTokenLimitError",
                },
            ],
        }
        for name, passed in (("one", 0), ("two", 1))
    }
    return curve, runs


def test_successes_and_regressions_are_paired_and_missing_usage_is_unknown():
    curve, runs = fixture()
    report = compare_completed_rounds(curve, runs)
    change = report["paired_changes"][0]
    assert change["new_passes"] == ["a"]
    assert change["regressions"] == ["b"]
    assert not report["strict_passes_increased"]
    assert report["strict_passes_nondecreasing"]
    assert change["per_task"][1]["reported_token_change"] is None
    assert not report["rounds"][0]["returned_task_usage_complete"]
    assert not report["rounds"][0]["model_timing_complete"]
    assert not report["causal_learning_benefit_established"]


def test_model_timing_counts_failed_calls_without_claiming_token_completeness():
    curve, runs = fixture()
    for run in runs.values():
        for task in run["results"]:
            task["model_call_costs"] = [{"elapsed_seconds": 2.5, "usage_available": False}]
    report = compare_completed_rounds(curve, runs)
    assert report["rounds"][0]["reported_model_seconds"] == 5
    assert report["rounds"][0]["model_timing_complete"]
    assert not report["rounds"][0]["returned_task_usage_complete"]


@pytest.mark.parametrize("failure", ["missing_task", "duplicate_task", "changed_controls"])
def test_incomparable_rounds_rejected(failure):
    curve, runs = fixture()
    if failure == "missing_task":
        runs["two"]["results"].pop()
    elif failure == "duplicate_task":
        runs["two"]["results"].append(copy.deepcopy(runs["two"]["results"][0]))
    else:
        runs["two"]["settings"]["budget"] = 100
    with pytest.raises(ValueError):
        compare_completed_rounds(curve, runs)


def test_no_completed_rounds_does_not_claim_a_trend():
    curve, _ = fixture()
    curve.update(rounds=[], complete=False)
    report = compare_completed_rounds(curve, {})
    assert report["strict_passes_nondecreasing"] is None
    assert report["strict_passes_increased"] is None


def test_execution_diagnostics_count_observations_not_successes():
    task = {
        "task_completed_correctly": 0,
        "ledger": {"environment_calls": 4, "by_purpose": {"discovery": 3, "task": 1}},
        "retrievals": [
            {"injected_claims": ["a"], "injected_chars": 100},
            {"injected_claims": [], "injected_chars": 20},
        ],
        "artifact_invocations": 2,
        "information_invocations": 1,
    }
    diagnostics = execution_diagnostics([task])
    assert diagnostics["environment_calls_by_purpose"] == {"discovery": 3, "task": 1}
    assert diagnostics["purpose_accounting_complete"]
    assert diagnostics["recorded_retrieval_decisions"] == 2
    assert diagnostics["decisions_with_injected_claims"] == 1
    assert diagnostics["recorded_injected_chars"] == 120
    assert diagnostics["artifact_invocations"] == 2
    assert diagnostics["invocation_accounting_complete"]


def test_execution_diagnostics_flag_missing_accounting():
    diagnostics = execution_diagnostics([{"ledger": {"environment_calls": 4}}])
    assert not diagnostics["purpose_accounting_complete"]
    assert not diagnostics["invocation_accounting_complete"]
