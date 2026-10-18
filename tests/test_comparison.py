import copy

import pytest

from worldmodel_harness.comparison import paired_report


def reports():
    arms = {}
    for name, calls in (
        ("flat_fixed", 8),
        ("graph_fixed", 7),
        ("flat_adaptive", 6),
        ("graph_adaptive", 3),
    ):
        arms[name] = {
            "results": [
                {
                    "task_name": "task",
                    "task_completed_correctly": 1,
                    "ledger": {"environment_calls": calls},
                    "token_usage": {"input": 100, "answer": 20, "reasoning": 10},
                }
            ]
        }
    return arms


def test_paired_interaction_and_costs_are_descriptive():
    result = paired_report(reports())
    contrasts = result["paired_tasks"][0]["contrasts"]
    assert contrasts["environment_calls"]["interaction"] == -2
    assert contrasts["success"]["interaction"] == 0
    assert not result["utility_comparison_complete"]


@pytest.mark.parametrize(
    "failure", ["missing_arm", "wrong_task", "execution_error", "duplicate_task"]
)
def test_incomplete_or_mismatched_comparisons_rejected(failure):
    arms = reports()
    if failure == "missing_arm":
        del arms["graph_fixed"]
    elif failure == "wrong_task":
        arms["graph_fixed"]["results"][0]["task_name"] = "different"
    elif failure == "execution_error":
        arms["graph_fixed"]["results"][0]["error_type"] = "TimeoutError"
    else:
        arms["graph_fixed"]["results"].append(copy.deepcopy(arms["graph_fixed"]["results"][0]))
    with pytest.raises(ValueError):
        paired_report(arms)


def test_different_model_or_budget_rejected():
    arms = reports()
    arms["graph_fixed"]["model_settings"] = {"name": "different-model"}
    with pytest.raises(ValueError, match="differs"):
        paired_report(arms)
    arms = reports()
    arms["graph_fixed"]["settings"] = {"max_environment_calls": 999}
    with pytest.raises(ValueError, match="budgets differ"):
        paired_report(arms)
