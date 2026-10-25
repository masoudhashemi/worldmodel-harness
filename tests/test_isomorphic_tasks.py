import json
from worldmodel_harness.isomorphic_tasks import (
    ISOMORPHIC_TASKS,
    get_isomorphic_tasks_for_domain,
)


def test_isomorphic_tasks_registry():
    assert len(ISOMORPHIC_TASKS) == 4
    expected_tasks = {
        "operations.twilio_afterhours_incident_isomorphic",
        "operations.contractor_badge_expiration_isomorphic",
        "operations.asana_fire_drill_isomorphic",
        "operations.monday_slack_inventory_isomorphic",
    }
    assert set(ISOMORPHIC_TASKS.keys()) == expected_tasks

    for task_name, task_fn in ISOMORPHIC_TASKS.items():
        task = task_fn()
        assert task["task"] == task_name
        assert "prompt" in task and len(task["prompt"]) >= 2
        assert "info" in task
        info = task["info"]
        assert "initial_state" in info
        assert "assertions" in info and len(info["assertions"]) >= 5
        assert isinstance(task["example_id"], int)


def test_get_isomorphic_tasks_for_domain():
    ops_tasks = get_isomorphic_tasks_for_domain("operations")
    assert len(ops_tasks) == 4
    for row in ops_tasks:
        assert "example_id" in row
        assert "prompt" in row
        assert "info" in row
        parsed_info = json.loads(row["info"])
        assert "initial_state" in parsed_info
        assert "assertions" in parsed_info
