import json

from worldmodel_harness.checks import (
    build_checks_note,
    learn_checks,
    trajectory_calls,
    write_profile,
)

ENDPOINTS = {
    ("GET", "https://slack/history"): "slack.conversations.history",
    ("GET", "https://slack/list"): "slack.conversations.list",
    ("POST", "https://slack/post"): "slack.chat.postMessage",
    ("GET", "https://zoom/meetings"): "zoom.meetings.list",
}


def resolve(method, url):
    return ENDPOINTS.get((method, url))


def _trajectory(tmp_path, name, calls):
    blocks = [
        {"kind": "tool_call", "name": "api_fetch", "arguments": json.dumps(c)} for c in calls
    ]
    path = tmp_path / name / "trajectory.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([[{"role": "assistant", "blocks": blocks}]]))
    return path


def test_reads_before_a_write_are_profiled_with_distinct_targets():
    calls = [
        {"method": "GET", "url": "https://slack/history", "params": '{"channel": "A"}'},
        {"method": "GET", "url": "https://slack/history", "params": '{"channel": "B"}'},
        {"method": "POST", "url": "https://slack/post"},
        {"method": "GET", "url": "https://slack/list"},  # after the write: not a check
        {"method": "GET", "url": "https://unknown/x"},
    ]
    assert write_profile(calls, resolve) == {
        "slack.chat.postMessage": {"slack.conversations.history": 2}
    }


def test_note_uses_other_tasks_only_and_the_task_services(tmp_path):
    history = {"method": "GET", "url": "https://slack/history", "params": '{"channel": "A"}'}
    zoom = {"method": "GET", "url": "https://zoom/meetings"}
    post = {"method": "POST", "url": "https://slack/post"}
    trajectories = {
        task: [_trajectory(tmp_path, task, [history, zoom, post])] for task in ("a", "b", "c")
    }
    table = learn_checks(trajectories, resolve)
    assert trajectory_calls(trajectories["a"][0]) == [history, zoom, post]
    note, info = build_checks_note(table, "Post a summary in Slack #general.", "a")
    # Two other tasks support the check; zoom is not a service this task names.
    assert info["checks"] == {"slack.chat.postMessage": ["slack.conversations.history"]}
    assert "(2 tasks)" in note and "zoom" not in note
    # With the task's own runs and one more removed, the evidence is too thin.
    thin = {"tasks": {k: v for k, v in table["tasks"].items() if k != "b"}}
    assert build_checks_note(thin, "Post a summary in Slack #general.", "a")[0] == ""
