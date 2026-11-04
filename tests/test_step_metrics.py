import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("step_metrics", Path(__file__).parent.parent / "scripts" / "step_metrics.py")
step_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step_metrics)

LIST = "https://slack.com/api/conversations.list"
POST = "https://slack.com/api/chat.postMessage"


def write(tmp_path, turns):
    """A trajectory: turns of (name, arguments, success, result)."""
    messages, n = [{"role": "user", "content": "Post the report to #ops"}], 0
    for calls in turns:
        blocks, results = [], []
        for name, arguments, success, result in calls:
            n += 1
            blocks.append({"kind": "tool_call", "name": name, "arguments": json.dumps(arguments), "tool_call_id": f"c{n}"})
            results.append({"role": "tool", "name": name, "tool_call_id": f"c{n}", "success": success, "content": result})
        messages += [{"role": "assistant", "blocks": blocks}, *results]
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps([messages]))
    return path


def test_steps_count_exploration_until_each_successful_fetch(tmp_path):
    path = write(tmp_path, [
        [("api_search", {"query": "slack"}, True, "[]"), ("api_search", {"query": "post"}, True, "[]")],
        [("api_fetch", {"method": "GET", "url": LIST + "?x=1"}, False, "{}")],
        [("api_fetch", {"method": "GET", "url": LIST}, True, '{"channels": [{"id": "C9", "name": "ops"}]}')],
        [("api_fetch", {"method": "POST", "url": POST, "body": {"channel": "C9", "user": "made"}}, True, "{}")],
    ])
    index = step_metrics.Index()
    index.add("ops.other", [step_metrics.Step("api_fetch", {"method": "GET", "url": LIST}, "{}", True)])
    (first, second), unfinished = step_metrics.steps_of("ops.report~v1", path, index)
    assert unfinished == 0
    assert (first.turns, first.calls, first.searches, first.first_try) == (2, 3, 2, False)
    assert first.level == 2 and first.traced is None  # transition start -> list seen in another workflow
    assert (second.turns, second.calls, second.level) == (0, 0, 0)
    assert second.traced == 0.5  # channel came from the listing, user was made up
