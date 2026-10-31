import json

from worldmodel_harness.fixes import LOOP_REPEATS, FixAdvisor, FixMemory, describe, endpoint, shape

SEND = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
NO_TO = json.dumps({"error": {"code": 400, "message": "Recipient address required. No 'to' was found"}})
SENT = json.dumps({"id": "m1", "labelIds": ["SENT"]})


def _post(url, body=None):
    return {"method": "POST", "url": url, **({"body": json.dumps(body)} if body is not None else {})}


def test_shape_keeps_formats_not_values():
    sent = _post(SEND, {"raw": "VG86IGFAYi5jb20KU3ViamVjdDogSGk"})
    assert shape(sent) == '{"body": {"raw": "<base64url>"}}'
    assert shape(_post(SEND)) == "{}"
    form = {"method": "POST", "url": "https://api.twilio.com/x", "body": "To=%2B1555&From=%2B1666&Body=Hi"}
    assert shape(form) == '{"body": "<non-JSON form-encoded text>"}'
    assert endpoint(_post("https://sheets.googleapis.com/v4/spreadsheets/ss_x/values/Tab!A2:B3")) == (
        "POST sheets.googleapis.com/{}/spreadsheets/{}/values/{}"
    )


def test_memory_learns_shape_fix_and_needs_support():
    episode = [(_post(SEND), NO_TO, False), (_post(SEND, {"raw": "VG86IGFAYi5jb20KU3ViamVjdA"}), SENT, True)]
    memory = FixMemory()
    memory.learn(episode)
    assert memory.lookup(_post(SEND), NO_TO) is None  # one episode is not enough
    memory.learn(episode)
    assert memory.lookup(_post(SEND), NO_TO) == '{"body": {"raw": "<base64url>"}}'


def test_advisor_prefers_same_task_success_and_flags_loops():
    memory = FixMemory()
    for _ in range(2):
        memory.learn([(_post(SEND), NO_TO, False), (_post(SEND, {"raw": "VG86IGFAYi5jb20KU3ViamVjdA"}), SENT, True)])
    advisor = FixAdvisor(memory)
    first = advisor.annotate(_post(SEND), NO_TO, False)
    assert first.startswith(NO_TO) and "had no `body` argument" in first
    assert """body='{"raw": "<base64url>"}'""" in first
    good = _post(SEND, {"raw": "VG86IGFAYi5jb20KU3ViamVjdA"})
    assert advisor.annotate(good, SENT, True) == SENT
    for _ in range(LOOP_REPEATS - 1):
        note = advisor.annotate(_post(SEND), NO_TO, False)
    assert "earlier call that worked" in note and "VG86IGFAYi5jb20KU3ViamVjdA" in note
    assert f"failed {LOOP_REPEATS} times" in note
    assert [e["kinds"] for e in advisor.log] == [["memory"], ["same_task"], ["same_task", "loop"]]


def test_unknown_error_is_unchanged_until_it_loops():
    advisor = FixAdvisor()
    error = json.dumps({"error": "boom"})
    assert advisor.annotate(_post(SEND), error, False) == error
    assert advisor.annotate(_post(SEND), error, False) == error
    assert "failed 3 times" in advisor.annotate(_post(SEND), error, False)


def test_describe_names_only_what_differs():
    raw = '{"body": {"raw": "<base64url>"}}'
    assert describe('{"body": {"raw": "<text>"}}', raw)[0] == (
        "`body.raw` was text; it must be base64url text (base64_encode produces it)."
    )
    both = '{"body": {"values": [["<text>"]]}, "params": {"valueInputOption": "<text>"}}'
    said = " ".join(describe('{"params": {"valueInputOption": "<text>"}}', both))
    assert "no `body` argument" in said and "params=" not in said
