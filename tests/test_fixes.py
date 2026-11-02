import json

from worldmodel_harness.fixes import (
    LOOP_REPEATS,
    MIN_DEAD_ENDS,
    FixAdvisor,
    FixMemory,
    SearchRepeats,
    describe,
    endpoint,
    shape,
    template,
)

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


SHEETS = "https://sheets.googleapis.com/v4/spreadsheets"
NO_HANDLER = json.dumps({"error": {"code": 404, "message": "No handler for GET sheets/v4/spreadsheets"}})
NO_ACCOUNT = json.dumps({"error": "No hubspot account is connected to this workspace"})


def test_route_edge_suggests_the_endpoint_that_worked_next():
    episode = [
        ({"method": "GET", "url": SHEETS}, NO_HANDLER, False),
        ({"method": "GET", "url": SHEETS + "/ss_budget"}, "{}", True),
    ]
    memory = FixMemory()
    memory.learn(episode)
    assert memory.route({"method": "GET", "url": SHEETS}, NO_HANDLER) is None  # needs support
    memory.learn(episode)
    note = FixAdvisor(memory).annotate({"method": "GET", "url": SHEETS}, NO_HANDLER, False)
    assert "url='https://sheets.googleapis.com/v4/spreadsheets/{id}'" in note and "ss_budget" not in note
    assert template({"method": "GET", "url": "https://api.intercom.io/contacts/6501"}).endswith("contacts/{id}")


def test_dead_end_only_when_the_service_never_recovered():
    hub = {"method": "GET", "url": "https://api.hubapi.com/crm/v3/objects/deals"}
    memory = FixMemory()
    for _ in range(MIN_DEAD_ENDS):
        memory.learn([(hub, NO_ACCOUNT, False), (hub, NO_ACCOUNT, False)])
    assert list(memory.outcomes.values()) == [[MIN_DEAD_ENDS, 0]]  # one count per episode
    advisor = FixAdvisor(memory)
    note = advisor.annotate(hub, NO_ACCOUNT, False)
    assert "Do not call api.hubapi.com again" in note and "another source" not in note
    assert advisor.log[0]["kinds"] == ["dead_end"]
    memory.learn([(hub, NO_ACCOUNT, False), (hub, "{}", True)])
    assert not memory.dead_end(hub, NO_ACCOUNT)


def test_unavailable_service_is_named_only_when_called_again():
    advisor = FixAdvisor()
    deals = {"method": "GET", "url": "https://api.hubapi.com/crm/v3/objects/deals"}
    contacts = {"method": "GET", "url": "https://api.hubapi.com/crm/v3/objects/contacts"}
    assert advisor.annotate(deals, NO_ACCOUNT, False) == NO_ACCOUNT
    note = advisor.annotate(contacts, NO_ACCOUNT, False)
    assert "api.hubapi.com already answered that no account is connected" in note
    assert advisor.log[-1]["kinds"] == ["unavailable"]


def test_repeated_search_is_answered_with_a_pointer():
    repeats = SearchRepeats()
    first = json.dumps({"results": [{"id": "gmail.send"}]})
    assert repeats.annotate({"query": "talent lead email"}, first) == first
    assert repeats.annotate({"query": "slack users"}, first) == first
    again = json.loads(repeats.annotate({"query": "Talent  lead EMAIL"}, first))
    assert again["results"] == [] and "search #1" in again["note"]
    assert repeats.annotate({"query": "talent lead email", "limit": 5}, first) == first
    assert repeats.log == [{"search": 3, "repeats": 1}]
