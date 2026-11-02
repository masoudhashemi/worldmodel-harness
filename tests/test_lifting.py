import json

from worldmodel_harness.lifting import Step, fill, lift_query, slots, source

SHEET = "https://sheets.googleapis.com/v4/spreadsheets/ss_budget/values/Q1!A1:D9"


def test_query_template_keeps_wording_and_types_literals():
    template, bindings = lift_query("Email ana@x.com about deal 006A12 for $1,500 in #sales.")
    assert template == "Email {email} about deal {id} for {money} in {channel}."
    assert bindings == [("email", "ana@x.com"), ("channel", "#sales"), ("money", "$1,500"), ("id", "006A12")]


def test_slots_keep_short_literals_only():
    body = json.dumps({"to": "ana@x.com", "text": "Hello team, the budget for this quarter is ready"})
    found = dict(slots({"method": "POST", "url": SHEET + "?valueInputOption=RAW", "body": body}))
    assert found["path[3]"] == "ss_budget" and found["query.valueInputOption"] == "RAW"
    assert found["body.to"] == "ana@x.com" and "body.text" not in found


def test_source_and_fill_round_trip_across_tasks():
    listing = Step("api_fetch", {}, json.dumps({"files": [{"id": "ss_budget", "name": "Budget"}]}), True)
    bindings = [("email", "ana@x.com")]
    assert source("ss_budget", bindings, [listing]) == "Resp(id) <- GET "
    assert source("ana@x.com", bindings, [listing]) == "Query(email,0)"
    assert source("RAW", bindings, [listing]) == "Free"
    other = Step("api_fetch", {}, json.dumps({"files": [{"id": "ss_hiring", "name": "Hiring"}]}), True)
    assert fill("Resp(id)", [("email", "bo@y.com")], [other]) == "ss_hiring"
    assert fill("Query(email,0)", [("email", "bo@y.com")], [other]) == "bo@y.com"
    assert fill("Const(RAW)", [], []) == "RAW" and fill("Free", [], []) is None


def test_record_choice_is_left_to_the_agent_when_the_query_does_not_name_it():
    files = [{"id": "ss_q1", "name": "Q1 Budget"}, {"id": "ss_hires", "name": "Recruiting Performance"}]
    listing = Step("api_fetch", {}, json.dumps({"files": files}), True)
    term = source("ss_hires", [], [listing])
    assert term == "Resp(id|?) <- GET "
    assert fill(term, [], [listing]) is None
    assert source("ss_q1", [("id", "ss_q1")], [listing]) == "Query(id,0)"


def test_response_value_comes_from_the_action_that_returned_it():
    sheet = Step("api_fetch", {"method": "GET", "url": SHEET}, json.dumps({"id": "ss_budget"}), True)
    user = Step("api_fetch", {"method": "GET", "url": "https://slack.com/api/users.info"}, json.dumps({"id": "U1"}), True)
    term = source("ss_budget", [], [sheet, user])
    assert term.startswith("Resp(id) <- GET sheets.googleapis.com")
    assert fill(term, [], [sheet, user]) == "ss_budget"


def test_record_named_in_the_query_is_anchored_without_its_value():
    query = "Pull the hires from the Recruiting Performance sheet and email ana@x.com"
    files = [{"id": "ss_q1", "name": "Q1 Budget"}, {"id": "ss_hires", "name": "Recruiting Performance"}]
    term = source("ss_hires", [], [Step("api_fetch", {}, json.dumps({"files": files}), True)], query)
    assert term == "Resp(id|name@query) <- GET "
    other = [{"id": "ss_9x", "name": "Hiring Pipeline"}, {"id": "ss_7y", "name": "Q1 Budget"}]
    listing = Step("api_fetch", {}, json.dumps({"files": other}), True)
    assert fill(term, [], [listing], "Copy the Q1 Budget sheet") == "ss_7y"


def test_query_anchor_fills_only_when_one_record_is_named():
    term = "Resp(id|name@query) <- GET "
    files = [{"id": "ss_a", "name": "Q1 Budget"}, {"id": "ss_b", "name": "Hiring Pipeline"}]
    listing = Step("api_fetch", {}, json.dumps({"files": files}), True)
    assert fill(term, [], [listing], "Compare the Q1 Budget with the Hiring Pipeline") is None
    assert fill(term, [], [listing], "Copy the Hiring Pipeline sheet") == "ss_b"
