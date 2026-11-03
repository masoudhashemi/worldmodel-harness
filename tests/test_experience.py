import json

from worldmodel_harness.experience import ExperienceAdvisor, Model
from worldmodel_harness.lifting import Episode, Step, lift_query

FILES = "https://www.googleapis.com/drive/v3/files"
SHEET = "https://sheets.googleapis.com/v4/spreadsheets/{}/values/A1:B9"


def run(task, query, files, chosen):
    listing = Step("api_fetch", {"method": "GET", "url": FILES}, json.dumps({"files": files}), True)
    read = Step("api_fetch", {"method": "GET", "url": SHEET.format(chosen)}, json.dumps({"values": []}), True)
    return Episode(task, query, [Step("api_search", {"query": "drive"}, "[]", True), listing, read])


def test_plan_fills_the_next_call_from_this_runs_state_without_old_values():
    old = [{"id": "ss_1a", "name": "Payroll"}, {"id": "ss_2b", "name": "Hiring Plan"}]
    graph = Model([run("hr.plan~v1", "Read the Hiring Plan sheet and email ana@x.com", old, "ss_2b")])
    advisor = ExperienceAdvisor(graph, "Read the Hiring Plan sheet and email bo@y.com")
    first = advisor.annotate_search("[]")
    assert "1. GET https://www.googleapis.com/drive/v3/files (1/1)" in first
    assert "2. GET https://sheets.googleapis.com/v4/spreadsheets/<id from call 1>/values/" in first
    new = [{"id": "ss_9z", "name": "Hiring Plan"}, {"id": "ss_8y", "name": "Payroll"}]
    advisor.observe({"method": "GET", "url": FILES}, json.dumps({"files": new}), True)
    second = advisor.annotate_search("[]")
    assert "spreadsheets/ss_9z/values/" in second and "ss_2b" not in second
    assert advisor.annotate_search("[]") == "[]"  # the same plan is not repeated


def test_no_plan_for_an_unseen_workflow():
    graph = Model([run("hr.plan~v1", "Read the Hiring Plan sheet", [{"id": "ss_1", "name": "x"}], "ss_1")])
    assert ExperienceAdvisor(graph, "Archive every closed ticket").annotate_search("[]") == "[]"


def test_source_hint_names_the_read_that_held_written_values_and_skips_made_up_ones():
    import base64

    inbox = "https://gmail.googleapis.com/gmail/v1/users/me/messages/m_1"
    send = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    def mail(task, to, found):
        raw = base64.urlsafe_b64encode(f"To: {to}\r\n\r\nHi".encode()).decode()
        steps = [Step("api_fetch", {"method": "GET", "url": inbox}, json.dumps({"body": f"CEO is {to}"}), True)] if found else []
        steps.append(Step("api_fetch", {"method": "POST", "url": send, "body": json.dumps({"raw": raw})}, "{}", True))
        return Episode(task, "Send the plan to the CEO", steps)

    graph = Model([mail("hr.p~v1", "a@x.com", True), mail("hr.p~v2", "made@up.com", False), mail("hr.p~v3", "z@made.com", False)])
    advisor = ExperienceAdvisor(graph, "Send the plan to the CEO")
    hint = advisor._source_hint()
    assert "the to of POST gmail.googleapis.com/gmail/{}/users/me/messages/send came from" in hint
    assert "GET https://gmail.googleapis.com/gmail/v1/users/me/messages/" in hint and "(1/1 runs)" in hint
    advisor.observe({"method": "GET", "url": inbox}, "{}", True)
    assert "came from" not in advisor._source_hint()  # already read: no value source left


def test_source_gate_counts_an_episode_once_when_it_wrote_found_and_made_up_values():
    import base64

    inbox = "https://gmail.googleapis.com/gmail/v1/users/me/messages/m_1"
    raw = base64.urlsafe_b64encode(b"To: a@x.com, made@up.com\r\n\r\nHi").decode()
    send = {"method": "POST", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send", "body": json.dumps({"raw": raw})}
    steps = [Step("api_fetch", {"method": "GET", "url": inbox}, '{"body": "a@x.com"}', True), Step("api_fetch", send, "{}", True)]
    graph = Model([Episode("hr.p~v1", "Send the plan", steps)])
    assert graph.sources(lift_query("Send the plan")[0], set())[0][3:] == (1, 1)


def test_hinted_read_is_chained_through_the_listing_its_id_came_from():
    import base64

    listing = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    raw = base64.urlsafe_b64encode(b"To: a@x.com\r\n\r\nHi").decode()
    steps = [
        Step("api_fetch", {"method": "GET", "url": listing}, json.dumps({"messages": [{"id": "msg_1"}]}), True),
        Step("api_fetch", {"method": "GET", "url": listing + "/msg_1"}, '{"body": "a@x.com"}', True),
        Step("api_fetch", {"method": "POST", "url": listing + "/send", "body": json.dumps({"raw": raw})}, "{}", True),
    ]
    advisor = ExperienceAdvisor(Model([Episode("hr.p~v1", "Send the plan", steps)]), "Send the plan")
    hint = advisor._source_hint()
    assert f"GET {listing} and then GET {listing}/<id from the first call>" in hint
    advisor.observe({"method": "GET", "url": listing}, json.dumps({"messages": [{"id": "msg_9"}]}), True)
    assert f"GET {listing}/msg_9 (1/1 runs)" in advisor._source_hint()  # listed: the id is filled in


def test_source_hint_waits_for_a_stall_and_is_shown_once():
    import base64

    listing = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    raw = base64.urlsafe_b64encode(b"To: a@x.com\r\n\r\nHi").decode()
    steps = [
        Step("api_fetch", {"method": "GET", "url": listing}, json.dumps({"messages": [{"id": "msg_1"}]}), True),
        Step("api_fetch", {"method": "GET", "url": listing + "/msg_1"}, '{"body": "a@x.com"}', True),
        Step("api_fetch", {"method": "POST", "url": listing + "/send", "body": json.dumps({"raw": raw})}, "{}", True),
    ]
    advisor = ExperienceAdvisor(Model([Episode("hr.p~v1", "Send the plan", steps)]), "Send the plan")
    assert advisor.source_hint() == ""  # one repeat is not a stall
    assert "and then GET" in advisor.source_hint()
    assert advisor.source_hint() == ""  # the same hint is not shown again
    advisor.observe({"method": "GET", "url": listing}, json.dumps({"messages": [{"id": "msg_9"}]}), True)
    assert advisor.source_hint() == ""  # progress resets the stall
    assert f"GET {listing}/msg_9" in advisor.source_hint()  # new content after progress


def test_branch_warning_names_where_runs_that_found_the_value_took_it_from():
    import base64

    sheet = "https://sheets.googleapis.com/v4/spreadsheets/ss_1"
    listing = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    send = listing + "/send"

    def run(task, found):
        to = "ceo@x.com" if found else f"{task[-2:]}@made.up"
        raw = base64.urlsafe_b64encode(f"To: {to}\r\n\r\nHi".encode()).decode()
        steps = [Step("api_fetch", {"method": "GET", "url": sheet}, "{}", True)]
        if found:
            steps += [
                Step("api_fetch", {"method": "GET", "url": listing}, json.dumps({"messages": [{"id": "msg_1"}]}), True),
                Step("api_fetch", {"method": "GET", "url": listing + "/msg_1"}, '{"body": "CEO: ceo@x.com"}', True),
            ]
        else:
            steps.append(Step("api_search", {"query": "contacts"}, "[]", True))
        steps.append(Step("api_fetch", {"method": "POST", "url": send, "body": json.dumps({"raw": raw})}, "{}", True))
        return Episode(task, "Send the plan to the CEO", steps)

    graph = Model([run(f"hr.p~v{i}", i < 4) for i in range(8)])
    advisor = ExperienceAdvisor(graph, "Send the plan to the CEO")
    advisor.observe({"method": "GET", "url": sheet}, "{}", True)
    assert advisor.warn_branch("api_fetch", {"method": "GET", "url": sheet}, "{}", True) == "{}"
    warned = advisor.warn_branch("api_search", {"query": "contact"}, "[]", True)
    assert "searched again at this point wrote a value found in no response they had read in 4/4 runs" in warned
    assert f"GET {listing} and then GET {listing}/<id from the first call> (4/4 runs)" in warned
    assert advisor.warn_branch("api_search", {"query": "contact"}, "[]", True) == "[]"  # once per state


def test_write_check_stops_a_near_copy_of_a_read_address_once():
    import base64

    sheet = "https://sheets.googleapis.com/v4/spreadsheets/ss_1/values/A1:B9"
    send = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    def mail(to):
        raw = base64.urlsafe_b64encode(f"To: {to}\r\n\r\nHi".encode()).decode()
        return {"method": "POST", "url": send, "body": json.dumps({"raw": raw})}

    def run(task, address):
        read = Step("api_fetch", {"method": "GET", "url": sheet}, json.dumps({"values": [["Ana", address]]}), True)
        return Episode(task, "Email the referrer", [read, Step("api_fetch", mail(address), "{}", True)])

    graph = Model([run(f"hr.r~v{i}", f"ref{i}@x.com") for i in range(4)])
    advisor = ExperienceAdvisor(graph, "Email the referrer")
    advisor.observe({"method": "GET", "url": sheet}, json.dumps({"values": [["Ximena", "ximena.r@szd.example"]]}), True)
    assert advisor.check_write(mail("ximena.r@szd.example")) is None  # found: sent
    stopped = advisor.check_write(mail("simena.r@sxd.example"))
    assert "you wrote simena.r@sxd.example; the response of " + sheet + " has ximena.r@szd.example" in json.loads(stopped)["message"]
    assert advisor.check_write(mail("simena.r@sxd.example")) is None  # sent again on purpose: let through
    assert advisor.check_write(mail("someone@else.example")) is None  # made up, but no near copy
    # Without runs that found this slot's values in reads, nothing is stopped.
    assert ExperienceAdvisor(Model([]), "Email the referrer").check_write(mail("simena.r@sxd.example")) is None
