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
    hint = advisor.source_hint()
    assert "the to of POST gmail.googleapis.com/gmail/{}/users/me/messages/send came from" in hint
    assert "GET https://gmail.googleapis.com/gmail/v1/users/me/messages/" in hint and "(1/1 runs)" in hint
    advisor.observe({"method": "GET", "url": inbox}, "{}", True)
    assert "came from" not in advisor.source_hint()  # already read: no value source left


def test_source_gate_counts_an_episode_once_when_it_wrote_found_and_made_up_values():
    import base64

    inbox = "https://gmail.googleapis.com/gmail/v1/users/me/messages/m_1"
    raw = base64.urlsafe_b64encode(b"To: a@x.com, made@up.com\r\n\r\nHi").decode()
    send = {"method": "POST", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send", "body": json.dumps({"raw": raw})}
    steps = [Step("api_fetch", {"method": "GET", "url": inbox}, '{"body": "a@x.com"}', True), Step("api_fetch", send, "{}", True)]
    graph = Model([Episode("hr.p~v1", "Send the plan", steps)])
    assert graph.sources(lift_query("Send the plan")[0], set())[0][3:] == (1, 1)
