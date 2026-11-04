import base64
import json

from worldmodel_harness.automationbench.domain import HttpDomain, endpoint
from worldmodel_harness.core.advisor import Advisor
from worldmodel_harness.core.graph import Graph
from worldmodel_harness.core.lifting import fill, lift_query, origin, source
from worldmodel_harness.core.trace import Episode, Step, ToolDomain

HTTP = HttpDomain()
FILES = "https://www.googleapis.com/drive/v3/files"
SHEET = "https://sheets.googleapis.com/v4/spreadsheets/{}/values/A1:B9"
INBOX = "https://gmail.googleapis.com/gmail/v1/users/me/messages"


def get(url, response):
    return Step("api_fetch", {"method": "GET", "url": url}, json.dumps(response), True)


def mail(to):
    raw = base64.urlsafe_b64encode(f"To: {to}\r\nSubject: Plan\r\n\r\nHi".encode()).decode()
    return Step("api_fetch", {"method": "POST", "url": INBOX + "/send", "body": json.dumps({"raw": raw})}, "{}", True)


def test_any_tools_work_with_the_generic_domain():
    lookup = Step("lookup", {"name": "Ana"}, json.dumps({"id": "u_17", "name": "Ana"}), True)
    notify = Step("notify", {"user": "u_17"}, "ok", True)
    graph = Graph(ToolDomain(), [Episode("t.notify~v1", "Notify Ana", [lookup, notify])])
    assert graph.plan(lift_query("Notify Ana")[0], []) == [("lookup(name)", 1, 1), ("notify(user)", 1, 1)]
    assert source("u_17", [], [lookup], "Notify Ana", ToolDomain()) == "Resp(id) <- lookup(name)"


def test_values_packed_in_encoded_arguments_are_slots():
    assert ("body.raw.decoded.email", "ceo@x.com") in HTTP.slots(mail("ceo@x.com"))
    assert endpoint({"method": "get", "url": SHEET.format("ss_1")}) == "GET sheets.googleapis.com/{}/spreadsheets/{}/values/{}"


def test_origin_and_source_trace_values_to_reads_not_writes():
    read = get(INBOX + "/m_1", {"body": "send it to CEO@x.com"})
    sent = Step("api_fetch", {"method": "POST", "url": "https://slack.com/api/chat.postMessage"}, '{"text": "ceo@x.com"}', True)
    assert origin("ceo@x.com", [read, sent], "", HTTP) == "GET gmail.googleapis.com/gmail/{}/users/me/messages/{}"
    assert origin("ceo@x.com", [sent], "", HTTP) == "Agent"
    assert origin("ceo@x.com", [read], "mail ceo@x.com", HTTP) == "Query"
    listing = get(FILES, {"files": [{"id": "ss_1a", "name": "Payroll"}, {"id": "ss_2b", "name": "Hiring Plan"}]})
    term = source("ss_2b", [], [listing], "Read the Hiring Plan", HTTP)
    assert term == "Resp(id|name@query) <- GET www.googleapis.com/drive/{}/files"
    renamed = get(FILES, {"files": [{"id": "ss_9z", "name": "Hiring Plan"}, {"id": "ss_8y", "name": "Payroll"}]})
    assert fill(term, [], [renamed], "Open the Hiring Plan", HTTP) == "ss_9z"


def run(task, files, chosen, query="Read the Hiring Plan sheet"):
    return Episode(task, query, [
        Step("api_search", {"query": "drive"}, "[]", True),
        get(FILES, {"files": files}),
        get(SHEET.format(chosen), {"values": []}),
    ])


def test_plan_is_filled_from_this_runs_state_without_old_values():
    old = [{"id": "ss_1a", "name": "Payroll"}, {"id": "ss_2b", "name": "Hiring Plan"}]
    # Two workflows, so the API version in the URL is learned as part of the call's shape.
    graph = Graph(HTTP, [run("hr.plan~v1", old, "ss_2b"), run("hr.other~v1", old, "ss_1a", "Read the Payroll sheet")])
    advisor = Advisor(graph, "Read the Hiring Plan sheet")
    first = advisor.after(Step("api_search", {"query": "drive"}, "[]", True))
    assert "- next GET https://www.googleapis.com/drive/v3/files (1/1 runs)" in first
    assert "<id from the call before>" in first
    new = [{"id": "ss_9z", "name": "Hiring Plan"}, {"id": "ss_8y", "name": "Payroll"}]
    second = advisor.after(get(FILES, {"files": new}))
    assert "spreadsheets/ss_9z/values/" in second and "ss_2b" not in second
    assert [e["kind"] for e in advisor.log if e["shown"] and e["kind"] == "plan"] == ["plan", "plan"]


def test_sources_name_the_read_written_values_came_from_and_chain_its_id():
    def episode(task, found):
        steps = [get(SHEET.format("ss_1"), {"values": []})]
        if found:
            steps += [get(INBOX, {"messages": [{"id": "m_1", "threadId": "t_1"}]}), get(INBOX + "/m_1", {"body": "CEO: ceo@x.com"})]
        return Episode(task, "Send the plan to the CEO", [*steps, mail("ceo@x.com" if found else "made@up.example")])

    graph = Graph(HTTP, [episode(f"hr.p~v{i}", i < 3) for i in range(5)])
    advisor = Advisor(graph, "Send the plan to the CEO")
    text = advisor.after(get(SHEET.format("ss_1"), {"values": []}))
    line = next(e for e in advisor.log if e["kind"] == "sources")["lines"][0]
    assert "came from GET https://gmail.googleapis.com/gmail/" in line and "/users/me/messages and then GET" in line
    assert "(3/3 runs)" in line and "ceo@x.com" not in line
    assert "[Experience]" in text


def test_recovery_is_the_call_that_worked_after_the_same_error():
    bad = Step("api_fetch", {"method": "POST", "url": INBOX + "/send"}, '{"error": {"message": "Missing raw"}}', False)
    good = mail("a@b.example")
    graph = Graph(HTTP, [Episode(f"t.x~v{i}", "Email them", [bad, good]) for i in range(3)])
    advisor = Advisor(graph, "Email them")
    text = advisor.after(bad)
    assert "after this error, runs made POST https://gmail.googleapis.com/gmail/" in text
    assert '{"body": {"raw": "<encoded text>"}}' in text and "(3/3 runs)" in text


def test_branch_records_where_runs_went_before_made_up_values():
    def episode(task, found):
        sheet = get(SHEET.format("ss_1"), {"values": []})
        if found:
            steps = [sheet, get(INBOX, {"messages": [{"id": "m_1", "x": "y"}]}), get(INBOX + "/m_1", {"to": "ceo@x.com"})]
        else:
            steps = [sheet, Step("api_search", {"query": "contacts"}, "[]", True)]
        return Episode(task, "Send the plan to the CEO", [*steps, mail("ceo@x.com" if found else f"{task[-1]}@made.up")])

    graph = Graph(HTTP, [episode(f"hr.p~v{i}", i < 4) for i in range(8)])
    template = lift_query("Send the plan to the CEO")[0]
    state = "GET sheets.googleapis.com/{}/spreadsheets/{}/values/{}"
    assert graph.bad_branch(template, state, "api_search")
    assert not graph.bad_branch(template, state, "GET gmail.googleapis.com/gmail/{}/users/me/messages")
    advisor = Advisor(graph, "Send the plan to the CEO")
    advisor.after(get(SHEET.format("ss_1"), {"values": []}))
    warned = advisor.after(Step("api_search", {"query": "contact"}, "[]", True))
    assert "found in no response in 4/4 runs" in warned


def test_the_first_visit_of_a_state_lasts_until_its_next_successful_call():
    sheet = get(SHEET.format("ss_1"), {"values": []})
    failed = Step("api_search", {"query": "contact"}, '{"error": "busy"}', False)
    search = Step("api_search", {"query": "contacts"}, "[]", True)
    graph = Graph(HTTP, [Episode("hr.p~v1", "Send the plan to the CEO", [sheet, failed, search, mail("a@made.up")])])
    state = (lift_query("Send the plan to the CEO")[0], "GET sheets.googleapis.com/{}/spreadsheets/{}/values/{}")
    taken = {b for branches in graph.branches[state].values() for b in branches}
    assert taken == {"fail api_search", "api_search"}
    advisor = Advisor(graph, "Send the plan to the CEO")
    advisor.after(sheet)
    advisor.after(failed)
    assert advisor.first_visit
    advisor.after(search)
    advisor.after(sheet)  # back to a state visited before
    assert not advisor.first_visit


def test_chains_reach_back_through_every_call_not_made_yet():
    def episode(i):
        return Episode(f"t.m~v{i}", "Mail the team lead", [
            Step("hello", {}, "{}", True),
            Step("lookup", {"team": "lead"}, json.dumps({"id": f"u_{i}1"}), True),
            Step("profile", {"user": f"u_{i}1"}, json.dumps({"email": f"p{i}@x.com"}), True),
            Step("send", {"to": f"p{i}@x.com"}, "ok", True),
        ])

    graph = Graph(ToolDomain(), [episode(i) for i in range(3)])
    advisor = Advisor(graph, "Mail the team lead")
    advisor.after(Step("hello", {}, "{}", True))
    unmade = next(e for e in advisor.log if e["kind"] == "unmade")["lines"]
    send = next(line for line in unmade if "send" in line.split(" and then ")[-1])
    assert [c.split("(")[0] for c in send.removeprefix("- runs made ").split(" and then ")] == ["lookup", "profile", "send"]


def test_show_new_does_not_repeat_the_same_text():
    graph = Graph(HTTP, [run("hr.plan~v1", [{"id": "ss_2b", "name": "Hiring Plan"}], "ss_2b")])
    advisor = Advisor(graph, "Read the Hiring Plan sheet")
    assert advisor.after(Step("api_search", {"query": "drive"}, "[]", True))
    assert advisor.after(Step("api_search", {"query": "drive"}, "[]", True)) == ""
    assert all(e["shown"] for e in advisor.log[:2]) and not any(e["shown"] for e in advisor.log[2:])
