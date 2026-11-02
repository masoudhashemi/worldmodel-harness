import json

from worldmodel_harness.experience import ExperienceAdvisor, Model
from worldmodel_harness.lifting import Episode, Step

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
