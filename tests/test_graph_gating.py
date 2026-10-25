from worldmodel_harness.graph_gating import (
    detect_services_from_prompt,
    extract_resource_literals,
    evaluate_graph_feasibility,
    sanitize_action_literals,
)
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.knowledge import Scope


def test_detect_services_from_prompt():
    prompt = "Check spreadsheet ss_ops and send SMS via Twilio then create a Jira ticket."
    services = detect_services_from_prompt(prompt)
    assert "sheets" in services
    assert "twilio" in services
    assert "jira" in services
    assert "zoom" not in services


def test_extract_resource_literals():
    prompt = "Check spreadsheet ss_badges (worksheet ws_rules) and project proj_facil."
    lits = extract_resource_literals(prompt)
    assert "ss_badges" in lits
    assert "ws_rules" in lits
    assert "proj_facil" in lits


def test_sanitize_action_literals():
    action = "api_fetch GET https://sheets.googleapis.com/v4/spreadsheets/ss_badges/values/A1:Z100"
    # When ss_badges is in allowed literals, it is preserved
    assert "ss_badges" in sanitize_action_literals(action, {"ss_badges"})

    # When ss_badges is not in allowed literals (e.g. current task wants ss_maint), it is sanitized
    sanitized = sanitize_action_literals(action, {"ss_maint"})
    assert "ss_badges" not in sanitized
    assert "{spreadsheet_id}" in sanitized


def test_evaluate_graph_feasibility_no_store():
    decision = evaluate_graph_feasibility("Check spreadsheet ss_test", None)
    assert decision.action == "without_graph"
    assert decision.reachability == 0.0


def test_evaluate_graph_feasibility_with_store(tmp_path):
    store = KnowledgeStore(tmp_path / "test_store.sqlite")
    # Empty store has 0 claims
    decision = evaluate_graph_feasibility("Check spreadsheet ss_test", store)
    assert decision.action == "without_graph"
    assert decision.reachability == 0.0
    store.close()
