import pytest
from unittest.mock import MagicMock

from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.reference_investigation import (
    advisory_reference_context,
    extract_unresolved_references,
)


def make_mock_adapter(prompt_text, episodes):
    adapter = MagicMock()
    adapter.task = MagicMock()
    adapter.task.prompt = [{"role": "user", "content": prompt_text}]
    adapter.scope = Scope(environment="operations", task_id="test_task", version="1")
    adapter.evidence = MagicMock()
    adapter.evidence.db = None
    adapter.task_state = MagicMock()
    adapter.task_state.observations = {e.episode_id: e for e in episodes}
    return adapter


def test_twilio_incident_references_detected_and_resolved():
    scope = Scope(environment="operations", task_id="test_twilio", version="1")
    queue_ep = Episode(
        episode_id="e1",
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_incident_queue/values/ws_queue"},
        response={
            "values": [
                ["Incident Title", "Severity", "Status", "Time", "Detected", "Notes"],
                ["Database Cluster Failover", "P1", "New", "After-Hours", "2026-01-29T02:15:00Z", "duplicate of INC-4921"],
                ["API Gateway Timeout", "P2", "New", "After-Hours", "2026-01-29T03:30:00Z", "severity under review - see monitoring team email"],
                ["CDN Edge Node Failure", "P1", "New", "After-Hours", "2026-01-29T01:20:00Z", "under investigation by SRE team - do not page"],
            ]
        },
        outcome="success",
    )

    adapter = make_mock_adapter("Handle the after-hours P1 incident.", [queue_ep])
    unresolved = extract_unresolved_references(adapter)

    # All three notes should be detected
    categories = [u["category"] for u in unresolved]
    assert "duplicate_reference" in categories
    assert "external_investigation" in categories
    assert "exclusion_note" in categories

    advice = advisory_reference_context(adapter)
    assert "API Gateway Timeout" in advice
    assert "monitoring team email" in advice
    assert "Database Cluster Failover" in advice
    assert "INC-4921" in advice
    assert "do not page" in advice

    # Now simulate inspecting the monitoring team email
    monitoring_ep = Episode(
        episode_id="e2",
        task_id=scope.task_id,
        scope=scope,
        operation="api_search",
        arguments={"query": "monitoring team email"},
        response={"messages": [{"id": "msg_mon_001", "snippet": "monitoring team escalation to P1"}]},
        outcome="success",
    )
    adapter2 = make_mock_adapter("Handle the after-hours P1 incident.", [queue_ep, monitoring_ep])
    unresolved2 = extract_unresolved_references(adapter2)
    advice2 = advisory_reference_context(adapter2)

    # Monitoring team email reference is now resolved!
    assert "monitoring team email" not in advice2
    assert "Database Cluster Failover" in advice2


def test_badge_policy_and_review_hold_detected():
    scope = Scope(environment="operations", task_id="test_badge", version="1")
    meta_ep = Episode(
        episode_id="e1",
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_badges"},
        response={
            "spreadsheetId": "ss_badges",
            "sheets": [
                {"properties": {"title": "Badge Tracker"}},
                {"properties": {"title": "Badge Notification Policy"}},
            ],
        },
        outcome="success",
    )
    tracker_ep = Episode(
        episode_id="e2",
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_badges/values/ws_contractors"},
        response={
            "values": [
                ["Contractor", "Badge Number", "Expiry", "Phone", "Company Contact", "Status", "Notes"],
                ["Wayne Ellis", "BDG-3001", "2026-02-18", "+15553001001", "ops@contractorsinc.com", "Active", ""],
                ["Marcus Webb", "BDG-3010", "2026-02-14", "+15553010010", "security@traptrust.com", "Active", "under SEC-2026-015 review"],
            ]
        },
        outcome="success",
    )

    adapter = make_mock_adapter("Flag anyone whose badge is expiring.", [meta_ep, tracker_ep])
    advice = advisory_reference_context(adapter)

    assert "Badge Notification Policy" in advice
    assert "Marcus Webb" in advice
    assert "SEC-2026-015" in advice

    # Now simulate reading Badge Notification Policy
    policy_ep = Episode(
        episode_id="e3",
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_badges/values/Badge Notification Policy"},
        response={"values": [["Rule", "Description"], ["Hold", "SEC- reference code must not receive badge renewal"]]},
        outcome="success",
    )
    adapter2 = make_mock_adapter("Flag anyone whose badge is expiring.", [meta_ep, tracker_ep, policy_ep])
    advice2 = advisory_reference_context(adapter2)

    # Worksheet unread warning is resolved
    assert "worksheet 'Badge Notification Policy' with special handling rules (not yet read)" not in advice2


def test_asana_prompt_policy_detected():
    scope = Scope(environment="operations", task_id="test_asana", version="1")
    prompt = (
        "Check the latest unread email from the facilities team about a fire drill. "
        "Before creating the task, check the email processing policy in spreadsheet ss_ops_policy (worksheet ws_email_rules)."
    )
    adapter = make_mock_adapter(prompt, [])
    advice = advisory_reference_context(adapter)

    assert "ss_ops_policy (ws_email_rules)" in advice
    assert "not yet read" in advice

    # Once policy is read, it resolves
    policy_ep = Episode(
        episode_id="e1",
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_ops_policy/values/ws_email_rules"},
        response={"values": [["Rule", "Condition"], ["Fire drill", "Only facilities@"]]},
        outcome="success",
    )
    adapter2 = make_mock_adapter(prompt, [policy_ep])
    advice2 = advisory_reference_context(adapter2)
    assert advice2 == ""


def test_unresolved_references_reroute_workflow_plan_to_investigation(tmp_path):
    from worldmodel_harness.context import _format_workflow_process_markdown
    from worldmodel_harness.knowledge import Claim, Dependency, Layer
    from worldmodel_harness.process_planner import extract_workflow_plan
    from worldmodel_harness.store import KnowledgeStore

    store = KnowledgeStore(tmp_path / "plan.sqlite")
    scope = Scope(environment="test", version="1", role="agent")

    # Add claims for sheets, gmail, and twilio
    c_sheets = Claim(
        claim_id="sheets.spreadsheets.values.get",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("sheets",),
        statement="Get sheet values",
        scope=scope,
        payload={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/{id}/values/{range}", "endpoint_id": "sheets.spreadsheets.values.get"},
    )
    c_gmail_list = Claim(
        claim_id="gmail.users.messages.list",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("gmail",),
        statement="List gmail messages",
        scope=scope,
        payload={"method": "GET", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages", "endpoint_id": "gmail.users.messages.list"},
    )
    c_twilio = Claim(
        claim_id="twilio.messages.create",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("twilio",),
        statement="Create twilio message",
        scope=scope,
        payload={"method": "POST", "url": "https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/Messages.json", "endpoint_id": "twilio.messages.create"},
    )
    store.put_claim(c_sheets)
    store.put_claim(c_gmail_list)
    store.put_claim(c_twilio)

    # Add transitions: sheets -> gmail.list, gmail.list -> twilio
    e1 = Episode(
        episode_id="e1",
        task_id="t1",
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_q/values/A:Z"},
        response={"values": [["Title", "Notes"], ["Task 1", "see email from monitoring"]]},
        outcome="success",
        origin="environment",
    )
    e2 = Episode(
        episode_id="e2",
        task_id="t1",
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages"},
        response={"messages": [{"id": "m1"}]},
        outcome="success",
        origin="environment",
    )
    e3 = Episode(
        episode_id="e3",
        task_id="t1",
        scope=scope,
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://api.twilio.com/2010-04-01/Accounts/AC/Messages.json"},
        response={"sid": "SM1"},
        outcome="success",
        origin="environment",
    )
    store.append_episode(e1)
    store.append_episode(e2)
    store.append_episode(e3)

    unresolved = [
        {
            "category": "external_investigation",
            "target": "Task 1",
            "advice": "Record 'Task 1' notes 'see email from monitoring' — inspect email before committing.",
        }
    ]

    plan = extract_workflow_plan(
        store,
        scope,
        "Handle P1 incident via twilio SMS",
        current_node="sheets.spreadsheets.values.get",
        unresolved_references=unresolved,
    )
    assert plan is not None
    assert "Investigate referenced evidence" in plan.goal
    assert "gmail.users.messages.list" in [s.operation for s in plan.steps]

    md = _format_workflow_process_markdown(plan, current_node="sheets.spreadsheets.values.get")
    assert "Active Evidence References & Disambiguation Constraints" in md
    assert "see email from monitoring" in md
    store.close()
