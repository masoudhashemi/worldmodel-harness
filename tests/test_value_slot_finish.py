"""Regressions from run 848f0ae6: a checked finish must verify the changed slot."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from worldmodel_harness.behavior import requested_change_formats, value_format
from worldmodel_harness.goal_checks import CheckedFinish, GoalTracker
from worldmodel_harness.knowledge import Episode, LiveState, Scope
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.verification_tools import CheckEffect, simple_check_tool

PHONE_TASK = (
    "Jordan Lee just emailed us with a new phone number. "
    "Can you find that email and update her phone number in Salesforce?"
)
EMAIL_TASK = (
    "Maria Santos sent us an email saying she has a new email address. "
    "Find her email and update her contact in Salesforce with the new address."
)


@pytest.fixture
def make_adapter(tmp_path):
    stores = []

    def build(task):
        store = KnowledgeStore(tmp_path / f"evidence{len(stores)}.sqlite")
        stores.append(store)
        scope = Scope(environment="test", version="1", task_id="task")
        adapter = SimpleNamespace(
            evidence=store,
            scope=scope,
            live=LiveState(scope.task_id),
            evidence_aliases={},
            task=SimpleNamespace(prompt=[{"content": task}]),
        )
        adapter.goal_tracker = GoalTracker(adapter)
        return adapter

    yield build
    for store in stores:
        store.close()


def read(adapter, url, response):
    event = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={"method": "GET", "url": url},
        response=response,
        coverage=tuple(response),
        outcome="success",
    )
    adapter.evidence.append_episode(event)
    adapter.live.observe(event)
    alias = f"e{len(adapter.evidence_aliases) + 1}"
    adapter.evidence_aliases[alias] = event.episode_id
    return alias


def check(adapter, requirement_id, alias, path, expected, task):
    return asyncio.run(
        simple_check_tool(adapter).executor(
            CheckEffect(
                requirement_id=requirement_id,
                task_quote=task,
                episode_id=alias,
                path=path,
                expected=expected,
                anchor_text=task,
            )
        )
    )


def test_task_names_the_changed_slot_not_the_message_medium():
    assert requested_change_formats(PHONE_TASK) == {"PHONE"}
    assert requested_change_formats(EMAIL_TASK) == {"EMAIL"}
    assert requested_change_formats("Update Avery Chen's contact with the value.") == set()


def test_matched_name_check_alone_cannot_finish_a_phone_update(make_adapter):
    adapter = make_adapter(PHONE_TASK)
    read(adapter, "https://mail.test/source", {"Name": "Jordan Lee", "body": "Use +1-415-555-0101"})
    contact = read(
        adapter,
        "https://crm.test/Contact/003001",
        {"Name": "Jordan Lee", "FirstName": "Jordan", "LastName": "Lee", "Phone": "+1-415-555-0101"},
    )
    assert check(adapter, "jordan_lee_name", contact, ("Name",), "Jordan Lee", PHONE_TASK).success
    tracker = adapter.goal_tracker
    refused = tracker.finish(CheckedFinish(summary="done"))
    assert not refused.success
    body = json.loads(refused.content)
    assert body["required_value_formats"] == ["PHONE"]
    assert body["missing_value_formats"] == ["PHONE"]
    assert body["matched_value_formats"] == ["NAME"]
    assert getattr(tracker, "completion_status", None) is None
    assert tracker.finish_audit[-1]["success"] is False

    assert check(adapter, "jordan_lee_phone", contact, ("Phone",), "+1-415-555-0101", PHONE_TASK).success
    assert tracker.finish(CheckedFinish(summary="done")).success
    assert tracker.completion_status == "checked"
    assert tracker.finish_audit[-1]["missing_value_formats"] == []


def test_honest_unresolved_finish_is_still_available(make_adapter):
    adapter = make_adapter(PHONE_TASK)
    contact = read(adapter, "https://crm.test/Contact/003001", {"Name": "Jordan Lee", "Phone": "x"})
    assert check(adapter, "name", contact, ("Name",), "Jordan Lee", PHONE_TASK).success
    result = adapter.goal_tracker.finish(CheckedFinish(summary="phone unverified", unresolved=True))
    assert result.success
    assert adapter.goal_tracker.completion_status == "unresolved"
    assert json.loads(result.content)["missing_value_formats"] == ["PHONE"]


def test_list_index_is_not_a_field_name():
    assert value_format("INBOX", 1) is None
    assert value_format("+1-415-555-0101", 3) == "PHONE"


def test_email_check_walks_lists_in_other_reads(make_adapter):
    adapter = make_adapter(EMAIL_TASK)
    new = "maria.santos@brightwave-global.example.com"
    source = read(
        adapter,
        "https://mail.test/messages/msg_3002",
        {
            "labelIds": ["INBOX", "UNREAD", "IMPORTANT"],
            "from": "Maria Santos",
            "body": f"Going forward please reach me at {new}.",
        },
    )
    contact = read(
        adapter,
        "https://crm.test/Contact/003002",
        {"Name": "Maria Santos", "FirstName": "Maria", "LastName": "Santos", "Email": new},
    )
    result = check(adapter, "maria_santos_email_update", contact, ("Email",), new, EMAIL_TASK)
    assert result.success, result.content
    outcome = adapter.goal_tracker.checks["maria_santos_email_update"].outcome
    assert outcome.source_episode_id == adapter.evidence_aliases[source]
    assert outcome.source_path == ("body",)
    assert adapter.goal_tracker.finish(CheckedFinish(summary="done")).success
    assert adapter.goal_tracker.completion_status == "checked"


def test_prose_source_body_grounds_exact_value_with_identity(make_adapter):
    adapter = make_adapter(PHONE_TASK)
    source = read(
        adapter,
        "https://mail.test/messages/msg_3001",
        {
            "id": "msg_3001",
            "from": "jordan.lee@horizon-analytics.example.com",
            "body_plain": (
                "Hi team, Quick note that my work mobile has changed. "
                "Please update my Salesforce contact record to +1-555-0101. "
                "Thanks, Jordan Lee"
            ),
        },
    )
    # Earlier read of the target record does not cause ambiguous_source or overshadow the message source.
    read(
        adapter,
        "https://crm.test/Contact/003001",
        {"Name": "Jordan Lee", "Phone": "+1-555-0100"},
    )
    contact = read(
        adapter,
        "https://crm.test/Contact/003001",
        {"Name": "Jordan Lee", "Phone": "+1-555-0101"},
    )
    result = check(adapter, "phone_update", contact, ("Phone",), "+1-555-0101", PHONE_TASK)
    assert result.success, result.content
    outcome = adapter.goal_tracker.checks["phone_update"].outcome
    assert outcome.source_episode_id == adapter.evidence_aliases[source]
    assert outcome.source_path == ("body_plain",)
    assert outcome.target.expected == "+1-555-0101"


def test_write_episode_gates_finish_for_non_phone_non_email_fields(make_adapter):
    task = "Check the message from Jordan Lee and update her opportunity in Salesforce."
    adapter = make_adapter(task)
    read(
        adapter,
        "https://mail.test/messages/msg_4001",
        {
            "from": "Jordan Lee",
            "body_plain": "Please move my opportunity stage to Closed Won. Thanks, Jordan Lee",
        },
    )
    write = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={
            "method": "PATCH",
            "url": "https://crm.test/Opportunity/006001",
            "body": json.dumps({"StageName": "Closed Won"}),
        },
        response={"success": True},
        coverage=("success",),
        outcome="success",
    )
    adapter.evidence.append_episode(write)
    adapter.live.observe(write)
    opp = read(
        adapter,
        "https://crm.test/Opportunity/006001",
        {"Name": "Jordan Lee", "StageName": "Closed Won"},
    )
    assert check(adapter, "opp_name", opp, ("Name",), "Jordan Lee", task).success
    refused = adapter.goal_tracker.finish(CheckedFinish(summary="done"))
    assert not refused.success
    body = json.loads(refused.content)
    assert body["unverified_written_fields"] == ["StageName"]
    assert check(adapter, "opp_stage", opp, ("StageName",), "Closed Won", task).success
    assert adapter.goal_tracker.finish(CheckedFinish(summary="done")).success


def test_non_person_record_uses_identifier_identity_without_name_field(make_adapter):
    task = "Update the website on record 001xx000003DGbYAAW from the onboarding message."
    adapter = make_adapter(task)
    source = read(
        adapter,
        "https://mail.test/messages/msg_5001",
        {
            "id": "001xx000003DGbYAAW",
            "body_plain": "Record 001xx000003DGbYAAW now uses https://acme.example.com/portal",
        },
    )
    record = read(
        adapter,
        "https://crm.test/Account/001xx000003DGbYAAW",
        {"Id": "001xx000003DGbYAAW", "Website": "https://acme.example.com/portal"},
    )
    result = check(
        adapter,
        "website_update",
        record,
        ("Website",),
        "https://acme.example.com/portal",
        task,
    )
    assert result.success, result.content
    outcome = adapter.goal_tracker.checks["website_update"].outcome
    assert outcome.identity_path == ("Id",)
    assert outcome.identity_anchor == "001xx000003DGbYAAW"
    assert outcome.target.expected == "https://acme.example.com/portal"
    assert outcome.source_episode_id == adapter.evidence_aliases[source]
    assert adapter.goal_tracker.finish(CheckedFinish(summary="done")).success


def test_single_record_query_wrapper_and_requirement_reuse(make_adapter):
    adapter = make_adapter(PHONE_TASK)
    read(
        adapter,
        "https://mail.test/messages/msg_3001",
        {
            "from": "Jordan Lee",
            "body_plain": "My new number is +1-555-0101. Thanks, Jordan Lee",
        },
    )
    queried = read(
        adapter,
        "https://crm.test/query?q=SELECT+Id,Name,Phone+FROM+Contact",
        {
            "totalSize": 1,
            "done": True,
            "records": [{"Id": "003001", "Name": "Jordan Lee", "Phone": "+1-555-0101"}],
        },
    )
    # First check on Name with requirement_id="req_1", then reuse "req_1" to check Phone without records[0] prefix.
    assert check(adapter, "req_1", queried, ("Name",), "Jordan Lee", PHONE_TASK).success
    repaired = check(adapter, "req_1", queried, ("Phone",), "+1-555-0101", PHONE_TASK)
    assert not repaired.success
    assert adapter.goal_tracker.checks["req_1"].outcome.target.expected == "Jordan Lee"
    assert check(adapter, "phone", queried, ("Phone",), "+1-555-0101", PHONE_TASK).success
    outcome = adapter.goal_tracker.checks["phone"].outcome
    assert outcome.target.path == ("records", 0, "Phone")
    assert outcome.identity_path == ("records", 0, "Name")
    assert adapter.goal_tracker.finish(CheckedFinish(summary="done")).success


def test_integer_and_numeric_string_effect_verification(make_adapter):
    task = "Update the number_of_employees field on Salesforce lead 00Q008 (Fatima Al-Rashid) to 250."
    adapter = make_adapter(task)
    write = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={
            "method": "PATCH",
            "url": "https://crm.test/sobjects/Lead/00Q008",
            "body": json.dumps({"NumberOfEmployees": 250}),
        },
        response={},
        coverage=(),
        outcome="success",
    )
    adapter.evidence.append_episode(write)
    adapter.live.observe(write)
    lead = read(
        adapter,
        "https://crm.test/sobjects/Lead/00Q008",
        {"Id": "00Q008", "Name": "Fatima Al-Rashid", "NumberOfEmployees": 250},
    )
    # Both string "250" and integer 250 should ground against the task quote and match integer 250 on the record.
    result = check(adapter, "lead_emp", lead, ("NumberOfEmployees",), "250", task)
    assert result.success, result.content
    assert adapter.goal_tracker.checks["lead_emp"].outcome.target.expected == 250
    assert adapter.goal_tracker.finish(CheckedFinish(summary="done")).success
    assert adapter.goal_tracker.completion_status == "checked"


def test_created_subscriber_email_slot_acts_as_self_identity_when_needed(make_adapter):
    task = "Add alex.rivera@pixelworks.example.com to our Newsletter mailing list in Mailchimp. The Newsletter list ID is list_001."
    adapter = make_adapter(task)
    write = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://mailchimp.test/3.0/lists/list_001/members",
            "body": json.dumps(
                {
                    "email_address": "alex.rivera@pixelworks.example.com",
                    "status": "subscribed",
                }
            ),
        },
        response={"id": "62128054e1e4f737abed2687d48a229b"},
        coverage=("id",),
        outcome="success",
    )
    adapter.evidence.append_episode(write)
    adapter.live.observe(write)
    member = read(
        adapter,
        "https://mailchimp.test/3.0/lists/list_001/members/alex.rivera@pixelworks.example.com",
        {
            "id": "62128054e1e4f737abed2687d48a229b",
            "email": "alex.rivera@pixelworks.example.com",
            "list_id": "list_001",
            "status": "subscribed",
        },
    )
    result = asyncio.run(
        simple_check_tool(adapter).executor(
            CheckEffect(
                requirement_id="sub_email",
                task_quote=task,
                episode_id=member,
                path=("email",),
                expected="alex.rivera@pixelworks.example.com",
                anchor_text="alex.rivera@pixelworks.example.com",
            )
        )
    )
    assert result.success, result.content
    assert adapter.goal_tracker.finish(CheckedFinish(summary="done")).success
    assert adapter.goal_tracker.completion_status == "checked"


def test_post_creation_response_cannot_replace_current_read(make_adapter):
    task = "Look for an email about dark mode and create an Asana task in workspace ws_prod."
    adapter = make_adapter(task)
    read(
        adapter,
        "https://mail.test/messages/msg_2002",
        {
            "id": "msg_2002",
            "subject": "Feature request: Add dark mode feature",
            "body_plain": "Could we create an Asana task titled 'Add dark mode feature' in ws_prod?",
        },
    )
    created = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://asana.test/api/1.0/tasks",
            "body": json.dumps({"name": "Add dark mode feature", "workspace": "ws_prod"}),
        },
        response={
            "data": {
                "gid": "12000999",
                "name": "Add dark mode feature",
                "workspace": "ws_prod",
            }
        },
        coverage=("data",),
        outcome="success",
    )
    adapter.evidence.append_episode(created)
    adapter.live.observe(created)
    ep_alias = "e2"
    adapter.evidence_aliases[ep_alias] = created.episode_id

    res_name = check(
        adapter, "task_name", ep_alias, ("data", "name"), "Add dark mode feature", task
    )
    assert not res_name.success
    assert json.loads(res_name.content)["result"] == "outcome_verification_requires_read"
    res_ws = check(adapter, "task_ws", ep_alias, ("data", "workspace"), "ws_prod", task)
    assert not res_ws.success
    assert not adapter.goal_tracker.finish(CheckedFinish(summary="done")).success


def test_shared_numbers_do_not_ground_arbitrary_summary(make_adapter):
    task = (
        "Post a formatted weekly metrics summary to #weekly-metrics (C02metrics). "
        "Data: signups 1,420 (+12% WoW), active users 18,350 (+4% WoW), revenue $48,200 (+8% WoW)."
    )
    adapter = make_adapter(task)
    adapter.task.prompt = [
        {"role": "system", "content": "You have a budget of ~50 tool-using turns."},
        {"role": "user", "content": task},
    ]
    post = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://slack.test/api/chat.postMessage",
            "body": json.dumps(
                {
                    "channel": "C02metrics",
                    "text": "*Weekly Metrics*\n- Signups: 1,420 (+12% WoW)\n- Active Users: 18,350 (+4% WoW)\n- Revenue: $48,200 (+8% WoW)",
                }
            ),
        },
        response={"ok": True},
        coverage=("ok",),
        outcome="success",
    )
    adapter.evidence.append_episode(post)
    adapter.live.observe(post)
    history = read(
        adapter,
        "https://slack.test/api/conversations.history?channel=C02metrics",
        {
            "ok": True,
            "messages": [
                {
                    "type": "message",
                    "text": "*Weekly Metrics*\n- Signups: 1,420 (+12% WoW)\n- Active Users: 18,350 (+4% WoW)\n- Revenue: $48,200 (+8% WoW)",
                }
            ],
        },
    )
    res = asyncio.run(
        simple_check_tool(adapter).executor(
            CheckEffect(
                requirement_id="metrics_post",
                task_quote="Post a formatted weekly metrics summary to #weekly-metrics (C02metrics).",
                episode_id=history,
                path=("messages", 0, "text"),
                expected="*Weekly Metrics*\n- Signups: 1,420 (+12% WoW)\n- Active Users: 18,350 (+4% WoW)\n- Revenue: $48,200 (+8% WoW)",
                anchor_text="1,420 (+12% WoW)",
            )
        )
    )
    assert not res.success
    assert not adapter.goal_tracker.finish(CheckedFinish(summary="done")).success





