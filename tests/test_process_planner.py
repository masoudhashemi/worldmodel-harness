"""Unit tests for graph-theoretic process planning and context integration."""

import json
import pytest

from worldmodel_harness.context import render_context
from worldmodel_harness.knowledge import Claim, Episode, Layer, Scope, WorkingGraph
from worldmodel_harness.process_planner import extract_workflow_plan
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def plan_store(tmp_path):
    store = KnowledgeStore(tmp_path / "plan_test.sqlite")
    scope = Scope(environment="test_env", version="1", role="test_role")

    # Add contracts
    contracts = [
        Claim(
            claim_id="c_drive",
            version=1,
            layer=Layer.INTERFACE,
            relation="endpoint_contract",
            participants=("google_drive.files.list",),
            statement="Query Google Drive files",
            scope=scope,
            payload={"id": "google_drive.files.list", "method": "GET", "url": "https://www.googleapis.com/drive/v3/files"},
            status="supported",
        ),
        Claim(
            claim_id="c_sheet_get",
            version=1,
            layer=Layer.INTERFACE,
            relation="endpoint_contract",
            participants=("sheets.spreadsheets.values.get",),
            statement="Read rows from Google Sheet",
            scope=scope,
            payload={
                "id": "sheets.spreadsheets.values.get",
                "method": "GET",
                "url": "https://sheets.googleapis.com/v4/spreadsheets/{spreadsheetId}/values/{range}",
            },
            status="supported",
        ),
        Claim(
            claim_id="c_mailchimp",
            version=1,
            layer=Layer.INTERFACE,
            relation="endpoint_contract",
            participants=("mailchimp.lists.members.delete",),
            statement="Archive or remove member from Mailchimp list",
            scope=scope,
            payload={
                "id": "mailchimp.lists.members.delete",
                "method": "DELETE",
                "url": "https://us1.api.mailchimp.com/3.0/lists/{list_id}/members/{subscriber_hash}",
            },
            status="supported",
        ),
        Claim(
            claim_id="c_gmail_send",
            version=1,
            layer=Layer.INTERFACE,
            relation="endpoint_contract",
            participants=("gmail.users.messages.send",),
            statement="Send email message via Gmail",
            scope=scope,
            payload={
                "id": "gmail.users.messages.send",
                "method": "POST",
                "url": "https://gmail.googleapis.com/gmail/v1/users/{userId}/messages/send",
            },
            status="supported",
        ),
    ]

    for c in contracts:
        store.db.execute("INSERT INTO claims VALUES (?, ?, ?)", (c.claim_id, c.version, c.model_dump_json()))

    # Add value flow claim
    vf = Claim(
        claim_id="vf_1",
        version=1,
        layer=Layer.BEHAVIOR,
        relation="observed_value_flow",
        participants=("google_drive.files.list", "sheets.spreadsheets.values.get"),
        statement="Reuse file.id as spreadsheetId",
        scope=scope,
        payload={
            "source_endpoint": "google_drive.files.list",
            "target_endpoint": "sheets.spreadsheets.values.get",
            "output_path": "files.*.id",
            "input_path": "path.spreadsheetId",
        },
        status="supported",
    )
    vf2 = Claim(
        claim_id="vf_2",
        version=1,
        layer=Layer.BEHAVIOR,
        relation="observed_value_flow",
        participants=("sheets.spreadsheets.values.get", "mailchimp.lists.members.delete"),
        statement="Reuse sheet values as subscriber_hash",
        scope=scope,
        payload={
            "source_endpoint": "sheets.spreadsheets.values.get",
            "target_endpoint": "mailchimp.lists.members.delete",
            "output_path": "values.*.*",
            "input_path": "path.subscriber_hash",
        },
        status="supported",
    )
    for c in (vf, vf2):
        store.db.execute("INSERT INTO claims VALUES (?, ?, ?)", (c.claim_id, c.version, c.model_dump_json()))

    # Add execution episodes linking drive -> sheets -> mailchimp -> base64 -> gmail
    episodes = [
        Episode(
            task_id="task_alpha",
            scope=scope,
            operation="api_fetch",
            arguments={"method": "GET", "url": "https://www.googleapis.com/drive/v3/files"},
            response={"files": [{"id": "ss_bounces"}]},
            outcome="success",
        ),
        Episode(
            task_id="task_alpha",
            scope=scope,
            operation="api_fetch",
            arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_bounces/values/A:B"},
            response={"values": [["bounced@example.com"]]},
            outcome="success",
        ),
        Episode(
            task_id="task_alpha",
            scope=scope,
            operation="api_fetch",
            arguments={"method": "DELETE", "url": "https://us1.api.mailchimp.com/3.0/lists/list_main/members/hash123"},
            response={"status": "deleted"},
            outcome="success",
        ),
        Episode(
            task_id="task_alpha",
            scope=scope,
            operation="base64_encode",
            arguments={"text": "Suppressed 1 member"},
            response="U3VwcHJlc3NlZCAxIG1lbWJlcg==",
            outcome="success",
        ),
        Episode(
            task_id="task_alpha",
            scope=scope,
            operation="api_fetch",
            arguments={"method": "POST", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"},
            response={"id": "msg_001"},
            outcome="success",
        ),
    ]
    for ep in episodes:
        store.append_episode(ep)

    return store, scope


def test_extract_workflow_plan_basic(plan_store):
    store, scope = plan_store
    query = (
        "We got a bounce report in 'Email Bounces' (ss_bounce). "
        "Clean up our Mailchimp list accordingly and notify ops@company.com via email."
    )
    plan = extract_workflow_plan(store, scope, query)
    assert plan is not None
    assert len(plan.steps) >= 4

    ops = [s.operation for s in plan.steps]
    # Verify sequential operational ordering
    assert ops[0] == "google_drive.files.list" or ops[0] == "sheets.spreadsheets.values.get"
    assert "mailchimp.lists.members.delete" in ops
    assert "gmail.users.messages.send" in ops
    assert ops[-1] == "finish"

    # Verify input flow was attached
    flows = [s.input_flow for s in plan.steps if s.input_flow]
    assert len(flows) >= 1
    assert "path.subscriber_hash" in flows[0]


def test_workflow_plan_paraphrase_invariance(plan_store):
    store, scope = plan_store
    q1 = "We got a bounce report in 'Email Bounces' (ss_bounce). Clean up our Mailchimp list accordingly and notify ops@company.com via email."
    q2 = "Check the spreadsheet for email bounces, remove those subscribers from Mailchimp, and send an email update to ops."

    plan1 = extract_workflow_plan(store, scope, q1)
    plan2 = extract_workflow_plan(store, scope, q2)

    assert plan1 is not None
    assert plan2 is not None

    ops1 = [s.operation for s in plan1.steps]
    ops2 = [s.operation for s in plan2.steps]
    assert ops1 == ops2


def test_render_context_injects_workflow_process(plan_store):
    store, scope = plan_store
    working = WorkingGraph(scope=scope, claims=tuple(store.claims(scope)))
    query = "Check the spreadsheet for bounces, remove subscribers from Mailchimp, and email ops."

    content, report = render_context(working, [], max_chars=12000, store=store, query=query)
    assert content != ""
    assert report["workflow_plan_steps"] >= 4

    # Verify JSON structure has workflow_process
    payload_json = content.split("\n", 1)[1]
    parsed = json.loads(payload_json)
    assert "workflow_process" in parsed
    assert "steps" in parsed["workflow_process"]
    assert len(parsed["workflow_process"]["steps"]) >= 4

    # Verify terminal step is finish
    assert parsed["workflow_process"]["steps"][-1]["operation"] == "finish"


def test_state_local_path_recalibration_with_current_node(plan_store):
    store, scope = plan_store
    query = "Check the spreadsheet for bounces, remove subscribers from Mailchimp, and email ops."

    # Initial state (t=0)
    plan_t0 = extract_workflow_plan(store, scope, query)
    assert plan_t0 is not None
    assert plan_t0.active_step_index == 1
    assert plan_t0.steps[0].status == "active"
    assert plan_t0.steps[1].status == "pending"

    # State transition: step 1 completed
    step1_op = plan_t0.steps[0].operation
    plan_t1 = extract_workflow_plan(store, scope, query, current_node=step1_op)
    assert plan_t1 is not None
    assert plan_t1.steps[0].status == "completed"
    assert plan_t1.steps[1].status == "active"
    assert plan_t1.active_step_index == 2

    # State transition: step 2 completed
    step2_op = plan_t1.steps[1].operation
    plan_t2 = extract_workflow_plan(
        store, scope, query, current_node=step2_op, completed_nodes={step1_op}
    )
    assert plan_t2 is not None
    assert plan_t2.steps[0].status == "completed"
    assert plan_t2.steps[1].status == "active"
    assert plan_t2.active_step_index == 2


def test_workflow_plan_includes_tried_actions_and_potential_solution_paths(plan_store):
    store, scope = plan_store
    query = "Check the spreadsheet for bounces, remove subscribers from Mailchimp, and email ops."

    plan = extract_workflow_plan(store, scope, query)
    assert plan is not None

    # Check candidate solution paths
    assert len(plan.potential_solution_paths) >= 1
    rec_path = plan.potential_solution_paths[0]
    assert rec_path.is_recommended
    assert "finish" in rec_path.steps
    assert rec_path.confidence > 0.0

    # Verify rendering context includes tried_from_this_state & potential_solution_paths
    working = WorkingGraph(scope=scope, claims=tuple(store.claims(scope)))
    content, _ = render_context(working, [], max_chars=12000, store=store, query=query)
    payload_json = content.split("\n", 1)[1]
    parsed = json.loads(payload_json)
    wf = parsed.get("workflow_process", {})

    assert "potential_solution_paths" in wf
    assert len(wf["potential_solution_paths"]) >= 1
    assert "exploration_guidance" in wf


def test_state_transition_memory_and_downstream_branching(plan_store):
    store, scope = plan_store
    query = "Check the spreadsheet for bounces, remove subscribers from Mailchimp, and email ops."

    # Add an error episode from sheets.spreadsheets.values.get to test downstream branching
    err_ep = Episode(
        task_id="task_beta",
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_bounces/values/invalid!range"},
        response={"error": {"message": "Unable to parse range: invalid!range"}},
        outcome="error",
    )
    store.append_episode(err_ep)

    plan = extract_workflow_plan(store, scope, query, current_node="google_drive.files.list")
    assert plan is not None

    from worldmodel_harness.context import _format_workflow_process_markdown
    md = _format_workflow_process_markdown(plan, current_node="google_drive.files.list")
    assert "### 🗺️ Empirical World Model — State Graph & Historical Transitions" in md
    assert "google_drive.files.list" in md
    assert "Recommended Transition Path" in md



