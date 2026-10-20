import json
from datetime import timedelta
import pytest
from stirrup.core.models import SystemMessage, ToolMessage, UserMessage

from worldmodel_harness.contrastive import (
    ContrastivePair,
    induce_contrastive_precondition,
    is_productive_episode,
    is_unproductive_episode,
    learn_contrastive_preconditions,
    mine_contrastive_pairs,
)
from worldmodel_harness.context import render_context
from worldmodel_harness.decisions import (
    ActionRequirements,
    CompactDecision,
    Decision,
    ProposedAction,
    check_decision,
)
from worldmodel_harness.experiment import _inject_operational_context
from worldmodel_harness.knowledge import Episode, LiveState, Scope, WorkingGraph, utc_now
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore(tmp_path / "contrastive.sqlite")
    yield s
    s.close()


def test_action_shorthand_coercion():
    # Tests that bare tool shorthand aliases are cleanly coerced into ProposedAction
    p1 = CompactDecision(action="a1")
    assert p1.action.tool == "a1"
    assert p1.action.arguments == {}

    p2 = CompactDecision(action="api_search")
    assert p2.action.tool == "api_search"
    assert p2.action.arguments == {}

    p3 = CompactDecision(action='{"tool": "api_fetch", "arguments": {"url": "https://example.com"}}')
    assert p3.action.tool == "api_fetch"
    assert p3.action.arguments == {"url": "https://example.com"}


def test_message_frontier_preservation():
    # Verifies that operational memory is injected into SystemMessage,
    # keeping the latest ToolMessage as the active conversation frontier.
    messages = [
        SystemMessage(content="You are an automation agent."),
        UserMessage(content="Task goal"),
        ToolMessage(content="Previous tool output", tool_call_id="call_1"),
    ]
    content = "Contract info and learned invariants."
    updated = _inject_operational_context(messages, content, "Operational Memory & Learned Invariants")

    # Front message must have the injected content
    assert updated[0].role == "system"
    assert "### Operational Memory & Learned Invariants" in updated[0].content
    assert content in updated[0].content

    # The LAST message must remain the ToolMessage
    assert updated[-1].role == "tool"
    assert updated[-1].content == "Previous tool output"


def test_unproductive_episode_detection(store):
    scope = Scope(environment="test", version="1")

    # 1. Explicit error
    e_err = Episode(
        task_id="t1",
        scope=scope,
        operation="api_fetch",
        arguments={"url": "https://api.test/crm"},
        response={"error": "Unauthorized", "code": 401},
        outcome="error",
    )
    unprod, kind = is_unproductive_episode(e_err)
    assert unprod is True
    assert kind == "explicit_error"

    # 2. Silent failure with 200 OK (updated: 0)
    e_silent = Episode(
        task_id="t2",
        scope=scope,
        operation="api_fetch",
        arguments={"url": "https://api.test/crm"},
        response={"success": True, "updated": 0},
        outcome="success",
    )
    unprod, kind = is_unproductive_episode(e_silent)
    assert unprod is True
    assert kind == "silent_failure"

    # 3. Productive episode
    e_prod = Episode(
        task_id="t3",
        scope=scope,
        operation="api_fetch",
        arguments={"url": "https://api.test/crm"},
        response={"success": True, "updated": 1, "id": "rec_123"},
        outcome="success",
    )
    unprod, _ = is_unproductive_episode(e_prod)
    assert unprod is False
    assert is_productive_episode(e_prod) is True


def test_contrastive_mining_and_precondition_induction(store):
    scope = Scope(environment="test", version="1")
    t0 = utc_now()

    # Task A: Skipped auth, called CRM, got silent failure (updated: 0)
    e1_fail = Episode(
        task_id="task_a",
        scope=scope,
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://api.test/customers/update", "status": "Active"},
        response={"success": True, "updated": 0, "warning": "No subscription"},
        outcome="success",
        timestamp=t0,
    )
    store.append_episode(e1_fail)

    # Task B: Ran auth/subscription first, then called CRM with subscription_id, succeeded
    e2_auth = Episode(
        task_id="task_b",
        scope=scope,
        operation="create_subscription",
        arguments={"customer": "Beta Corp"},
        response={"subscription_id": "sub_999"},
        outcome="success",
        timestamp=t0 + timedelta(seconds=1),
    )
    e2_succ = Episode(
        task_id="task_b",
        scope=scope,
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://api.test/customers/update",
            "status": "Active",
            "subscription_id": "sub_999",
        },
        response={"success": True, "updated": 1},
        outcome="success",
        timestamp=t0 + timedelta(seconds=2),
    )
    store.append_episode(e2_auth)
    store.append_episode(e2_succ)

    # Mine contrastive pairs
    pairs = mine_contrastive_pairs(store, scope)
    assert len(pairs) >= 1
    pair = pairs[0]
    assert pair.target_tool == "api_fetch"
    assert pair.failure_kind == "silent_failure"

    # Induce contrastive precondition claim
    claim = induce_contrastive_precondition(pair, store, scope)
    assert claim.relation == "contrastive_precondition"
    assert claim.payload["required_preceding_action"] == "create_subscription"
    assert claim.payload["required_argument"] == "subscription_id"
    assert "create_subscription" in claim.statement

    # Ingest into store
    created = learn_contrastive_preconditions(store, scope)
    assert len(created) >= 1
    stored_claim = store.claim(created[0])
    assert stored_claim.relation == "contrastive_precondition"


def test_contrastive_precondition_gating_in_check_decision(store):
    scope = Scope(environment="test", version="1", task_id="task_c")
    live = LiveState("task_c")

    ep_neg = Episode(
        task_id="t_neg",
        scope=scope,
        operation="activate_user",
        arguments={"user_id": "u1"},
        response={"success": False, "error": "No subscription"},
        outcome="error",
    )
    ep_pos = Episode(
        task_id="t_pos",
        scope=scope,
        operation="activate_user",
        arguments={"user_id": "u1", "subscription_id": "s1"},
        response={"success": True, "updated": 1},
        outcome="success",
    )
    store.append_episode(ep_neg)
    store.append_episode(ep_pos)

    # Store a contrastive precondition claim
    claim = induce_contrastive_precondition(
        ContrastivePair(
            pair_id="p1",
            target_tool="activate_user",
            target_route=None,
            negative_episode_id=ep_neg.episode_id,
            positive_episode_id=ep_pos.episode_id,
            failure_kind="silent_failure",
            divergence_summary="diverged",
        ),
        store,
        scope,
    )
    # Target requires preceding action "create_subscription" and argument "subscription_id"
    claim = claim.model_copy(
        update={
            "payload": {
                "target_tool": "activate_user",
                "required_preceding_action": "create_subscription",
                "required_argument": "subscription_id",
            }
        }
    )
    store.put_claim(claim)

    # 1. Propose action without subscription_id argument -> gate blocks with missing argument gap
    decision1 = Decision(
        goal="activate user",
        rationale="call activate",
        action=ProposedAction(tool="activate_user", arguments={"user_id": "u1"}),
    )
    res1 = check_decision(
        decision1,
        mandatory=ActionRequirements(),
        store=store,
        scope=scope,
        live=live,
        allowed_tools={"activate_user", "create_subscription"},
    )
    assert not res1.allowed
    assert any("missing_argument:subscription_id" in g for g in res1.gaps)

    # 2. Propose action with subscription_id argument, but create_subscription has not run -> blocks with action gap
    decision2 = Decision(
        goal="activate user",
        rationale="call activate",
        action=ProposedAction(tool="activate_user", arguments={"user_id": "u1", "subscription_id": "s1"}),
    )
    res2 = check_decision(
        decision2,
        mandatory=ActionRequirements(),
        store=store,
        scope=scope,
        live=live,
        allowed_tools={"activate_user", "create_subscription"},
    )
    assert not res2.allowed
    assert any("contrastive_precondition:" in g for g in res2.gaps)

    # 3. Now simulate that create_subscription successfully ran in the task
    ep_sub = Episode(
        task_id="task_c",
        scope=scope,
        operation="create_subscription",
        arguments={"user_id": "u1"},
        response={"status": "created"},
        outcome="success",
    )
    store.append_episode(ep_sub)

    # Gate now allows execution
    res3 = check_decision(
        decision2,
        mandatory=ActionRequirements(),
        store=store,
        scope=scope,
        live=live,
        allowed_tools={"activate_user", "create_subscription"},
    )
    assert res3.allowed


def test_contrastive_precondition_rendered_in_context():
    from worldmodel_harness.knowledge import Claim, Layer, Scope
    scope = Scope(environment="test", version="1")
    claim = Claim(
        claim_id="inv1",
        layer=Layer.BEHAVIOR,
        relation="contrastive_precondition",
        participants=("crm.update_status", "billing.create_subscription"),
        statement="Setting status=Active silently fails unless subscription is created first.",
        scope=scope,
        payload={"target_tool": "crm.update_status"},
        status="supported",
    )
    working = WorkingGraph(scope=scope, claims=(claim,), gaps=())
    rendered, sel = render_context(working, [], max_chars=5000)

    # Verify that the rendered JSON contains learned_invariants
    data = json.loads(rendered.split("\n", 1)[1])
    assert "learned_invariants" in data
    assert claim.statement in data["learned_invariants"]


def test_contrastive_precondition_route_specific_matching(store):
    from worldmodel_harness.decisions import ActionRequirements, Decision, ProposedAction, check_decision
    from worldmodel_harness.knowledge import Claim, Layer, LiveState, Scope

    scope = Scope(environment="test", version="1", task_id="task_route")
    live = LiveState("task_route")

    from worldmodel_harness.knowledge import Episode
    ep = Episode(
        task_id="task_route",
        scope=scope,
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"},
        response={"id": "m1"},
        outcome="success",
    )
    store.append_episode(ep)

    # Add a precondition targeting gmail.users.messages.send requiring base64_encode
    claim = Claim(
        claim_id="gmail_send_precond",
        layer=Layer.BEHAVIOR,
        relation="contrastive_precondition",
        participants=("gmail.users.messages.send", "base64_encode"),
        statement="Calling POST gmail send requires base64_encode",
        scope=scope,
        supporting_episodes=(ep.episode_id,),
        payload={
            "target_operation": "gmail.users.messages.send",
            "required_preceding_action": "base64_encode",
        },
        status="supported",
    )
    store.put_claim(claim)

    # 1. GET request to Drive files -> MUST NOT BE BLOCKED
    dec_drive = Decision(
        goal="list drive files",
        rationale="fetch files",
        action=ProposedAction(
            tool="api_fetch",
            arguments={"method": "GET", "url": "https://www.googleapis.com/drive/v3/files"},
        ),
    )
    res_drive = check_decision(
        dec_drive,
        mandatory=ActionRequirements(operation_id="google_drive.files.list"),
        store=store,
        scope=scope,
        live=live,
        allowed_tools={"api_fetch"},
    )
    assert res_drive.allowed

    # 2. GET request to Gmail messages -> MUST NOT BE BLOCKED
    dec_gmail_get = Decision(
        goal="list gmail messages",
        rationale="fetch messages",
        action=ProposedAction(
            tool="api_fetch",
            arguments={"method": "GET", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages"},
        ),
    )
    res_gmail_get = check_decision(
        dec_gmail_get,
        mandatory=ActionRequirements(operation_id="gmail.users.messages.list"),
        store=store,
        scope=scope,
        live=live,
        allowed_tools={"api_fetch"},
    )
    assert res_gmail_get.allowed

    # 3. POST request to Gmail send -> MUST BE BLOCKED
    dec_gmail_send = Decision(
        goal="send email",
        rationale="post email",
        action=ProposedAction(
            tool="api_fetch",
            arguments={"method": "POST", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"},
        ),
    )
    res_gmail_send = check_decision(
        dec_gmail_send,
        mandatory=ActionRequirements(operation_id="gmail.users.messages.send"),
        store=store,
        scope=scope,
        live=live,
        allowed_tools={"api_fetch", "base64_encode"},
    )
    assert not res_gmail_send.allowed
    assert any("Calling POST gmail send requires base64_encode" in g for g in res_gmail_send.gaps)

