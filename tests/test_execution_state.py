from worldmodel_harness.execution_state import advance_frontier, environment_actions
from worldmodel_harness.knowledge import Claim, Episode, Scope
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.trajectory_graph import compile_trajectory_graph


def test_linked_actions_and_graph_ignore_wrapper_success(tmp_path):
    store = KnowledgeStore(tmp_path / "state.sqlite")
    scope = Scope(environment="test", version="1", role="agent", task_id="task")
    for name in ("read", "send"):
        claim = Claim(
            claim_id=name,
            version=1,
            layer="interface",
            relation="endpoint_contract",
            participants=(name,),
            statement=name,
            scope=scope,
            status="supported",
            payload={
                "id": name,
                "method": "GET" if name == "read" else "POST",
                "url": f"https://example.com/{name}",
            },
        )
        store.db.execute("INSERT INTO claims VALUES (?, ?, ?)", (name, 1, claim.model_dump_json()))

    def action(name, outcome="success"):
        return Episode(
            task_id="task",
            scope=scope,
            interaction_id="call",
            operation="api_fetch",
            outcome=outcome,
            arguments={
                "method": "GET" if name == "read" else "POST",
                "url": f"https://example.com/{name}",
            },
            response={},
        )

    read, failed, sent = action("read"), action("send", "error"), action("send")

    def wrapper(*episodes):
        return Episode(
            task_id="task",
            scope=scope,
            interaction_id="call",
            origin="harness",
            operation="tool_call:execute_decision",
            arguments={},
            response={},
            outcome="success",
            related_episode_ids=tuple(e.episode_id for e in episodes),
        )

    for episode in (read, wrapper(read), failed, wrapper(failed), sent, wrapper(sent)):
        store.append_episode(episode)
    completed = set()
    current = advance_frontier(store, scope, store, wrapper(read), None, completed)
    assert current == "read"
    assert advance_frontier(store, scope, store, wrapper(failed), current, completed) == "read"
    assert completed == {"read"}
    assert advance_frontier(store, scope, store, wrapper(), current, completed) == "read"
    assert advance_frontier(store, scope, store, wrapper(read, sent), current, completed) == "send"
    assert completed == {"read", "send"}
    assert not compile_trajectory_graph(store, scope).edges
    graph = compile_trajectory_graph(store, scope, successful_transitions_only=True)
    assert graph.edges["read"]["send"].count == 1
    assert all(not n.startswith("tool_call:") for n in graph.edges)
    missing = wrapper(read).model_copy(update={"related_episode_ids": ("missing",)})
    assert environment_actions(missing, store) == []
    wrong_task = wrapper(read).model_copy(update={"task_id": "other"})
    assert environment_actions(wrong_task, store) == []
    store.close()


def test_experimental_defaults_are_disabled():
    from worldmodel_harness.experiment import ExperimentSettings

    settings = ExperimentSettings()
    assert not settings.resolve_execution_frontier
    assert not settings.successful_transition_graph


def test_frontier_and_graph_switches_are_independent(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from stirrup.core.models import AssistantMessage, TextBlock

    from worldmodel_harness import experiment
    from worldmodel_harness.knowledge import LiveState

    store = KnowledgeStore(tmp_path / "switches.sqlite")
    scope = Scope(environment="test", version="1", role="agent", task_id="task")
    action = Episode(
        task_id="task",
        scope=scope,
        interaction_id="call",
        operation="base64_encode",
        arguments={},
        response={},
        outcome="success",
    )
    store.append_episode(action)
    wrapper = Episode(
        task_id="task",
        scope=scope,
        interaction_id="call",
        origin="harness",
        operation="tool_call:execute_decision",
        arguments={},
        response={},
        outcome="success",
        related_episode_ids=(action.episode_id,),
    )
    captured = []
    monkeypatch.setattr(
        experiment, "retrieve", lambda *a, **kw: SimpleNamespace(claims=[], gaps=[])
    )

    def render(*args, **kwargs):
        captured.append(kwargs)
        return "", {}

    monkeypatch.setattr(experiment, "render_context", render)

    class ActingClient:
        async def generate(self, messages, tools):
            return AssistantMessage(blocks=[TextBlock(text="done")])

    for frontier in (False, True):
        for graph in (False, True):
            adapter = SimpleNamespace(
                task=SimpleNamespace(prompt=[{"role": "user", "content": "Process data"}]),
                scope=scope,
                evidence=store,
                last_interaction=wrapper,
                live=LiveState("task"),
                ledger=SimpleNamespace(compute_seconds=0),
                prediction_assessments=[],
            )
            client = experiment.KnowledgeClient(
                ActingClient(),
                adapter,
                store,
                experiment.ExperimentSettings(
                    resolve_execution_frontier=frontier, successful_transition_graph=graph
                ),
            )
            asyncio.run(client.generate([], {}))
            assert captured[-1]["successful_transition_graph"] is graph
            expected = "base64_encode" if frontier else "tool_call:execute_decision"
            assert captured[-1]["current_node"] == expected
            assert captured[-1]["completed_nodes"] == {expected}
    store.close()


def test_entity_tracker_obligation_lifecycle_and_retrieval(tmp_path):
    from types import SimpleNamespace
    from worldmodel_harness.execution_state import EntityTracker, advance_frontier
    from worldmodel_harness.retrieval import decision_focus
    from worldmodel_harness.context import render_context
    from worldmodel_harness.knowledge import WorkingGraph

    tracker = EntityTracker(required_channels=("sms", "email", "slack"))
    scope = Scope(environment="test", version="1", role="agent", task_id="task")

    # 1. Simulate spreadsheet read episode with active and terminated contractors
    read_resp = {
        "valueRanges": [
            {
                "values": [
                    ["Contractor", "Badge Number", "Expiry", "Phone", "Company Contact", "Status"],
                    ["Wayne Ellis", "BDG-3001", "2026-02-18", "+15553001001", "ops@contractorsinc.com", "Active"],
                    ["Fiona Hart", "BDG-3002", "2026-02-12", "+15553002002", "admin@techstaff.com", "Terminated"],
                    ["Luis Moreno", "BDG-3003", "2026-02-22", "+15553003003", "hr@buildright.com", "Active"],
                ]
            }
        ]
    }
    read_ep = Episode(
        episode_id="e5",
        task_id="task",
        scope=scope,
        interaction_id="int1",
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_badges/values:batchGet"},
        response=read_resp,
        outcome="success",
    )
    new_ents = tracker.register_from_read(read_ep, task_text="send sms, email company contact, post slack; skip terminated contractors")
    assert "BDG-3001" in tracker.entities
    assert "BDG-3003" in tracker.entities
    assert "BDG-3002" not in tracker.entities  # Terminated was filtered out!

    # 2. Check initial active entity and obligations
    active = tracker.active_entity()
    assert active is not None
    assert active.name == "Wayne Ellis"
    assert active.next_pending_channel() == "sms"
    assert all(ob.status == "pending" for ob in active.obligations.values())

    # 3. Simulate Twilio SMS queued to Wayne Ellis
    sms_ep = Episode(
        episode_id="e8",
        task_id="task",
        scope=scope,
        interaction_id="int2",
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://api.twilio.com/2010-04-01/Accounts/AC/Messages.json",
            "body": '{"To": "+15553001001", "Body": "Hi Wayne Ellis, badge BDG-3001 is expiring"}',
        },
        response={"sid": "SM123", "status": "queued", "to": "+15553001001"},
        outcome="success",
    )
    matched = tracker.record_action(sms_ep)
    assert matched == "BDG-3001"
    assert active.obligations["sms"].status == "accepted"
    assert active.obligations["sms"].details.get("queued") is True  # Queued != delivered
    assert active.obligations["sms"].evidence_id == "e8"
    assert active.next_pending_channel() == "email"

    # 4. Verify decision_focus conditions retrieval on Wayne's next requirement: email
    adapter = SimpleNamespace(
        entity_tracker=tracker,
        last_interaction=None,
        active_decision=None,
    )
    entities, subgoal = decision_focus(adapter)
    assert "gmail" in entities
    assert "ops@contractorsinc.com" in entities
    assert "Wayne Ellis" in entities
    assert "email notification" in subgoal.lower()

    # 5. Simulate Gmail send for Wayne Ellis
    email_ep = Episode(
        episode_id="e10",
        task_id="task",
        scope=scope,
        interaction_id="int3",
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            "body": '{"to": "ops@contractorsinc.com", "raw": "..."}',
        },
        response={"id": "msg_001"},
        outcome="success",
    )
    tracker.record_action(email_ep)
    assert active.obligations["email"].status == "accepted"
    assert active.next_pending_channel() == "slack"

    # 6. Simulate Slack post for Wayne Ellis
    slack_ep = Episode(
        episode_id="e12",
        task_id="task",
        scope=scope,
        interaction_id="int4",
        operation="api_fetch",
        arguments={
            "method": "POST",
            "url": "https://slack.com/api/chat.postMessage",
            "body": '{"channel": "security", "text": "Wayne Ellis badge BDG-3001 renewal"}',
        },
        response={"ok": True},
        outcome="success",
    )
    tracker.record_action(slack_ep)
    assert active.is_complete()

    # 7. Active entity advances to Luis Moreno!
    next_active = tracker.active_entity()
    assert next_active is not None
    assert next_active.name == "Luis Moreno"
    assert next_active.next_pending_channel() == "sms"

    # 8. Check context rendering injection includes entity_obligations
    working = WorkingGraph(scope=scope, claims=(), reasons={}, stages=(), gaps=())
    text, _ = render_context(working, [], max_chars=4000, entity_tracker=tracker)
    import json
    payload = json.loads(text.split("\n", 1)[1])
    assert "entity_obligations" in payload
    assert payload["entity_obligations"]["active_entity"]["entity"] == "Luis Moreno"
    assert payload["entity_obligations"]["completed_count"] == 1
    assert payload["entity_obligations"]["pending_count"] == 1


def test_predicate_filtering_resolved_duplicates_and_task_exclusions():
    from worldmodel_harness.execution_state import EntityTracker
    from worldmodel_harness.knowledge import Episode, Scope

    scope = Scope(environment="test", version="1", role="agent", task_id="task")
    tracker = EntityTracker(required_channels=("sms", "slack"))

    # Spreadsheets read with:
    # Row 1: Resolved incident with "duplicate of INC-4921"
    # Row 2: Active after-hours incident
    # Row 3: Annex facility drill (which task prompt explicitly excludes)
    # Row 4: Main facility drill
    read_resp = {
        "values": [
            ["Incident ID", "Title", "Time", "Status", "Notes"],
            ["INC-4920", "Database Cluster Failover", "14:15", "Resolved", "duplicate of INC-4921"],
            ["INC-4922", "API Gateway Timeout", "03:30", "New", "severity under review"],
            ["DRILL-01", "Annex generator failure", "10:00", "Open", "routine drill"],
            ["DRILL-02", "Main plant water pump", "11:00", "Open", "scheduled test"],
        ]
    }
    ep = Episode(
        task_id="task",
        scope=scope,
        interaction_id="call1",
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/queue/values"},
        response=read_resp,
        outcome="success",
    )

    task_prompt = "Handle the after-hours P1 incident and process drill alarms for all facilities except the Annex. Skip duplicate records."
    new_ents = tracker.register_from_read(ep, task_text=task_prompt)

    # 1. INC-4920 must be filtered out (Status: Resolved, Notes: duplicate of INC-4921)
    assert "INC-4920" not in tracker.entities

    # 2. INC-4922 must be registered as active entity
    assert "INC-4922" in tracker.entities
    active = tracker.active_entity()
    assert active is not None
    assert active.entity_id == "INC-4922"

    # 3. DRILL-01 (Annex) must be filtered out by task exclusion "except the Annex"
    assert "DRILL-01" not in tracker.entities

    # 4. DRILL-02 (Main plant) must be registered
    assert "DRILL-02" in tracker.entities

    # 5. Summary reflects negative constraints
    summary = tracker.summary()
    assert "negative_constraints" in summary
    assert any("annex" in c for c in summary["negative_constraints"])


def test_json_record_list_registration():
    from worldmodel_harness.execution_state import EntityTracker
    from worldmodel_harness.knowledge import Episode, Scope

    scope = Scope(environment="test", version="1", role="agent", task_id="task")
    tracker = EntityTracker(required_channels=("sms",))

    # REST API response returning list of issues
    jira_resp = {
        "issues": [
            {
                "key": "OPS-101",
                "fields": {
                    "summary": "Storage Full",
                    "status": {"name": "Closed"},
                    "notes": "already resolved",
                },
            },
            {
                "key": "OPS-102",
                "fields": {
                    "summary": "Memory Leak",
                    "status": {"name": "Open"},
                    "phone": "+15551234567",
                },
            },
        ]
    }
    ep = Episode(
        task_id="task",
        scope=scope,
        interaction_id="call1",
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://jira.example.com/rest/api/2/search"},
        response=jira_resp,
        outcome="success",
    )
    tracker.register_from_read(ep, task_text="dispatch urgent alerts via sms; skip already resolved incidents")

    assert "OPS-101" not in tracker.entities  # Closed & already resolved
    assert "OPS-102" in tracker.entities
    assert tracker.entities["OPS-102"].phone == "+15551234567"


def test_record_action_unmatched_recipient_remains_unbound():
    from worldmodel_harness.execution_state import ChannelObligation, EntityTracker, TrackedEntity

    tracker = EntityTracker()
    tracker.entities["alice_id"] = TrackedEntity(
        entity_id="alice_id",
        name="Alice",
        phone="+15551111111",
        obligations={"sms": ChannelObligation(channel="sms", status="pending")},
    )
    scope = Scope(environment="test", version="1", role="ops", task_id="task")
    # SMS sent to Bob (+15552222222)
    sms_bob = Episode(
        episode_id="ep_bob",
        task_id="task",
        scope=scope,
        interaction_id="int_bob",
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://api.twilio.com/2010-04-01/Accounts/AC1/Messages.json", "body": '{"To": "+15552222222", "Body": "Hi Bob"}'},
        response={"sid": "msg_bob", "status": "queued"},
        outcome="success",
    )
    res = tracker.record_action(sms_bob)
    # Must remain unbound and NOT complete Alice's obligation
    assert res is None
    assert tracker.entities["alice_id"].obligations["sms"].status == "pending"


def test_explicit_task_intent_allows_closed_ticket_reopening():
    from worldmodel_harness.execution_state import EntityTracker

    scope = Scope(environment="test", version="1", role="ops", task_id="task")
    tracker = EntityTracker()
    jira_resp = {
        "issues": [
            {
                "key": "ticket-123",
                "fields": {
                    "summary": "Database Crash",
                    "status": {"name": "Closed"},
                    "phone": "+15550001111",
                },
            }
        ]
    }
    ep = Episode(
        episode_id="ep_read",
        task_id="task",
        scope=scope,
        interaction_id="call1",
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://jira.example.com/rest/api/2/issue/ticket-123"},
        response=jira_resp,
        outcome="success",
    )
    # The task explicitly asks to reopen closed ticket-123
    tracker.register_from_read(ep, task_text="Reopen closed ticket ticket-123 and notify via sms")
    assert "ticket-123" in tracker.entities
    assert tracker.entities["ticket-123"].entity_id == "ticket-123"





def test_positive_state_change_requests_do_not_create_exclusions():
    from worldmodel_harness.execution_state import EntityTracker, extract_task_exclusions

    scope = Scope(environment="test", version="1")
    for text, status, notes in [
        ("Reopen already closed ticket ticket-123", "closed", "already resolved"),
        ("Reactivate previously deactivated account ticket-123", "deactivated", ""),
        ("Restore archived ticket ticket-123", "archived", "no action required"),
    ]:
        tracker = EntityTracker()
        assert extract_task_exclusions(text) == []
        episode = Episode(task_id="t", scope=scope, operation="api_fetch", arguments={},
                          response={"items": [{"id": "ticket-123", "name": "Restore access",
                                               "status": status, "notes": notes}]},
                          outcome="success")
        assert tracker.register_from_read(episode, task_text=text) == ["ticket-123"]
    assert extract_task_exclusions("Skip already closed tickets") == ["closed"]


def test_recipient_fields_override_mentions_and_shared_addresses_remain_pending():
    import base64
    import json

    from worldmodel_harness.execution_state import ChannelObligation, EntityTracker, TrackedEntity

    scope = Scope(environment="test", version="1")
    tracker = EntityTracker()
    tracker.entities["alice"] = TrackedEntity(
        entity_id="alice", name="Alice", phone="+15551111111", email="alice@example.com",
        obligations={ch: ChannelObligation(ch) for ch in ("sms", "email")})
    def action(channel, body):
        return Episode(task_id="t", scope=scope, operation="api_fetch", outcome="success",
                       arguments={"method": "POST", "url": f"https://{channel}.example.com/send",
                                  "body": json.dumps(body)}, response={"status": "queued"})
    assert tracker.record_action(action("twilio", {"To": "+15552222222", "Body": "Contact Alice"})) is None
    assert tracker.record_action(action("gmail", {"to": "bob@example.com", "body": "Alice: alice@example.com"})) is None
    assert all(ob.status == "pending" for ob in tracker.entities["alice"].obligations.values())
    assert tracker.record_action(action("twilio", {"To": "+15551111111", "Body": "Hello Bob"})) == "alice"
    raw = base64.urlsafe_b64encode(b'To: Alice <alice@example.com>\r\nSubject: Notice\r\n\r\nHello').decode()
    assert tracker.record_action(action("gmail", {"raw": raw})) == "alice"
    tracker.entities["other"] = TrackedEntity(entity_id="other", email="alice@example.com",
                                              obligations={"email": ChannelObligation("email")})
    assert tracker.record_action(action("gmail", {"to": "alice@example.com"})) is None
    assert tracker.entities["other"].obligations["email"].status == "pending"
