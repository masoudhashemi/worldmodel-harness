import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from worldmodel_harness.goal_checks import CheckedFinish, GoalTracker
from worldmodel_harness.knowledge import Episode, LiveState, Scope
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.verification_tools import (
    CheckEffect,
    VerifyGoal,
    simple_check_tool,
    verification_tool,
)


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    scope = Scope(environment="test", version="1", task_id="task")
    task = "Find Avery Chen's message and update their contact with the requested value."
    adapter = SimpleNamespace(
        evidence=store,
        scope=scope,
        live=LiveState(scope.task_id),
        evidence_aliases={},
        task=SimpleNamespace(prompt=[{"content": task}]),
    )
    adapter.goal_tracker = GoalTracker(adapter)
    yield adapter, store, task
    store.close()


def observe(
    adapter,
    store,
    *,
    name="Avery Chen",
    value="new-value",
    method="GET",
    url="https://target.test/items/1",
):
    event = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={"method": method, "url": url},
        response={"Name": name, "Value": value},
        coverage=("Name", "Value"),
        outcome="success",
    )
    store.append_episode(event)
    adapter.live.observe(event)
    alias = f"e{len(adapter.evidence_aliases) + 1}"
    adapter.evidence_aliases[alias] = event.episode_id
    return alias, event


def source(adapter, store, task):
    alias, event = observe(
        adapter,
        store,
        name="avery.chen@example.test",
        value="Please use new-value.",
        url="https://source.test/message",
    )
    request = VerifyGoal(
        requirement_id="message",
        task_quote=task,
        kind="source",
        episode_id=alias,
        path=("Value",),
        anchor_text="Avery Chen",
        anchor_path=("Name",),
    )
    result = asyncio.run(verification_tool(adapter).executor(request))
    assert result.success
    return event


def effect(task, alias):
    return VerifyGoal(
        requirement_id="update",
        task_quote=task,
        kind="effect",
        episode_id=alias,
        path=("Value",),
        expected="new-value",
        resource_url="https://target.test/items/1",
        source_requirement_id="message",
        identity_path=("Name",),
    )


def test_omitted_effect_source_attaches_the_one_matching_phone(world):
    adapter, store, task = world
    phone = "+1-415-555-0100"
    alias, _ = observe(
        adapter, store, name="Avery Chen", value=phone, url="https://source.test/message"
    )
    registered = VerifyGoal(
        requirement_id="message",
        task_quote=task,
        kind="source",
        episode_id=alias,
        path=("Value",),
        anchor_text="Avery Chen",
        anchor_path=("Name",),
    )
    assert asyncio.run(verification_tool(adapter).executor(registered)).success
    target, _ = observe(adapter, store, value=phone)
    omitted = VerifyGoal(
        requirement_id="update",
        task_quote=task,
        kind="effect",
        episode_id=target,
        path=("Value",),
        expected=phone,
        resource_url="https://target.test/items/1",
        identity_path=("Name",),
    )
    assert asyncio.run(verification_tool(adapter).executor(omitted)).success
    assert adapter.goal_tracker.checks["update"].outcome.source_requirement_id == "message"
    audit = adapter.goal_tracker.verification_audit[-1]
    assert audit["original_request"]["source_requirement_id"] is None
    assert audit["inferred_binding"]["source_requirement_id"] == "message"


@pytest.mark.parametrize("case", ["wrong_value", "wrong_person", "two_sources", "prose"])
def test_auto_source_requires_unique_exact_value_and_identity(world, case):
    adapter, store, task = world
    phone = "+1-415-555-0100"
    tool = verification_tool(adapter)
    for i in range(2 if case == "two_sources" else 1):
        content = "Someone else's number is " + phone if case == "prose" else phone
        alias, _ = observe(adapter, store, value=content, url=f"https://source.test/{i}")
        assert asyncio.run(tool.executor(VerifyGoal(
            requirement_id=f"source{i}", task_quote=task, kind="source", episode_id=alias,
            path=("Value",), anchor_text="Avery Chen", anchor_path=("Name",),
        ))).success
    target, _ = observe(adapter, store, value=phone,
                        name="Other Person" if case == "wrong_person" else "Avery Chen")
    request = VerifyGoal(requirement_id="effect", task_quote=task, kind="effect",
        episode_id=target, path=("Value",), identity_path=("Name",),
        resource_url="https://target.test/items/1",
        expected="+1-415-555-0199" if case == "wrong_value" else phone)
    assert not asyncio.run(tool.executor(request)).success
    assert adapter.goal_tracker.verification_audit[-1]["inferred_binding"] is None


def test_kind_predicate_cannot_establish_task_outcome(world):
    adapter, store, task = world
    source(adapter, store, task)
    alias, _ = observe(adapter, store, value="+1-415-555-0100")
    request = effect(task, alias).model_copy(update={"expected": {"$kind": "phone"}})
    result = asyncio.run(verification_tool(adapter).executor(request))
    assert not result.success
    assert "insufficient_outcome_predicate" in result.content


def test_direct_source_and_effect_without_body_equality_or_second_source(world):
    adapter, store, task = world
    event = source(adapter, store, task)
    adapter.live.invalidate({f.key for f in adapter.live.facts()}, "write")
    assert adapter.goal_tracker.assess(adapter.goal_tracker.checks["message"]) == "matched"
    alias, _ = observe(adapter, store)
    assert asyncio.run(verification_tool(adapter).executor(effect(task, alias))).success
    assert adapter.goal_tracker.finish(CheckedFinish(summary="verified")).success
    contract = adapter.goal_tracker.checks["update"].outcome
    assert contract.source_episode_id == event.episode_id
    assert len(store.episodes(adapter.scope)) == 2
    assert len(adapter.goal_tracker.verification_audit) == 2


@pytest.mark.parametrize(
    "name,value,method,expected",
    [
        ("Other Person", "new-value", "GET", "target_identity_mismatch"),
        ("Avery Chen", "old-value", "GET", "contradicted"),
        ("Avery Chen", "new-value", "PATCH", "outcome_verification_requires_read"),
    ],
)
def test_effect_not_satisfied_by_wrong_identity_value_or_write(
    world, name, value, method, expected
):
    adapter, store, task = world
    source(adapter, store, task)
    alias, _ = observe(adapter, store, name=name, value=value, method=method)
    result = asyncio.run(verification_tool(adapter).executor(effect(task, alias)))
    assert not result.success and expected in result.content
    assert not adapter.goal_tracker.finish(CheckedFinish(summary="done")).success


def test_invalid_path_can_repair_then_become_stale(world):
    adapter, store, task = world
    source(adapter, store, task)
    alias, _ = observe(adapter, store)
    tool = verification_tool(adapter)
    valid = effect(task, alias)
    invalid = valid.model_copy(update={"path": ("absent",)})
    result = asyncio.run(tool.executor(invalid))
    assert "invalid_evidence_path" in result.content
    assert "available_keys" in result.content
    assert asyncio.run(tool.executor(valid)).success
    adapter.live.invalidate({f.key for f in adapter.live.facts()}, "mutation")
    result = asyncio.run(tool.executor(valid))
    assert not result.success and "stale_evidence" in result.content


def test_pending_goal_survives_rejected_update_and_kind_cannot_weaken(world):
    adapter, store, task = world
    source(adapter, store, task)
    alias, _ = observe(adapter, store)
    bad = effect(task, alias).model_copy(update={"expected": "invented"})
    tool = verification_tool(adapter)
    assert not asyncio.run(tool.executor(bad)).success
    assert "update" in adapter.goal_tracker.checks
    assert adapter.goal_tracker.checks["update"].outcome is None
    assert not adapter.goal_tracker.finish(CheckedFinish(summary="done")).success
    changed = VerifyGoal(
        requirement_id="update",
        task_quote=task,
        kind="source",
        episode_id=alias,
        path=("Value",),
        anchor_text="Avery Chen",
        anchor_path=("Name",),
    )
    assert not asyncio.run(tool.executor(changed)).success
    assert asyncio.run(tool.executor(effect(task, alias))).success


def test_source_needs_task_relevance_and_nonempty_content(world):
    adapter, store, task = world
    alias, _ = observe(adapter, store, name="Someone Else")
    request = VerifyGoal(
        requirement_id="message",
        task_quote=task,
        kind="source",
        episode_id=alias,
        path=("Value",),
        anchor_text="Avery Chen",
        anchor_path=("Name",),
    )
    result = asyncio.run(verification_tool(adapter).executor(request))
    assert not result.success and "source_anchor_mismatch" in result.content


def test_flat_schema_rejects_nested_action_and_duplicated_observation():
    properties = VerifyGoal.model_json_schema()["properties"]
    assert (
        "action" not in properties
        and "observation" not in properties
        and "outcome" not in properties
    )
    with pytest.raises(ValidationError):
        VerifyGoal.model_validate({"action": '{"tool":"record_goal_checks"}'})


def test_identity_and_value_cannot_select_different_records(world):
    adapter, store, task = world
    source(adapter, store, task)
    alias, _ = observe(adapter, store)
    request = effect(task, alias).model_copy(
        update={"path": ("records", 0, "Value"), "identity_path": ("records", 1, "Name")}
    )
    result = asyncio.run(verification_tool(adapter).executor(request))
    assert not result.success and "identity_record_mismatch" in result.content


def test_direct_verification_in_real_sdk_session(world, tmp_path):
    from stirrup.core.models import AssistantMessage, ToolCall

    from worldmodel_harness.repairing_agent import RepairingAgent

    adapter, store, task = world
    source_alias, _ = observe(
        adapter,
        store,
        name="avery.chen@example.test",
        value="Please use new-value.",
        url="https://source.test/message",
    )
    target_alias, _ = observe(adapter, store)
    requests = [
        VerifyGoal(
            requirement_id="message",
            task_quote=task,
            kind="source",
            episode_id=source_alias,
            path=("Value",),
            anchor_text="Avery Chen",
            anchor_path=("Name",),
        ),
        effect(task, target_alias),
    ]

    class Client:
        model_slug = "offline-scripted"
        context_window_tokens = 32000
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls <= 2:
                return AssistantMessage(
                    blocks=[
                        ToolCall(
                            name="verify_goal", arguments=requests[self.calls - 1].model_dump_json()
                        )
                    ]
                )
            return AssistantMessage(
                blocks=[ToolCall(name="finish", arguments=json.dumps({"summary": "verified"}))]
            )

    async def run():
        tool = verification_tool(adapter)
        agent = RepairingAgent(
            client=Client(),
            name="direct-verifier",
            tools=[tool],
            max_turns=4,
            finish_tool=adapter.goal_tracker.finish_tool(),
            repair_schemas={"verify_goal": VerifyGoal},
        )
        async with agent.session(output_dir=tmp_path / "session") as session:
            finish, _, _ = await session.run(task)
        assert not agent.outer_validation_audit
        return finish

    assert asyncio.run(run()).summary == "verified"
    assert adapter.goal_tracker.completion_status == "checked"


def test_invalid_anchor_can_be_repaired_without_new_requirement(world):
    adapter, store, task = world
    alias, _ = observe(
        adapter, store, name="avery.chen@example.test", value="Please use new-value."
    )
    tool = verification_tool(adapter)
    invalid = VerifyGoal(
        requirement_id="source",
        task_quote="message",
        kind="source",
        episode_id=alias,
        path=("Value",),
        anchor_text="new-value",
        anchor_path=("Value",),
    )
    result = asyncio.run(tool.executor(invalid))
    assert not result.success and "missing_task_anchor" in result.content
    feedback = json.loads(result.content)
    assert feedback["field"] == "anchor_text" and feedback["public_task"] == task
    corrected = invalid.model_copy(update={"anchor_text": "Avery Chen", "anchor_path": ("Name",)})
    assert asyncio.run(tool.executor(corrected)).success
    assert list(adapter.goal_tracker.checks) == ["source"]
    # Once supported, its identity cannot be weakened to a different anchor.
    changed = corrected.model_copy(update={"anchor_text": "Avery"})
    assert not asyncio.run(tool.executor(changed)).success


def test_task_literal_effect_requires_task_identity(world):
    adapter, store, _ = world
    task = "Set Avery Chen's value to new-value."
    adapter.task.prompt = [{"content": task}]
    alias, _ = observe(adapter, store, name="Other Person")
    request = VerifyGoal(
        requirement_id="literal",
        task_quote="value to new-value.",
        kind="effect",
        episode_id=alias,
        path=("Value",),
        expected="new-value",
        resource_url="https://target.test/items/1",
        anchor_text="Avery Chen",
        identity_path=("Name",),
    )
    result = asyncio.run(verification_tool(adapter).executor(request))
    assert not result.success and "target_identity_mismatch" in result.content


def test_simple_check_accepts_exact_value_and_rejects_another_number(world):
    adapter, store, _ = world
    phone = "+1-415-555-0100"
    other = "+1-415-555-0199"
    source_alias, source_event = observe(
        adapter, store, name="Avery Chen", value=phone, url="https://source.test/message"
    )
    target_alias, target_event = observe(adapter, store, name="Avery Chen", value=phone)
    tool = simple_check_tool(adapter)
    matched = asyncio.run(
        tool.executor(
            CheckEffect(
                requirement_id="phone",
                task_quote="Avery Chen",
                episode_id=target_alias,
                path=("Value",),
                expected=phone,
                anchor_text="Avery Chen",
            )
        )
    )
    assert matched.success
    outcome = adapter.goal_tracker.checks["phone"].outcome
    assert outcome.resource_url == target_event.arguments["url"]
    assert outcome.identity_path == ("Name",)
    assert outcome.source_episode_id == source_event.episode_id
    assert source_alias
    assert adapter.goal_tracker.finish(CheckedFinish(summary="verified")).success
    rejected = json.loads(
        asyncio.run(
            tool.executor(
                CheckEffect(
                    requirement_id="other",
                    task_quote="Avery Chen",
                    episode_id=target_alias,
                    path=("Value",),
                    expected=other,
                    anchor_text="Avery Chen",
                )
            )
        ).content
    )
    assert rejected["result"] == "ungrounded_outcome_value"
    assert "other" not in adapter.goal_tracker.checks
    plain = json.loads(
        asyncio.run(
            tool.executor(
                CheckEffect(
                    requirement_id="plain",
                    task_quote="Avery Chen",
                    episode_id=target_alias,
                    path=("Value",),
                    expected="not-a-phone",
                    anchor_text="Avery Chen",
                )
            )
        ).content
    )
    assert plain["result"] == "ungrounded_outcome_value"
    kind = json.loads(
        asyncio.run(
            tool.executor(
                CheckEffect(
                    requirement_id="kind",
                    task_quote="Avery Chen",
                    episode_id=target_alias,
                    path=("Value",),
                    expected={"$kind": "phone"},
                    anchor_text="Avery Chen",
                )
            )
        ).content
    )
    assert kind["result"] == "insufficient_outcome_predicate"


@pytest.mark.parametrize(
    ("value", "other", "fmt"),
    [
        ("avery@example.com", "other@example.com", "EMAIL"),
        ("https://example.com/a", "https://example.com/b", "URL"),
        ("2020-01-02", "2021-02-03", "DATE"),
        ("003xx0000000001AAA", "003xx0000000001AAB", "IDENTIFIER"),
    ],
)
def test_wrong_same_format_literal_is_rejected_for_every_slot(world, value, other, fmt):
    adapter, store, _ = world
    observe(adapter, store, name="Avery Chen", value=value, url="https://source.test/message")
    target_alias, _ = observe(adapter, store, name="Avery Chen", value=value)
    result = asyncio.run(
        simple_check_tool(adapter).executor(
            CheckEffect(
                requirement_id="slot",
                task_quote="Avery Chen",
                episode_id=target_alias,
                path=("Value",),
                expected=other,
                anchor_text="Avery Chen",
            )
        )
    )
    assert not result.success
    assert json.loads(result.content)["result"] == "ungrounded_outcome_value"


def test_split_name_is_one_identity_without_a_name_in_the_anchor(world):
    adapter, store, _ = world
    event = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://target.test/items/1"},
        response={
            "FirstName": "Avery",
            "LastName": "Chen",
            "Name": "Avery Chen",
            "Email": "avery@example.com",
        },
        coverage=("FirstName", "LastName", "Name", "Email"),
        outcome="success",
    )
    store.append_episode(event)
    adapter.live.observe(event)
    alias = "e1"
    adapter.evidence_aliases[alias] = event.episode_id
    result = asyncio.run(
        simple_check_tool(adapter).executor(
            CheckEffect(
                requirement_id="email",
                task_quote="update their contact",
                episode_id=alias,
                path=("Email",),
                expected="other@example.com",
                anchor_text="update her contact in Salesforce with the new address",
            )
        )
    )
    assert not result.success
    assert json.loads(result.content)["field"] == "anchor_text"


def test_sentence_anchor_matches_name_format_not_the_literal(world):
    adapter, store, _ = world
    phone = "+1-415-555-0100"
    source_alias, _ = observe(
        adapter,
        store,
        name="Avery Chen",
        value=phone,
        url="https://source.test/message",
    )
    event = Episode(
        scope=adapter.scope,
        task_id=adapter.scope.task_id,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://target.test/items/1"},
        response={
            "Name": "Avery Chen",
            "FirstName": "Avery",
            "LastName": "Chen",
            "Value": phone,
        },
        coverage=("Name", "FirstName", "LastName", "Value"),
        outcome="success",
    )
    store.append_episode(event)
    adapter.live.observe(event)
    alias = f"e{len(adapter.evidence_aliases) + 1}"
    adapter.evidence_aliases[alias] = event.episode_id
    sentence = (
        "We updated Avery Chen's contact in Salesforce with the new value and verified the update."
    )
    result = asyncio.run(
        simple_check_tool(adapter).executor(
            CheckEffect(
                requirement_id="contact",
                task_quote="Avery Chen",
                episode_id=alias,
                path=("Value",),
                expected=phone,
                anchor_text=sentence,
            )
        )
    )
    assert not result.success
    assert json.loads(result.content)["field"] == "anchor_text"
    repaired = asyncio.run(simple_check_tool(adapter).executor(CheckEffect(
        requirement_id="contact", task_quote="Avery Chen", episode_id=alias,
        path=("Value",), expected=phone, anchor_text="Avery Chen")))
    assert repaired.success
    assert adapter.goal_tracker.checks["contact"].outcome.identity_anchor == "Avery Chen"
    assert source_alias


def test_captured_incomplete_request_names_one_field_and_ids_to_copy(world):
    """Replay of learned-email req_find_email. No model or environment call."""
    adapter, store, _ = world
    captured_quote = "Find her email"
    adapter.task.prompt = [{"content": "Find her email and update the contact."}]
    captured = {
        "requirement_id": "req_find_email",
        "task_quote": captured_quote,
        "kind": "source",
        "episode_id": "8af2a170d7f246719f2605733aa6bf31",
        "path": ["body_plain"],
        "anchor_text": "maria.santos@brightwave-global.example.com",
        "anchor_path": [],
        "expected": None,
        "resource_url": None,
        "source_requirement_id": None,
        "identity_path": [],
    }
    result = asyncio.run(verification_tool(adapter).executor(VerifyGoal.model_validate(captured)))
    body = json.loads(result.content)
    assert not result.success
    assert body["result"] == "missing_source_fields"
    assert body["field"] == "anchor_path"
    assert body["required"] == ["anchor_path"]
    assert body["requirement_id"] == "req_find_email"
    assert body["episode_id"] == captured["episode_id"]
    assert body["copy"] == {
        "requirement_id": "req_find_email",
        "episode_id": captured["episode_id"],
    }
    assert "maria.santos@brightwave-global.example.com" not in result.content
    assert store.episodes(adapter.scope) == []

    phone = "+1-415-555-0100"
    alias, _ = observe(
        adapter, store, name="Avery Chen", value=phone, url="https://source.test/message"
    )
    registered = VerifyGoal(
        requirement_id="message",
        task_quote=captured_quote,
        kind="source",
        episode_id=alias,
        path=("Value",),
        anchor_text="Avery Chen",
        anchor_path=("Name",),
    )
    adapter.task.prompt = [{"content": "Find her email from Avery Chen."}]
    registered = registered.model_copy(update={"task_quote": "Avery Chen"})
    assert asyncio.run(verification_tool(adapter).executor(registered)).success
    target, _ = observe(adapter, store, value=phone)
    partial = VerifyGoal(
        requirement_id="update",
        task_quote="Avery Chen",
        kind="effect",
        episode_id=target,
        path=("Value",),
        expected=phone,
        resource_url="https://target.test/items/1",
    )
    repaired = json.loads(asyncio.run(verification_tool(adapter).executor(partial)).content)
    assert repaired["result"] == "missing_target_identity"
    assert repaired["field"] == "identity_path"
    assert repaired["matched_source_id"] == "message"
    assert repaired["copy"]["source_requirement_id"] == "message"
    assert repaired["copy"]["episode_id"] == target
    assert adapter.goal_tracker.checks["update"].outcome is None


@pytest.mark.parametrize("failure", ["wrong_person", "ambiguous_source", "wrong_value", "wrong_tenant"])
def test_exact_effect_binding_rejects_unsafe_sources(world, failure):
    adapter, store, _ = world
    phone = "+1-415-555-0100"
    _, source_event = observe(adapter, store, name="Other Person" if failure == "wrong_person" else "Avery Chen",
                             value=phone, url="https://source.test/message")
    if failure == "wrong_tenant":
        # Replace the fixture's source with evidence from another tenant.
        store.db.execute("DELETE FROM episodes WHERE id=?", (source_event.episode_id,))
        store.append_episode(source_event.model_copy(update={"scope": adapter.scope.model_copy(update={"tenant": "other"})}))
    if failure == "ambiguous_source":
        observe(adapter, store, name="Avery Chen", value=phone, url="https://source.test/second")
    alias, _ = observe(adapter, store, name="Avery Chen", value=phone)
    result = asyncio.run(simple_check_tool(adapter).executor(CheckEffect(
        requirement_id="phone", task_quote="Avery Chen", episode_id=alias,
        path=("Value",), expected="+1-415-555-9999" if failure == "wrong_value" else phone,
        anchor_text="Avery Chen")))
    assert not result.success
    assert "phone" not in adapter.goal_tracker.checks


def test_exact_quote_repair_keeps_original_request_and_accepts_corrected_quote(world):
    adapter, store, task = world
    phone = "+1-415-555-0100"
    observe(adapter, store, name="Avery Chen", value=phone, url="https://source.test/message")
    alias, _ = observe(adapter, store, name="Avery Chen", value=phone)
    tool = simple_check_tool(adapter)
    request = CheckEffect(requirement_id="phone", task_quote="Find Avery's message", episode_id=alias,
                          path=("Value",), expected=phone, anchor_text="Avery Chen")
    rejected = asyncio.run(tool.executor(request))
    feedback = json.loads(rejected.content)
    assert not rejected.success
    assert feedback["public_task"] == task
    assert feedback["field"] == "task_quote"
    assert asyncio.run(tool.executor(request.model_copy(update={"task_quote": "Avery Chen"}))).success
    original = adapter.goal_tracker.checks["phone"]
    changed = request.model_copy(update={"task_quote": "Avery Chen", "expected": "Avery Chen", "path": ("Name",)})
    assert not asyncio.run(tool.executor(changed)).success
    assert adapter.goal_tracker.checks["phone"] == original


def test_finish_repair_names_direct_tool_and_preserves_empty_ledger(world):
    adapter, _, task = world
    result = adapter.goal_tracker.finish(CheckedFinish(summary="done"))
    assert not result.success
    feedback = json.loads(result.content)
    assert feedback["verification_tool"] == "check_effect"
    assert feedback["public_task"] == task
    assert "exact expected value" in feedback["next"]
    assert adapter.goal_tracker.checks == {}

    # Second finish without pending entity obligations or written fields allows soft completion
    second = adapter.goal_tracker.finish(CheckedFinish(summary="done"))
    assert second.success
    assert adapter.goal_tracker.completion_status == "unverified"
