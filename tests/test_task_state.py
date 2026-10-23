import asyncio
import json
from types import SimpleNamespace

from worldmodel_harness.decisions import ProposedAction
from worldmodel_harness.goal_checks import (
    CheckedFinish,
    GoalCheck,
    GoalTracker,
    GoalUpdate,
    OutcomeContract,
    SourceBinding,
)
from worldmodel_harness.knowledge import Episode, LiveState, OutputExpectation, Scope
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.task_state import TargetProposal, TaskState


def setup_state(tmp_path):
    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    scope = Scope(environment="test", version="1", task_id="task")
    adapter = SimpleNamespace(
        evidence=store,
        knowledge=store,
        scope=scope,
        live=LiveState("task"),
        evidence_aliases={},
        task=SimpleNamespace(
            prompt=[
                {
                    "role": "user",
                    "content": "Use the facilities email. Create Task A, assign February, apply Compliance, and notify Slack.",
                }
            ]
        ),
    )
    adapter.goal_tracker = GoalTracker(adapter)
    adapter.entity_tracker = TaskState(adapter)
    return adapter


def read(adapter, response, url="https://example.com/source"):
    ep = Episode(
        task_id="task",
        scope=adapter.scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": url},
        response=response,
        coverage=tuple(response),
        outcome="success",
    )
    adapter.evidence.append_episode(ep)
    adapter.live.observe(ep)
    adapter.entity_tracker.register_from_read(ep)
    return ep


def proposal(source):
    selection = GoalCheck(
        requirement_id="sender",
        task_quote="facilities email",
        kind="source",
        evidence_mode="historical_observation",
        episode_id=source.episode_id,
        observation=OutputExpectation(path=("body",), predicate="type", expected="string"),
        source_binding=SourceBinding(anchor_text="facilities", anchor_path=("from",)),
    )
    effects = tuple(
        GoalCheck(requirement_id=key, task_quote=quote)
        for key, quote in [
            ("create", "Create Task A"),
            ("section", "assign February"),
            ("tag", "apply Compliance"),
            ("notify", "notify Slack"),
        ]
    )
    return TargetProposal(
        target_id="task-a",
        source_episode_id=source.episode_id,
        selection_checks=(selection,),
        effects=effects,
    )


def test_reads_are_observations_not_targets(tmp_path):
    adapter = setup_state(tmp_path)
    for response in [
        {"values": [["Rule", "Description"], ["Draft", "Do not action draft emails"]]},
        {"data": [{"gid": "sec_feb", "name": "February"}]},
    ]:
        read(adapter, response)
    state = adapter.entity_tracker
    assert len(state.observations) == 2
    assert state.targets == {} and state.pending_entities() == []
    assert state.unresolved() == []
    # No invented policy-row gate; normal evidence requirements still apply.
    result = adapter.goal_tracker.finish(CheckedFinish(summary="Not yet complete"))
    assert json.loads(result.content)["status"] == "needs_verification"
    adapter.evidence.close()


def test_wrong_sender_is_unselected_and_cannot_authorize_write(tmp_path):
    adapter = setup_state(tmp_path)
    ep = read(adapter, {"from": "maintenance@example.com", "body": "Task A"})
    p = proposal(ep)
    result = asyncio.run(adapter.entity_tracker.tool().executor(p))
    assert not result.success
    assert json.loads(result.content)["selection_checks"]["sender"] == "source_anchor_mismatch"
    action = ProposedAction(
        tool="api_fetch",
        target_id="task-a",
        arguments={"method": "POST", "url": "https://example.com/task"},
    )
    assert adapter.entity_tracker.action_error(action)
    result = adapter.goal_tracker.finish(CheckedFinish(summary="Done"))
    assert adapter.goal_tracker.finish_audit[-1]["shadow_unresolved"]
    # Explicitly unresolved completion is allowed, not reported as success.
    assert adapter.goal_tracker.finish(
        CheckedFinish(summary="Missing evidence", unresolved=True)
    ).success
    # Repair rejected selection with a matching record, retaining all requirements.
    correct = read(adapter, {"from": "facilities@example.com", "body": "Task A"})
    assert asyncio.run(adapter.entity_tracker.tool().executor(proposal(correct))).success
    assert adapter.entity_tracker.action_error(action) is None
    adapter.evidence.close()


def test_each_effect_stays_pending_until_its_own_verification(tmp_path):
    adapter = setup_state(tmp_path)
    ep = read(adapter, {"from": "facilities@example.com", "body": "Task A"})
    assert asyncio.run(adapter.entity_tracker.tool().executor(proposal(ep))).success
    write = Episode(
        task_id="task",
        scope=adapter.scope,
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://example.com/task"},
        response={"id": "created"},
        outcome="success",
    )
    adapter.entity_tracker.record_action(write)
    assert set(adapter.entity_tracker.summary()["targets"][0]["effects"].values()) == {
        "missing_evidence"
    }
    current = read(adapter, {"name": "Task A"}, "https://example.com/task")
    expected = OutputExpectation(path=("name",), predicate="equals", expected="Task A")
    check = GoalCheck(
        requirement_id="create",
        task_quote="Create Task A",
        episode_id=current.episode_id,
        observation=expected,
        outcome=OutcomeContract(
            target=expected,
            resource_url="https://example.com/task",
            source_episode_id=ep.episode_id,
            source_path=("body",),
            source_requirement_id="sender",
            identity_path=("name",),
            source_identity_path=("body",),
        ),
    )
    assert asyncio.run(adapter.goal_tracker.tool().executor(GoalUpdate(checks=(check,)))).success
    row = adapter.entity_tracker.summary()["targets"][0]
    assert row["effects"]["create"] == "matched"
    assert row["effects"]["tag"] == "missing_evidence" and not row["complete"]
    for _ in range(2):
        assert not adapter.goal_tracker.finish(
            CheckedFinish(summary="Created, tagged, and notified")
        ).success
    # A bare success claim or a different source cannot satisfy the missing tag.
    unbound = check.model_copy(
        update={
            "requirement_id": "tag",
            "task_quote": "apply Compliance",
            "outcome": check.outcome.model_copy(update={"source_requirement_id": None}),
        }
    )
    assert not asyncio.run(
        adapter.goal_tracker.tool().executor(GoalUpdate(checks=(unbound,)))
    ).success
    adapter.evidence.close()


def test_selection_predicate_and_contract_cannot_be_weakened(tmp_path):
    adapter = setup_state(tmp_path)
    ep = read(adapter, {"from": "facilities@example.com", "body": "Wrong Task"})
    p = proposal(ep)
    check = p.selection_checks[0].model_copy(
        update={
            "observation": OutputExpectation(path=("body",), predicate="equals", expected="Task A")
        }
    )
    p = p.model_copy(update={"selection_checks": (check,)})
    assert not asyncio.run(adapter.entity_tracker.tool().executor(p)).success
    weakened = check.model_copy(
        update={"observation": OutputExpectation(path=("body",), predicate="present")}
    )
    assert not asyncio.run(
        adapter.goal_tracker.tool().executor(GoalUpdate(checks=(weakened,)))
    ).success
    assert not asyncio.run(
        adapter.entity_tracker.tool().executor(
            p.model_copy(update={"selection_checks": (weakened,)})
        )
    ).success
    adapter.evidence.close()


def test_selection_cannot_combine_sibling_records(tmp_path):
    adapter = setup_state(tmp_path)
    ep = read(
        adapter,
        {
            "data": [
                {"from": "facilities@example.com", "body": "Task B"},
                {"from": "maintenance@example.com", "body": "Task A"},
            ]
        },
    )
    p = proposal(ep)
    check = p.selection_checks[0].model_copy(
        update={
            "source_binding": SourceBinding(
                anchor_text="facilities", anchor_path=("data", 0, "from")
            ),
            "observation": OutputExpectation(
                path=("data", 1, "body"), predicate="equals", expected="Task A"
            ),
        }
    )
    result = asyncio.run(
        adapter.entity_tracker.tool().executor(p.model_copy(update={"selection_checks": (check,)}))
    )
    assert not result.success
    assert json.loads(result.content)["selection_checks"]["sender"] == "different_record"
    adapter.evidence.close()


def test_dispatch_evaluates_target_selection_in_shadow_mode(tmp_path):
    from pydantic import BaseModel
    from stirrup.core.models import Tool, ToolResult

    from worldmodel_harness.decision_runtime import dispatch_decision
    from worldmodel_harness.decisions import ActionRequirements, Decision

    class Fetch(BaseModel):
        method: str
        url: str

    adapter = setup_state(tmp_path)
    calls, audit = [], []

    async def invoke(params):
        calls.append(params)
        return ToolResult(content="accepted")

    tool = Tool(name="api_fetch", description="API", parameters=Fetch, executor=invoke)
    decision = Decision(
        goal="Create Task A",
        rationale="Create requested task",
        action=ProposedAction(
            tool="api_fetch", arguments={"method": "POST", "url": "https://example.com/tasks"}
        ),
    )
    result = asyncio.run(
        dispatch_decision(
            decision,
            tools={tool.name: tool},
            requirements_for=lambda _: ActionRequirements(),
            store=adapter.knowledge,
            scope=adapter.scope,
            live=adapter.live,
            audit=audit,
            entity_tracker=adapter.entity_tracker,
        )
    )
    assert result.success and len(calls) == 1 and audit[0]["invoked"]
    assert audit[0]["shadow_target_check"]["allowed"] is False
    assert audit[0]["shadow_target_check"]["diagnostic"]["status"] == "target_selection_required"
    wrong = read(adapter, {"from": "maintenance@example.com", "body": "Task A"})
    assert not asyncio.run(adapter.entity_tracker.tool().executor(proposal(wrong))).success
    correct = read(adapter, {"from": "facilities@example.com", "body": "Task A"})
    assert asyncio.run(adapter.entity_tracker.tool().executor(proposal(correct))).success
    repaired = decision.model_copy(
        update={"action": decision.action.model_copy(update={"target_id": "task-a"})}
    )
    result = asyncio.run(
        dispatch_decision(
            repaired,
            tools={tool.name: tool},
            requirements_for=lambda _: ActionRequirements(),
            store=adapter.knowledge,
            scope=adapter.scope,
            live=adapter.live,
            audit=audit,
            entity_tracker=adapter.entity_tracker,
        )
    )
    assert result.success and len(calls) == 2 and audit[-1]["invoked"]
    assert audit[-1]["shadow_target_check"]["allowed"] is True
    adapter.evidence.close()


def test_public_target_tool_derives_internal_evidence_modes(tmp_path):
    adapter = setup_state(tmp_path)
    source = read(adapter, {"from": "facilities@example.com", "body": "Task A"})
    tool = adapter.entity_tracker.tool()
    params = tool.parameters.model_validate(
        {
            "target_id": "task-a",
            "source_episode_id": source.episode_id,
            "selection_checks": [
                {
                    "requirement_id": "sender",
                    "task_quote": "facilities email",
                    "anchor_text": "facilities",
                    "anchor_path": ["from"],
                    "observation": {"path": ["body"], "predicate": "equals", "expected": "Task A"},
                }
            ],
            "effects": [{"requirement_id": "create", "task_quote": "Create Task A"}],
        }
    )
    assert asyncio.run(tool.executor(params)).success
    assert adapter.goal_tracker.checks["sender"].evidence_mode == "historical_observation"
    assert adapter.goal_tracker.checks["create"].evidence_mode == "current_state"
    adapter.evidence.close()


def test_rejected_paths_can_be_repaired_without_weakening_conditions(tmp_path):
    adapter = setup_state(tmp_path)
    source = read(adapter, {"data": [{"from": "facilities@example.com", "body": "Task A"}]})
    state = adapter.entity_tracker
    initial = proposal(source)
    rejected = asyncio.run(state.tool().executor(initial))
    diagnostic = json.loads(rejected.content)["diagnostic"]
    assert diagnostic["failures"][0]["anchor"]["candidate_paths"] == [["data", 0, "from"]]
    action = ProposedAction(
        tool="api_fetch",
        target_id=initial.target_id,
        arguments={"method": "POST", "url": "https://example.com/tasks"},
    )
    assert state.action_error(action)["status"] == "target_selection_required"
    retry = state.action_error(action)
    assert retry["status"] == "unchanged_blocked_retry" and retry["unchanged_attempts"] == 2
    c = initial.selection_checks[0]
    corrected = initial.model_copy(
        update={
            "record_path": ("data", 0),
            "selection_checks": (
                c.model_copy(
                    update={
                        "source_binding": c.source_binding.model_copy(
                            update={"anchor_path": ("data", 0, "from")}
                        ),
                        "observation": c.observation.model_copy(
                            update={"path": ("data", 0, "body")}
                        ),
                    }
                ),
            ),
        }
    )
    assert asyncio.run(state.tool().executor(corrected)).success
    assert state.action_error(action) is None
    assert adapter.goal_tracker.checks["sender"].observation.expected == "string"
    assert state.summary()["targets"][0]["effects"]["create"] == "missing_evidence"
    # An accepted target cannot be relocated to a different source.
    other = read(adapter, {"data": [{"from": "facilities@example.com", "body": "Task B"}]})
    replacement = corrected.model_copy(
        update={
            "source_episode_id": other.episode_id,
            "selection_checks": (
                corrected.selection_checks[0].model_copy(update={"episode_id": other.episode_id}),
            ),
        }
    )
    assert not asyncio.run(state.tool().executor(replacement)).success
    adapter.evidence.close()


def test_diagnostic_reports_mismatch_without_inventing_new_expectation(tmp_path):
    adapter = setup_state(tmp_path)
    source = read(adapter, {"from": "maintenance@example.com", "body": "Task A"})
    state = adapter.entity_tracker
    original = proposal(source)
    result = asyncio.run(state.tool().executor(original))
    failure = json.loads(result.content)["diagnostic"]["failures"][0]
    assert failure["anchor"]["value"] == "maintenance@example.com"
    weakened = original.model_copy(
        update={
            "selection_checks": (
                original.selection_checks[0].model_copy(
                    update={
                        "source_binding": SourceBinding(
                            anchor_text="maintenance", anchor_path=("from",)
                        )
                    }
                ),
            )
        }
    )
    assert not asyncio.run(state.tool().executor(weakened)).success
    assert state.targets[original.target_id] == original
    correct = read(adapter, {"from": "facilities@example.com", "body": "Task A"})
    assert asyncio.run(state.tool().executor(proposal(correct))).success
    adapter.evidence.close()


def test_missing_expected_value_is_actionable_schema_error(tmp_path):
    import pytest
    from pydantic import ValidationError

    from worldmodel_harness.task_state import SelectionCondition

    with pytest.raises(ValidationError, match="Supply observation.expected explicitly"):
        SelectionCondition(
            requirement_id="sender",
            task_quote="facilities email",
            anchor_text="facilities",
            anchor_path=("from",),
            observation={"path": ["from"], "predicate": "equals"},
        )


def test_execution_coverage_tracks_omitted_channels_and_warns(tmp_path):
    adapter = setup_state(tmp_path)
    state = adapter.entity_tracker
    state.requested_channels = ["sms", "email", "slack"]

    # Action 1: SMS to Wayne
    sms_ep = Episode(
        task_id="task",
        scope=adapter.scope,
        origin="environment",
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://api.twilio.com/2010-04-01/Accounts/AC1/Messages",
                   "body": {"To": "+15553001001", "Body": "BDG-3001 renewal"}},
        response={"sid": "SM1"},
        outcome="success",
    )
    state.record_action(sms_ep)

    # Action 2: Slack post
    slack_ep = Episode(
        task_id="task",
        scope=adapter.scope,
        origin="environment",
        operation="api_fetch",
        arguments={"method": "POST", "url": "https://slack.com/api/chat.postMessage",
                   "body": {"channel": "+15553001001", "text": "BDG-3001 renewal"}},
        response={"ok": True},
        outcome="success",
    )
    state.record_action(slack_ep)

    coverage = state.coverage_audit()
    assert coverage["records"][0]["entity"] == "+15553001001"
    assert coverage["records"][0]["completed"] == ["slack", "sms"]
    assert coverage["records"][0]["omitted"] == ["email"]

    # Finish attempt without unresolved should trigger warning
    result = adapter.goal_tracker.finish(CheckedFinish(summary="All done"))
    res_data = json.loads(result.content)
    assert res_data["status"] == "incomplete_effects"
    assert res_data["omitted_channels"] == ["email"]

    # Subsequent finish with unresolved=True succeeds
    res_unresolved = adapter.goal_tracker.finish(CheckedFinish(summary="Cannot email", unresolved=True))
    assert res_unresolved.success
    adapter.evidence.close()
