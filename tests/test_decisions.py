from datetime import timedelta

import pytest

from worldmodel_harness.decisions import (
    ActionRequirements,
    Decision,
    LiveRequirement,
    ProposedAction,
    check_decision,
)
from worldmodel_harness.knowledge import Claim, Dependency, Episode, LiveState, Scope, utc_now
from worldmodel_harness.store import KnowledgeStore


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="test", version="1", tenant="tenant", task_id="task")
    live = LiveState("task")
    episode = Episode(
        task_id="task",
        scope=scope,
        operation="lookup",
        arguments={},
        response={"ready": True},
        coverage=("ready",),
        outcome="success",
    )
    store.append_episode(episode)
    live.observe(episode)
    claim = Claim(
        claim_id="contract",
        layer="interface",
        relation="contract",
        participants=("write",),
        statement="write requires ready",
        scope=scope,
        status="supported",
        supporting_episodes=(episode.episode_id,),
        conditions={"lookup:ready": True},
    )
    store.put_claim(claim)
    yield store, scope, live, episode, claim
    store.close()


def check(world, *, mandatory=None, decision=None, **overrides):
    store, scope, live, _, _ = world
    args = {
        "store": store,
        "scope": scope,
        "live": live,
        "allowed_tools": {"write", "lookup"},
        "mandatory": mandatory
        or ActionRequirements(claims=(Dependency(claim_id="contract", version=1),)),
    }
    args.update(overrides)
    return check_decision(
        decision
        or Decision(
            goal="update",
            rationale="ready observation",
            action=ProposedAction(tool="write", arguments={}),
        ),
        **args,
    )


def test_supported_prerequisites_pass_without_mutating_store(world):
    store, *_ = world
    before = store.db.total_changes
    result = check(world)
    assert result.allowed
    assert result.checked_claims == ("contract",)
    assert result.checked_facts == ("lookup:ready",)
    assert store.db.total_changes == before


def test_frozen_decision_accepts_current_evidence_and_historical_evidence_separately(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    scope = Scope(environment="test", version="1", tenant="tenant", task_id="current")
    knowledge = KnowledgeStore(path)
    historical = Episode(
        task_id="old",
        scope=scope.model_copy(update={"task_id": "old"}),
        operation="read",
        arguments={},
        response={},
        outcome="success",
    )
    knowledge.append_episode(historical)
    knowledge.close()
    knowledge = KnowledgeStore(path, frozen=True)
    evidence = KnowledgeStore(tmp_path / "evidence.sqlite")
    current = Episode(
        task_id="current",
        scope=scope,
        operation="read",
        arguments={},
        response={"ready": True},
        coverage=("ready",),
        outcome="success",
    )
    evidence.append_episode(current)
    live = LiveState("current")
    live.observe(current)
    decision = Decision(
        goal="act",
        rationale="use current and historical evidence",
        action=ProposedAction(tool="write", arguments={}),
        evidence_ids=(historical.episode_id, current.episode_id),
    )
    kwargs = {
        "mandatory": ActionRequirements(),
        "store": knowledge,
        "evidence_store": evidence,
        "scope": scope,
        "live": live,
        "allowed_tools": {"write"},
    }
    assert check_decision(decision, **kwargs).allowed
    with pytest.raises(KeyError):
        knowledge.episode(current.episode_id)
    assert not check_decision(
        decision.model_copy(update={"evidence_ids": ("invented",)}), **kwargs
    ).allowed
    other = Episode(
        task_id="other",
        scope=scope.model_copy(update={"tenant": "other"}),
        operation="read",
        arguments={},
        response={},
        outcome="success",
    )
    evidence.append_episode(other)
    assert not check_decision(
        decision.model_copy(update={"evidence_ids": (other.episode_id,)}), **kwargs
    ).allowed
    assert knowledge.db.total_changes == 0
    knowledge.close()
    evidence.close()


def test_model_omission_cannot_waive_mandatory_live_requirement(world):
    world[2].invalidate({"lookup:ready"}, "write")
    result = check(world)
    assert not result.allowed
    assert "refresh:lookup:ready" in result.gaps


@pytest.mark.parametrize(
    "change,gap",
    [
        ({"status": "disputed"}, "dependency:contract@1"),
        ({"valid_until": utc_now() - timedelta(seconds=1)}, "refresh_claim:contract"),
    ],
)
def test_recheck_after_retrieval_blocks_stale_or_disputed_claim(world, change, gap):
    store, _, _, _, claim = world
    store.put_claim(claim.model_copy(update={"version": 2, **change}))
    assert gap in check(world).gaps


def test_dependency_closure_checks_missing_claim_not_in_context(world):
    store, _, _, _, claim = world
    store.put_claim(
        claim.model_copy(
            update={
                "version": 2,
                "dependencies": (Dependency(claim_id="permission", version=1),),
            }
        )
    )
    result = check(
        world, mandatory=ActionRequirements(claims=(Dependency(claim_id="contract", version=2),))
    )
    assert not result.allowed
    assert "missing_claim:permission" in result.gaps


def test_unknown_result_is_not_valid_live_evidence(world):
    store, _, live, episode, _ = world
    uncertain = episode.model_copy(
        update={
            "episode_id": "uncertain",
            "event_id": "uncertain",
            "outcome": "unknown",
        }
    )
    store.append_episode(uncertain)
    live.observe(uncertain)
    assert "untrusted_live:lookup:ready" in check(world).gaps


def test_type_sensitive_prerequisite_does_not_equate_true_and_one(world):
    result = check(
        world,
        mandatory=ActionRequirements(facts=(LiveRequirement(key="lookup:ready", expected=1),)),
    )
    assert "prerequisite:lookup:ready" in result.gaps


def test_monitoring_disputed_claim_does_not_assume_it_true(world):
    store, _, _, _, claim = world
    store.put_claim(claim.model_copy(update={"version": 2, "status": "disputed"}))
    result = check(
        world,
        mandatory=ActionRequirements(
            claims=(Dependency(claim_id="contract", version=1, mode="monitors"),)
        ),
    )
    assert result.allowed


def test_wrong_task_and_unregistered_tool_rejected(world):
    result = check(world, live=LiveState("other"), allowed_tools={"lookup"})
    assert "scope:live_state" in result.gaps
    assert "tool:write" in result.gaps


def test_invented_evidence_reference_is_not_accepted(world):
    result = check(
        world,
        decision=Decision(
            goal="update",
            rationale="unsupported citation",
            evidence_ids=("invented",),
            action=ProposedAction(tool="write", arguments={}),
        ),
    )
    assert "evidence:invented" in result.gaps


def test_compact_action_accepts_only_schema_valid_serialized_objects():
    import json

    import pytest
    from pydantic import ValidationError

    from worldmodel_harness.decisions import CompactDecision

    action = {"tool": "api_fetch", "arguments": {"params": '{"literal":"keep as string"}'}}
    parsed = CompactDecision(action=json.dumps(action))
    assert parsed.action.arguments == action["arguments"]
    for invalid in ("not json", "[]", "null", '{"tool":"x","arguments":{},"extra":true}'):
        with pytest.raises(ValidationError):
            CompactDecision(action=invalid)


def test_compact_decision_hoists_nested_goal_and_allows_omitted_tool():
    from worldmodel_harness.decisions import CompactDecision

    payload = {
        "action": {
            "arguments": {
                "method": "GET",
                "url": "https://api.airtable.com/v0/meta/bases",
            },
            "goal": "List all Airtable bases to find the one containing Q1 P&L and Targets.",
        },
        "output_expectations": [
            {
                "path": "bases",
                "type": "array",
            }
        ],
    }

    decision = CompactDecision.model_validate(payload)
    assert decision.goal == "List all Airtable bases to find the one containing Q1 P&L and Targets."
    assert decision.action.tool == ""
    assert decision.action.arguments["method"] == "GET"
    assert decision.action.arguments["url"] == "https://api.airtable.com/v0/meta/bases"
    assert decision.output_expectations[0].path == ("bases",)
    assert decision.output_expectations[0].predicate == "type"
    assert decision.output_expectations[0].expected == "array"


def test_compact_decision_absorbs_misplaced_parameter_siblings():
    from worldmodel_harness.decisions import CompactDecision

    payload = {
        "action": {
            "arguments": {
                "method": "PUT",
                "params": '{"valueInputOption": "USER_ENTERED"}',
                "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_candidates/values/F2:F6",
            },
            "body": '{"values": [["Duplicate"]]}',
            "tool": "api_fetch",
        },
        "goal": "Flag duplicate candidates in the spreadsheet.",
    }

    decision = CompactDecision.model_validate(payload)
    assert decision.action.tool == "api_fetch"
    assert decision.action.arguments["method"] == "PUT"
    assert decision.action.arguments["body"] == '{"values": [["Duplicate"]]}'
    assert decision.action.arguments["url"].endswith("F2:F6")


def test_compact_decision_absorbs_root_tool_and_preserves_name():
    from worldmodel_harness.decisions import CompactDecision

    payload = {
        "action": {
            "arguments": {
                "method": "GET",
                "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_incident_queue/values/ws_queue!A1:Z100",
            },
            "goal": "Read the incident queue to find the P1 incident details.",
        },
        "tool": "a2:api_fetch",
    }

    decision = CompactDecision.model_validate(payload)
    assert decision.action.tool == "a2:api_fetch"
    assert decision.action.arguments["method"] == "GET"
    assert decision.goal == "Read the incident queue to find the P1 incident details."


def test_compact_decision_preserves_both_tool_declarations_for_registry():
    import json
    from worldmodel_harness.decisions import CompactDecision

    # 1. Bare string action preserves root_tool and action.tool
    cd1 = CompactDecision.model_validate({"tool": "api_fetch", "action": "api_search"})
    assert cd1.root_tool == "api_fetch"
    assert cd1.action.tool == "api_search"

    # 2. Serialized arguments-only action inherits root_tool
    cd2 = CompactDecision.model_validate({
        "tool": "api_fetch",
        "action": json.dumps({"arguments": {"method": "GET"}}),
    })
    assert cd2.root_tool == "api_fetch"
    assert cd2.action.tool == "api_fetch"
    assert cd2.action.arguments["method"] == "GET"

    # 3. Serialized action with conflicting tool preserves both
    cd3 = CompactDecision.model_validate({
        "tool": "api_search",
        "action": json.dumps({"tool": "api_fetch", "arguments": {"method": "DELETE"}}),
    })
    assert cd3.root_tool == "api_search"
    assert cd3.action.tool == "api_fetch"

    # 4. Equivalent alias preserves both declarations
    cd4 = CompactDecision.model_validate({
        "tool": "a2:api_fetch",
        "action": {"tool": "api_fetch", "arguments": {}},
    })
    assert cd4.root_tool == "a2:api_fetch"
    assert cd4.action.tool == "api_fetch"


