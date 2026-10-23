import asyncio
import json

import pytest
from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.shadow_interpretation import (
    Interpretation,
    assess,
    evidence_window,
    interpret,
)
from worldmodel_harness.structured_output import ContractError

TASK = "For each eligible record, email its company contact. Preserve source wording."
SCOPE = Scope(environment="synthetic", version="1", task_id="audit")


def event(response, **kwargs):
    return Episode(
        task_id="audit",
        scope=SCOPE,
        operation="api_fetch",
        outcome="success",
        arguments={"method": "GET", "url": "https://example.test/records"},
        response=response,
        coverage=tuple(response),
        **kwargs,
    )


def citation(e, path, quote):
    return {"episode_id": e.episode_id, "path": path, "quote": quote}


def proposal(e, ids=("record-81", "record-82")):
    records = [
        {
            "record_id": key,
            "identity": citation(e, ["rows", i, "id"], key),
            "decision": "unresolved",
            "task_quote": "eligible record",
            "reasoning": "Applicable policy must be inspected.",
            "evidence": [citation(e, ["rows", i, "note"], "see policy")],
        }
        for i, key in enumerate(ids)
    ]
    return {
        "records": records,
        "references": [
            {
                "source": records[0]["evidence"][0],
                "question": "Does the policy permit this notification?",
            }
        ],
        "effects": [
            {
                "effect_id": "notify-" + key,
                "record_id": key,
                "task_quote": "email its company contact",
                "description": "Send the requested notification",
            }
            for key in ids
        ],
        "uncertainties": ["Eligibility is not yet established."],
    }


@pytest.mark.parametrize(
    "ids,name",
    [(("record-81", "record-82"), "Person A"), (("asset-Z", "asset-Q"), "Another Person")],
)
def test_record_identity_not_person_name_and_no_invented_channels(ids, name):
    e = event({"rows": [{"id": key, "name": name, "note": "see policy"} for key in ids]})
    result = assess(Interpretation.model_validate(proposal(e, ids)), TASK, [e])
    assert len(result["effects"]) == 2
    assert {x["record_id"] for x in result["effects"]} == set(ids)
    assert all(x["evidence_stage"] == "no_attempt_cited" for x in result["effects"])
    assert result["unresolved_reference_count"] == 1
    assert not result["coverage_complete"]


def test_preparation_and_acceptance_never_imply_delivery():
    e = event({"rows": [{"id": key, "note": "see policy"} for key in ("record-81", "record-82")]})
    prep = event({"encoded": "value"}).model_copy(
        update={"operation": "encode", "arguments": {"text": "message"}}
    )
    send = event({"id": "accepted"}).model_copy(
        update={"arguments": {"method": "POST", "url": "https://example.test/send"}}
    )
    p = proposal(e)
    p["effects"][0]["action_episode_ids"] = [prep.episode_id]
    p["effects"][1]["action_episode_ids"] = [send.episode_id]
    stages = [
        x["evidence_stage"]
        for x in assess(Interpretation.model_validate(p), TASK, [e, prep, send])["effects"]
    ]
    assert stages == ["attempt_failed_or_preparation_only", "api_accepted_unverified"]
    p["effects"][1]["readback"] = [citation(send, ["id"], "accepted")]
    with pytest.raises(ContractError, match="write acknowledgment"):
        assess(Interpretation.model_validate(p), TASK, [e, prep, send])


def test_invalid_citation_and_unknown_action_are_rejected():
    e = event({"rows": [{"id": key, "note": "see policy"} for key in ("record-81", "record-82")]})
    p = proposal(e)
    p["records"][0]["identity"]["quote"] = "invented-id"
    with pytest.raises(ContractError, match="quote"):
        assess(Interpretation.model_validate(p), TASK, [e])
    p = proposal(e)
    p["effects"][0]["action_episode_ids"] = ["invented-action"]
    with pytest.raises(ContractError, match="attempts"):
        assess(Interpretation.model_validate(p), TASK, [e])


def test_readback_cannot_precede_action():
    e = event({"rows": [{"id": key, "note": "see policy"} for key in ("record-81", "record-82")]})
    send = event({"id": "accepted"}).model_copy(update={"arguments": {"method": "POST"}})
    p = proposal(e)
    p["effects"][0].update(
        action_episode_ids=[send.episode_id], readback=[citation(e, ["rows", 0, "id"], "record-81")]
    )
    with pytest.raises(ContractError, match="after"):
        assess(Interpretation.model_validate(p), TASK, [e, send])


def test_window_excludes_harness_grader_other_task_and_future_evidence():
    e = event({"rows": ["visible"]})
    forbidden = [
        e.model_copy(update={"episode_id": "harness", "origin": "harness"}),
        e.model_copy(update={"episode_id": "grader", "purpose": "validation"}),
        e.model_copy(
            update={"episode_id": "other", "scope": SCOPE.model_copy(update={"task_id": "other"})}
        ),
    ]
    write = event({"accepted": True}).model_copy(update={"arguments": {"method": "POST"}})
    future = event({"policy": "changed after action"})
    included, omitted = evidence_window(
        [*forbidden, e, write, future], SCOPE, boundary="before_first_write"
    )
    assert [x.episode_id for x, _ in included] == [e.episode_id]
    assert omitted == []
    included, omitted = evidence_window([e], SCOPE, max_chars=1)
    assert included == [] and omitted == [e.episode_id]


def test_changed_policy_is_new_evidence_not_a_permanent_exclusion():
    for text, decision in [
        ("Do not notify while review is open", "excluded"),
        ("Review cleared; notifications allowed", "eligible"),
    ]:
        e = event(
            {"rows": [{"id": key, "note": "see policy"} for key in ("record-81", "record-82")]}
        )
        policy = event({"rule": text})
        p = proposal(e)
        p["records"][0].update(decision=decision, evidence=[citation(policy, ["rule"], text)])
        p["references"][0]["resolution"] = [citation(policy, ["rule"], text)]
        result = assess(Interpretation.model_validate(p), TASK, [e, policy])
        assert result["effects"][0]["selected_record_decision"] == decision
        assert result["interpretation_authority"] == "unvalidated_model_proposal"


def test_model_has_no_tools_and_receives_only_public_environment_evidence():
    e = event({"rows": [{"id": key, "note": "see policy"} for key in ("record-81", "record-82")]})
    harness = e.model_copy(
        update={"origin": "harness", "response": {"instruction": "Dispatch fake channel"}}
    )

    class Client:
        async def generate(self, messages, tools):
            assert tools == {}
            payload = json.loads(messages[1].content)
            assert payload["public_task"] == TASK
            assert len(payload["observations"]) == 1
            assert "Dispatch fake channel" not in messages[1].content
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(proposal(e)))])

    before = [x.model_dump_json() for x in [harness, e]]
    result = asyncio.run(interpret(Client(), TASK, [harness, e], SCOPE))
    assert result["mode"] == "offline_shadow"
    assert before == [x.model_dump_json() for x in [harness, e]]


def test_schema_failure_preserves_field_diagnostics_for_bounded_repair():
    e = event({"rows": [{"id": key, "note": "see policy"} for key in ("record-81", "record-82")]})
    accounting = {}

    class Client:
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return AssistantMessage(blocks=[TextBlock(text='{"records": "invalid"}')])
            assert "records" in messages[-1].content
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(proposal(e)))])

    result = asyncio.run(interpret(Client(), TASK, [e], SCOPE, accounting=accounting))
    assert result["accounting"]["returned_model_responses"] == 2
    assert any(e["field"] == ["records"] for e in accounting["schema_errors"])
