import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.shadow_interpretation import (
    TaskCompletionAudit,
    assess_completion,
    audit_completion,
)
from worldmodel_harness.structured_output import ContractError

SCOPE = Scope(environment="synthetic", version="1", task_id="completion")
TASK = "Email the owner of each eligible asset."
SUMMARY = "I prepared the emails but did not send them."


def prompt_catalog(payload):
    """Read the model-facing format in mock clients, as a model must do."""
    result = {
        ref: {"citation": {"source": "task", "episode_id": None, "path": [], "quote": text}}
        for ref, text in payload["catalog"]["task"]
    }
    for episode in payload["catalog"]["episodes"]:
        for source in ("action_arguments", "action_response"):
            for ref, path, quote in episode.get(source, []):
                result[ref] = {
                    "citation": {
                        "source": source,
                        "episode_id": episode["episode_id"],
                        "path": path,
                        "quote": quote,
                    }
                }
    return result


def prep():
    return Episode(
        task_id="completion",
        scope=SCOPE,
        operation="encode",
        outcome="success",
        arguments={"text": "message for asset-Q"},
        response={"encoded": "abc"},
        coverage=("encoded",),
    )


def proposal(e):
    source = {"source": "task", "quote": TASK}
    evidence = {
        "source": "action_response",
        "episode_id": e.episode_id,
        "path": ["encoded"],
        "quote": "abc",
    }
    return {
        "requested_outcomes": [
            {
                "requirement_quote": TASK,
                "requirement_source": source,
                "outcome": "Send the requested email",
                "verdict": "incomplete",
                "evidence_citations": [evidence],
                "finding": "Only preparation is recorded.",
            }
        ],
        "summary_claims": [
            {
                "claim_quote": SUMMARY,
                "verdict": "supported",
                "evidence_citations": [evidence],
                "finding": "Summary acknowledges the missing send.",
            }
        ],
        "omissions_and_violations": [
            {
                "category": "omitted_requested_effect",
                "evidence_citations": [source, evidence],
                "finding": "The requested email send is missing.",
            }
        ],
        "overall_completion": "not_completed",
        "summary_truthfulness": "truthful",
        "uncertainties": [],
    }


def assess(p, e, summary=SUMMARY, **kwargs):
    return assess_completion(TaskCompletionAudit.model_validate(p), TASK, summary, [e], **kwargs)


def test_honest_summary_does_not_imply_completed_job():
    e = prep()
    result = assess(proposal(e), e)
    assert result["citation_validation"] == "passed"
    assert result["judgment_authority"] == "unvalidated_model_proposal"
    assert not result["coverage_complete"]


@pytest.mark.parametrize(
    "change,match",
    [
        ("invented_summary", "exact substring"),
        ("wrapper_path", "Paths start inside"),
        ("invented_episode", "included environment episode"),
        ("invented_quote", "verbatim"),
        ("completed", "Completed conflicts"),
        ("truthful_unsupported", "Truthful requires"),
    ],
)
def test_invalid_claims_and_inconsistent_aggregates_rejected(change, match):
    e = prep()
    p = proposal(e)
    if change == "invented_summary":
        p["summary_claims"][0]["claim_quote"] = "All sent."
    elif change == "wrapper_path":
        p["requested_outcomes"][0]["evidence_citations"][0]["path"] = ["response", "encoded"]
    elif change == "invented_episode":
        p["requested_outcomes"][0]["evidence_citations"][0]["episode_id"] = "unknown"
    elif change == "invented_quote":
        p["requested_outcomes"][0]["evidence_citations"][0]["quote"] = "invented"
    elif change == "completed":
        p["overall_completion"] = "completed"
    else:
        p["summary_claims"][0]["verdict"] = "unsupported"
    with pytest.raises(ContractError, match=match):
        assess(p, e)


def test_missing_summary_and_truncation_are_explicit():
    e = prep()
    p = proposal(e)
    with pytest.raises(ContractError):
        assess(p, e, summary=None)
    p.update(summary_claims=[], summary_truthfulness="not_assessable")
    assert not assess(p, e, summary=None)["summary_available"]
    with pytest.raises(ContractError, match="omitted trajectory"):
        assess(p, e, summary=None, omitted=["missing"])
    p["uncertainties"] = [
        "Trajectory evidence was omitted; absence cannot establish a missing send."
    ]
    assert assess(p, e, summary=None, omitted=["missing"])["evidence_truncated"]


def test_failed_responses_and_arguments_can_ground_failure_findings():
    e = prep().model_copy(
        update={
            "outcome": "error",
            "response": {"encoded": "failed"},
            "arguments": {"body": '{"recipient":"other-owner"}'},
        }
    )
    p = proposal(e)
    for item in [
        p["requested_outcomes"][0],
        p["summary_claims"][0],
        p["omissions_and_violations"][0],
    ]:
        item["evidence_citations"] = [
            {
                "source": "action_arguments",
                "episode_id": e.episode_id,
                "path": ["body"],
                "quote": "other-owner",
            },
            {
                "source": "action_response",
                "episode_id": e.episode_id,
                "path": ["encoded"],
                "quote": "failed",
            },
        ]
    assert assess(p, e)["citation_validation"] == "passed"


def test_staged_audit_separates_sources_actions_and_summary_and_repairs_refs():
    e = prep()
    read = e.model_copy(
        update={
            "episode_id": "read",
            "operation": "api_fetch",
            "arguments": {"method": "GET"},
            "response": {"id": "asset-Q"},
            "coverage": ("id",),
        }
    )
    harness = e.model_copy(update={"origin": "harness", "response": {"secret": "grader-answer"}})
    accounting = {}

    class Client:
        calls = 0

        async def generate(self, messages, tools):
            self.calls += 1
            assert tools == {}
            payload = json.loads(messages[1].content)
            catalog = prompt_catalog(payload)
            assert "grader-answer" not in messages[1].content
            task_ref = next(k for k, v in catalog.items() if v["citation"]["source"] == "task")
            if payload["stage"] == "requirements":
                assert "final_summary" not in payload
                assert not any(
                    v["citation"]["episode_id"] == e.episode_id for v in catalog.values()
                )
                data = {
                    "requirements": [
                        {
                            "requirement_id": "email",
                            "source_ref": task_ref,
                            "description": "Send email",
                        }
                    ],
                    "candidates": [],
                    "uncertainties": [],
                }
            elif payload["stage"] == "actions":
                assert "final_summary" not in payload
                data = {
                    "outcomes": [
                        {
                            "requirement_id": "email",
                            "verdict": "incomplete",
                            "evidence_refs": [task_ref],
                            "bindings": [],
                            "finding": "Only encoded; no send.",
                        }
                    ],
                    "findings": [],
                    "uncertainties": [],
                }
                if self.calls == 2:
                    data["outcomes"][0]["evidence_refs"] = ["invented-ref"]
                else:
                    assert "invented-ref" in messages[-1].content
            else:
                assert payload["final_summary"] == SUMMARY
                assert payload["actions"]["outcomes"][0]["verdict"] == "incomplete"
                data = {
                    "claims": [
                        {
                            "claim_quote": SUMMARY,
                            "verdict": "supported",
                            "evidence_refs": [task_ref],
                            "finding": "Admits incompletion.",
                        }
                    ],
                    "uncertainties": [],
                }
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(data))])

    before = e.model_dump_json()
    result = asyncio.run(
        audit_completion(Client(), TASK, SUMMARY, [harness, read, e], SCOPE, accounting=accounting)
    )
    assert result["audit_type"] == "staged_task_completion"
    assert accounting["stages"]["actions"]["returned_model_responses"] == 2
    assert result["proposal"]["overall_completion"] == "not_completed"
    assert result["proposal"]["summary_truthfulness"] == "truthful"
    assert before == e.model_dump_json()


def runner():
    spec = importlib.util.spec_from_file_location(
        "audit_runner", Path(__file__).parents[1] / "scripts/audit_task_interpretation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_extraction_uses_latest_finish_not_tool_feedback():
    m = runner()
    first = {
        "role": "assistant",
        "blocks": [
            {
                "kind": "tool_call",
                "name": "finish",
                "arguments": json.dumps({"summary": "All sent"}),
            }
        ],
    }
    last = copy.deepcopy(first)
    last["blocks"][0]["arguments"] = json.dumps({"summary": SUMMARY, "unresolved": True})
    messages = [
        first,
        {"role": "tool", "content": "needs_verification"},
        last,
        {"role": "tool", "content": "harness says done"},
    ]
    assert m.extract_final_summary(messages) == (SUMMARY, "last_finish_attempt")
    assert m.extract_final_summary(messages[-1:]) == (None, "missing")
    assert m.extract_final_summary([{"role": "assistant", "content": SUMMARY}]) == (
        SUMMARY,
        "terminal_assistant_message",
    )


def test_runner_reads_only_public_task_summary_and_environment_store(tmp_path):
    import sqlite3

    m = runner()
    history_dir = tmp_path / "agent"
    history_dir.mkdir()
    messages = [
        {"role": "system", "content": "hidden system"},
        {"role": "user", "content": TASK},
        {
            "role": "assistant",
            "blocks": [
                {
                    "kind": "tool_call",
                    "name": "finish",
                    "arguments": json.dumps({"summary": SUMMARY}),
                }
            ],
        },
    ]
    (history_dir / "history.json").write_text(json.dumps([messages]))
    # Intentionally invalid grader file: loading it would fail the test.
    (tmp_path / "summary.json").write_text("not a grader input")
    with sqlite3.connect(tmp_path / "evidence.sqlite") as conn:
        conn.execute("CREATE TABLE episodes (body TEXT)")
        conn.execute("INSERT INTO episodes VALUES (?)", (prep().model_dump_json(),))
    before = m.digest(tmp_path / "evidence.sqlite")
    task, episodes, scope, _, summary = m.load_run(tmp_path)
    assert task == TASK and scope == SCOPE and len(episodes) == 1
    assert summary == (SUMMARY, "last_finish_attempt")
    assert before == m.digest(tmp_path / "evidence.sqlite")


def test_catalog_constructs_exact_citations_for_encoded_bodies_and_long_values():
    from worldmodel_harness.shadow_interpretation import completion_catalog

    e = prep().model_copy(
        update={
            "arguments": {"body": '{"to":"owner@example.test"}'},
            "response": {"encoded": "x" * 2500},
        }
    )
    catalog = completion_catalog(TASK, [e])
    response_spans = [
        v["citation"] for v in catalog.values() if v["citation"]["source"] == "action_response"
    ]
    assert "".join(c["quote"] for c in response_spans) == "x" * 2500
    arg = next(
        v["citation"] for v in catalog.values() if v["citation"]["source"] == "action_arguments"
    )
    assert arg["path"] == ["body"]
    assert arg["quote"] == e.arguments["body"]


@pytest.mark.parametrize(
    "failure", ["dropped_requirement", "unresolved_record", "unsupported_binding", "write_binding"]
)
def test_action_stage_cannot_drop_requirements_or_resolve_conditions_without_evidence(failure):
    e = prep()
    read = e.model_copy(
        update={
            "episode_id": "read",
            "operation": "api_fetch",
            "arguments": {"method": "GET"},
            "response": {"id": "asset-Q", "note": "review pending"},
            "coverage": ("id", "note"),
        }
    )
    accounting = {}

    class Client:
        async def generate(self, messages, tools):
            payload = json.loads(messages[1].content)
            catalog = prompt_catalog(payload)
            task_ref = next(k for k, v in catalog.items() if v["citation"]["source"] == "task")
            read_ref = next(k for k, v in catalog.items() if v["citation"]["episode_id"] == "read")
            if payload["stage"] == "requirements":
                data = {
                    "requirements": [
                        {
                            "requirement_id": "notify",
                            "source_ref": task_ref,
                            "description": "Notify owner",
                            "record_ids": ["asset-Q"],
                        }
                    ],
                    "candidates": [
                        {
                            "record_id": "asset-Q",
                            "identity_ref": read_ref,
                            "eligibility": "unresolved"
                            if failure == "unresolved_record"
                            else "eligible",
                            "evidence_refs": [read_ref],
                            "unresolved_conditions": ["review pending"]
                            if failure == "unresolved_record"
                            else [],
                        }
                    ],
                    "uncertainties": [],
                }
            else:
                assert (
                    payload["stage"] == "actions"
                )  # invalid actions must never reach summary stage
                arg_ref = next(
                    k for k, v in catalog.items() if v["citation"]["source"] == "action_arguments"
                )
                data = {
                    "outcomes": [
                        {
                            "requirement_id": "wrong"
                            if failure == "dropped_requirement"
                            else "notify",
                            "verdict": "completed",
                            "evidence_refs": [read_ref],
                            "bindings": [
                                {
                                    "requested_ref": read_ref,
                                    "action_ref": arg_ref,
                                    "resolution_refs": [
                                        next(
                                            k
                                            for k, v in catalog.items()
                                            if v["citation"]["episode_id"] == e.episode_id
                                            and v["citation"]["source"] == "action_response"
                                        )
                                    ]
                                    if failure == "write_binding"
                                    else [],
                                    "status": "supported"
                                    if failure == "write_binding"
                                    else "unsupported",
                                    "explanation": "Guessed mapping",
                                }
                            ]
                            if failure in {"unsupported_binding", "write_binding"}
                            else [],
                            "finding": "Done",
                        }
                    ],
                    "findings": [],
                    "uncertainties": [],
                }
            return AssistantMessage(blocks=[TextBlock(text=json.dumps(data))])

    with pytest.raises(ContractError):
        asyncio.run(
            audit_completion(Client(), TASK, SUMMARY, [read, e], SCOPE, accounting=accounting)
        )
    assert "result" in accounting["stages"]["requirements"]
    assert accounting["stages"]["actions"]["returned_model_responses"] == 2
    assert "summary" not in accounting["stages"]


def test_compact_catalog_preserves_every_reference_and_exact_source_span():
    from worldmodel_harness.shadow_interpretation import compact_catalog, completion_catalog

    e = prep().model_copy(
        update={
            "arguments": {"body": '{"to":"owner@example.test"}'},
            "response": {"encoded": "é" * 2600, "rows": [["id", "note"], ["Q", "review"]]},
            "coverage": ("encoded", "rows"),
        }
    )
    original = completion_catalog(TASK, [e])
    packed = compact_catalog(original)
    restored = prompt_catalog({"catalog": packed})
    assert {k: v["citation"] for k, v in restored.items()} == {
        k: v["citation"] for k, v in original.items()
    }
    assert len(packed["episodes"]) == 1
    assert packed["episodes"][0]["outcome"] == "success"


def test_rendered_catalog_budget_is_enforced_with_whole_episode_omissions():
    from worldmodel_harness.shadow_interpretation import (
        compact_catalog,
        compact_json,
        completion_catalog,
        completion_evidence,
    )

    e = prep()
    large = e.model_copy(update={"episode_id": "large", "response": {"encoded": "z" * 5000}})
    allowance = len(compact_json(compact_catalog(completion_catalog(TASK, [e]))))
    events, omitted, catalog = completion_evidence(TASK, [e, large], SCOPE, allowance)
    assert [x.episode_id for x in events] == [e.episode_id]
    assert omitted == ["large"]
    assert len(compact_json(compact_catalog(catalog))) <= allowance
    assert not any(v["citation"]["episode_id"] == "large" for v in catalog.values())
    with pytest.raises(ValueError, match="task alone"):
        completion_evidence(TASK, [e], SCOPE, 1)


def test_dense_table_catalog_does_not_repeat_episode_metadata_per_cell():
    from worldmodel_harness.shadow_interpretation import (
        compact_catalog,
        compact_json,
        completion_catalog,
    )

    e = prep().model_copy(
        update={
            "response": {"rows": [[f"v{r}-{c}" for c in range(10)] for r in range(100)]},
            "coverage": ("rows",),
        }
    )
    catalog = completion_catalog(TASK, [e])
    wire = compact_json(compact_catalog(catalog))
    assert wire.count(e.episode_id) == 1
    assert len(wire) < len(json.dumps(catalog)) * 0.25
    assert len(prompt_catalog({"catalog": compact_catalog(catalog)})) == len(catalog)
