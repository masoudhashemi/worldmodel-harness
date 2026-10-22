"""Model-proposed distinguishing actions; observed outcomes never imply causality."""

import json
from typing import Literal

from pydantic import Field, TypeAdapter
from stirrup.core.models import SystemMessage, UserMessage

from .knowledge import OutputExpectation, Record
from .observable_predictions import compare_expectations
from .semantic_learning import BlockerConditions
from .structured_output import generate_checked


class RecoveryExperiment(Record):
    explanation: str = Field(min_length=1, max_length=600)
    tool: str
    arguments: dict
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    alternatives: tuple[str, str]
    prediction_target: Literal["harness_response", "tool_result"] = Field(
        default="harness_response", description="Root for prediction paths: harness_response is the full wrapper; "
        "tool_result is its parsed result member. Raw environment observations are not the assessment target.")
    predictions: tuple[OutputExpectation, OutputExpectation]


class RecoveryPause(Record):
    kind: Literal["pause"]
    explanation: str = Field(min_length=1, max_length=600)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    blocker: BlockerConditions
    remaining_alternatives: tuple[str, ...] = Field(min_length=1, max_length=6)


def canonical_arguments(tool, arguments):
    return tool.parameters.model_validate(arguments).model_dump(mode="json")


async def propose_experiment(adapter, client, episodes, hypotheses, accounting):
    tools = adapter.public_tools
    selected, size = [], 0
    for event in reversed(episodes):
        cost = len(event.model_dump_json())
        if size + cost <= 18000:
            selected.insert(0, event)
            size += cost
    if not selected:
        raise ValueError("No complete observations fit the experiment allowance")
    references = {f"observation_{i + 1}": e for i, e in enumerate(selected)}
    observations = [{"evidence_ref": ref, **event.model_dump(mode="json", include={
        "origin", "operation", "arguments", "response", "outcome", "coverage",
        "environment_called", "effect_kind",
    })} for ref, event in references.items()]
    decision_type = TypeAdapter(RecoveryExperiment | RecoveryPause)
    schema = decision_type.json_schema()
    for name in ("RecoveryExperiment", "RecoveryPause"):
        schema["$defs"][name]["properties"]["evidence_ids"]["items"] = {
            "type": "string", "enum": list(references)}

    def validate(data):
        proposal = decision_type.validate_python(data)
        if not set(proposal.evidence_ids) <= references.keys():
            raise ValueError("Use supplied tools and evidence only")
        resolved = tuple(references[r].episode_id for r in proposal.evidence_ids)
        if isinstance(proposal, RecoveryPause):
            from .contrastive import is_unproductive_episode

            if proposal.blocker.origin == "environment" and not any(
                references[r].origin == "environment"
                and (
                    references[r].outcome in {"error", "blocked"}
                    or references[r].purpose == "discovery"
                    or references[r].operation == "api_search"
                    or is_unproductive_episode(references[r])[0]
                )
                and references[r].environment_called is not False for r in proposal.evidence_ids
            ):
                raise ValueError("Environment blockers require environment failure evidence or uninformative discovery")
            return proposal.model_copy(update={"evidence_ids": resolved})
        if proposal.tool not in tools:
            raise ValueError("Use supplied tools only")
        args = canonical_arguments(tools[proposal.tool], proposal.arguments)
        if (proposal.alternatives[0] == proposal.alternatives[1]
            or proposal.predictions[0] == proposal.predictions[1]
            or proposal.predictions[0].path != proposal.predictions[1].path
            or any(p.predicate != "equals" for p in proposal.predictions)):
            raise ValueError("Require distinct alternatives and distinguishable predictions")
        for event in episodes:
            if event.operation == "tool_call:" + proposal.tool:
                try:
                    prior = canonical_arguments(tools[proposal.tool], event.arguments)
                except ValueError:
                    continue
                if prior == args:
                    raise ValueError("A diagnostic proposal must change the observed action")
        return proposal.model_copy(update={"arguments": args,
            "evidence_ids": resolved})

    return await generate_checked(client, [SystemMessage(content=
        "Choose a distinguishing next action or a scoped pause after stagnation. Use only public observations. "
        "Choose kind=pause when available evidence suggests another attempt adds no information under "
        "current conditions; explain stop and reopening conditions and remaining alternatives. "
        "A pause concerns the blocked branch, not completion or universal impossibility; continue "
        "unaffected work. Repetition alone does not prove a blocker. "
        "Diagnose alternatives and select a schema-valid action that will distinguish them. "
        "Do not repeat the observed action, invent IDs, declare completion or retry an uncertain "
        "write. Set prediction_target explicitly: tool_result for paths relative to parsed tool content, "
        "or harness_response for paths relative to the entire wrapper (result holds parsed tool content). "
        "The assessment always uses the next matching tool-call wrapper, not a raw environment episode. "
        "Predict two different values at the same path using equals, one for each "
        "alternative. Cite only observation evidence_ref values in evidence_ids. Changing actions is an experiment, not proof of a remedy. Return JSON."),
        UserMessage(content=json.dumps({"schema": schema,
            "public_task": adapter.task.prompt, "hypotheses": hypotheses,
            "observations": observations,
            "tools": {name: {"description": t.description,
                "parameters": t.parameters.model_json_schema()} for name, t in tools.items()}}))],
        validate, accounting)


def assess_experiment(adapter, proposal, event):
    if event.operation != "tool_call:" + proposal.tool:
        return {"result": "different_action", "episode_id": event.episode_id}
    try:
        args = canonical_arguments(adapter.public_tools[proposal.tool], event.arguments)
    except ValueError:
        return {"result": "invalid_action", "episode_id": event.episode_id}
    if args != proposal.arguments:
        return {"result": "different_arguments", "episode_id": event.episode_id}
    predictions = proposal.predictions
    if proposal.prediction_target == "tool_result":
        predictions = tuple(p.model_copy(update={"path": ("result", *p.path)}) for p in predictions)
    checks = compare_expectations(event.model_copy(update={"output_expectations": predictions}))
    return {"result": "executed", "episode_id": event.episode_id,
            "prediction_target": proposal.prediction_target, "predictions": checks, "interpretation": "Observation, not causal proof or task success"}
