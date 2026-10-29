"""LLM-based trajectory symbolization and workflow parameterization.

Abstracts concrete execution traces (instance IDs, emails, hashes, record keys)
into generalized, parameterized operational workflow templates and data flows.
Decoupled from benchmark task IDs and internal metadata.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from stirrup.core.models import SystemMessage, UserMessage

from .knowledge import Claim, Episode, Layer, Record, Scope, utc_now
from .runtime import MeteredClient
from .store import KnowledgeStore
from .structured_output import ContractError, generate_checked


class SymbolizedStep(Record):
    step_index: int = Field(ge=1, description="1-based sequence index of the step.")
    operation: str = Field(description="Operational tool or endpoint name, e.g. 'api_fetch', 'base64_encode'.")
    action_template: str = Field(
        min_length=1,
        max_length=600,
        description="Abstract parameterized action template with placeholders, e.g. 'POST https://api.airtable.com/v0/{base_id}/{table_name}/{record_id}/comments'.",
    )
    input_bindings: dict[str, str] = Field(
        default_factory=dict,
        description="Data flow source for each placeholder, e.g. {'record_id': 'extracted from message body in step 1'}.",
    )
    abstract_intent: str = Field(
        min_length=1,
        max_length=400,
        description="Concise description of what this step accomplishes in the workflow.",
    )

    @model_validator(mode="after")
    def validate_no_concrete_ids(self) -> SymbolizedStep:
        # Prevent obvious concrete Airtable, Google, or Salesforce record leaks in templates
        leaks = re.findall(r"\b(?:rec_[a-zA-Z0-9]{5,}|00Q[a-zA-Z0-9]{10,}|ss_[a-zA-Z0-9_-]{5,})\b", self.action_template)
        if leaks:
            raise ContractError(f"Action template leaks concrete instance literals: {leaks}. Parameterize as {{record_id}} or {{resource_id}}.")
        return self


class TrajectoryAbstraction(Record):
    goal_intent: str = Field(
        min_length=1,
        max_length=400,
        description="Generalizable abstract purpose of this operational workflow without task-specific instance details.",
    )
    steps: tuple[SymbolizedStep, ...] = Field(
        default=(),
        description="Sequence of parameterized steps comprising the workflow.",
    )
    invariants: tuple[str, ...] = Field(
        default=(),
        description="Key operational invariants learned from execution (e.g. 'Must fetch message before updating record').",
    )


def format_episodes_for_abstraction(episodes: list[Episode], max_chars: int = 12000) -> tuple[str, list[str]]:
    """Format a sequence of environment execution episodes for LLM symbolization."""
    lines: list[str] = []
    episode_ids: list[str] = []

    for idx, ep in enumerate(episodes, 1):
        if ep.origin != "environment":
            continue
        episode_ids.append(ep.episode_id)
        args = ep.arguments or {}
        method = args.get("method", "").upper() if isinstance(args, dict) else ""
        url = args.get("url", "") if isinstance(args, dict) else ""
        outcome = ep.outcome

        # Compact representation of arguments
        arg_summary = {}
        if isinstance(args, dict):
            for k in ("params", "body", "ranges", "sheet", "range"):
                if k in args and args[k]:
                    val = str(args[k])
                    arg_summary[k] = val[:120] if len(val) > 120 else val

        res = ep.response
        res_summary = ""
        if isinstance(res, dict):
            keys = list(res.keys())[:4]
            sample = {k: str(res[k])[:80] for k in keys}
            res_summary = f"keys={keys}, sample={sample}"
        elif isinstance(res, list):
            res_summary = f"list of {len(res)} items"
        else:
            res_summary = str(res)[:100]

        line = (
            f"Step {len(episode_ids)}: operation={ep.operation} method={method} url={url}\n"
            f"  args={json.dumps(arg_summary)}\n"
            f"  outcome={outcome} response={res_summary}"
        )
        lines.append(line)

    payload = "\n\n".join(lines)
    if len(payload) > max_chars:
        payload = payload[:max_chars] + "\n...[truncated]"
    return payload, episode_ids


async def symbolize_trajectory(
    client: MeteredClient,
    episodes: list[Episode],
    *,
    accounting: dict[str, Any] | None = None,
    max_chars: int = 12000,
) -> TrajectoryAbstraction | None:
    """Prompt the LLM to generalize and symbolize an execution trajectory into parameterized workflow templates."""
    if accounting is None:
        accounting = {}

    formatted_trace, episode_ids = format_episodes_for_abstraction(episodes, max_chars=max_chars)
    if not episode_ids:
        return None

    schema = TrajectoryAbstraction.model_json_schema()
    system_prompt = (
        "You are an expert operational world-model abstractor.\n"
        "Your task is to analyze the following sequence of real API tool interactions from a workflow execution "
        "and generalize them into an abstract, parameterized workflow template.\n\n"
        "CRITICAL RULES:\n"
        "1. Parameterize ALL instance-specific entities, IDs, row numbers, email addresses, and hashes.\n"
        "   - e.g. Replace 'rec_104' or 'rec_105' with '{record_id}'\n"
        "   - e.g. Replace 'base_ops' with '{base_id}'\n"
        "   - e.g. Replace 'ss_badges' with '{spreadsheet_id}'\n"
        "   - e.g. Replace 'msg_nda_001' with '{message_id}'\n"
        "   - e.g. Replace specific user emails with '{recipient_email}'\n"
        "2. Never leak concrete entity literals or person names into action_template or goal_intent.\n"
        "3. Explicitly document input_bindings showing where each placeholder is sourced from (e.g. extracted from email body in previous step).\n"
        "4. Return valid JSON matching the supplied schema exactly."
    )

    user_prompt = (
        f"Execution Trace ({len(episode_ids)} steps):\n\n"
        f"{formatted_trace}\n\n"
        f"Return the abstract TrajectoryAbstraction JSON according to the schema:\n{json.dumps(schema, indent=2)}"
    )

    def validate(data: Any) -> TrajectoryAbstraction:
        accounting["proposed_abstraction"] = data
        return TrajectoryAbstraction.model_validate(data)

    try:
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ]
        abstraction = await generate_checked(client, messages, validate, accounting)
        accounting["symbolized_step_count"] = len(abstraction.steps)
        return abstraction
    except Exception as exc:
        accounting["symbolization_error"] = str(exc)
        return None


def store_trajectory_abstraction(
    store: KnowledgeStore,
    scope: Scope,
    abstraction: TrajectoryAbstraction,
    supporting_episode_ids: list[str],
) -> Claim:
    """Persist an LLM-symbolized workflow abstraction as a supported world model claim."""
    import hashlib
    from uuid import uuid4

    claim_id = hashlib.sha256(
        json.dumps([scope.model_dump(), abstraction.goal_intent, len(abstraction.steps)], sort_keys=True).encode()
    ).hexdigest()[:24]

    payload = abstraction.model_dump(mode="json")
    claim = Claim(
        claim_id=claim_id,
        version=1,
        layer=Layer.BEHAVIOR,
        relation="symbolized_workflow_template",
        participants=tuple(f"step_{s.step_index}:{s.operation}" for s in abstraction.steps[:8]) or ("workflow_root",),
        statement=f"Parameterized operational workflow: {abstraction.goal_intent}",
        scope=scope,
        payload=payload,
        status="supported",
        origin="inferred",
        supporting_episodes=tuple(supporting_episode_ids),
        contradicting_episodes=(),
        dependencies=(),
        observed_at=utc_now(),
        last_verified_at=utc_now(),
        needs_review=False,
    )

    store.put_claim(claim)
    return claim
