"""One bounded, costed repair of invalid data; never replay an executed action."""

import json

from pydantic import ValidationError
from stirrup.core.models import UserMessage

from .adaptation import parse_proposal


class ContractError(ValueError):
    """Controller-authored feedback safe to expose; never include external data."""


def validation_feedback(exc):
    if isinstance(exc, ContractError):
        return [str(exc)[:600]]
    if isinstance(exc, ValidationError):
        return [str(error["ctx"]["error"])[:600]
                if isinstance(error.get("ctx", {}).get("error"), ContractError)
                else f"Schema validation failed: {error['type']}"
                for error in exc.errors(include_input=False, include_url=False)[:8]]
    return ["Output failed validation; follow the supplied schema and constraints."]


async def generate_checked(client, messages, validate, accounting):
    total = {}
    for attempt in range(2):
        response = await client.generate(messages, {})
        usage = response.token_usage.model_dump()
        for key, value in usage.items():
            total[key] = total.get(key, 0) + value
        accounting["token_usage"] = total
        accounting["returned_model_responses"] = attempt + 1
        try:
            return validate(parse_proposal(response))
        except (ValueError, TypeError) as exc:
            accounting.setdefault("invalid_outputs", []).append(type(exc).__name__)
            text = "".join(b.text for b in response.blocks if b.kind == "text")
            accounting["rejected_proposal_text"] = text
            if attempt:
                raise
            # No provider error text or unbounded rejected output in a repair prompt.
            messages = [
                *messages,
                UserMessage(
                    content=json.dumps(
                        {
                            "repair": "One repair attempt. Return schema-valid JSON only. Use exact supplied "
                            "catalog keys for citations. No actions have executed. An empty batch is allowed.",
                            "validation_feedback": validation_feedback(exc),
                            "rejected_output": text[:18000],
                        }
                    )
                ),
            ]
