"""One bounded, costed repair of invalid data; never replay an executed action."""

import json

from stirrup.core.models import UserMessage

from .adaptation import parse_proposal


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
                            "rejected_output": text,
                        }
                    )
                ),
            ]
