"""Stirrup's agent with Gemma text tool-call parsing and fuller schema-error feedback."""

import json
from typing import Any, Union, get_args, get_origin

from pydantic import ValidationError
from stirrup import Agent
from stirrup.core.models import AssistantMessage, TextBlock, ToolCall, joined_text, tool_call_blocks

from .runtime import parse_gemma_calls

GEMMA_CALL_MARKERS = ("<|tool_call>call:", "<|tool_call|>call:", "<tool_call>call:")


def schema_skeleton(model_cls: Any, depth: int = 0) -> Any:
    """A minimal example object for a Pydantic model."""
    if depth > 3 or not hasattr(model_cls, "model_fields"):
        return {}
    skeleton = {}
    for name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        if get_origin(annotation) is Union:
            non_null = [a for a in get_args(annotation) if a is not type(None)]
            annotation = non_null[0] if non_null else annotation
        origin = get_origin(annotation)
        if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
            skeleton[name] = schema_skeleton(annotation, depth + 1)
        elif origin in (list, tuple):
            skeleton[name] = []
        elif origin is dict:
            skeleton[name] = {"<parameter>": "<value>"}
        else:
            skeleton[name] = f"<{name}>"
    return skeleton


def parse_text_tool_calls(msg):
    """Turn tool calls Gemma wrote as text into real tool-call blocks."""
    if not isinstance(msg, AssistantMessage) or tool_call_blocks(msg.blocks):
        return msg
    text = joined_text(msg.blocks) or ""
    if not any(marker in text for marker in GEMMA_CALL_MARKERS):
        return msg
    clean_text, extracted = parse_gemma_calls(text)
    blocks = [TextBlock(text=clean_text)] if clean_text.strip() else []
    blocks += [
        ToolCall(tool_call_id=f"text_call_{i}", name=name, arguments=json.dumps(args))
        for i, (name, args) in enumerate(extracted)
    ]
    return msg.model_copy(update={"blocks": blocks})


class TextCallClient:
    def __init__(self, client):
        self._inner = client

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def generate(self, messages, tools):
        return parse_text_tool_calls(await self._inner.generate(messages, tools))


class RepairingAgent(Agent):
    def __init__(self, *, repair_schemas, **kwargs):
        super().__init__(**kwargs)
        self.repair_schemas = repair_schemas
        self.outer_validation_audit = []
        self._client = TextCallClient(self._client)
        # Stirrup sessions share the instance dictionary but use SessionAgent's
        # class, not the subclass. Bind the execution hook on this instance.
        self.run_tool = self._run_tool_with_schema_feedback

    async def _run_tool_with_schema_feedback(self, tool_call, run_metadata):
        result = await super().run_tool(tool_call, run_metadata)
        schema = self.repair_schemas.get(tool_call.name)
        if result.args_was_valid or schema is None:
            return result
        try:
            schema.model_validate_json(tool_call.arguments if tool_call.arguments.strip() else "{}")
        except ValidationError as exc:
            # Only invalid *inputs* establish that nothing was dispatched.
            repair = {
                "status": "blocked",
                "origin": "outer_tool_schema",
                "environment_called": False,
                "tool": tool_call.name,
                "validation_errors": [
                    {"field": list(e["loc"]), "error_type": e["type"], "message": e["msg"]}
                    for e in exc.errors(
                        include_input=False, include_context=False, include_url=False
                    )
                ],
                "checked_schema": schema.model_json_schema(),
                "expected_format": schema_skeleton(schema),
                "instruction": (
                    "Return JSON matching expected_format. Ensure fields are placed at their "
                    "declared schema nesting level. Do not repeat the unchanged invalid call; "
                    "no action was executed."
                ),
            }
            self.outer_validation_audit.append({"tool_call_id": tool_call.tool_call_id, **repair})
            return result.model_copy(update={"content": json.dumps(repair)})
        return result
