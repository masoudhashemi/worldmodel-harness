"""Preserve Stirrup execution, enrich its pre-dispatch schema rejection feedback."""

import json
from typing import Any, Union, get_args, get_origin

from pydantic import ValidationError
from stirrup import Agent


def generate_schema_skeleton(model_cls: Any, depth: int = 0) -> Any:
    """Dynamically generate a minimal example skeleton from any Pydantic model class."""
    if depth > 3 or not hasattr(model_cls, "model_fields"):
        return {}
    skeleton = {}
    for name, field_info in model_cls.model_fields.items():
        if name in ("requirements", "rationale", "evidence_ids"):
            continue
        annotation = field_info.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Union:
            non_null = [a for a in args if a is not type(None)]
            if non_null:
                annotation = non_null[0]
                origin = get_origin(annotation)

        if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
            skeleton[name] = generate_schema_skeleton(annotation, depth + 1)
        elif origin in (list, tuple):
            skeleton[name] = []
        elif origin is dict:
            skeleton[name] = {"<parameter>": "<value>"}
        elif annotation is str or origin is None:
            skeleton[name] = f"<{name}>"
        else:
            skeleton[name] = f"<{name}>"
    return skeleton


def suggested_arguments(schema, arguments):
    """Suggest a validated native object for one missing terminal action brace.

    This never executes or changes values. Incomplete strings, interior syntax
    errors and missing nested containers remain errors for the actor to repair.
    """
    try:
        payload = json.loads(arguments)
        action = payload.get("action") if isinstance(payload, dict) else None
        if not isinstance(action, str):
            return None
        try:
            json.loads(action)
            return None
        except json.JSONDecodeError as exc:
            if exc.msg != "Expecting ',' delimiter" or exc.pos != len(action.rstrip()):
                return None
        payload["action"] = json.loads(action + "}")
        return schema.model_validate(payload).model_dump(mode="json", exclude_unset=True)
    except (ValueError, TypeError):
        return None


def normalize_assistant_tool_calls(msg):
    """Ensure tool calls from raw text syntax or misrouted execute_decision finish calls are parsed cleanly."""
    from stirrup.core.models import AssistantMessage, TextBlock, ToolCall, joined_text, tool_call_blocks

    if not isinstance(msg, AssistantMessage):
        return msg

    blocks = list(msg.blocks)
    existing_calls = tool_call_blocks(blocks)

    if not existing_calls:
        text = joined_text(blocks) or ""
        if any(marker in text for marker in ("<|tool_call>call:", "<|tool_call|>call:", "<tool_call>call:")):
            from .runtime import parse_gemma_calls

            clean_text, extracted = parse_gemma_calls(text)
            new_text_blocks = [TextBlock(text=clean_text)] if clean_text.strip() else []
            new_tool_blocks = [
                ToolCall(
                    tool_call_id=f"text_call_{idx}",
                    name=fn_name,
                    arguments=json.dumps(fn_args),
                )
                for idx, (fn_name, fn_args) in enumerate(extracted)
            ]
            blocks = new_text_blocks + new_tool_blocks

    new_blocks = []
    for b in blocks:
        if isinstance(b, ToolCall) and b.name == "execute_decision":
            try:
                args = json.loads(b.arguments) if isinstance(b.arguments, str) else (b.arguments or {})
                action = args.get("action")
                if isinstance(action, str):
                    try:
                        action = json.loads(action)
                    except Exception:
                        pass
                if isinstance(action, dict):
                    action_args = action.setdefault("arguments", {})
                    # Repair misplaced https/http top-level keys
                    for scheme in ("https", "http"):
                        if scheme in args:
                            val = str(args.pop(scheme))
                            if not val.startswith(f"{scheme}:"):
                                val = f"{scheme}:{val}" if val.startswith("//") else f"{scheme}://{val}"
                            action_args["url"] = val
                            action.setdefault("tool", "api_fetch")
                            if "method" not in action_args:
                                action_args["method"] = "POST"

                    # Repair misplaced body on action or malformed trailing delimiter
                    for k in ("body",):
                        target = action_args if k in action_args else action
                        if k in target:
                            val = str(target[k])
                            for marker in ("},tool:", "},method:", "},url:", "},params:", "},headers:"):
                                if marker in val:
                                    val = val.split(marker)[0] + "}"
                            action_args[k] = val
                            if target is action:
                                action.pop(k, None)

                    if action.get("tool") in ("finish", "finish_task"):
                        finish_args = action.get("arguments") or {}
                        new_blocks.append(
                            ToolCall(
                                tool_call_id=b.tool_call_id,
                                name="finish",
                                arguments=json.dumps(finish_args),
                            )
                        )
                        continue

                    args["action"] = action
                    b = b.model_copy(update={"arguments": json.dumps(args)})
            except Exception:
                pass
        new_blocks.append(b)

    return msg.model_copy(update={"blocks": new_blocks})


class InterceptingClient:
    def __init__(self, client):
        self._inner = client

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def generate(self, messages, tools):
        msg = await self._inner.generate(messages, tools)
        return normalize_assistant_tool_calls(msg)


class RepairingAgent(Agent):
    def __init__(self, *, repair_schemas, interaction_recorder=None, **kwargs):
        super().__init__(**kwargs)
        self.repair_schemas = repair_schemas
        self.outer_validation_audit = []
        self.interaction_recorder = interaction_recorder
        self._client = InterceptingClient(self._client)
        # Stirrup sessions share the instance dictionary but use SessionAgent's
        # class, not the subclass. Bind the public execution hook on this instance
        # so the same hook is present in the session; no global SDK monkeypatch.
        self.run_tool = self._run_tool_with_repairs

    async def _run_tool_with_repairs(self, tool_call, run_metadata):
        recorder = self.interaction_recorder
        if recorder is None:
            return await self._run_tool_with_schema_feedback(tool_call, run_metadata)
        interaction_id, token = recorder.begin()
        try:
            result = await self._run_tool_with_schema_feedback(tool_call, run_metadata)
        except BaseException as exc:
            recorder.end(interaction_id, token, tool_call, error_type=type(exc).__name__)
            raise
        recorder.end(interaction_id, token, tool_call, result=result)
        return result

    async def _run_tool_with_schema_feedback(self, tool_call, run_metadata):
        result = await super().run_tool(tool_call, run_metadata)
        schema = self.repair_schemas.get(tool_call.name)
        if result.args_was_valid or schema is None:
            return result
        try:
            schema.model_validate_json(tool_call.arguments if tool_call.arguments.strip() else "{}")
        except ValidationError as exc:
            # Revalidating is pure. Only invalid *inputs* establish no dispatch;
            # a ValidationError from inside an executor is not labeled unexecuted.
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
                "expected_format": generate_schema_skeleton(schema),
                "instruction": (
                    "Return JSON matching expected_format. Ensure fields are placed at their declared schema nesting level. "
                    "Do not repeat the unchanged invalid call; no action was executed."
                ),
            }
            candidate = suggested_arguments(schema, tool_call.arguments)
            if candidate is not None:
                repair["suggested_arguments"] = candidate
                repair["instruction"] = (
                    "The serialized action is missing its final closing brace. No action executed. "
                    "Review suggested_arguments and resubmit it as a native JSON object, not an escaped string. "
                    "It preserves your supplied values; ordinary schema and execution gates still apply."
                )
            self.outer_validation_audit.append({"tool_call_id": tool_call.tool_call_id, **repair})
            return result.model_copy(update={"content": json.dumps(repair)})
        return result
