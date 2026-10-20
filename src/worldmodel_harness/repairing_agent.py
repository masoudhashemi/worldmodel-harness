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


class RepairingAgent(Agent):
    def __init__(self, *, repair_schemas, **kwargs):
        super().__init__(**kwargs)
        self.repair_schemas = repair_schemas
        self.outer_validation_audit = []
        # Stirrup sessions share the instance dictionary but use SessionAgent's
        # class, not the subclass. Bind the public execution hook on this instance
        # so the same hook is present in the session; no global SDK monkeypatch.
        self.run_tool = self._run_tool_with_repairs

    async def _run_tool_with_repairs(self, tool_call, run_metadata):
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
