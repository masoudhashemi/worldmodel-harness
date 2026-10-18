"""Preserve Stirrup execution, enrich its pre-dispatch schema rejection feedback."""

import json

from pydantic import ValidationError
from stirrup import Agent


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
                "instruction": "Return JSON matching this schema. Object-valued fields must be objects, not escaped JSON strings. Put fields at the schema's specified level. Do not repeat the unchanged invalid call; no action was executed.",
            }
            self.outer_validation_audit.append({"tool_call_id": tool_call.tool_call_id, **repair})
            return result.model_copy(update={"content": json.dumps(repair)})
        return result
