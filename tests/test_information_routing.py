import asyncio

import pytest
from pydantic import BaseModel
from stirrup.core.models import Tool, ToolResult

from worldmodel_harness.information import information_route_instruction, tracked_information_tool


def test_prompt_routes_contract_inspection_through_operational_decision():
    operational = information_route_instruction(operational=True)
    assert "execute_decision" in operational
    assert "action.tool='inspect_contract'" in operational
    assert "not a top-level tool" in operational
    assert "execute_decision" not in information_route_instruction(operational=False)


@pytest.mark.parametrize("name", ["inspect_contract", "investigate_hypothesis"])
@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_information_audit_counts_returned_and_interrupted_calls(name, failure):
    class Input(BaseModel):
        pass

    async def invoke(params):
        if failure:
            raise failure("private SDK detail")
        return ToolResult(content="Local refusal", success=False)

    audit = []
    tool = tracked_information_tool(
        Tool(name=name, description="inspect", parameters=Input, executor=invoke), audit
    )
    if failure:
        with pytest.raises(failure):
            asyncio.run(tool.executor(Input()))
    else:
        assert not asyncio.run(tool.executor(Input())).success
    assert len(audit) == 1 and audit[0]["tool"] == name
    assert audit[0]["returned"] == (failure is None)
    assert "private" not in str(audit)
