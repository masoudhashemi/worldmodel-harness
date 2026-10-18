"""Explicit, bounded information requests through existing discovery tools.

This first surface inspects API contracts, not arbitrary experimental mutations.
Declared distinguishing outcomes are hypotheses, never automatically scored as
resolved or promoted to claims. Ordinary diagnosis remains available in all arms.
"""

import inspect
from contextvars import ContextVar
from typing import Literal

from pydantic import Field
from stirrup.core.models import Tool, ToolResult

from .knowledge import InformationRequest

CURRENT_INFORMATION: ContextVar[InformationRequest | None] = ContextVar(
    "current_information", default=None
)


def tracked_information_tool(tool: Tool, audit: list[dict]) -> Tool:
    """Count typed invocations, including rejected and interrupted probes."""

    async def execute(params):
        record = {"tool": tool.name, "returned": False}
        audit.append(record)
        try:
            result = tool.executor(params)
            if inspect.isawaitable(result):
                result = await result
            record.update(returned=True, success=result.success)
            return result
        except BaseException as exc:
            record["error_type"] = type(exc).__name__
            raise

    return Tool(
        name=tool.name, description=tool.description, parameters=tool.parameters, executor=execute
    )


def information_route_instruction(*, operational: bool) -> str:
    if operational:
        return (
            "Use execute_decision with action.tool='inspect_contract' and its inputs in "
            "action.arguments for API contract uncertainty. inspect_contract is not a "
            "top-level tool. Use investigate_hypothesis directly only when it is offered "
            "and a retained hypothesis needs investigation. "
        )
    return "Use inspect_contract for API contract uncertainty. "


class InspectContract(InformationRequest):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


def information_tool(adapter, *, policy: Literal["task", "future"], max_calls: int) -> Tool:
    """A secondary cap never replaces the shared environment-call budget."""
    used = 0

    async def inspect(params: InspectContract) -> ToolResult:
        nonlocal used
        from automationbench.tools.api import api_search

        if params.intended_use == "future_tasks" and policy != "future":
            return ToolResult(
                content="Future-only probes are disabled; task diagnosis remains allowed.",
                success=False,
            )
        if used >= max_calls:
            return ToolResult(
                content="Information-call allowance exhausted; ordinary tools remain available within the shared budget.",
                success=False,
            )
        if len(set(params.distinguishing_outcomes)) < 2:
            return ToolResult(
                content="Name at least two distinct possible observations.", success=False
            )
        if params.target_claim_id is not None:
            try:
                claim = adapter.knowledge.claim(params.target_claim_id)
            except KeyError:
                return ToolResult(
                    content="Unknown target claim; discover using a query without a target ID.",
                    success=False,
                )
            if not claim.scope.applies_to(adapter.scope) or claim.relation != "endpoint_contract":
                return ToolResult(
                    content="This tool refreshes only in-scope endpoint contracts, not behavior or live records.",
                    success=False,
                )
        request = InformationRequest.model_validate(params.model_dump(exclude={"query", "top_k"}))
        token = CURRENT_INFORMATION.set(request)
        try:
            used += 1
            return await adapter.execute(
                api_search,
                {"query": params.query, "top_k": params.top_k},
                purpose="refresh" if params.target_claim_id else "probe",
            )
        finally:
            CURRENT_INFORMATION.reset(token)

    return Tool(
        name="inspect_contract",
        parameters=InspectContract,
        executor=inspect,
        description="Inspect an API contract to resolve a named uncertainty. State alternative "
        "observable outcomes and whether this helps the current task or only future tasks. "
        "Provide target_claim_id to refresh an existing endpoint contract, otherwise omit it "
        "to discover unknown operations. Uses the same api_search and shared call budget; "
        "results are documentation, not proof of live state or task completion.",
    )
