"""Costed, read-only investigation of retained uncertainty; not automatic proof."""

from pydantic import Field
from stirrup.core.models import Tool, ToolResult

from .decisions import Decision, ProposedAction
from .information import CURRENT_INFORMATION
from .knowledge import InformationRequest
from .operation_semantics import known_effect


class HypothesisProbe(InformationRequest):
    target_claim_id: str = Field(min_length=1)
    action: ProposedAction


def hypothesis_probe_tool(adapter, dispatch, *, policy, max_calls):
    used = 0

    async def investigate(params: HypothesisProbe):
        nonlocal used
        from automationbench.tools.api import api_fetch, api_search

        if policy == "off" or (params.intended_use == "future_tasks" and policy != "future"):
            return ToolResult(
                content="This information policy does not permit the probe.", success=False
            )
        if used >= max_calls:
            return ToolResult(content="Hypothesis-probe allowance exhausted.", success=False)
        if len(set(params.distinguishing_outcomes)) < 2:
            return ToolResult(content="Specify distinct possible observations.", success=False)
        try:
            claim = adapter.knowledge.claim(params.target_claim_id)
        except KeyError:
            return ToolResult(content="Unknown hypothesis.", success=False)
        if (
            not claim.scope.applies_to(adapter.scope)
            or claim.status != "hypothesis"
            or not claim.relation.startswith("possible_")
        ):
            return ToolResult(
                content="Probe requires an in-scope semantic hypothesis.", success=False
            )
        action = params.action
        if action.tool not in {"api_search", "api_fetch"}:
            return ToolResult(content="Probe supports discovery or API reads only.", success=False)
        if action.tool == "api_fetch" and (
            known_effect(adapter, action.arguments)[0] != "read"
        ):
            return ToolResult(
                content="Mutating probes are not permitted by this tool.", success=False
            )
        base = {t.name: t for t in adapter.tools()}[action.tool]
        function = api_search if action.tool == "api_search" else api_fetch

        async def execute(arguments):
            nonlocal used
            used += 1
            return await adapter.execute(function, arguments.model_dump(), purpose="probe")

        tool = Tool(
            name=base.name,
            description=base.description,
            parameters=base.parameters,
            executor=execute,
        )
        request = InformationRequest.model_validate(params.model_dump(exclude={"action"}))
        token = CURRENT_INFORMATION.set(request)
        before = used
        try:
            return await dispatch(
                Decision(
                    goal=params.uncertainty,
                    rationale="Gather a distinguishing observation, not assume the hypothesis",
                    action=action,
                ),
                {tool.name: tool},
            )
        finally:
            CURRENT_INFORMATION.reset(token)
            # Request fresh screening after actual attempted observation, including
            # errors/unknown outcomes. Neither a successful read nor a failed read
            # automatically confirms/refutes the proposition.
            if used > before and not adapter.knowledge.frozen:
                current = adapter.knowledge.claim(claim.claim_id)
                if current.status == "hypothesis" and not current.needs_review:
                    adapter.knowledge.put_claim(
                        current.model_copy(
                            update={
                                "version": current.version + 1,
                                "needs_review": True,
                            }
                        )
                    )

    return Tool(
        name="investigate_hypothesis",
        parameters=HypothesisProbe,
        executor=investigate,
        description="Investigate an in-scope retained semantic hypothesis with one API read "
        "or contract discovery query. Name competing observable outcomes and current/future "
        "task value. Choose arguments from observed contracts and current task bindings. "
        "Uses ordinary prerequisite checks and the shared environment budget. No writes, "
        "automatic remedies, or automatic promotion; observed results may remain inconclusive.",
    )
