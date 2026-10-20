"""Agent-directed, scoped local retrieval; never substitutes historical live values."""

from time import perf_counter

from pydantic import Field
from stirrup.core.models import Tool, ToolResult

from .context import render_context, select_procedures
from .knowledge import Record
from .retrieval import decision_focus, retrieve


class LookupKnowledge(Record):
    query: str = Field(min_length=1, max_length=1000)
    max_chars: int = Field(default=4000, ge=1, le=12000)


def knowledge_lookup_tool(
    adapter, *, max_chars, expand, adaptive, allow_experimental, audit, max_calls=8
):
    used = 0

    async def lookup(params: LookupKnowledge):
        nonlocal used
        budget = min(params.max_chars, max_chars)
        if used >= max_calls:
            message = "Local retrieval allowance exhausted; unresolved prerequisites still require evidence."
            return ToolResult(content=message if len(message) <= budget else "", success=False)
        used += 1
        adapter.ledger.knowledge_queries += 1
        started = perf_counter()
        try:
            working = retrieve(
                adapter.knowledge,
                adapter.scope,
                params.query,
                live=adapter.live,
                max_chars=budget,
                expand=expand,
                focus_entities=decision_focus(adapter)[0],
                subgoal=params.query,
            )
            procedures = []
            if adaptive:
                procedures, _ = select_procedures(
                    adapter.knowledge,
                    adapter.scope,
                    adapter.live,
                    params.query,
                    allow_experimental=allow_experimental,
                    linked_claim_ids={c.claim_id for c in working.claims},
                )
            content, selection = render_context(
                working,
                procedures,
                max_chars=budget,
                store=adapter.knowledge,
                query=params.query,
            )
            audit.append(
                {
                    "query": params.query,
                    "effective_max_chars": budget,
                    "gaps": working.gaps,
                    **selection,
                }
            )
            return ToolResult(content=content, success=bool(content))
        finally:
            adapter.ledger.compute_seconds += perf_counter() - started

    return Tool(
        name="lookup_knowledge",
        parameters=LookupKnowledge,
        executor=lookup,
        description="Search retained operational knowledge for a specific capability, endpoint, "
        "data relationship, or procedure before rediscovering it. This is local retrieval, not "
        "an environment call or a live-state refresh. Results obey scope, freshness, dependency "
        "and context limits. Gaps remain unknown; use permitted discovery or refresh when needed. "
        "Queries incur compute and model-context costs and have a per-task allowance.",
    )
