"""Small diagnostic fixture, not an implementation of a published benchmark.

The grader is deliberately separate from the agent-visible tools. Observations
record the visible response, never the hidden fact that a timed-out write committed.
"""

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from stirrup.core.models import Tool, ToolResult


class TicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    external_id: str


class Observation(BaseModel):
    sequence: int
    task_id: str
    operation: str
    arguments: dict[str, str]
    result: dict
    outcome: Literal["success", "unknown"]


@dataclass
class TicketFixture:
    """Create commits but times out once; lookup reveals the resulting ticket."""

    task_id: str = field(default_factory=lambda: uuid4().hex)
    call_budget: int = 6
    observations: list[Observation] = field(default_factory=list, init=False)
    _tickets: list[dict[str, str]] = field(default_factory=list, init=False)

    def _check_budget(self) -> None:
        if len(self.observations) >= self.call_budget:
            raise RuntimeError("Environment call budget exhausted")

    def _record(self, operation, params, result, outcome) -> ToolResult:
        event = Observation(
            sequence=len(self.observations) + 1,
            task_id=self.task_id,
            operation=operation,
            arguments=params.model_dump(),
            result=result,
            outcome=outcome,
        )
        self.observations.append(event)
        return ToolResult(content=event.model_dump_json(), success=outcome == "success")

    async def create(self, params: TicketInput) -> ToolResult:
        self._check_budget()
        ticket = {"id": uuid4().hex, "external_id": params.external_id}
        self._tickets.append(ticket)
        if len(self._tickets) == 1:
            return self._record(
                "create_ticket",
                params,
                {"error": "timeout", "message": "Request timed out; outcome unknown."},
                "unknown",
            )
        return self._record("create_ticket", params, dict(ticket), "success")

    async def lookup(self, params: TicketInput) -> ToolResult:
        self._check_budget()
        matches = [dict(t) for t in self._tickets if t["external_id"] == params.external_id]
        return self._record("lookup_ticket", params, {"tickets": matches}, "success")

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="create_ticket",
                description="Create a ticket with an external ID.",
                parameters=TicketInput,
                executor=self.create,
            ),
            Tool(
                name="lookup_ticket",
                description="Find tickets matching an external ID.",
                parameters=TicketInput,
                executor=self.lookup,
            ),
        ]

    def grade(self, external_id: str, reported_id: str) -> bool:
        matches = [t for t in self._tickets if t["external_id"] == external_id]
        return len(matches) == 1 and matches[0]["id"] == reported_id
