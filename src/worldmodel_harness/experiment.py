"""Run one AutomationBench task on Stirrup: the baseline, or with the experience graph."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field
from stirrup.core.models import SystemMessage, TokenUsage, Tool, ToolResult, UserMessage

from .automationbench import AutomationBenchAdapter, task_row
from .budget import Ledger
from .config import load_api_key, load_settings
from .core.advisor import Advisor
from .core.graph import Graph
from .core.policy import LearnedPolicy
from .repairing_agent import RepairingAgent
from .runtime import build_client


class TrialSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: str
    # Which trial of the task this is (e.g. its round), so repeated trials draw afresh.
    trial: str = ""
    # With the graph passed to run_trial, its advisor adds to tool results.
    experience: bool = False
    # Overrides configs/model.toml, for comparing output limits on the same day.
    max_tokens: int | None = Field(default=None, ge=1)
    max_turns: int = Field(default=50, ge=1)
    max_environment_calls: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=1200, ge=1)


class Finish(BaseModel):
    summary: str


class CostClient:
    """Records the cost of every model call."""

    def __init__(self, client):
        self.client = client
        self.model_call_costs = []
        self.last_messages = []

    @property
    def model_slug(self):
        return self.client.model_slug

    @property
    def context_window_tokens(self):
        return self.client.context_window_tokens

    async def generate(self, messages, tools):
        self.last_messages = list(messages)
        started = perf_counter()
        cost = {"usage_available": False}
        try:
            response = await self.client.generate(messages, tools)
            cost["token_usage"] = response.token_usage.model_dump()
            cost["usage_available"] = True
            self.last_messages.append(response)
            return response
        except BaseException as exc:
            # Include cancellation and failed calls without retaining SDK messages.
            cost["error_type"] = type(exc).__name__
            raise
        finally:
            cost["elapsed_seconds"] = perf_counter() - started
            self.model_call_costs.append(cost)


async def run_trial(
    root: Path, settings: TrialSettings, output_dir: Path, graph: Graph | None = None, policy: LearnedPolicy | None = None
) -> dict:
    """One task. With `settings.experience`, the graph advises through tool results, and
    `policy` (if given) decides which kinds of message this run is shown; the caller
    decides what the graph and the policy learn afterwards."""
    if settings.experience and graph is None:
        raise ValueError("Experience needs a graph")
    model_settings = load_settings(root)
    if settings.max_tokens is not None:
        model_settings = model_settings.model_copy(update={"max_tokens": settings.max_tokens})
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(max_environment_calls=settings.max_environment_calls)
    adapter = AutomationBenchAdapter(task_row(settings.task_name), settings.task_name.split(".", 1)[0], ledger)
    result = {"task_name": adapter.task.task_id}
    client = CostClient(build_client(model_settings, load_api_key(root, model_settings)))
    if settings.experience:
        query = next(m["content"] for m in adapter.task.prompt if m["role"] == "user")
        run_policy = policy.for_run(settings.task_name, settings.trial) if policy is not None else None
        adapter.advisor = Advisor(graph, query, run_policy)
        result["experience"] = adapter.advisor.log
        if run_policy is not None:
            result["policy"] = run_policy.decisions
    tools = adapter.tools()
    finish_tool = Tool(
        name="finish",
        description="Finish the task and summarize the result.",
        parameters=Finish,
        executor=lambda _: ToolResult(content="Finished"),
    )
    agent = RepairingAgent(
        repair_schemas={t.name: t.parameters for t in [*tools, finish_tool]},
        client=client,
        name="automationbench",
        max_turns=settings.max_turns,
        tools=tools,
        finish_tool=finish_tool,
        recover_from_context_overflow=False,
        context_summarization_cutoff=1.0,
    )
    messages = [
        SystemMessage(content=m["content"]) if m["role"] == "system" else UserMessage(content=m["content"])
        for m in adapter.task.prompt
    ]
    try:
        async with agent.session(output_dir=output_dir, cache_on_interrupt=False) as session:
            async with asyncio.timeout(settings.timeout_seconds):
                finish, history, metadata = await session.run(messages)
        result.update(
            finished=finish is not None,
            token_usage=sum(metadata["token_usage"], TokenUsage()).model_dump(),
        )
        trajectory = [[m.model_dump(mode="json") for m in group] for group in history]
    except Exception as exc:  # noqa: BLE001 -- keep SDK error details out of artifacts
        result.update(finished=False, error_type=type(exc).__name__)
        trajectory = [[m.model_dump(mode="json") for m in client.last_messages]]
    (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    result["model_call_costs"] = client.model_call_costs
    result["outer_validation_audit"] = agent.outer_validation_audit
    # Score only after interaction; the score is never given to the graph.
    result.update(adapter.score())
    result["grading_details"] = adapter.grading_details()
    result["ledger"] = asdict(ledger)
    if adapter.advisor is not None:  # the calls the advisor saw, to check logs against traces
        result["advisor_calls"] = len(adapter.advisor.steps)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
