"""Opt-in live fixture run through Stirrup; no shell, browser, or file tools."""

import asyncio
import json
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel
from stirrup import Agent
from stirrup.core.models import TokenUsage, Tool, ToolResult

from .config import load_api_key, load_settings
from .fixture import TicketFixture
from .runtime import build_client


class FinishInput(BaseModel):
    ticket_id: str


async def run_smoke(root: Path) -> dict:
    settings = load_settings(root, default_profile="configs/model-free-test.toml")
    client = build_client(settings, load_api_key(root, settings))
    fixture = TicketFixture()
    external_id = "smoke-" + uuid4().hex
    agent = Agent(
        client=client,
        name="ticket-smoke",
        max_turns=6,
        tools=fixture.tools(),
        finish_tool=Tool(
            name="finish",
            description="Report the verified ticket ID and finish.",
            parameters=FinishInput,
            executor=lambda _: ToolResult(content="Finished"),
        ),
        system_prompt="Use the provided tools to complete the task, then call finish.",
        recover_from_context_overflow=False,
        context_summarization_cutoff=1.0,
    )
    output = root / "runs" / fixture.task_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        async with asyncio.timeout(180):
            async with agent.session(output_dir=output, cache_on_interrupt=False) as session:
                finish, history, metadata = await session.run(
                    f"Create exactly one ticket with external_id {external_id}. "
                    "Verify its existence and report its ticket ID. Do not create duplicates."
                )
        reported_id = finish.ticket_id if isinstance(finish, FinishInput) else ""
        report = {
            "run_id": fixture.task_id,
            "model": settings.name,
            "passed": fixture.grade(external_id, reported_id),
            "environment_calls": len(fixture.observations),
            "message_count": sum(len(group) for group in history),
            "model_calls_with_usage": len(metadata["token_usage"]),
            "token_usage": sum(metadata["token_usage"], TokenUsage()).model_dump(),
            "versions": {name: version(name) for name in ("stirrup", "litellm")},
            "protocol": {
                "max_turns": 6,
                "environment_call_budget": fixture.call_budget,
                "wall_timeout_seconds": 180,
                "context_summarization_cutoff": 1.0,
                "recover_from_context_overflow": False,
                "reasoning_effort": "provider_default",
                "litellm_num_retries": 0,
                "stirrup_transient_max_attempts": 3,
            },
            "settings": settings.model_dump(),
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        # Persist only agent-visible boundary evidence, including on interrupted runs.
        with (output / "observations.jsonl").open("x") as stream:
            for observation in fixture.observations:
                stream.write(observation.model_dump_json() + "\n")
