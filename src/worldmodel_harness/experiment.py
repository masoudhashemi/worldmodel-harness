"""Run one AutomationBench task on Stirrup, as the baseline or with advisory memory."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field
from stirrup.core.models import SystemMessage, TokenUsage, Tool, ToolResult, UserMessage

from .advisory import build_note, detect_loop
from .automationbench import AutomationBenchAdapter, task_row
from .budget import Ledger
from .config import load_api_key, load_settings
from .repairing_agent import RepairingAgent
from .runtime import build_client
from .store import KnowledgeStore


class TrialSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: str
    # Baseline when False: no store, no note, nothing learned.
    advisory: bool = False
    max_turns: int = Field(default=50, ge=1)
    max_environment_calls: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=1200, ge=1)


class Finish(BaseModel):
    summary: str


def _with_note(messages: list, note: str) -> list:
    updated = list(messages)
    header = "\n\n### Advisory Memory\n"
    if updated and updated[0].role == "system":
        updated[0] = SystemMessage(content=updated[0].content + header + note)
    else:
        updated.insert(0, SystemMessage(content="### Advisory Memory\n" + note))
    return updated


class AdvisoryClient:
    """Records per-call model cost; in advisory mode adds the fixed note until the agent loops."""

    def __init__(self, client, note: str = ""):
        self.client = client
        self.note = note
        self.withdrawn = None
        self.model_call_costs = []
        self.last_messages = []

    @property
    def model_slug(self):
        return self.client.model_slug

    @property
    def context_window_tokens(self):
        return self.client.context_window_tokens

    async def generate(self, messages, tools):
        if self.note and self.withdrawn is None:
            loop = detect_loop(messages)
            if loop is not None:
                self.withdrawn = {**loop, "model_call_index": len(self.model_call_costs)}
        sent = _with_note(messages, self.note) if self.note and self.withdrawn is None else list(messages)
        self.last_messages = list(messages)
        started = perf_counter()
        cost = {"usage_available": False}
        try:
            response = await self.client.generate(sent, tools)
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
    root: Path, settings: TrialSettings, output_dir: Path, store_path: Path | None = None
) -> dict:
    """One task. With advisory memory the store at `store_path` is read and updated."""
    if settings.advisory and store_path is None:
        raise ValueError("Advisory memory needs a store")
    model_settings = load_settings(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    row = task_row(settings.task_name)
    store = KnowledgeStore(store_path) if settings.advisory else None
    try:
        ledger = Ledger(max_environment_calls=settings.max_environment_calls)
        adapter = AutomationBenchAdapter(row, settings.task_name.split(".", 1)[0], ledger, store)
        result = {"task_name": adapter.task.task_id, "scope": adapter.scope.model_dump()}
        note = ""
        if store is not None:
            started = perf_counter()
            task_text = " ".join(m["content"] for m in adapter.task.prompt)
            note, info = build_note(store, adapter.scope, task_text)
            ledger.compute_seconds += perf_counter() - started
            result["advisory"] = {**info, "note": note}
        client = AdvisoryClient(build_client(model_settings, load_api_key(root, model_settings)), note)
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
        if "advisory" in result:
            result["advisory"]["withdrawn"] = client.withdrawn
        result["model_call_costs"] = client.model_call_costs
        result["outer_validation_audit"] = agent.outer_validation_audit
        # Score only after interaction; the score is never written to the store.
        result.update(adapter.score())
        result["grading_details"] = adapter.grading_details()
        result["ledger"] = asdict(ledger)
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if store is not None:
            store.close()
