import asyncio
import json

import pytest
from pydantic import BaseModel, ConfigDict
from stirrup.core.models import AssistantMessage, TextBlock, Tool, ToolResult

from worldmodel_harness.decision_runtime import dispatch_decision, propose_decision
from worldmodel_harness.decisions import (
    ActionRequirements,
    Decision,
    LiveRequirement,
    ProposedAction,
)
from worldmodel_harness.knowledge import LiveState, Scope
from worldmodel_harness.store import KnowledgeStore


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 1


def proposal(tool="unfamiliar_tool", arguments=None):
    return Decision(
        goal="inspect state",
        rationale="observe before deciding",
        action=ProposedAction(tool=tool, arguments=arguments or {}),
    )


@pytest.mark.parametrize("reason", ["missing_fact", "bad_arguments", "unknown_tool"])
def test_blocked_dispatch_never_calls_environment(tmp_path, reason):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    calls, audit = [], []
    tool = Tool(
        name="unfamiliar_tool",
        description="inspect",
        parameters=Inputs,
        executor=lambda p: calls.append(p),
    )
    decision = proposal(
        tool="unknown" if reason == "unknown_tool" else tool.name,
        arguments={"not_allowed": 1} if reason == "bad_arguments" else {},
    )
    result = asyncio.run(
        dispatch_decision(
            decision,
            tools={tool.name: tool},
            requirements_for=lambda _: ActionRequirements(
                facts=(LiveRequirement(key="ready", expected=True),)
            ),
            store=store,
            scope=Scope(environment="new", version="1", task_id="task"),
            live=LiveState("task"),
            audit=audit,
        )
    )
    assert not result.success
    blocked = json.loads(result.content)
    assert blocked["origin"] == "local_precondition_gate"
    assert blocked["environment_called"] is False
    assert "not an API response" in blocked["interpretation"]
    assert calls == []
    assert not audit[0]["invoked"]
    store.close()


def test_resolver_receives_normalized_arguments_and_runs_on_every_dispatch(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    calls, resolved, audit = [], [], []

    async def invoke(params):
        calls.append(params.value)
        return ToolResult(content="observed")

    def requirements(action):
        resolved.append(action.arguments)
        return (
            ActionRequirements()
            if len(resolved) == 1
            else ActionRequirements(facts=(LiveRequirement(key="new_requirement", expected=True),))
        )

    tool = Tool(name="unfamiliar_tool", description="inspect", parameters=Inputs, executor=invoke)
    args = {
        "tools": {tool.name: tool},
        "requirements_for": requirements,
        "store": store,
        "scope": Scope(environment="new", version="1", task_id="task"),
        "live": LiveState("task"),
        "audit": audit,
    }
    assert asyncio.run(dispatch_decision(proposal(), **args)).success
    assert not asyncio.run(dispatch_decision(proposal(), **args)).success
    assert calls == [1]
    assert resolved == [{"value": 1}, {"value": 1}]
    assert [row["invoked"] for row in audit] == [True, False]
    store.close()


def test_model_proposes_data_without_executing_or_promoting_it():
    calls = []
    tool = Tool(
        name="unfamiliar_tool",
        description="inspect",
        parameters=Inputs,
        executor=lambda p: calls.append(p),
    )

    class Client:
        async def generate(self, messages, tools):
            prompt = json.loads(messages[1].content)
            assert tools == {}
            assert "unfamiliar_tool" in prompt["tools"]
            assert prompt["working_context"] == "Outcome unknown; inspect state."
            return AssistantMessage(blocks=[TextBlock(text=proposal().model_dump_json())])

    accounting = {}
    result = asyncio.run(
        propose_decision(
            Client(),
            task="inspect",
            working_context="Outcome unknown; inspect state.",
            tools={tool.name: tool},
            accounting=accounting,
        )
    )
    assert result == proposal()
    assert calls == []
    assert "token_usage" in accounting
    assert accounting["decision_seconds"] >= 0
    with pytest.raises(ValueError, match="budget"):
        asyncio.run(
            propose_decision(
                None,
                task="inspect",
                working_context="x" * 100,
                tools={},
                accounting={},
                max_prompt_chars=1,
            )
        )


def test_upstream_dispatch_keeps_evidence_and_shared_budget(tmp_path):
    pytest.importorskip("automationbench")
    from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
    from worldmodel_harness.budget import BudgetExceeded, Ledger
    from worldmodel_harness.learning import Learner

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", store, Ledger(max_environment_calls=1), Learner(store)
    )
    audit = []
    args = {
        "tools": {t.name: t for t in adapter.tools()},
        "requirements_for": lambda _: ActionRequirements(),
        "store": store,
        "scope": adapter.scope,
        "live": adapter.live,
        "audit": audit,
    }
    decision = proposal("api_search", {"query": "contacts", "top_k": 1})
    assert asyncio.run(dispatch_decision(decision, **args)).success
    assert adapter.ledger.environment_calls == 1
    assert len(store.episodes(adapter.scope)) == 1
    assert store.claims(adapter.scope)
    with pytest.raises(BudgetExceeded):
        asyncio.run(dispatch_decision(decision, **args))
    assert adapter.ledger.environment_calls == 1
    assert audit[-1]["error_type"] == "BudgetExceeded"
    store.close()
