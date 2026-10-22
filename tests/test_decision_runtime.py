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


def test_dispatch_decision_dynamically_resolves_omitted_tool(tmp_path):
    import asyncio
    from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
    from worldmodel_harness.budget import Ledger
    from worldmodel_harness.decisions import ActionRequirements, Decision, ProposedAction
    from worldmodel_harness.learning import Learner

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    adapter = AutomationBenchAdapter(
        dataset("simple")[0], "simple", store, Ledger(max_environment_calls=5), Learner(store)
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
    decision = Decision(
        goal="Search contacts",
        rationale="Need to find contacts",
        action=ProposedAction(tool="", arguments={"query": "contacts", "top_k": 1}),
    )
    result = asyncio.run(dispatch_decision(decision, **args))
    assert result.success
    assert audit[-1]["decision"]["action"]["tool"] == "api_search"
    store.close()


def test_dispatch_decision_catches_unhandled_tool_exceptions(tmp_path):
    import asyncio
    from stirrup.core.models import Tool
    from pydantic import BaseModel
    from worldmodel_harness.decisions import ActionRequirements, Decision, ProposedAction

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")

    class CrashParams(BaseModel):
        val: str

    def crashing_executor(p):
        raise TypeError("got multiple values for keyword argument 'mock_kwarg'")

    crash_tool = Tool(name="crash_tool", description="crashes", parameters=CrashParams, executor=crashing_executor)
    audit = []
    args = {
        "tools": {"crash_tool": crash_tool},
        "requirements_for": lambda _: ActionRequirements(),
        "store": store,
        "scope": Scope(environment="test", version="1", role="agent", task_id="test_task"),
        "live": LiveState("test_task"),
        "audit": audit,
    }
    decision = Decision(
        goal="Call crash tool",
        rationale="Testing exception safety",
        action=ProposedAction(tool="crash_tool", arguments={"val": "test"}),
    )
    result = asyncio.run(dispatch_decision(decision, **args))
    assert not result.success
    content = json.loads(result.content)
    assert content["error"]["type"] == "TypeError"
    assert "mock_kwarg" in content["error"]["message"]
    assert audit[-1]["error_type"] == "TypeError"
    store.close()


def test_considered_hypothesis_is_audited_without_becoming_a_prerequisite(tmp_path):
    from worldmodel_harness.knowledge import Claim, Layer

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="new", version="1", task_id="task")
    store.put_claim(Claim(claim_id="h", scope=scope, layer=Layer.BEHAVIOR,
        relation="possible_routing", participants=("unfamiliar_tool",),
        statement="A model hypothesis", status="hypothesis"))
    decision = proposal().model_copy(update={"considered_claim_ids": ("h", "missing")})
    tool = Tool(name="unfamiliar_tool", description="inspect", parameters=Inputs,
                executor=lambda _: ToolResult(content="observed"))
    audit = []
    result = asyncio.run(dispatch_decision(decision, tools={tool.name: tool},
        requirements_for=lambda _: ActionRequirements(), store=store, scope=scope,
        live=LiveState("task"), audit=audit))
    assert result.success
    assert audit[0]["considered_claims"][0]["status"] == "hypothesis"
    assert audit[0]["considered_claims"][0]["version"] == 1
    assert audit[0]["considered_claims"][1]["result"] == "unknown"
    assert store.claim("h").status == "hypothesis"
    store.close()


def test_malformed_http_method_diagnosis_suppresses_discovery(tmp_path):
    from worldmodel_harness.knowledge import Claim, Episode, Layer
    from worldmodel_harness.operational_requirements import api_requirements

    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    scope = Scope(environment="test", version="1", role="agent", task_id="task")
    ep = Episode(
        episode_id="ep1",
        task_id="task",
        scope=scope,
        interaction_id="int1",
        operation="api_fetch",
        arguments={},
        response={},
        outcome="success",
    )
    store.append_episode(ep)
    store.put_claim(
        Claim(
            claim_id="gmail.read",
            scope=scope,
            layer=Layer.INTERFACE,
            relation="endpoint_contract",
            participants=("gmail.read",),
            statement="Read message",
            status="supported",
            supporting_episodes=("ep1",),
            payload={"id": "gmail.read", "method": "GET", "url": "https://gmail.googleapis.com/v1/users/me/messages"},
        )
    )
    # 1. Malformed method: method='api_fetch'
    action_malformed = ProposedAction(
        tool="api_fetch",
        arguments={"method": "api_fetch", "url": "https://gmail.googleapis.com/v1/users/me/messages"},
    )
    from worldmodel_harness.decisions import UnresolvedRequirements
    with pytest.raises(UnresolvedRequirements) as exc_info:
        api_requirements(store, scope, action_malformed)
    assert exc_info.value.details["reason"] == "malformed_http_method"
    assert exc_info.value.details["discovery_action"] is None
    assert "not an HTTP method" in exc_info.value.details["instruction"]

    # 2. Unsupported method: method='DELETE' on read-only endpoint
    action_unsupported = ProposedAction(
        tool="api_fetch",
        arguments={"method": "DELETE", "url": "https://gmail.googleapis.com/v1/users/me/messages"},
    )
    with pytest.raises(UnresolvedRequirements) as exc_info2:
        api_requirements(store, scope, action_unsupported)
    assert exc_info2.value.details["reason"] == "unsupported_method"
    assert exc_info2.value.details["discovery_action"] is None
    assert "supports: GET, but not 'DELETE'" in exc_info2.value.details["instruction"]

    # 3. Verify dispatch_decision returns this exact diagnostic to the model
    class ApiParams(BaseModel):
        method: str
        url: str

    tool = Tool(name="api_fetch", description="api", parameters=ApiParams, executor=lambda _: ToolResult(content="ok"))
    audit = []
    res = asyncio.run(
        dispatch_decision(
            proposal("api_fetch", {"method": "api_fetch", "url": "https://gmail.googleapis.com/v1/users/me/messages"}),
            tools={"api_fetch": tool},
            requirements_for=lambda a: api_requirements(store, scope, a),
            store=store,
            scope=scope,
            live=LiveState("task"),
            audit=audit,
        )
    )
    assert not res.success
    feedback = json.loads(res.content)
    assert feedback["status"] == "blocked"
    assert not feedback["environment_called"]
    assert feedback["repair"]["reason"] == "malformed_http_method"
    assert "not an HTTP method" in feedback["next_step"]
    store.close()


def test_stagnant_query_blocked_after_consecutive_empty_results(tmp_path):
    from worldmodel_harness.knowledge import Episode

    store = KnowledgeStore(tmp_path / "stagnant.sqlite")
    scope = Scope(environment="pipefy", version="1", role="ops", task_id="task1")

    class FetchParams(BaseModel):
        method: str = "GET"
        url: str
        params: str = ""

    calls = []
    tool = Tool(
        name="api_fetch",
        description="fetch",
        parameters=FetchParams,
        executor=lambda p: calls.append(p) or ToolResult(content=json.dumps({"records": []})),
    )

    query_args = {
        "method": "GET",
        "url": "https://api.pipefy.com/tables/ops_table/records:find",
        "params": json.dumps({"title": "Forklift Attachment"}),
    }

    # Record 2 prior empty episodes for this exact query
    for i in range(2):
        ep = Episode(
            task_id="task1",
            scope=scope,
            interaction_id=f"int_{i}",
            operation="api_fetch",
            arguments=query_args,
            response={"records": []},
            outcome="success",
        )
        store.append_episode(ep)

    decision = proposal("api_fetch", query_args)
    audit = []
    result = asyncio.run(
        dispatch_decision(
            decision,
            tools={"api_fetch": tool},
            requirements_for=lambda _: ActionRequirements(),
            store=store,
            evidence_store=store,
            scope=scope,
            live=LiveState("task1"),
            audit=audit,
        )
    )

    # 3rd attempt must be blocked locally by stagnant query guard
    assert not result.success
    assert len(calls) == 0  # Environment was never called!
    feedback = json.loads(result.content)
    assert feedback["status"] == "blocked"
    assert "action:stagnant_query" in feedback["gaps"]
    assert "Do not repeat the same query" in feedback["next_step"]
    store.close()


def test_mutation_on_excluded_entity_is_blocked(tmp_path):
    from worldmodel_harness.execution_state import EntityTracker, TrackedEntity, ChannelObligation

    store = KnowledgeStore(tmp_path / "excluded.sqlite")
    scope = Scope(environment="test", version="1", role="ops", task_id="task1")

    class MutateParams(BaseModel):
        summary: str

    calls = []
    tool = Tool(
        name="create_issue",
        description="create",
        parameters=MutateParams,
        executor=lambda p: calls.append(p) or ToolResult(content="ok"),
    )

    tracker = EntityTracker(required_channels=("action",))
    tracker.excluded_entities["Database Cluster Failover"] = "duplicate of INC-4921"
    tracker.entities["API Gateway Timeout"] = TrackedEntity(
        entity_id="API Gateway Timeout",
        name="API Gateway Timeout",
        obligations={"action": ChannelObligation(channel="action", status="pending")},
    )

    decision = proposal("create_issue", {"summary": "P1 Incident: Database Cluster Failover"})
    audit = []
    result = asyncio.run(
        dispatch_decision(
            decision,
            tools={"create_issue": tool},
            requirements_for=lambda _: ActionRequirements(),
            store=store,
            evidence_store=store,
            scope=scope,
            live=LiveState("task1"),
            audit=audit,
            entity_tracker=tracker,
        )
    )

    assert not result.success
    assert len(calls) == 0  # Environment mutation was prevented!
    feedback = json.loads(result.content)
    assert feedback["status"] == "blocked"
    assert "entity:excluded:Database Cluster Failover" in feedback["gaps"]
    assert "The active eligible entity to process is 'API Gateway Timeout'" in feedback["next_step"]
    store.close()






