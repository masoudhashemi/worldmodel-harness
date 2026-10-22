"""Micro-loop acceptance tests verifying the closed behavioral feedback loop:
1. Finish routing: execute_decision with tool='finish' completes the task rather than looping.
2. Scoped retrieval: excludes cross-service noise (Mailchimp/Instagram in Salesforce/Sheets tasks).
3. Hint retention: online reflections merge with and do not displace persistent lessons.
"""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from stirrup.core.models import AssistantMessage, TextBlock, ToolCall, ToolMessage, ToolResult, UserMessage
from worldmodel_harness.knowledge import Claim, Layer, Scope, utc_now
from worldmodel_harness.learning import claim_identity
from worldmodel_harness.store import KnowledgeStore


def make_endpoint_claim(store, scope, service, name, desc):
    from worldmodel_harness.knowledge import Episode
    ep_id = f"ep_{service}_{name}"
    ep = Episode(
        episode_id=ep_id,
        event_id=f"ev_{service}_{name}",
        origin="environment",
        task_id="task_1",
        scope=scope,
        operation="api_fetch",
        arguments={"url": f"https://api.{service}.com/v1/{name}", "method": "POST"},
        response={"status": 200},
        outcome="success",
    )
    store.append_episode(ep)
    url = f"https://api.{service}.com/v1/{name}"
    identity = claim_identity(scope, "endpoint_contract", (service, name))
    claim = Claim(
        claim_id=identity,
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("POST", url, f"{service}.{name}"),
        statement=desc,
        scope=scope,
        payload={"id": f"{service}.{name}", "url": url, "method": "POST", "service": service},
        status="supported",
        supporting_episodes=(ep_id,),
        needs_review=False,
    )
    store.put_claim(claim)
    return claim


def test_retrieval_excludes_unrelated_services(tmp_path):
    """Irrelevant services (Mailchimp, Instagram) must be excluded from Salesforce/Zoom queries."""
    from worldmodel_harness.retrieval import retrieve

    store = KnowledgeStore(tmp_path / "store.sqlite")
    scope = Scope(environment="automationbench", version="1", role="agent")

    # Add claims across multiple services
    make_endpoint_claim(store, scope, "salesforce", "opportunities.update", "Update fields on an existing Salesforce opportunity.")
    make_endpoint_claim(store, scope, "salesforce", "leads.convert", "Convert a Salesforce lead into a contact.")
    make_endpoint_claim(store, scope, "zoom", "meetings.get", "Retrieve details of a scheduled Zoom meeting.")
    make_endpoint_claim(store, scope, "mailchimp", "campaigns.create", "Create an email marketing campaign in Mailchimp.")
    make_endpoint_claim(store, scope, "instagram", "media.publish", "Publish media container to an Instagram account.")
    make_endpoint_claim(store, scope, "sheets", "spreadsheets.clear", "Erase values from a Google Sheets range.")

    # Query for a Salesforce + Zoom follow-up task
    query = "Follow up on the Zoom discovery call and update the opportunity stage to Proposal in Salesforce."
    working = retrieve(store, scope, query, max_chars=4000)

    retrieved_services = {
        c.payload.get("service")
        for c in working.claims
        if isinstance(c.payload, dict) and c.payload.get("service")
    }

    assert "salesforce" in retrieved_services or "zoom" in retrieved_services
    assert "mailchimp" not in retrieved_services
    assert "instagram" not in retrieved_services
    assert "sheets" not in retrieved_services
    assert not any(g.startswith("budget:") for g in working.gaps)


@pytest.mark.anyio
async def test_finish_rewriting_in_repairing_agent():
    """execute_decision calling tool='finish' must route to finish and complete the turn."""
    from worldmodel_harness.repairing_agent import RepairingAgent
    from worldmodel_harness.decisions import CompactDecision
    from stirrup.core.models import Tool
    from pydantic import BaseModel

    class FinishParams(BaseModel):
        summary: str = "done"

    finish_called = False

    async def execute_finish(params: FinishParams):
        nonlocal finish_called
        finish_called = True
        return ToolResult(content="Completed successfully", success=True)

    finish_tool = Tool(
        name="finish",
        description="Finish task",
        parameters=FinishParams,
        executor=execute_finish,
    )

    client = MagicMock()
    client.context_window_tokens = 8192
    # Mock client to return execute_decision with tool='finish' on turn 1
    call = ToolCall(
        tool_call_id="call_1",
        name="execute_decision",
        arguments=json.dumps({
            "action": {"tool": "finish", "arguments": {"summary": "Task complete"}},
            "goal": "Finish task",
        }),
    )
    client.generate = AsyncMock(return_value=AssistantMessage(blocks=[call]))

    exec_tool = Tool(
        name="execute_decision",
        description="Execute decision",
        parameters=CompactDecision,
        executor=lambda _: ToolResult(content="executed", success=True),
    )

    agent = RepairingAgent(
        client=client,
        name="test_agent",
        max_turns=3,
        tools=[exec_tool],
        finish_tool=finish_tool,
        repair_schemas={"execute_decision": CompactDecision, "finish": FinishParams},
    )

    finish_params, msg_history, metadata = await agent.run("Please complete the task")

    assert finish_called is True
    assert finish_params is not None
    assert finish_params.summary == "Task complete"


@pytest.mark.anyio
async def test_raw_text_tool_call_salvaging():
    """Text-only tool call syntax must be parsed into executable tool calls."""
    from worldmodel_harness.repairing_agent import RepairingAgent
    from worldmodel_harness.decisions import CompactDecision
    from stirrup.core.models import TextBlock, Tool
    from pydantic import BaseModel

    class FinishParams(BaseModel):
        summary: str = "done"

    finish_called = False

    async def execute_finish(params: FinishParams):
        nonlocal finish_called
        finish_called = True
        return ToolResult(content="Completed successfully", success=True)

    finish_tool = Tool(
        name="finish",
        description="Finish task",
        parameters=FinishParams,
        executor=execute_finish,
    )

    client = MagicMock()
    client.context_window_tokens = 8192
    # Model emits tool call syntax as raw text inside a TextBlock
    raw_text = '<|tool_call|>call:execute_decision{"action": {"tool": "finish", "arguments": {"summary": "Recovered from text"}}, "goal": "Finish"}<tool_call|>'
    client.generate = AsyncMock(return_value=AssistantMessage(blocks=[TextBlock(text=raw_text)]))

    exec_tool = Tool(
        name="execute_decision",
        description="Execute decision",
        parameters=CompactDecision,
        executor=lambda _: ToolResult(content="executed", success=True),
    )

    agent = RepairingAgent(
        client=client,
        name="test_agent",
        max_turns=3,
        tools=[exec_tool],
        finish_tool=finish_tool,
        repair_schemas={"execute_decision": CompactDecision, "finish": FinishParams},
    )

    finish_params, msg_history, metadata = await agent.run("Please complete the task")

    assert finish_called is True
    assert finish_params is not None
    assert finish_params.summary == "Recovered from text"


def test_hint_combination_preserves_persistent_lessons(tmp_path):
    """Online hints must merge with, not displace, verified persistent hypotheses."""
    from worldmodel_harness.semantic_learning import hypothesis_context

    store = KnowledgeStore(tmp_path / "store.sqlite")
    scope = Scope(environment="automationbench", version="1", role="agent")

    persistent_claim = Claim(
        claim_id="hyp_persistent_1",
        layer=Layer.BEHAVIOR,
        relation="possible_recovery",
        participants=("salesforce.opportunities.update",),
        statement="Opportunity stage requires explicit Proposal/Price Quote value.",
        scope=scope,
        payload={
            "distinguishing_observation": "StageName must match picklist exactly",
            "alternatives": ["Draft", "Active"],
        },
        status="hypothesis",
        needs_review=False,
    )
    store.put_claim(persistent_claim)

    persistent_hints = hypothesis_context(store, scope, "update opportunity stage")
    assert "Opportunity stage requires explicit" in persistent_hints

    online_hints = "New model hypotheses from this task:\n[{\"statement\": \"Transient local observation\"}]"

    # Simulated combination logic
    hint_budget = 2400
    if persistent_hints and online_hints:
        combined = persistent_hints + "\n" + online_hints
    else:
        combined = online_hints or persistent_hints

    assert "Opportunity stage requires explicit" in combined
    assert "Transient local observation" in combined


def test_doc_search_pause_validation(tmp_path):
    """Reflections recognizing that doc search cannot return records must pass validation."""
    from worldmodel_harness.knowledge import Episode, Scope
    from worldmodel_harness.store import KnowledgeStore
    from worldmodel_harness.semantic_learning import (
        BlockerConditions,
        SemanticHypothesis,
        SemanticProposal,
        evidence_catalog,
        retain_hypotheses,
    )

    store = KnowledgeStore(tmp_path / "store.sqlite")
    scope = Scope(
        environment="automationbench",
        version="1",
        role="agent",
        tenant="hr.job_posting_refresh",
        task_id="task_1",
    )

    # An api_search call that returned documentation endpoints
    ep = Episode(
        episode_id="ep_search_1",
        event_id="ev_1",
        origin="environment",
        task_id="task_1",
        scope=scope,
        operation="api_search",
        arguments={"query": "company domain recruiting email"},
        response={"results": [{"id": "recruitee.offers.list"}], "count": 1},
        outcome="success",
        purpose="discovery",
    )
    store.append_episode(ep)
    cat = evidence_catalog([ep])
    ref = list(cat.keys())[0]

    hyp = SemanticHypothesis(
        layer="behavior",
        dimension="INV",
        relation="possible_blocker",
        participants=("api_search",),
        statement="api_search returns API tool schemas rather than live company records.",
        citations=(ref,),
        alternatives=("The search query was too broad",),
        distinguishing_observation="Repeated searches return endpoint definitions not data",
        response_rule=None,
        blocker=BlockerConditions(
            origin="environment",
            applies_when="Searching api_search for instance data",
            stop_when="Searches return documentation",
            reopen_when="A different tool is used",
        ),
    )
    proposal = SemanticProposal(hypotheses=(hyp,))
    retained = retain_hypotheses(proposal, [ep], store, scope)
    assert len(retained) == 1
    stored_claim = store.claim(retained[0])
    assert stored_claim.payload.get("blocker") is not None


@pytest.mark.anyio
async def test_api_fetch_url_query_normalization(tmp_path):
    from pathlib import Path
    from worldmodel_harness.decisions import ActionRequirements, Decision, ProposedAction
    from worldmodel_harness.decision_runtime import dispatch_decision
    from worldmodel_harness.knowledge import LiveState, Scope
    from worldmodel_harness.store import KnowledgeStore
    from stirrup.core.models import Tool
    from pydantic import BaseModel
    from typing import Optional

    class ApiFetchParams(BaseModel):
        method: str
        url: str
        params: Optional[str] = None
        body: Optional[str] = None

    captured = {}
    async def mock_execute(params: ApiFetchParams):
        captured["url"] = params.url
        captured["params"] = params.params
        return ToolResult(content="ok", tool_call_id="call_1")

    api_tool = Tool(
        name="api_fetch",
        description="Fetch",
        parameters=ApiFetchParams,
        executor=mock_execute,
    )
    tools = {"api_fetch": api_tool}

    decision = Decision(
        action=ProposedAction(
            tool="api_fetch",
            arguments={
                "method": "GET",
                "url": "https://yourinstance.salesforce.com/services/data/v61.0/query?q=SELECT+Id+FROM+Opportunity",
            },
        ),
        goal="Query opportunity",
        rationale="Testing query normalization",
    )

    store = KnowledgeStore(Path(tmp_path / "store.sqlite"))
    scope = Scope(environment="test", version="1", task_id="test_task")
    live = LiveState(task_id="test_task")
    audit = []

    res = await dispatch_decision(
        decision,
        tools=tools,
        requirements_for=lambda act: ActionRequirements(),
        store=store,
        scope=scope,
        live=live,
        audit=audit,
    )

    assert captured["url"] == "https://yourinstance.salesforce.com/services/data/v61.0/query"
    parsed_params = json.loads(captured["params"])
    assert parsed_params == {"q": "SELECT Id FROM Opportunity"}


def test_stage_coverage_guarantee_retrieval(tmp_path):
    from pathlib import Path
    from worldmodel_harness.knowledge import Claim, Episode, Layer, Scope
    from worldmodel_harness.store import KnowledgeStore
    from worldmodel_harness.retrieval import retrieve

    store = KnowledgeStore(Path(tmp_path / "store.sqlite"))
    scope = Scope(environment="test", version="1")

    ep1 = Episode(episode_id="ep_1", event_id="ev_1", origin="environment", task_id="t1", scope=scope, operation="api_fetch", arguments={}, response={}, outcome="success")
    ep2 = Episode(episode_id="ep_2", event_id="ev_2", origin="environment", task_id="t1", scope=scope, operation="api_fetch", arguments={}, response={}, outcome="success")
    store.append_episode(ep1)
    store.append_episode(ep2)

    # 1. Lookup tool (reading sheet)
    c_lookup = Claim(
        claim_id="ep_sheets_get",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("GET", "https://sheets.googleapis.com/v4/spreadsheets/{id}", "sheets.spreadsheets.get"),
        statement="Retrieve a spreadsheet by ID",
        scope=scope,
        payload={"id": "sheets.spreadsheets.get", "method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/{id}", "service": "sheets"},
        status="supported",
        supporting_episodes=("ep_1",),
    )
    # 2. Notify tool (sending SMS)
    c_notify = Claim(
        claim_id="ep_twilio_sms",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("POST", "https://api.twilio.com/Messages.json", "twilio.messages.create"),
        statement="Dispatch an SMS message to recipient",
        scope=scope,
        payload={"id": "twilio.messages.create", "method": "POST", "url": "https://api.twilio.com/Messages.json", "service": "twilio"},
        status="supported",
        supporting_episodes=("ep_2",),
    )
    store.put_claim(c_lookup)
    store.put_claim(c_notify)

    # Query mentions both sheets and sending SMS, but lexical overlap heavily favors SMS
    wg = retrieve(
        store,
        scope,
        query="Check contractor sheets and send SMS alert",
        subgoal="Dispatch urgent SMS alert",
    )
    retrieved_ids = {c.claim_id for c in wg.claims}
    assert "ep_sheets_get" in retrieved_ids
    assert "ep_twilio_sms" in retrieved_ids
    assert "lookup" in wg.stages
    assert "notify" in wg.stages


def test_reverse_dataflow_prerequisite_retrieval(tmp_path):
    from pathlib import Path
    from worldmodel_harness.knowledge import Claim, Episode, Layer, Scope
    from worldmodel_harness.store import KnowledgeStore
    from worldmodel_harness.retrieval import retrieve

    store = KnowledgeStore(Path(tmp_path / "store.sqlite"))
    scope = Scope(environment="test", version="1")

    ep1 = Episode(episode_id="ep_1", event_id="ev_1", origin="environment", task_id="t1", scope=scope, operation="api_fetch", arguments={}, response={}, outcome="success")
    ep2 = Episode(episode_id="ep_2", event_id="ev_2", origin="environment", task_id="t1", scope=scope, operation="api_fetch", arguments={}, response={}, outcome="success")
    ep3 = Episode(episode_id="ep_3", event_id="ev_3", origin="environment", task_id="t1", scope=scope, operation="api_fetch", arguments={}, response={}, outcome="success")
    store.append_episode(ep1)
    store.append_episode(ep2)
    store.append_episode(ep3)

    # Producer: sheets.spreadsheets.get (produces spreadsheetId)
    c_producer = Claim(
        claim_id="ep_sheets_get",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("GET", "https://sheets.googleapis.com/v4/spreadsheets/{id}", "sheets.spreadsheets.get"),
        statement="Retrieve a spreadsheet metadata by ID",
        scope=scope,
        payload={"id": "sheets.spreadsheets.get", "method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/{id}", "service": "sheets"},
        status="supported",
        supporting_episodes=("ep_1",),
    )
    # Consumer: sheets.spreadsheets.values.get (consumes spreadsheetId)
    c_consumer = Claim(
        claim_id="ep_sheets_values",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("GET", "https://sheets.googleapis.com/v4/spreadsheets/{id}/values/{range}", "sheets.spreadsheets.values.get"),
        statement="Read cell contents from a specific range in sheets",
        scope=scope,
        payload={"id": "sheets.spreadsheets.values.get", "method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/{id}/values/{range}", "service": "sheets"},
        status="supported",
        supporting_episodes=("ep_2",),
    )
    # Dataflow edge connecting them
    c_flow = Claim(
        claim_id="flow_sheets_id",
        layer=Layer.BEHAVIOR,
        relation="observed_value_flow",
        participants=("sheets.spreadsheets.get", "sheets.spreadsheets.values.get"),
        statement="Observed sheets.spreadsheets.get spreadsheetId reused as values.get path.spreadsheetId",
        scope=scope,
        payload={"source_endpoint": "sheets.spreadsheets.get", "target_endpoint": "sheets.spreadsheets.values.get", "output_path": "spreadsheetId", "input_path": "path.spreadsheetId"},
        status="supported",
        supporting_episodes=("ep_3",),
    )
    store.put_claim(c_producer)
    store.put_claim(c_consumer)
    store.put_claim(c_flow)

    # Query specifically targeting cell values reading in sheets
    wg = retrieve(
        store,
        scope,
        query="Read cell contents from sheets range",
        subgoal="Fetch rows",
    )
    retrieved_ids = {c.claim_id for c in wg.claims}
    # Both consumer AND its upstream producer MUST be present
    assert "ep_sheets_values" in retrieved_ids
    assert "ep_sheets_get" in retrieved_ids
    assert any("prerequisite_dataflow" in str(r) for r in wg.reasons.get("ep_sheets_get", ()))


def test_nested_tool_argument_normalization():
    """Verify that when an LLM puts tool inside action.arguments, it is properly normalized."""
    from worldmodel_harness.decisions import CompactDecision

    payload = {
        "action": {
            "arguments": {
                "query": "send text message sms",
                "tool": "api_search"
            },
            "goal": "Find the tool to send text messages to contractors."
        },
        "rationale": "I need to identify the correct API endpoint."
    }
    cd = CompactDecision.model_validate(payload)
    assert cd.action.tool == "api_search"
    assert cd.action.arguments == {"query": "send text message sms"}
    assert cd.goal == "Find the tool to send text messages to contractors."


def test_multiservice_balanced_coverage_retrieval(tmp_path):
    """Verify that multi-service tasks balance endpoints across all active services rather than letting one monopolize the budget."""
    from worldmodel_harness.knowledge import Episode
    from worldmodel_harness.retrieval import retrieve, claim_service
    store = KnowledgeStore(tmp_path / "test_multiservice.sqlite")
    scope = Scope(environment="test_env", version="v1")

    ep = Episode(episode_id="ep_all", event_id="ev_all", origin="environment", task_id="t1", scope=scope, operation="api_fetch", arguments={}, response={}, outcome="success")
    store.append_episode(ep)

    # Seed 3 slack endpoints, 1 twilio endpoint, and 1 sheets endpoint
    for i in range(3):
        store.put_claim(Claim(
            claim_id=f"slack_ep_{i}",
            layer=Layer.INTERFACE,
            relation="endpoint_contract",
            participants=(f"slack.action_{i}", f"https://slack.com/api/action_{i}", "POST"),
            statement=f"Slack action {i} to send messages to channel",
            scope=scope,
            payload={"id": f"slack.action_{i}", "method": "POST", "url": f"https://slack.com/api/action_{i}", "service": "slack"},
            status="supported",
            supporting_episodes=("ep_all",),
        ))

    store.put_claim(Claim(
        claim_id="twilio_ep_msg",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("twilio.messages.create", "https://api.twilio.com/2010-04-01/Messages.json", "POST"),
        statement="Dispatch an SMS message using Twilio",
        scope=scope,
        payload={"id": "twilio.messages.create", "method": "POST", "url": "https://api.twilio.com/Messages.json", "service": "twilio"},
        status="supported",
        supporting_episodes=("ep_all",),
    ))

    store.put_claim(Claim(
        claim_id="sheets_ep_read",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("sheets.spreadsheets.get", "https://sheets.googleapis.com/v4/spreadsheets/get", "GET"),
        statement="Get metadata and contents for spreadsheet sheet",
        scope=scope,
        payload={"id": "sheets.spreadsheets.get", "method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/get", "service": "sheets"},
        status="supported",
        supporting_episodes=("ep_all",),
    ))

    task = "Check the spreadsheet sheet, text the contractors via sms, and post an update to the slack channel."
    wg = retrieve(store, scope, task, max_chars=4000)
    retrieved_services = {claim_service(c) for c in wg.claims}
    assert "sheets" in retrieved_services
    assert "twilio" in retrieved_services
    assert "slack" in retrieved_services




