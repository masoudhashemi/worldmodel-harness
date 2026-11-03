import asyncio
import json

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset, task_row
from worldmodel_harness.budget import BudgetExceeded, Ledger

INBOX = "https://gmail.googleapis.com/gmail/v1/users/me/messages"


def test_baseline_adapter_exposes_upstream_tools():
    async def scenario():
        from automationbench.tools.api import api_search

        row = dataset("simple")[0]
        adapter = AutomationBenchAdapter(row, "simple", Ledger())
        tools = adapter.tools()
        assert {t.name for t in tools} == {"api_search", "api_fetch", "base64_encode"}
        assert all("world" not in t.parameters.model_fields for t in tools)
        assert not hasattr(adapter.task, "assertions")
        result = await adapter.execute(api_search, {"query": "salesforce contacts", "top_k": 2})
        assert json.loads(result.content) == json.loads(api_search("salesforce contacts", 2))
        assert adapter.ledger.environment_calls == 1
        assert adapter.ledger.by_purpose == {"discovery": 1}
        assert adapter.advisor is None
        assert set(adapter.score()) == {"partial_credit", "task_completed_correctly"}
        # Task initialization never mutates the dataset's state.
        assert row == dataset("simple")[0]

    asyncio.run(scenario())


def test_budget_applies_before_environment_call():
    async def scenario():
        from automationbench.tools.api import api_search

        adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", Ledger(max_environment_calls=0))
        with pytest.raises(BudgetExceeded):
            await adapter.execute(api_search, {"query": "salesforce contacts"})
        assert adapter.ledger.environment_calls == 0

    asyncio.run(scenario())


def test_tool_exception_is_returned_to_the_agent_like_upstream():
    async def scenario():
        from automationbench.tools.api import api_fetch

        adapter = AutomationBenchAdapter(task_row("operations.asana_fire_drill"), "operations", Ledger())
        # A path parameter repeated in params makes the upstream route raise TypeError.
        arguments = {
            "method": "GET",
            "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_ops_policy/values/ws_email_rules!A1:Z100",
            "params": json.dumps({"range": "ws_email_rules!A1:Z100", "spreadsheetId": "ss_ops_policy"}),
        }
        with pytest.raises(TypeError) as upstream:
            api_fetch(world=adapter._state["world"].model_copy(deep=True), **arguments)
        result = await adapter.execute(api_fetch, arguments)
        assert not result.success
        assert result.content == str(upstream.value)

    asyncio.run(scenario())


def test_query_string_in_url_is_moved_into_params():
    async def scenario():
        from automationbench.tools.api import api_fetch

        adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", Ledger())
        split = await adapter.execute(api_fetch, {"method": "GET", "url": INBOX + "?q=in:inbox"})
        explicit = await adapter.execute(
            api_fetch, {"method": "GET", "url": INBOX, "params": '{"q":"in:inbox"}'}
        )
        assert split.success and json.loads(split.content) == json.loads(explicit.content)

    asyncio.run(scenario())
