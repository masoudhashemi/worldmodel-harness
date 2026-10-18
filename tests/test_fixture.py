import asyncio
import json

import pytest

from worldmodel_harness.fixture import TicketFixture, TicketInput


def test_timeout_is_unknown_and_lookup_resolves_it():
    async def scenario():
        fixture = TicketFixture()
        params = TicketInput(external_id="incident-1")
        result = await fixture.create(params)
        assert not result.success
        assert fixture.observations[0].outcome == "unknown"
        assert "id" not in fixture.observations[0].result
        await fixture.lookup(params)
        ticket_id = fixture.observations[1].result["tickets"][0]["id"]
        assert fixture.grade("incident-1", ticket_id)
        assert len(fixture.observations) == 2

    asyncio.run(scenario())


def test_blind_retry_creates_duplicate_and_fails_grade():
    async def scenario():
        fixture = TicketFixture()
        params = TicketInput(external_id="incident-1")
        await fixture.create(params)
        result = await fixture.create(params)
        ticket_id = json.loads(result.content)["result"]["id"]
        assert not fixture.grade("incident-1", ticket_id)

    asyncio.run(scenario())


def test_budget_blocks_before_side_effect():
    async def scenario():
        fixture = TicketFixture(call_budget=1)
        params = TicketInput(external_id="incident-1")
        await fixture.create(params)
        with pytest.raises(RuntimeError, match="budget"):
            await fixture.create(params)
        assert len(fixture._tickets) == 1
        assert len(fixture.observations) == 1

    asyncio.run(scenario())


def test_new_task_has_no_live_state_from_previous_task():
    async def scenario():
        first, second = TicketFixture(), TicketFixture()
        params = TicketInput(external_id="incident-1")
        await first.create(params)
        await second.lookup(params)
        assert second.observations[0].result == {"tickets": []}
        assert first.task_id != second.task_id

    asyncio.run(scenario())
