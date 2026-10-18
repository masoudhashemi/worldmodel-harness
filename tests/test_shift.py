import asyncio
import json

import pytest

from worldmodel_harness.shift import ResponseShift, ShiftController

URL = "https://yourinstance.salesforce.com/services/data/v61.0/query"


def test_shift_is_targeted_delayed_and_does_not_mutate_original():
    controller = ShiftController(
        ResponseShift(shift_id="test", url=URL, field="count", after_matching_calls=1)
    )
    original = {"count": 2, "records": []}
    assert controller.apply("api_fetch", {"url": "other"}, original, "success") == (original, False)
    assert controller.apply("api_fetch", {"url": URL}, original, "success") == (original, False)
    changed, applied = controller.apply("api_fetch", {"url": URL}, original, "success")
    assert applied and changed["count"] == "2"
    assert original["count"] == 2
    assert controller.apply("api_fetch", {"url": URL}, {"count": True}, "success")[1] is False
    assert controller.apply("api_fetch", {"url": URL}, original, "unknown")[1] is False
    assert controller.report()["detection_delay_calls"] is None


def test_unannounced_upstream_shift_disputes_claim_and_stops_procedure(tmp_path):
    pytest.importorskip("automationbench")
    from automationbench.tools.api import api_fetch, api_search

    from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
    from worldmodel_harness.budget import Ledger
    from worldmodel_harness.knowledge import Dependency, Scope
    from worldmodel_harness.learning import Learner
    from worldmodel_harness.procedures import Procedure, Step, ValidationCase, execute
    from worldmodel_harness.store import KnowledgeStore

    async def scenario():
        store = KnowledgeStore(tmp_path / "knowledge.sqlite")
        learner = Learner(store)
        args = {"method": "GET", "url": URL, "params": '{"q":"SELECT Id FROM Contact"}'}
        cases = []
        for row in list(dataset("simple"))[:2]:
            adapter = AutomationBenchAdapter(row, "simple", store, Ledger(), learner)
            await adapter.execute(api_search, {"query": "salesforce query", "top_k": 5})
            result = await adapter.execute(api_fetch, args, purpose="validation")
            assert type(json.loads(result.content)["count"]) is int
            event = store.episodes(adapter.scope)[-1]
            cases.append(
                ValidationCase(
                    task_id=event.task_id,
                    task_name=adapter.task.task_id,
                    passed=True,
                    isolated_execution=True,
                    evidence_ids=(event.episode_id,),
                )
            )
        count = next(
            c
            for c in store.claims()
            if c.relation == "observed_output_type" and c.payload["field"] == "count"
        )
        assert count.status == "supported"
        shifted = AutomationBenchAdapter(
            dataset("simple")[2],
            "simple",
            store,
            Ledger(),
            learner,
            shift=ResponseShift(shift_id="count-change", url=URL, field="count"),
        )
        tools = {t.name: t for t in shifted.tools()}
        artifact = Procedure(
            procedure_id="query-count",
            goal_pattern="query count",
            scope=Scope(environment=shifted.scope.environment, version=shifted.scope.version),
            parameter_types={},
            required_claims=(Dependency(claim_id=count.claim_id, version=count.version),),
            steps=(
                Step(
                    name="fetch", kind="call", tool="api_fetch", arguments=args, next_step="verify"
                ),
                Step(
                    name="verify",
                    kind="verify",
                    value={"$ref": "steps.fetch.data.count"},
                    predicate="type",
                    expected="integer",
                    next_step="stop",
                ),
                Step(name="stop", kind="stop"),
            ),
            validation=tuple(cases),
            estimated_reuses=100,
            estimated_saving_per_use=1,
            status="active",
        )
        store.put_procedure(artifact, tools=tools)
        result = await execute(artifact, {}, tools, store, shifted.scope, shifted.live)
        assert not result.completed
        assert result.executed_steps == ("fetch",)
        assert store.claim(count.claim_id).status == "disputed"
        assert store.procedures()[0].status == "suspended"
        report = shifted.shift.report()
        assert report["detection_delay_calls"] == 0
        assert report["notice_supplied"] is False
        assert report["events"][0]["detected_claim_ids"] == [count.claim_id]
        event = store.episodes(shifted.scope)[-1]
        assert isinstance(event.response["count"], str)
        assert "shift_id" not in event.model_dump_json()
        assert shifted.ledger.environment_calls == 1
        store.close()

    asyncio.run(scenario())
