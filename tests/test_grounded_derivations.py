import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from worldmodel_harness.goal_checks import GoalTracker
from worldmodel_harness.grounding import EvidenceValue, GroundedDerivation, IdentityJoin
from worldmodel_harness.knowledge import Episode, LiveState, Scope, utc_now
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.verification_tools import CheckEffect, simple_check_tool


@pytest.fixture
def world(tmp_path):
    store = KnowledgeStore(tmp_path / "evidence.sqlite")
    scope = Scope(environment="test", version="1", task_id="task")
    task = "Calculate the gross margin percentage for each active product and update its report."
    adapter = SimpleNamespace(evidence=store, knowledge=store, scope=scope, live=LiveState("task"),
                              task=SimpleNamespace(prompt=[{"role": "user", "content": task}]))
    adapter.goal_tracker = GoalTracker(adapter)
    start = utc_now()

    def observe(response, seconds, method="GET", resource="source"):
        event = Episode(task_id="task", scope=scope, operation="api_fetch",
            arguments={"method": method, "url": "https://example.test/" + resource},
            response=response, coverage=tuple(response), outcome="success",
            timestamp=start + timedelta(seconds=seconds))
        store.append_episode(event)
        adapter.live.observe(event)
        return event

    source = observe({"id": "product-1", "revenue": 100, "cost": 60, "active": True}, 0)
    unrelated = observe({"id": "product-2", "revenue": 100, "cost": 20, "active": True}, 1)
    write = observe({"id": "product-1"}, 2, "PATCH", "report")
    target = observe({"id": "product-1", "margin": 40}, 3, resource="report")
    proof = GroundedDerivation(
        inputs={name: EvidenceValue(episode_id=source.episode_id, path=(name,))
                for name in ("id", "revenue", "cost", "active")},
        expression="round((revenue - cost) / revenue * 100)", selection="active",
        joins=(IdentityJoin(source_name="id", target_path=("id",)),),
        write_episode_id=write.episode_id,
    )
    request = CheckEffect(requirement_id="margin", task_quote=task, episode_id=target.episode_id,
                          path=("margin",), expected=40, anchor_text="active product", derivation=proof)
    yield adapter, request, source, unrelated, target
    store.close()


def test_model_supplied_arithmetic_and_discovered_identity_can_be_checked(world):
    adapter, request, *_ = world
    result = asyncio.run(simple_check_tool(adapter).executor(request))
    assert result.success, result.content
    assert adapter.goal_tracker.assess(adapter.goal_tracker.checks["margin"]) == "matched"
    # Neither the product identifier nor the calculated value appears in the task.
    assert "product-1" not in request.task_quote and "40" not in request.task_quote


@pytest.mark.parametrize("case", ["wrong_value", "target_as_source", "wrong_identity", "code", "constant"])
def test_derivation_rejects_invalid_provenance_or_computation(world, case):
    adapter, request, _source, unrelated, _target = world
    proof = request.derivation
    target = _target
    if case == "wrong_value":
        request = request.model_copy(update={"expected": 41})
    elif case == "target_as_source":
        proof = proof.model_copy(update={"inputs": {**proof.inputs,
            "cost": EvidenceValue(episode_id=target.episode_id, path=("margin",))}})
    elif case == "wrong_identity":
        proof = proof.model_copy(update={"inputs": {**proof.inputs,
            "id": EvidenceValue(episode_id=unrelated.episode_id, path=("id",))}})
    elif case == "code":
        proof = proof.model_copy(update={"expression": "__import__('os').system(revenue)"})
    else:
        proof = proof.model_copy(update={"expression": "40"})
    request = request.model_copy(update={"derivation": proof})
    result = asyncio.run(simple_check_tool(adapter).executor(request))
    assert not result.success
    assert json.loads(result.content)["result"] != "matched"
    assert not adapter.goal_tracker.checks


def test_derivation_rejects_values_from_an_unjoined_record(world):
    adapter, request, _source, unrelated, _target = world
    # Arithmetic still yields 40, but mixes product-1's revenue with product-2's cost.
    # Identity verification of product-1 alone must not bless this unrelated input.
    proof = request.derivation.model_copy(update={
        "inputs": {**request.derivation.inputs,
                   "cost": EvidenceValue(episode_id=unrelated.episode_id, path=("cost",))},
        "expression": "round((revenue - cost * 3) / revenue * 100)",
    })
    result = asyncio.run(simple_check_tool(adapter).executor(
        request.model_copy(update={"derivation": proof})))
    assert not result.success, "An unjoined source record must not ground the target's value"


def test_derivation_accepts_transitive_source_identity_join(world):
    adapter, request, _source, unrelated, _target = world
    # A separate observed record carries a foreign key to the first record.
    detail = unrelated.model_copy(update={"episode_id": "detail", "event_id": "detail-event",
        "response": {"product_id": "product-1", "cost": 60},
        "coverage": ("product_id", "cost")})
    adapter.evidence.append_episode(detail)
    proof = request.derivation.model_copy(update={
        "inputs": {**request.derivation.inputs,
            "detail_id": EvidenceValue(episode_id="detail", path=("product_id",)),
            "cost": EvidenceValue(episode_id="detail", path=("cost",))},
        "joins": (*request.derivation.joins,
                  IdentityJoin(source_name="detail_id", other_source_name="id")),
    })
    result = asyncio.run(simple_check_tool(adapter).executor(
        request.model_copy(update={"derivation": proof})))
    assert result.success, result.content


@pytest.mark.parametrize("container", [False, True])
def test_derivation_does_not_join_other_rows_in_same_response(world, container):
    adapter, request, _source, unrelated, _target = world
    source = _source
    rows = source.model_copy(update={"episode_id": "rows", "event_id": "rows-event",
        "response": {"rows": [source.response, unrelated.response]}, "coverage": ("rows",)})
    adapter.evidence.append_episode(rows)
    inputs = {name: EvidenceValue(episode_id="rows", path=("rows", 0, name))
              for name in ("id", "revenue", "cost", "active")}
    inputs["cost"] = EvidenceValue(episode_id="rows", path=("rows",) if container else ("rows", 1, "cost"))
    proof = request.derivation.model_copy(update={"inputs": inputs,
        "expression": "round((revenue - cost[1]['cost'] * 3) / revenue * 100)" if container
                      else "round((revenue - cost * 3) / revenue * 100)"})
    result = asyncio.run(simple_check_tool(adapter).executor(
        request.model_copy(update={"derivation": proof})))
    assert not result.success
