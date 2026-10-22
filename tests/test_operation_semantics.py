import asyncio
import json
from types import SimpleNamespace

from stirrup.core.models import AssistantMessage, TextBlock

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.goal_checks import GoalTracker
from worldmodel_harness.knowledge import Episode
from worldmodel_harness.learning import Learner
from worldmodel_harness.operation_semantics import is_observation_read
from worldmodel_harness.runtime import MeteredClient
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.verification_tools import CheckEffect, simple_check_tool


class DocumentationClient:
    def __init__(self, uncertain=False):
        self.uncertain = uncertain

    async def generate(self, messages, tools):
        prompt = json.loads(messages[1].content)
        if "proposal" in prompt:
            result = {"verdict": "uncertain" if self.uncertain else "supported_by_documentation",
                      "reason": "Review of the explicit non-mutating statement."}
        else:
            result = {"effect": "read", "quotes": [prompt["contract"]["description"]],
                      "reason": "The documented operation retrieves state without changing it."}
        return AssistantMessage(blocks=[TextBlock(text=json.dumps(result))])


def make_adapter(tmp_path, uncertain=False):
    store = KnowledgeStore(tmp_path / "store.sqlite")
    adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", store, Ledger(), Learner(store))
    description = "Returns the current records without changing any state."
    payload = {"id": "records.query", "method": "POST", "url": "https://example.test/query",
               "description": description}
    doc = Episode(task_id=adapter.scope.task_id, scope=adapter.scope, operation="api_search",
                  arguments={"query": "records"}, response={"results": [payload]},
                  coverage=("results",), outcome="success", purpose="discovery")
    store.append_episode(doc)
    contract = adapter.learner.document(doc, relation="endpoint_contract",
        participants=(payload["id"], payload["url"], "POST"), statement=description,
        payload=payload, scope=adapter.scope.model_copy(update={"task_id": None, "tenant": None}))
    adapter.semantics_client = MeteredClient(DocumentationClient(uncertain))
    return adapter, store, contract


def test_documented_post_read_can_verify_and_contract_change_invalidates(tmp_path):
    adapter, store, contract = make_adapter(tmp_path)
    task = "Set record-1 status to done."
    adapter.task = SimpleNamespace(prompt=[{"role": "user", "content": task}])
    adapter.goal_tracker = GoalTracker(adapter)

    def api_fetch(method, url):
        return json.dumps({"id": "record-1", "status": "done"})

    asyncio.run(adapter.execute(api_fetch, {"method": "POST", "url": "https://example.test/query"}))
    event = next(e for e in store.episodes(adapter.scope) if e.operation == "api_fetch")
    assert is_observation_read(adapter, event)
    assert event.effect_source_claim_id
    assert len(adapter.semantics_client.calls) == 2
    assert adapter.operation_semantics_audit[0]["evidence_level"] == "reviewed_model_document_interpretation"
    request = CheckEffect(requirement_id="status", task_quote=task, episode_id=event.episode_id,
                          path=("status",), expected="done", anchor_text="record-1")
    result = asyncio.run(simple_check_tool(adapter).executor(request))
    assert result.success, result.content
    store.put_claim(contract.model_copy(update={"version": 2, "status": "disputed"}))
    assert not is_observation_read(adapter, event)
    assert adapter.goal_tracker.assess(adapter.goal_tracker.checks["status"]) == "outcome_verification_requires_read"
    store.close()


def test_uncertain_review_does_not_authorize_read_or_repeat_model_calls(tmp_path):
    adapter, store, _ = make_adapter(tmp_path, uncertain=True)

    def api_fetch(method, url):
        return json.dumps({"id": "record-1", "status": "done"})

    for _ in range(2):
        asyncio.run(adapter.execute(api_fetch, {"method": "POST", "url": "https://example.test/query"}))
    events = [e for e in store.episodes(adapter.scope) if e.operation == "api_fetch"]
    assert all(not is_observation_read(adapter, e) and e.effect_kind == "unknown" for e in events)
    assert len(adapter.semantics_client.calls) == 2
    assert len(adapter.operation_semantics_audit) == 1
    store.close()
