import asyncio
import json

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset
from worldmodel_harness.budget import Ledger
from worldmodel_harness.checkpoint import copy_checkpoint
from worldmodel_harness.decisions import ProposedAction
from worldmodel_harness.learning import Learner
from worldmodel_harness.operational_requirements import api_requirements
from worldmodel_harness.store import KnowledgeStore


def test_frozen_discovery_contract_is_usable_locally_and_cannot_reach_siblings(tmp_path):
    source = tmp_path / "checkpoint.sqlite"
    KnowledgeStore(source).close()
    before = source.read_bytes()
    local = tmp_path / "local.sqlite"
    copy_checkpoint(source, local)
    store = KnowledgeStore(local)
    evidence = KnowledgeStore(tmp_path / "evidence.sqlite")
    adapter = AutomationBenchAdapter(dataset("simple")[0], "simple", evidence,
                                    Ledger(), knowledge=store)
    adapter.documentation_learner = Learner(store)

    def api_search(query):
        return json.dumps({"results": [{"id": "novel.read", "method": "GET",
            "url": "https://example.test/records", "description": "Read records"}]})

    asyncio.run(adapter.execute(api_search, {"query": "records"}))
    action = ProposedAction(tool="api_fetch", arguments={"method": "GET",
                              "url": "https://example.test/records"})
    assert api_requirements(store, adapter.scope, action).operation_id == "novel.read"
    assert {c.relation for c in store.claims()} == {"endpoint_contract"}
    assert adapter.learner is None
    sibling = tmp_path / "sibling.sqlite"
    copy_checkpoint(source, sibling)
    with_store = KnowledgeStore(sibling)
    assert with_store.claims() == []
    assert source.read_bytes() == before
    with_store.close()
    store.close()
    evidence.close()
