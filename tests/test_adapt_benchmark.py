import asyncio
import json

import pytest

pytest.importorskip("automationbench")

from worldmodel_harness import adapt_benchmark
from worldmodel_harness.adaptation import ValidationTrial
from worldmodel_harness.config import ModelSettings
from worldmodel_harness.store import KnowledgeStore


def test_failed_proposal_charges_elapsed_cost_and_suppresses_error_details(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge.sqlite"
    KnowledgeStore(knowledge).close()
    monkeypatch.setattr(adapt_benchmark, "load_settings", lambda _: ModelSettings())
    monkeypatch.setattr(adapt_benchmark, "load_api_key", lambda *_: "not-a-real-key")
    monkeypatch.setattr(adapt_benchmark, "build_client", lambda *_: object())
    times = iter((100.0, 102.5))
    monkeypatch.setattr(adapt_benchmark, "perf_counter", lambda: next(times))

    async def failing_proposal(*args, **kwargs):
        raise RuntimeError("sensitive-provider-error")

    monkeypatch.setattr(adapt_benchmark, "propose", failing_proposal)
    spec = adapt_benchmark.AdaptationSpec(
        goal="lookup",
        parameter_types={},
        trials=tuple(
            ValidationTrial(task_name=name, inputs={})
            for name in (
                "simple.email_sf_contact_title_update",
                "simple.email_sf_contact_city_update",
            )
        ),
        estimated_reuses=100,
        estimated_saving_seconds_per_use=1,
    )
    with pytest.raises(RuntimeError):
        asyncio.run(adapt_benchmark.adapt_automationbench(tmp_path, knowledge, spec))
    path = next((tmp_path / "runs").glob("*/failure.json"))
    report = json.loads(path.read_text())
    assert report["ledger"]["adaptation_seconds"] == 2.5
    assert report["failed_provider_token_usage"] == "unavailable"
    assert "sensitive-provider-error" not in path.read_text()
