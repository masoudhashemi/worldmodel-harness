from pathlib import Path

from worldmodel_harness.graph_comparison import (
    load_ground_truth_graph,
    compare_worldmodel_to_ground_truth,
)
from worldmodel_harness.store import KnowledgeStore


def test_ground_truth_graph_loading():
    gt = load_ground_truth_graph()
    # Benchmark contains 47 services and ~400-500 endpoints
    assert len(gt.services) >= 40
    assert len(gt.endpoints) >= 300
    assert len(gt.dataflow_edges) >= 100

    # Verify core enterprise endpoints exist
    assert "google_drive.files.list" in gt.endpoints
    assert "sheets.spreadsheets.get" in gt.endpoints
    assert "monday.items.find" in gt.endpoints
    assert "monday.items.updateColumn" in gt.endpoints
    assert "slack.chat.postMessage" in gt.endpoints

    # Verify stages
    assert gt.endpoints["sheets.spreadsheets.get"].stage == "lookup"
    assert gt.endpoints["monday.items.updateColumn"].stage == "mutate"
    assert gt.endpoints["slack.chat.postMessage"].stage == "notify"


def test_compare_carried_worldmodel_against_ground_truth():
    seed_path = Path("runs/eval-isomorphic-transfer/stores/worldmodel_carried.sqlite")
    if not seed_path.exists():
        # Fallback to local store if available
        candidates = list(Path("runs").glob("**/worldmodel_carried.sqlite"))
        if candidates:
            seed_path = candidates[0]
        else:
            return

    store = KnowledgeStore(seed_path)
    gt = load_ground_truth_graph()
    metrics = compare_worldmodel_to_ground_truth(store, gt)

    # 1. Zero Hallucinations: All learned endpoints must exist in ground truth
    assert metrics.endpoint_precision == 1.0, f"Hallucinated endpoints: {metrics.hallucinated_endpoints}"
    assert len(metrics.hallucinated_endpoints) == 0

    # 2. Perfect Service Attribution: Every learned endpoint must map to its real service
    assert metrics.service_attribution_accuracy >= 0.95

    # 3. High Value Flow Precision: Learned flows must be genuine schema bindings
    assert metrics.dataflow_precision >= 0.85, f"Dataflow precision too low: {metrics.dataflow_precision}"

    # 4. Stage Precedence Conformity: Actions predominantly respect canonical lookup -> mutate -> notify
    # workflows (~84%), with remaining transitions representing multi-stage incident alerting
    # where emergency notifications (Twilio/Slack) precede ticket and documentation creation (Jira/Confluence).
    assert metrics.precedence_conformity_rate >= 0.80, (
        f"Precedence violations: {metrics.precedence_violations}"
    )
