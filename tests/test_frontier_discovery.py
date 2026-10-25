"""Unit tests for Active Frontier Discovery and Schema Exploration (Task 4)."""

from pathlib import Path
import pytest

from worldmodel_harness.frontier_discovery import discover_and_index_frontier
from worldmodel_harness.graph_comparison import load_ground_truth_graph, compare_worldmodel_to_ground_truth
from worldmodel_harness.knowledge import Scope
from worldmodel_harness.store import KnowledgeStore


def test_targeted_frontier_discovery(tmp_path: Path):
    store = KnowledgeStore(tmp_path / "frontier.sqlite")
    scope = Scope(environment="automationbench", version="1")
    gt = load_ground_truth_graph()

    # Target specific unvisited enterprise services: zendesk and salesforce
    target_services = {"zendesk", "salesforce"}
    report = discover_and_index_frontier(store, scope, target_services=target_services, gt=gt)

    assert report.initial_known_endpoints == 0
    assert report.newly_discovered_endpoints > 0
    assert report.total_known_endpoints == report.newly_discovered_endpoints
    assert set(report.discovered_services) == target_services

    # Verify that all indexed endpoints belong to the target services
    claims = list(store.claims(scope))
    for c in claims:
        assert c.payload.get("service") in target_services
        assert c.status == "supported"
        assert c.origin == "documented"


def test_full_frontier_discovery_and_ground_truth_fidelity(tmp_path: Path):
    store = KnowledgeStore(tmp_path / "full_frontier.sqlite")
    scope = Scope(environment="automationbench", version="1")
    gt = load_ground_truth_graph()

    # Actively explore the entire API catalog
    report = discover_and_index_frontier(store, scope, target_services=None, gt=gt)

    assert report.initial_known_endpoints == 0
    assert report.total_known_endpoints == len(gt.endpoints)
    assert report.final_coverage == 1.0
    assert len(report.discovered_services) == len(gt.services)

    # Validate against ground truth comparison engine
    metrics = compare_worldmodel_to_ground_truth(store, gt)

    # 1. Zero Hallucinations across the entire 758-endpoint catalog
    assert metrics.endpoint_precision == 1.0
    assert len(metrics.hallucinated_endpoints) == 0

    # 2. Perfect 100% Endpoint Recall
    assert metrics.endpoint_recall == 1.0

    # 3. Perfect Service Attribution & Stage Alignment
    assert metrics.service_attribution_accuracy == 1.0
    assert metrics.stage_classification_accuracy == 1.0
