"""Unit and integration tests for Continual Learning & Multi-Iteration Consolidation (Task 3).

Validates that when the World Model experiences sequential learning across isomorphic tasks
(k=1 -> k=2 -> k=3):
1. Value flow claims graduate from 'hypothesis' to 'supported' once confirmed across 2+ distinct tenants/tasks.
2. Abstract parameter slots consolidate multiple distinct runtime values without concrete literal anchoring.
3. The store avoids catastrophic forgetting of the source task capabilities.
4. Trajectory graph activates structural reachability edges only once claims are consolidated.
"""

from pathlib import Path
import pytest

from worldmodel_harness.composition import CompositionLearner
from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.trajectory_graph import compile_trajectory_graph


def setup_continual_environment(tmp_path: Path):
    store = KnowledgeStore(tmp_path / "continual.sqlite")
    learner = Learner(store)
    scope = Scope(environment="automationbench", version="1")
    for endpoint in (
        {"id": "sheets.spreadsheets.get", "url": "https://sheets.googleapis.com/v4/spreadsheets/{spreadsheetId}", "method": "GET"},
        {"id": "sheets.spreadsheets.values.get", "url": "https://sheets.googleapis.com/v4/spreadsheets/{spreadsheetId}/values/{range}", "method": "GET"},
    ):
        event = Episode(
            task_id="schema",
            scope=scope,
            operation="tool_schema",
            arguments={},
            response=endpoint,
            outcome="success",
        )
        store.append_episode(event)
        learner.document(
            event,
            relation="endpoint_contract",
            participants=(endpoint["id"],),
            statement=endpoint["id"],
            payload=endpoint,
            scope=scope,
        )
    return store, CompositionLearner(learner), scope


def create_isomorphic_episode_pair(store, scope, *, spreadsheet_id: str):
    source = Episode(
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"},
        response={"spreadsheetId": spreadsheet_id, "sheets": [{"properties": {"title": "Contractors"}}]},
        coverage=("spreadsheetId",),
        outcome="success",
    )
    store.append_episode(source)

    target = Episode(
        task_id=scope.task_id,
        scope=scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/Contractors!A:D"},
        response={"values": [["ID", "Name"], ["101", "Wayne Ellis"]]},
        coverage=("values",),
        outcome="success",
        pre_evidence=(source.episode_id,),
    )
    store.append_episode(target)
    return source, target


def test_continual_learning_and_consolidation_across_isomorphic_runs(tmp_path: Path):
    store, comp_learner, base_scope = setup_continual_environment(tmp_path)

    # Iteration 1 (k = 1): Run on source task (contractor_badge_expiration)
    scope_k1 = base_scope.model_copy(update={"tenant": "operations.badge_source", "task_id": "badge_task_001"})
    src_k1, tgt_k1 = create_isomorphic_episode_pair(store, scope_k1, spreadsheet_id="ss_badges_alpha")
    
    claims_k1 = comp_learner.observe(tgt_k1)
    assert len(claims_k1) == 1
    claim_k1 = claims_k1[0]

    # At k = 1: Single tenant and single observed value -> hypothesis
    assert claim_k1.status == "hypothesis"
    assert claim_k1.payload["source_endpoint"] == "sheets.spreadsheets.get"
    assert claim_k1.payload["target_endpoint"] == "sheets.spreadsheets.values.get"
    assert "ss_badges_alpha" not in claim_k1.statement

    # Trajectory graph at k = 1: Empirical observation exists, but NO structural value flow edge attached
    graph_k1 = compile_trajectory_graph(store, scope_k1, successful_transitions_only=True)
    edge_k1 = graph_k1.edges.get("sheets.spreadsheets.get", {}).get("sheets.spreadsheets.values.get")
    assert edge_k1 is not None
    assert len(edge_k1.value_flows) == 0  # Hypothesis claim is not attached as structural flow

    # Iteration 2 (k = 2): Run on isomorphic task (contractor_badge_expiration_isomorphic)
    # Different tenant, different task_id, different spreadsheet ID
    scope_k2 = base_scope.model_copy(update={"tenant": "operations.badge_isomorphic", "task_id": "badge_task_002"})
    src_k2, tgt_k2 = create_isomorphic_episode_pair(store, scope_k2, spreadsheet_id="ss_drill_beta_99")

    claims_k2 = comp_learner.observe(tgt_k2)
    assert len(claims_k2) == 1
    claim_k2 = claims_k2[0]

    # At k = 2: 2 distinct tenants, 2 distinct tasks, 2 distinct values -> Promoted to "supported"!
    assert claim_k2.status == "supported"
    assert len(claim_k2.supporting_episodes) == 4
    # Neither literal leaked into the consolidated claim
    assert "ss_badges_alpha" not in claim_k2.statement
    assert "ss_drill_beta_99" not in claim_k2.statement

    # Trajectory graph at k = 2: Structural value flow is now ATTACHED to the edge!
    graph_k2 = compile_trajectory_graph(store, scope_k1, successful_transitions_only=True)
    edge_k2 = graph_k2.edges.get("sheets.spreadsheets.get", {}).get("sheets.spreadsheets.values.get")
    assert edge_k2 is not None
    assert len(edge_k2.value_flows) == 1
    assert edge_k2.value_flows[0].input_path == "path.spreadsheetId"

    # Iteration 3 (k = 3): Retention check / Non-catastrophic forgetting
    # Re-evaluating source task has full access to consolidated capability without losing source memory
    all_episodes = store.historical_episodes(scope_k1)
    source_episodes = [e for e in all_episodes if e.task_id == "badge_task_001"]
    assert len(source_episodes) == 2
    assert "ss_badges_alpha" in source_episodes[0].arguments["url"]
    
    store.close()
