"""Unit tests for empirical trajectory graph compilation and pathfinding."""

import pytest

from worldmodel_harness.knowledge import Claim, Episode, Layer, Scope
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.trajectory_graph import (
    OperationNode,
    TrajectoryGraph,
    TransitionEdge,
    ValueFlowInfo,
    compile_trajectory_graph,
)


@pytest.fixture
def memory_store(tmp_path):
    return KnowledgeStore(tmp_path / "test.sqlite")


@pytest.fixture
def sample_scope():
    return Scope(environment="test_env", version="1", role="test_role")


def test_trajectory_graph_shortest_path():
    graph = TrajectoryGraph()
    n1 = OperationNode(node_id="drive.files.list", operation="api_fetch", method="GET", action_type="read")
    n2 = OperationNode(node_id="sheets.values.get", operation="api_fetch", method="GET", action_type="read")
    n3 = OperationNode(node_id="sheets.values.update", operation="api_fetch", method="PUT", action_type="mutate")
    n4 = OperationNode(node_id="gmail.messages.send", operation="api_fetch", method="POST", action_type="mutate")

    for n in (n1, n2, n3, n4):
        graph.add_node(n)

    graph.add_edge(TransitionEdge(source="drive.files.list", target="sheets.values.get", count=5, cost=0.2))
    graph.add_edge(TransitionEdge(source="sheets.values.get", target="sheets.values.update", count=4, cost=0.3))
    graph.add_edge(TransitionEdge(source="sheets.values.update", target="gmail.messages.send", count=3, cost=0.1))

    # Test direct path
    res = graph.shortest_path("drive.files.list", "gmail.messages.send")
    assert res is not None
    path, cost = res
    assert path == ["drive.files.list", "sheets.values.get", "sheets.values.update", "gmail.messages.send"]
    assert pytest.approx(cost, 0.001) == 0.6

    # Test disconnected path
    assert graph.shortest_path("gmail.messages.send", "drive.files.list") is None


def test_trajectory_graph_path_covering():
    graph = TrajectoryGraph()
    n_start = OperationNode(node_id="start_read", operation="api_fetch", method="GET", action_type="read")
    n_mid = OperationNode(node_id="mid_op", operation="api_fetch", method="POST", action_type="mutate")
    n_target1 = OperationNode(node_id="target1", operation="api_fetch", method="POST", action_type="mutate")
    n_target2 = OperationNode(node_id="target2", operation="api_fetch", method="POST", action_type="mutate")

    for n in (n_start, n_mid, n_target1, n_target2):
        graph.add_node(n)

    graph.add_edge(TransitionEdge(source="start_read", target="mid_op", count=5, cost=0.1))
    graph.add_edge(TransitionEdge(source="mid_op", target="target1", count=4, cost=0.1))
    graph.add_edge(TransitionEdge(source="target1", target="target2", count=3, cost=0.1))

    # Even if targets are passed in reverse order [target2, target1], covering finds the valid sequence
    res = graph.find_path_covering(starts=["start_read"], targets=["target2", "target1"])
    assert res is not None
    path, cost = res
    assert path == ["start_read", "mid_op", "target1", "target2"]


def test_compile_trajectory_graph_from_episodes(memory_store, sample_scope):
    # Register contracts
    c1 = Claim(
        claim_id="claim_c1",
        version=1,
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("google_drive.files.list",),
        statement="List files in Google Drive",
        scope=sample_scope,
        payload={"id": "google_drive.files.list", "method": "GET", "url": "https://www.googleapis.com/drive/v3/files"},
        status="supported",
    )
    c2 = Claim(
        claim_id="claim_c2",
        version=1,
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("sheets.spreadsheets.values.get",),
        statement="Get sheet values",
        scope=sample_scope,
        payload={
            "id": "sheets.spreadsheets.values.get",
            "method": "GET",
            "url": "https://sheets.googleapis.com/v4/spreadsheets/{spreadsheetId}/values/{range}",
        },
        status="supported",
    )
    c3 = Claim(
        claim_id="claim_c3",
        version=1,
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("mailchimp.lists.members.delete",),
        statement="Delete list member in Mailchimp",
        scope=sample_scope,
        payload={
            "id": "mailchimp.lists.members.delete",
            "method": "DELETE",
            "url": "https://us1.api.mailchimp.com/3.0/lists/{list_id}/members/{subscriber_hash}",
        },
        status="supported",
    )
    for c in (c1, c2, c3):
        memory_store.db.execute(
            "INSERT INTO claims VALUES (?, ?, ?)", (c.claim_id, c.version, c.model_dump_json())
        )

    # Append episodes simulating task execution
    task1_eps = [
        Episode(
            task_id="t1",
            scope=sample_scope,
            operation="api_fetch",
            arguments={"method": "GET", "url": "https://www.googleapis.com/drive/v3/files"},
            response={"files": [{"id": "s1"}]},
            outcome="success",
        ),
        Episode(
            task_id="t1",
            scope=sample_scope,
            operation="api_fetch",
            arguments={"method": "GET", "url": "https://sheets.googleapis.com/v4/spreadsheets/s1/values/A1:B10"},
            response={"values": [["alice@example.com"]]},
            outcome="success",
        ),
        Episode(
            task_id="t1",
            scope=sample_scope,
            operation="api_fetch",
            arguments={"method": "DELETE", "url": "https://us1.api.mailchimp.com/3.0/lists/l1/members/hash1"},
            response={"status": "deleted"},
            outcome="success",
        ),
    ]
    for ep in task1_eps:
        memory_store.append_episode(ep)

    graph = compile_trajectory_graph(memory_store, sample_scope)
    assert "google_drive.files.list" in graph.nodes
    assert "sheets.spreadsheets.values.get" in graph.nodes
    assert "mailchimp.lists.members.delete" in graph.nodes

    # Verify transitions were captured
    assert "sheets.spreadsheets.values.get" in graph.edges["google_drive.files.list"]
    assert "mailchimp.lists.members.delete" in graph.edges["sheets.spreadsheets.values.get"]

    # Verify shortest path traversal
    path_result = graph.shortest_path("google_drive.files.list", "mailchimp.lists.members.delete")
    assert path_result is not None
    assert path_result[0] == ["google_drive.files.list", "sheets.spreadsheets.values.get", "mailchimp.lists.members.delete"]
