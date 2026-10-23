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


def test_trajectory_graph_tracks_tried_attempts_and_multi_paths(memory_store, sample_scope):
    # Setup 3 nodes: start, branch_a, branch_b
    n1 = OperationNode(node_id="read_start", operation="api_fetch", method="GET", action_type="read")
    n2 = OperationNode(node_id="mutate_a", operation="api_fetch", method="POST", action_type="mutate")
    n3 = OperationNode(node_id="mutate_b", operation="api_fetch", method="POST", action_type="mutate")

    graph = TrajectoryGraph()
    for n in (n1, n2, n3):
        graph.add_node(n)

    # Path A: read_start -> mutate_a -> mutate_b (Cost: 0.2)
    graph.add_edge(TransitionEdge(source="read_start", target="mutate_a", count=3, cost=0.1))
    graph.add_edge(TransitionEdge(source="mutate_a", target="mutate_b", count=3, cost=0.1))

    # Path B: read_start -> mutate_b -> mutate_a (Cost: 0.45)
    graph.add_edge(TransitionEdge(source="read_start", target="mutate_b", count=1, cost=0.15))
    graph.add_edge(TransitionEdge(source="mutate_b", target="mutate_a", count=1, cost=0.3))

    # Test multi-path candidate extraction
    candidate_paths = graph.find_all_candidate_paths(
        starts=["read_start"], targets=["mutate_a", "mutate_b"], max_paths=3
    )
    assert len(candidate_paths) == 2
    # Recommended Path A has lower cost and higher confidence
    path_a, cost_a, conf_a, supp_a = candidate_paths[0]
    path_b, cost_b, conf_b, supp_b = candidate_paths[1]

    assert path_a == ["read_start", "mutate_a", "mutate_b"]
    assert path_b == ["read_start", "mutate_b", "mutate_a"]
    assert cost_a < cost_b
    assert conf_a > conf_b
    assert supp_a == 3
    assert supp_b == 1

    # Now verify compilation records error and success attempts from nodes
    # Add an error episode from read_start (e.g. invalid range syntax)
    err_ep = Episode(
        task_id="t_err",
        scope=sample_scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/test", "params": '{"ranges": "Bad!A:Z"}'},
        response={"error": {"code": 400, "message": "Unable to parse range: Bad!A:Z"}},
        outcome="error",
    )
    success_ep = Episode(
        task_id="t_err",
        scope=sample_scope,
        operation="api_fetch",
        arguments={"method": "GET", "url": "https://sheets.googleapis.com/test", "params": '{"ranges": "A:Z"}'},
        response={"values": [["1", "2"]]},
        outcome="success",
    )
    memory_store.append_episode(err_ep)
    memory_store.append_episode(success_ep)

    compiled = compile_trajectory_graph(memory_store, sample_scope)
    # Check attempts recorded under root / initial state
    root_attempts = compiled.node_attempts.get("root", [])
    assert len(root_attempts) >= 2
    err_att = next(a for a in root_attempts if a.outcome == "error")
    succ_att = next(a for a in root_attempts if a.outcome == "success")

    assert "Unable to parse range" in err_att.result_summary
    assert "syntax error" in err_att.lesson
    assert "returned 1 rows" in succ_att.result_summary


def test_bayesian_certainty_increases_with_success_and_usage():
    """Verify certainty increases monotonically with successful usage without hard thresholds."""
    counts = [1, 2, 3, 5, 10, 20]
    edges = [
        TransitionEdge(source="u", target="v", count=c, success_count=c, failure_count=0)
        for c in counts
    ]

    certainties = [e.certainty() for e in edges]
    variances = [e.posterior_variance() for e in edges]

    # Certainty must strictly increase
    for i in range(len(certainties) - 1):
        assert certainties[i] < certainties[i + 1], f"Certainty did not increase: {certainties}"

    # Posterior variance (uncertainty) must strictly decrease
    for i in range(len(variances) - 1):
        assert variances[i] > variances[i + 1], f"Variance did not decrease: {variances}"

    # Check realistic boundaries
    assert certainties[0] > 0.0  # s=1 has positive certainty
    assert certainties[-1] > 0.90  # s=20 reaches high certainty (~0.93)
    assert variances[-1] < 0.003  # variance concentrated near 0


def test_thompson_sampling_favors_high_confidence_while_exploring():
    """Verify Thompson Sampling samples proven paths more often while allowing probabilistic exploration."""
    edge_high = TransitionEdge(source="u", target="v1", count=10, success_count=10, failure_count=0)
    edge_low = TransitionEdge(source="u", target="v2", count=1, success_count=1, failure_count=0)

    high_wins = 0
    total_draws = 1000
    for _ in range(total_draws):
        s_high = edge_high.sample_beta()
        s_low = edge_low.sample_beta()
        if s_high > s_low:
            high_wins += 1

    win_rate = high_wins / total_draws
    # High-confidence path should win roughly 70-85% of draws
    assert 0.65 <= win_rate <= 0.90
    # Low-confidence path should still be explored roughly 10-35% of the time (no hard lock-out)
    assert (1.0 - win_rate) >= 0.10


def test_candidate_path_rich_attributes_and_tuple_unpacking():
    from worldmodel_harness.trajectory_graph import CandidatePath

    cand = CandidatePath(
        path=["node1", "node2"],
        cost=0.25,
        confidence=0.88,
        support_count=5,
        certainty=0.72,
        posterior_variance=0.015,
        sample_score=0.89,
    )

    # 4-tuple unpacking for backward compatibility
    nodes, cost, conf, supp = cand
    assert nodes == ["node1", "node2"]
    assert cost == 0.25
    assert conf == 0.88
    assert supp == 5
    assert len(cand) == 4

    # Rich attributes
    assert cand.certainty == 0.72
    assert cand.posterior_variance == 0.015
    assert cand.sample_score == 0.89
    assert cand.path == ["node1", "node2"]


def test_pathfinding_excludes_unauthorized_destructive_mutations():
    graph = TrajectoryGraph()
    n_read = OperationNode(node_id="slack.history", operation="api_fetch", method="GET", action_type="read")
    n_delete = OperationNode(node_id="crm.delete", operation="api_fetch", method="DELETE", action_type="mutate")
    n_post = OperationNode(node_id="slack.post", operation="api_fetch", method="POST", action_type="mutate")

    for n in (n_read, n_delete, n_post):
        graph.add_node(n)

    graph.add_edge(TransitionEdge(source="slack.history", target="crm.delete", count=1, cost=0.1))
    graph.add_edge(TransitionEdge(source="crm.delete", target="slack.post", count=1, cost=0.1))

    # Authorized mutations only includes slack.post (not crm.delete)
    res = graph.shortest_path("slack.history", "slack.post", authorized_mutations={"slack.post"})
    assert res is None  # crm.delete is an unauthorized destructive intermediate step and must be rejected!

    # When crm.delete is authorized or a verified prerequisite, it is permitted
    graph.prerequisites["slack.post"].add("crm.delete")
    res_prereq = graph.shortest_path("slack.history", "slack.post", authorized_mutations={"slack.post"})
    assert res_prereq is not None
    assert res_prereq[0] == ["slack.history", "crm.delete", "slack.post"]


def test_bayesian_confidence_does_not_count_invented_successes(tmp_path):
    from worldmodel_harness.knowledge import Claim, Episode, Layer, Scope
    from worldmodel_harness.store import KnowledgeStore
    from worldmodel_harness.trajectory_graph import compile_trajectory_graph

    store = KnowledgeStore(tmp_path / "test_bayes.sqlite")
    scope = Scope(environment="test", version="1", role="ops", task_id="task")

    # Add discovery/evidence episode
    store.append_episode(
        Episode(
            episode_id="ep_contract_discovery",
            task_id="task",
            scope=scope,
            operation="api_search",
            arguments={"query": "gmail send"},
            response={"results": []},
            outcome="success",
        )
    )

    # Single verified contract
    c_contract = Claim(
        claim_id="claim_send",
        version=1,
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("gmail.messages.send",),
        statement="Send email endpoint",
        scope=scope,
        status="supported",
        supporting_episodes=("ep_contract_discovery",),
        payload={"id": "gmail.messages.send", "method": "POST", "url": "https://gmail.googleapis.com/v1/users/me/messages/send"},
    )
    # Verified contrastive precondition with ZERO execution episodes
    c_prereq = Claim(
        claim_id="claim_prereq",
        version=1,
        layer=Layer.BEHAVIOR,
        relation="contrastive_precondition",
        participants=("base64_encode", "gmail.messages.send"),
        statement="base64_encode required before gmail.messages.send",
        scope=scope,
        status="supported",
        supporting_episodes=("ep_contract_discovery",),
        payload={"source_operation": "base64_encode", "target_operation": "gmail.messages.send"},
    )
    # Hypothetical prerequisite that must be ignored
    c_hypo = Claim(
        claim_id="claim_hypo",
        version=1,
        layer=Layer.BEHAVIOR,
        relation="contrastive_precondition",
        participants=("random_op", "gmail.messages.send"),
        statement="hypothetical prerequisite",
        scope=scope,
        status="hypothesis",
        payload={"source_operation": "random_op", "target_operation": "gmail.messages.send"},
    )

    store.put_claim(c_contract)
    store.put_claim(c_prereq)
    store.put_claim(c_hypo)

    graph = compile_trajectory_graph(store, scope)

    # Hypothesis claim must NOT create an edge
    assert "random_op" not in graph.edges

    # Verified precondition edge exists for structural routing, but has ZERO empirical observations
    edge = graph.edges["base64_encode"]["gmail.messages.send"]
    assert edge.success_count == 0  # Does NOT invent 5 successes!
    assert edge.failure_count == 0
    assert edge.alpha == 1.0
    assert edge.beta == 1.0
    assert edge.posterior_mean() == 0.5  # Neutral uniform prior, NOT 0.857!
    assert pytest.approx(edge.posterior_variance(), 0.001) == 0.0833
    assert edge.certainty() == 0.0  # Zero empirical certainty

