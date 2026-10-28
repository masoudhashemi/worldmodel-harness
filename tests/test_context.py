import json
from types import SimpleNamespace

from worldmodel_harness.context import PREFIX, render_context, render_episodes, select_procedures
from worldmodel_harness.knowledge import (
    Claim,
    Dependency,
    Episode,
    Layer,
    LiveState,
    Scope,
    WorkingGraph,
)
from worldmodel_harness.procedures import Procedure, Step
from worldmodel_harness.store import KnowledgeStore


def artifact(**updates):
    base = {
        "procedure_id": "salesforce-query",
        "scope": Scope(environment="test", version="1"),
        "goal_pattern": "Salesforce query",
        "parameter_types": {},
        "required_claims": (),
        "status": "active",
        "steps": (Step(name="stop", kind="stop"),),
    }
    return Procedure(**(base | updates))


def test_selection_excludes_stale_inapplicable_and_irrelevant():
    good = artifact()
    unavailable = artifact(procedure_id="unavailable", prerequisites={"permission": True})
    suspended = artifact(procedure_id="suspended", status="suspended")
    irrelevant = artifact(procedure_id="email", goal_pattern="Send email")
    store = SimpleNamespace(procedures=lambda _: [good, unavailable, suspended, irrelevant])
    chosen, rejected = select_procedures(store, good.scope, LiveState("task"), "Salesforce query")
    assert chosen == [good]
    assert rejected == {
        "unavailable": "unresolved_prerequisite",
        "suspended": "inactive_or_out_of_scope",
    }


def test_linked_procedure_is_offered_without_shared_goal_words():
    from worldmodel_harness.knowledge import Dependency

    linked = artifact(
        procedure_id="bind-message",
        goal_pattern="Read the first mailbox entry",
        required_claims=(Dependency(claim_id="gmail-get", version=1),),
    )
    unrelated = artifact(procedure_id="email", goal_pattern="Send email")
    claim = Claim(
        claim_id="gmail-get",
        layer=Layer.INTERFACE,
        relation="endpoint_contract",
        participants=("gmail",),
        statement="Fetch one Gmail message",
        scope=linked.scope,
        status="supported",
    )
    def claim_for(claim_id):
        if claim_id != "gmail-get":
            raise KeyError(claim_id)
        return claim

    store = SimpleNamespace(
        procedures=lambda _: [linked, unrelated],
        claim=claim_for,
        dependency_compatible=lambda dependency: True,
    )
    chosen, rejected = select_procedures(
        store,
        linked.scope,
        LiveState("task"),
        "quarterly revenue workbook",
        linked_claim_ids={"gmail-get"},
    )
    assert chosen == [linked]
    assert rejected == {}


def test_whole_message_budget_includes_artifacts_and_metadata():
    procedure = artifact(goal_pattern="Salesforce " * 1000)
    working = WorkingGraph(scope=procedure.scope, gaps=tuple("missing" for _ in range(1000)))
    content, report = render_context(working, [procedure], max_chars=700)
    assert len(content) <= 700
    assert report["artifact_candidates"] == []
    assert "1000" in content
    assert render_context(working, [procedure], max_chars=1)[0] == ""


def test_dependency_bundle_is_not_split_by_rendering_budget():
    scope = Scope(environment="test", version="1")
    dependency = Claim(
        claim_id="contract",
        layer=Layer.INTERFACE,
        relation="schema",
        participants=("tool",),
        statement="Schema",
        scope=scope,
    )
    consumer = Claim(
        claim_id="consumer",
        layer=Layer.BEHAVIOR,
        relation="produces",
        participants=("tool",),
        statement="Produces ID",
        scope=scope,
        dependencies=(Dependency(claim_id="contract", version=1),),
    )
    working = WorkingGraph(scope=scope, claims=(consumer, dependency))
    for budget in (500, 1000, 2000, 5000):
        content, report = render_context(working, [], max_chars=budget)
        assert len(content) <= budget
        if "consumer" in report["injected_claims"]:
            assert "contract" in report["injected_claims"]
        if content:
            assert json.loads(content[len(PREFIX) :])["claims"] is not None


def test_historical_episode_retrieval_is_not_live_state(tmp_path):
    store = KnowledgeStore(tmp_path / "memory.sqlite")
    previous = Scope(environment="test", version="1", tenant="old", task_id="old-task")
    episode = Episode(
        task_id="old-task",
        scope=previous,
        operation="lookup",
        arguments={},
        response={"id": "old-record"},
        outcome="success",
        coverage=("id",),
    )
    store.append_episode(episode)
    current = Scope(environment="test", version="1", tenant="new", task_id="new-task")
    content, selected = render_episodes(store, current, "lookup record", max_chars=2000)
    assert selected["injected_episodes"] == [episode.episode_id]
    assert "NOT current task state" in content
    assert store.episodes(current) == []
    assert render_episodes(store, current, "lookup", max_chars=1)[0] == ""
    different = current.model_copy(update={"version": "2"})
    assert render_episodes(store, different, "lookup", max_chars=2000)[1]["injected_episodes"] == []
    store.close()


def test_workflow_plan_json_serializable_and_markdown_formatting():
    from worldmodel_harness.context import _format_workflow_process_markdown
    from worldmodel_harness.process_planner import (
        DownstreamConsequence,
        SolutionPath,
        TriedActionSummary,
        WorkflowProcess,
        WorkflowStep,
    )

    plan = WorkflowProcess(
        goal="Resolve incident",
        estimated_steps=2,
        steps=[
            WorkflowStep(
                step=1,
                operation="gmail.users.messages.get",
                status="pending",
                description="Inspect message",
            )
        ],
        tried_from_this_state=[
            TriedActionSummary(
                action="datadog.incidents.list",
                outcome="success",
                count=2,
                result="Found 2 incidents",
                lesson="Incident 202 is duplicate of 101",
                next_state="datadog.incidents.get",
                downstream_consequences=[
                    DownstreamConsequence(
                        action="pagerduty.incidents.create",
                        outcome="error",
                        result="Paged duplicate",
                        lesson="Paging duplicate ticket leads to downstream failure",
                    )
                ],
            )
        ],
        potential_solution_paths=[
            SolutionPath(
                path_id="path_best",
                steps=["gmail.users.messages.get", "sheets.spreadsheets.values.get"],
                confidence=0.9,
                certainty=0.85,
                posterior_variance=0.05,
                estimated_steps=2,
                strategy_description="Inspect email first",
            )
        ],
    )

    scope = Scope(environment="test", version="1")
    working = WorkingGraph(scope=scope)

    # 1. Render context with plan
    content, report = render_context(working, [], max_chars=5000, plan=plan)
    assert report["workflow_plan"] is not None
    assert isinstance(report["workflow_plan"], dict)

    # 2. Must be JSON serializable without TypeError
    dumped = json.dumps(report)
    assert "gmail.users.messages.get" in dumped

    # 3. _format_workflow_process_markdown must work with both Pydantic model and dict
    md_from_model = _format_workflow_process_markdown(plan)
    md_from_dict = _format_workflow_process_markdown(report["workflow_plan"])
    assert "Empirical World Model" in md_from_model
    assert "Empirical World Model" in md_from_dict
    assert "duplicate of 101" in md_from_model
    assert "duplicate of 101" in md_from_dict


def test_workflow_plan_compacts_under_tight_budget():
    from worldmodel_harness.process_planner import (
        DownstreamConsequence,
        SolutionPath,
        TriedActionSummary,
        WorkflowProcess,
        WorkflowStep,
    )

    long_summary = [
        TriedActionSummary(
            action=f"action_{i}",
            outcome="error",
            count=i + 1,
            result="a" * 200,
            lesson="b" * 200,
            next_state=f"state_{i}",
            downstream_consequences=[
                DownstreamConsequence(
                    action=f"sub_action_{i}",
                    outcome="error",
                    result="c" * 200,
                    lesson="d" * 200,
                )
            ],
        )
        for i in range(10)
    ]
    paths = [
        SolutionPath(
            path_id=f"path_{i}",
            steps=[f"step_{j}" for j in range(10)],
            confidence=0.8,
            certainty=0.7,
            posterior_variance=0.1,
            estimated_steps=10,
            strategy_description="s" * 150,
        )
        for i in range(5)
    ]
    plan = WorkflowProcess(
        goal="Test heavy plan",
        estimated_steps=5,
        steps=[
            WorkflowStep(step=i, operation=f"op_{i}", status="pending", description="d")
            for i in range(5)
        ],
        tried_from_this_state=long_summary,
        potential_solution_paths=paths,
    )

    scope = Scope(environment="test", version="1")
    working = WorkingGraph(scope=scope)

    # With a 1500 char budget, heavy plan would exceed uncompacted (>4000 chars),
    # but with graceful compaction it should compact and fit without dropping to empty.
    content, report = render_context(working, [], max_chars=1500, plan=plan)
    assert content != ""
    assert len(content) <= 1500
    assert report["workflow_plan"] is not None


def test_prune_historical_tool_outputs_bounds_latency():
    from stirrup.core.models import AssistantMessage, ToolMessage, UserMessage
    from worldmodel_harness.experiment import _prune_historical_tool_outputs

    messages = [
        UserMessage(content="Start task"),
    ]
    # Add 6 rounds of assistant + tool messages
    for i in range(6):
        messages.append(AssistantMessage(content=f"Call {i}"))
        messages.append(ToolMessage(content="X" * 3000, tool_call_id=f"c_{i}", name="api_fetch"))

    pruned = _prune_historical_tool_outputs(messages, keep_recent=4, max_chars=1500)
    assert len(pruned) == len(messages)

    # First two tool messages (indices 2 and 4) must be truncated
    assert len(pruned[2].content) < 2000
    assert "Truncated" in pruned[2].content
    assert pruned[2].tool_call_id == "c_0"

    assert len(pruned[4].content) < 2000
    assert "Truncated" in pruned[4].content

    # The last 4 tool messages (indices 6, 8, 10, 12) must remain unpruned
    assert len(pruned[6].content) == 3000
    assert len(pruned[8].content) == 3000
    assert len(pruned[10].content) == 3000
    assert len(pruned[12].content) == 3000


