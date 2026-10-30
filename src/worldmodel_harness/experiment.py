"""Stirrup task streams with explicit reset, learning, and evidence boundaries."""

import asyncio
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from stirrup.core.models import SystemMessage, TokenUsage, Tool, ToolResult, UserMessage

from .automationbench import UPSTREAM_COMMIT, AutomationBenchAdapter, dataset
from .budget import Ledger
from .config import load_api_key, load_settings
from .context import render_context, render_episodes, select_procedures
from .information import information_route_instruction, information_tool, tracked_information_tool
from .learning import Learner
from .procedures import artifact_tool
from .repairing_agent import RepairingAgent
from .retrieval import retrieve
from .runtime import MeteredClient, build_client
from .shift import ResponseShift
from .store import KnowledgeStore


def learns_from_trajectories(phase: str) -> bool:
    """Train and test both update the environment graph from tool observations.

    A validation score is not an input. Frozen evaluation is the one phase that
    keeps a store read-only.
    """
    return phase != "frozen"


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: str = "simple"
    task_names: tuple[str, ...] = ()
    limit: int = Field(default=1, ge=1)
    offset: int = Field(default=0, ge=0)
    phase: Literal["development", "frozen", "online", "checkpoint", "train", "test"] = "development"
    memory: Literal["none", "episodes", "flat", "graph"] = "graph"
    resolve_execution_frontier: bool = False
    successful_transition_graph: bool = False
    adaptive: bool = False
    operational: bool = False
    semantic_learning: bool = False
    max_online_reflections: int = Field(default=2, ge=0, le=8)
    max_operation_interpretations: int = Field(default=2, ge=0, le=8)
    semantic_max_tokens: int = Field(default=16384, ge=1)
    semantic_timeout_seconds: int = Field(default=180, ge=1)
    semantic_reasoning_effort: Literal["xhigh", "medium", "low", "none"] = "low"
    semantic_json_mode: bool = True
    allow_experimental: bool = False
    information_policy: Literal["off", "task", "future"] = "off"
    max_information_calls: int = Field(default=4, ge=0)
    max_turns: int = Field(default=20, ge=1)
    max_environment_calls: int | None = Field(default=40, ge=1)
    retrieval_chars: int = Field(default=12000, ge=1)
    timeout_seconds: int | None = Field(default=300, ge=1)
    gated_graph: bool = False
    # When the operational layers trap the agent, finish the episode as the baseline.
    baseline_fallback: bool = False
    # Baseline tools and loop plus one small, fixed note of confirmed routes.
    advisory: bool = False
    sampling_seed: int = 0
    shift: ResponseShift | None = None

    @model_validator(mode="after")
    def supported_arms(self):
        if self.phase == "checkpoint" and (
            self.limit != 1
            or len(self.task_names) != 1
            or self.offset != 0
            or not self.operational
            or self.memory != "graph"
            or self.adaptive
            or self.semantic_learning
            or self.information_policy != "off"
        ):
            raise ValueError(
                "Checkpoint measurement requires one explicit operational graph task, without semantic/adaptive/probe updates"
            )
        if self.semantic_learning and not self.operational:
            raise ValueError("Semantic learning currently requires operational mode")
        if self.advisory and (self.operational or self.memory not in ("flat", "graph")):
            raise ValueError("Advisory memory uses the baseline tools and a claim store")
        if self.operational and self.memory not in ("flat", "graph"):
            raise ValueError("Operational decisions require a claim-based working graph")
        if self.allow_experimental and (self.phase != "online" or not self.adaptive):
            raise ValueError(
                "Experimental procedures are restricted to explicit adaptive online runs"
            )
        if self.adaptive and self.memory in ("none", "episodes"):
            raise ValueError("Adaptive execution requires flat or graph operational memory")
        if self.information_policy == "future" and self.phase == "frozen":
            raise ValueError("Future-task probes require a learning phase, not frozen evaluation")
        if self.information_policy == "future" and self.memory == "none":
            raise ValueError("Future-task probes require persistent memory")
        return self


def attach_validation_score(result: dict, adapter) -> None:
    """Record the environment rubric on the run report when the task supplies assertions.

    The returned numbers stay in this result dict. They are not written to the
    knowledge store and are not passed to the learner.
    """
    result["validation_scores_used_for_learning"] = False
    assertions = adapter._state.get("info", {}).get("assertions") or []
    if not assertions:
        result["validation_available"] = False
        return
    result.update(adapter.score())
    result["grading_details"] = adapter.grading_details()
    result["validation_available"] = True


class Finish(BaseModel):
    summary: str


def _inject_operational_context(messages: list, content: str, title: str) -> list:
    if not content:
        return list(messages)
    updated = list(messages)
    header = f"\n\n### {title}\n"
    if updated and updated[0].role == "system":
        updated[0] = SystemMessage(content=updated[0].content + header + content)
    else:
        updated.insert(0, SystemMessage(content=f"### {title}\n" + content))
    return updated


def _prune_historical_tool_outputs(
    messages: list, keep_recent: int = 4, max_chars: int = 1500
) -> list:
    """Prune large historical tool outputs from earlier turns to prevent context bloat and latency spikes."""
    tool_indices = [i for i, m in enumerate(messages) if getattr(m, "role", None) == "tool"]
    if len(tool_indices) <= keep_recent:
        return list(messages)
    prune_indices = set(tool_indices[:-keep_recent])
    pruned = []
    for i, m in enumerate(messages):
        if i in prune_indices and getattr(m, "content", None) and len(str(m.content)) > max_chars:
            trunc_notice = (
                f"\n... [Truncated {len(str(m.content)) - max_chars} characters of previous tool output "
                "for response latency and context bounds. Full details are recorded in task evidence.]"
            )
            truncated_content = str(m.content)[:max_chars] + trunc_notice
            if hasattr(m, "model_copy"):
                pruned.append(m.model_copy(update={"content": truncated_content}))
            else:
                pruned.append(m)
        else:
            pruned.append(m)
    return pruned


class KnowledgeClient:
    """Inject a fresh working view before each model decision; keep Stirrup's loop."""

    def __init__(self, client, adapter, knowledge, settings):
        self.client, self.adapter = client, adapter
        self.knowledge, self.settings = knowledge, settings
        self.retrievals = []
        self.gating_decision = None
        self.model_call_costs = []
        self.last_messages = []
        self.online_learner = None
        self._last_working_claims = set()
        self._completed_operations = set()
        self._current_operation = None
        self._last_processed_interaction_index = 0
        self.advisory = None

    def _advisory_messages(self, messages):
        """Inject the fixed advisory note; withdraw it for good once the agent loops."""
        from .advisory import build_advisory_block, recent_tool_rows
        from .baseline_fallback import detect_trap

        if self.advisory is None:
            started = perf_counter()
            task_text = " ".join(m["content"] for m in self.adapter.task.prompt)
            block, info = build_advisory_block(self.knowledge, self.adapter.scope, task_text)
            self.advisory = {**info, "block": block, "withdrawn": None}
            self.adapter.ledger.compute_seconds += perf_counter() - started
        if self.advisory["withdrawn"] is None:
            trap = detect_trap(recent_tool_rows(messages))
            if trap is not None:
                self.advisory["withdrawn"] = {**trap, "model_call_index": len(self.model_call_costs)}
        if self.advisory["withdrawn"] is not None or not self.advisory["block"]:
            return list(messages)
        return _inject_operational_context(messages, self.advisory["block"], "Advisory Memory")

    async def _generate(self, messages, tools):
        self.last_messages = list(messages)
        started = perf_counter()
        cost = {"usage_available": False}
        try:
            response = await self.client.generate(messages, tools)
            cost["token_usage"] = response.token_usage.model_dump()
            cost["usage_available"] = True
            self.last_messages.append(response)
            return response
        except BaseException as exc:
            # Include cancellation and failed calls without retaining SDK messages.
            cost["error_type"] = type(exc).__name__
            raise
        finally:
            cost["elapsed_seconds"] = perf_counter() - started
            self.model_call_costs.append(cost)

    @property
    def model_slug(self):
        return self.client.model_slug

    @property
    def context_window_tokens(self):
        return self.client.context_window_tokens

    async def generate(self, messages, tools):
        if self.settings.advisory:
            return await self._generate(self._advisory_messages(messages), tools)
        fallback = getattr(self.adapter, "baseline_fallback", None)
        if fallback is not None and fallback.active:
            # `tools` is the agent's live tool map; the swap persists for later turns.
            fallback.install(tools)
            return await self._generate(list(messages), tools)
        if self.online_learner is not None:
            await self.online_learner.before_decision()
        # Pruning is part of the operational execution layer; the baseline keeps the
        # upstream loop's full history.
        selected_messages = (
            _prune_historical_tool_outputs(messages) if self.settings.operational else list(messages)
        )
        if self.settings.memory != "none":
            started = perf_counter()
            user_msgs = [m["content"] for m in self.adapter.task.prompt if m.get("role") != "system"]
            query = " ".join(user_msgs or [m["content"] for m in self.adapter.task.prompt])
            # Unexpected tool output can seed a new neighborhood, without reading hidden state.
            recent = [str(m.content) for m in messages[-3:] if m.role == "tool"]
            if self.settings.memory == "episodes":
                content, selection = render_episodes(
                    self.knowledge,
                    self.adapter.scope,
                    query + " " + " ".join(recent),
                    max_chars=self.settings.retrieval_chars,
                )
                self.adapter.ledger.compute_seconds += perf_counter() - started
                self.retrievals.append(selection)
                if content:
                    selected_messages = _inject_operational_context(
                        selected_messages, content, "Operational Episodes"
                    )
                return await self._generate(selected_messages, tools)
            from .retrieval import decision_focus

            if self.settings.resolve_execution_frontier or self.settings.operational:
                from .execution_state import advance_frontier
    
                history = getattr(self.adapter, "interaction_history", None)
                if history:
                    unprocessed = history[self._last_processed_interaction_index:]
                    task = getattr(self.adapter, "task", None)
                    task_text = " ".join(m["content"] for m in task.prompt) if task else ""
                    entity_tracker = getattr(self.adapter, "entity_tracker", None)
                    for inter in unprocessed:
                        self._current_operation = advance_frontier(
                            self.knowledge, self.adapter.scope, self.adapter.evidence,
                            inter, self._current_operation, self._completed_operations,
                            entity_tracker=entity_tracker,
                            task_text=task_text,
                        )
                    self._last_processed_interaction_index = len(history)
                else:
                    last_interaction = getattr(self.adapter, "last_interaction", None)
                    if last_interaction is not None:
                        task = getattr(self.adapter, "task", None)
                        task_text = " ".join(m["content"] for m in task.prompt) if task else ""
                        entity_tracker = getattr(self.adapter, "entity_tracker", None)
                        self._current_operation = advance_frontier(
                            self.knowledge, self.adapter.scope, self.adapter.evidence,
                            last_interaction, self._current_operation, self._completed_operations,
                            entity_tracker=entity_tracker,
                            task_text=task_text,
                        )
                current_node = self._current_operation
            else:
                current_node = None
                last_interaction = getattr(self.adapter, "last_interaction", None)
                if last_interaction is not None:
                    if last_interaction.operation == "api_fetch":
                        from .behavior import endpoint_contract
    
                        contract = endpoint_contract(
                            self.knowledge,
                            self.adapter.scope,
                            last_interaction.arguments.get("method", "GET"),
                            last_interaction.arguments.get("url", ""),
                        )
                        if contract is not None:
                            current_node = contract.payload.get("id") or contract.claim_id
                        else:
                            current_node = last_interaction.operation
                    else:
                        current_node = last_interaction.operation
    
                    if last_interaction.outcome == "success" and current_node:
                        self._completed_operations.add(current_node)

            focus, subgoal = decision_focus(self.adapter)
            # Response bodies can contain unrelated words and instructions. In the
            # operational arm use explicit action bindings and its current subgoal.
            retrieval_query = query if self.settings.operational else query + " " + " ".join(recent)
            gating_decision_record = None
            if self.settings.gated_graph:
                from .graph_gating import evaluate_graph_feasibility
                gating = evaluate_graph_feasibility(query, self.knowledge, self.adapter.scope)
                gating_decision_record = {
                    "action": gating.action,
                    "reachability": gating.reachability,
                    "collision_hazard": gating.collision_hazard,
                    "required_services": gating.required_services,
                    "missing_services": gating.missing_services,
                    "colliding_entities": gating.colliding_entities,
                    "reason": gating.reason,
                }
                self.gating_decision = gating_decision_record
                if gating.action == "without_graph":
                    from .knowledge import WorkingGraph
                    working = WorkingGraph(scope=self.adapter.scope, claims=[])
                else:
                    working = retrieve(
                        self.knowledge,
                        self.adapter.scope,
                        retrieval_query,
                        live=self.adapter.live,
                        max_chars=self.settings.retrieval_chars,
                        expand=self.settings.memory == "graph",
                        focus_entities=focus,
                        subgoal=subgoal,
                        current_node=current_node,
                        completed_nodes=set(self._completed_operations),
                    )
            else:
                working = retrieve(
                    self.knowledge,
                    self.adapter.scope,
                    retrieval_query,
                    live=self.adapter.live,
                    max_chars=self.settings.retrieval_chars,
                    expand=self.settings.memory == "graph",
                    focus_entities=focus,
                    subgoal=subgoal,
                    current_node=current_node,
                    completed_nodes=set(self._completed_operations),
                )
            procedures, rejected = [], {}
            if self.settings.adaptive:
                procedures, rejected = select_procedures(
                    self.knowledge,
                    self.adapter.scope,
                    self.adapter.live,
                    query + " " + " ".join(recent),
                    allow_experimental=self.settings.allow_experimental,
                    linked_claim_ids={c.claim_id for c in working.claims},
                )
            hints = ""
            if self.settings.semantic_learning:
                from .semantic_learning import hypothesis_context, investigation_context

                hint_budget = min(2400, self.settings.retrieval_chars // 5)
                pending = (
                    investigation_context(
                        self.knowledge,
                        self.adapter.scope,
                        query + " " + " ".join(recent),
                        max_chars=hint_budget // 2,
                    )
                    if self.settings.information_policy != "off"
                    else ""
                )
                hints = hypothesis_context(
                    self.knowledge,
                    self.adapter.scope,
                    query + " " + " ".join(recent),
                    max_chars=hint_budget - len(pending) - (1 if pending else 0),
                )
                if pending:
                    hints = (hints + "\n" if hints else "") + pending
                if self.online_learner is not None:
                    online_budget = max(400, hint_budget // 2) if hints else hint_budget
                    online_hints = self.online_learner.context(max_chars=online_budget)
                    if online_hints:
                        if hints:
                            combined = hints + "\n\n" + online_hints
                            if len(combined) <= hint_budget:
                                hints = combined
                            else:
                                rem = max(0, hint_budget - len(hints) - 2)
                                if rem > 200:
                                    trimmed = self.online_learner.context(max_chars=rem)
                                    if trimmed:
                                        hints = hints + "\n\n" + trimmed
                        else:
                            hints = online_hints
            feedback = ""
            if self.settings.operational:
                from .observable_predictions import prediction_feedback

                feedback = prediction_feedback(
                    self.adapter.prediction_assessments,
                )
            tracker = getattr(self.adapter, "goal_tracker", None)
            goals = tracker.context() if tracker else ""
            reserved = sum(len(part) + 1 for part in (hints, feedback, goals) if part)
            if reserved >= self.settings.retrieval_chars:
                from .context_compression import MandatoryContextOverflow

                self.retrievals.append(
                    {
                        "error": "mandatory_context_overflow",
                        "required_chars": reserved,
                        "available_chars": self.settings.retrieval_chars,
                        "goal_check_count": len(tracker.checks) if tracker else 0,
                        "omitted_goal_checks": 0,
                    }
                )
                raise MandatoryContextOverflow(
                    "Complete compressed task state exceeds the context allowance; "
                    "increase the declared allowance. No requirement was silently omitted."
                )
            unresolved_refs = []
            if self.settings.operational or self.settings.semantic_learning:
                from .reference_investigation import extract_unresolved_references

                unresolved_refs = extract_unresolved_references(self.adapter)

            content, selection = render_context(
                working,
                procedures,
                max_chars=self.settings.retrieval_chars - reserved,
                store=self.knowledge,
                query=retrieval_query,
                current_node=current_node,
                completed_nodes=set(self._completed_operations),
                successful_transition_graph=self.settings.successful_transition_graph or self.settings.operational,
                entity_tracker=getattr(self.adapter, "entity_tracker", None),
                unresolved_references=unresolved_refs,
            )
            current_claim_ids = {c.claim_id for c in working.claims}
            newly_injected = current_claim_ids - self._last_working_claims
            retired = self._last_working_claims - current_claim_ids
            self._last_working_claims = current_claim_ids

            selection["current_node"] = current_node
            selection["completed_nodes"] = list(self._completed_operations)
            selection["newly_injected_claim_count"] = len(newly_injected)
            selection["retired_claim_count"] = len(retired)
            if hints:
                content += "\n" + hints
                selection["hypothesis_chars"] = len(hints)
            if feedback:
                content += "\n" + feedback
                selection["prediction_feedback_chars"] = len(feedback)
            if goals:
                content += "\n" + goals
                selection["goal_check_chars"] = len(goals)
            if selection.get("workflow_plan_active_step") and newly_injected and current_node:
                active_st = selection["workflow_plan_active_step"]
                frontier_note = f"\n[State Frontier: Active step {active_st}. Dynamic frontier updated with {len(newly_injected)} contracts.]"
                content += frontier_note
            if selection.get("workflow_plan"):
                from .context import _format_workflow_process_markdown

                md_transitions = _format_workflow_process_markdown(
                    selection["workflow_plan"], current_node=current_node
                )
                if md_transitions:
                    content += "\n\n" + md_transitions
                    selection["workflow_plan_md_chars"] = len(md_transitions)
            if self.settings.operational or self.settings.semantic_learning:
                from .reference_investigation import advisory_reference_context

                ref_guidance = advisory_reference_context(self.adapter, max_chars=1200)
                plan_has_refs = bool(selection.get("workflow_plan") and selection["workflow_plan"].get("unresolved_references"))
                if ref_guidance and not plan_has_refs:
                    content += "\n\n" + ref_guidance
                    selection["reference_guidance_chars"] = len(ref_guidance)
            selection["injected_chars"] = len(content)
            self.adapter.ledger.compute_seconds += perf_counter() - started
            self.retrievals.append(
                {
                    "claims": [c.claim_id for c in working.claims],
                    "focus_entities": focus,
                    "subgoal": subgoal,
                    "gaps": working.gaps,
                    "rejected_artifacts": rejected,
                    **selection,
                }
            )
            if content:
                selected_messages = _inject_operational_context(
                    selected_messages, content, "Operational Memory & Learned Invariants"
                )
        return await self._generate(selected_messages, tools)


async def run_automationbench(
    root: Path,
    settings: ExperimentSettings,
    knowledge_path: Path | None = None,
    output_dir: Path | None = None,
    trial_label: str | None = None,
) -> dict:
    """`trial_label` separates repeated trials of one task (arm, iteration) under output_dir."""
    model_settings = load_settings(root)
    client = build_client(model_settings, load_api_key(root, model_settings))
    semantic_client = None
    if settings.semantic_learning and learns_from_trajectories(settings.phase):
        semantic_settings = type(model_settings).model_validate(
            {
                **model_settings.model_dump(),
                "max_tokens": settings.semantic_max_tokens,
                "reasoning_effort": settings.semantic_reasoning_effort,
                "json_mode": settings.semantic_json_mode,
            }
        )
        semantic_client = MeteredClient(build_client(semantic_settings, load_api_key(root, semantic_settings)))
    if output_dir is not None:
        output = output_dir
        output.mkdir(parents=True, exist_ok=True)
        run_id = output.name
    else:
        run_id = uuid4().hex
        output = root / "runs" / run_id
        output.mkdir(parents=True, exist_ok=False)
    checkpoint_metadata = None
    checkpoint_source = None
    if settings.phase == "checkpoint":
        from .checkpoint import copy_checkpoint

        if knowledge_path is None:
            raise ValueError("Checkpoint measurement requires an explicit immutable source")
        checkpoint_source = knowledge_path
        knowledge_path = output / "knowledge.sqlite"
        digest = copy_checkpoint(checkpoint_source, knowledge_path)
        checkpoint_metadata = {
            "source": str(checkpoint_source),
            "sha256": digest,
            "evaluation": "in-sample-checkpoint",
            "within_task_updates": "disposable; never carried to sibling trials",
        }
    if settings.phase == "frozen" and knowledge_path is None:
        raise ValueError("Frozen evaluation requires an explicit development knowledge store")
    knowledge_start = "continued" if knowledge_path is not None else "cold"
    knowledge_path = knowledge_path or output / "knowledge.sqlite"
    knowledge = KnowledgeStore(knowledge_path, frozen=not learns_from_trajectories(settings.phase))
    frozen_source = knowledge if settings.phase == "frozen" else None
    evidence = KnowledgeStore(output / "evidence.sqlite")
    results = []
    report = {
        "run_id": run_id,
        "harness_revision": "interaction-learning-v4",
        "feedback_protocol": "trajectory-only",
        "validation_scores_used_for_learning": False,
        "knowledge_start": knowledge_start,
        "benchmark": "automationbench",
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol": "custom-response-shift-stirrup-stream"
        if settings.shift
        else "public-api-toolset-custom-stirrup-stream",
        "settings": settings.model_dump(mode="json"),
        "model_settings": model_settings.model_dump(),
        "versions": {p: version(p) for p in ("stirrup", "litellm", "automation-bench")},
        "results": results,
        "checkpoint": checkpoint_metadata,
    }
    try:
        rows = list(dataset(settings.domain))
        if settings.task_names:
            requested = set(settings.task_names)
            rows = [r for r in rows if json.loads(r["info"])["task_name"] in requested]
            found = {json.loads(r["info"])["task_name"] for r in rows}
            if found != requested:
                raise ValueError("Unknown task name")
            by_name = {json.loads(r["info"])["task_name"]: r for r in rows}
            rows = [by_name[name] for name in settings.task_names]
        rows = rows[settings.offset : settings.offset + settings.limit]
        if not rows:
            raise ValueError("Empty task selection")
        for task_index, row in enumerate(rows):
            if frozen_source is not None:
                from .checkpoint import copy_checkpoint

                # Current documentation is usable locally, never carried to siblings.
                if knowledge is not frozen_source:
                    knowledge.close()
                local_path = output / f"task-contracts-{task_index}.sqlite"
                copy_checkpoint(knowledge_path, local_path)
                knowledge = KnowledgeStore(local_path)
            learning_call_start = len(semantic_client.calls) if semantic_client else 0
            ledger = Ledger(max_environment_calls=settings.max_environment_calls)
            learner = (
                Learner(knowledge)
                if learns_from_trajectories(settings.phase) and settings.memory in ("flat", "graph")
                else None
            )
            adapter = AutomationBenchAdapter(
                row,
                settings.domain,
                evidence,
                ledger,
                learner,
                knowledge,
                retain_interactions=learns_from_trajectories(settings.phase)
                and settings.memory == "episodes",
                shift=settings.shift
                if settings.shift and task_index >= settings.shift.start_task_index
                else None,
            )
            if settings.phase == "frozen" and adapter.task.task_id in knowledge.learning_tasks():
                raise ValueError(
                    "Frozen evaluation task overlaps development/online learning tasks"
                )
            if frozen_source is not None:
                adapter.documentation_learner = Learner(knowledge)
            adapter.semantics_client = semantic_client
            adapter.max_operation_interpretations = settings.max_operation_interpretations
            if learns_from_trajectories(settings.phase) and settings.memory != "none":
                knowledge.register_learning_task(adapter.task.task_id, settings.phase)
            wrapper = KnowledgeClient(client, adapter, knowledge, settings)
            if semantic_client is not None and settings.max_online_reflections:
                from .online_learning import OnlineLearner

                wrapper.online_learner = OnlineLearner(
                    adapter, semantic_client, max_reflections=settings.max_online_reflections,
                    timeout_seconds=settings.semantic_timeout_seconds,
                )
            tools = adapter.tools()
            information_audit = []
            if settings.information_policy != "off":
                tools.append(
                    tracked_information_tool(
                        information_tool(
                            adapter,
                            policy=settings.information_policy,
                            max_calls=settings.max_information_calls,
                        ),
                        information_audit,
                    )
                )
            decision_audit, refresh_audit, lookup_audit = [], [], []
            plain_finish = Tool(
                name="finish",
                description="Finish the task and summarize the result.",
                parameters=Finish,
                executor=lambda _: ToolResult(content="Finished"),
            )
            if settings.operational:
                from .baseline_fallback import BaselineFallback
                from .goal_checks import CheckedFinish

                # Stirrup validates finish arguments against the session's registered
                # finish schema, so the unchecked fallback finish accepts the same one.
                adapter.baseline_fallback = BaselineFallback(
                    settings.baseline_fallback,
                    baseline_tools=tools,
                    finish_tool=plain_finish.model_copy(update={"parameters": CheckedFinish}),
                )
                from .operational_tools import operational_tools

                tools = operational_tools(
                    adapter,
                    tools,
                    adaptive=settings.adaptive,
                    allow_experimental=settings.allow_experimental,
                    audit=decision_audit,
                    refresh_audit=refresh_audit,
                    lookup_audit=lookup_audit,
                    retrieval_chars=settings.retrieval_chars,
                    expand=settings.memory == "graph",
                    hypothesis_policy=settings.information_policy
                    if settings.semantic_learning
                    else "off",
                    max_hypothesis_probes=settings.max_information_calls,
                    information_audit=information_audit,
                )
            elif settings.adaptive:
                tools.append(
                    artifact_tool(
                        knowledge,
                        adapter.scope,
                        adapter.live,
                        {t.name: t for t in tools},
                        allow_experimental=settings.allow_experimental,
                    )
                )
            finish_tool = plain_finish
            if settings.operational:
                finish_tool = adapter.goal_tracker.finish_tool()
            from .interactions import InteractionRecorder

            interaction_recorder = InteractionRecorder(adapter)
            interaction_recorder.register_tools([*tools, finish_tool])
            agent = RepairingAgent(
                interaction_recorder=interaction_recorder,
                repair_schemas={t.name: t.parameters for t in [*tools, finish_tool]},
                client=wrapper,
                name="automationbench",
                max_turns=settings.max_turns,
                tools=tools,
                finish_tool=finish_tool,
                recover_from_context_overflow=False,
                context_summarization_cutoff=1.0,
            )
            messages = [
                SystemMessage(content=m["content"])
                if m["role"] == "system"
                else UserMessage(content=m["content"])
                for m in adapter.task.prompt
            ]
            if settings.information_policy != "off":
                messages.append(
                    UserMessage(
                        content="Information-gathering policy: "
                        + settings.information_policy
                        + ". "
                        + information_route_instruction(operational=settings.operational)
                        + "Name possible distinguishing observations; "
                        "the call is not proof the uncertainty was resolved. Ordinary task "
                        "diagnosis remains allowed. "
                        + (
                            "Future-only information gathering is disabled."
                            if settings.information_policy == "task"
                            else "You may also gather information expected to help later tasks; "
                            "do not sacrifice the current task or exhaust the shared budget."
                        )
                    )
                )
            if output_dir is not None:
                task_output = output / "trajectories" / adapter.task.task_id
                if trial_label:
                    task_output = task_output / trial_label
            else:
                task_output = output / adapter.scope.task_id
            task_output.mkdir(parents=True, exist_ok=True)
            result = {"task_name": adapter.task.task_id, "scope": adapter.scope.model_dump()}
            from .trajectory_graph import SAMPLING_SALT

            SAMPLING_SALT.set(f"{settings.sampling_seed}:{adapter.task.task_id}")
            try:
                async with agent.session(
                    output_dir=task_output, cache_on_interrupt=False
                ) as session:
                    if settings.timeout_seconds is None:
                        finish, history, metadata = await session.run(messages)
                    else:
                        async with asyncio.timeout(settings.timeout_seconds):
                            finish, history, metadata = await session.run(messages)
                result.update(
                    {
                        "finished": finish is not None,
                        "token_usage": sum(metadata["token_usage"], TokenUsage()).model_dump(),
                        "model_calls_with_usage": len(metadata["token_usage"]),
                        "artifact_invocations": sum(
                            m.role == "tool" and getattr(m, "name", None) == "run_procedure"
                            for group in history
                            for m in group
                        ),
                        "information_invocations": sum(
                            m.role == "tool" and getattr(m, "name", None) == "inspect_contract"
                            for group in history
                            for m in group
                        ),
                    }
                )
                traj_json = json.dumps(
                    [[m.model_dump(mode="json") for m in group] for group in history], indent=2
                )
                (task_output / "history.json").write_text(traj_json)
                (task_output / "trajectory.json").write_text(traj_json)
            except Exception as exc:  # noqa: BLE001 -- keep SDK error details out of artifacts
                result.update({"finished": False, "error_type": type(exc).__name__})
                partial_json = json.dumps(
                    [m.model_dump(mode="json") for m in wrapper.last_messages], indent=2
                ) + "\n"
                (task_output / "partial_history.json").write_text(partial_json)
                (task_output / "trajectory.json").write_text(partial_json)
                (task_output / "history.json").write_text(partial_json)
            result["model_call_costs"] = wrapper.model_call_costs
            result["outer_validation_audit"] = agent.outer_validation_audit
            result["online_learning"] = (
                wrapper.online_learner.audit if wrapper.online_learner is not None else []
            )
            result["operation_semantics"] = getattr(adapter, "operation_semantics_audit", [])
            if settings.semantic_learning and learner is not None:
                from .semantic_learning import learn_semantics
                from .semantic_review import review_semantics

                semantic_call_start = len(semantic_client.calls)
                semantic = {}
                try:
                    async with asyncio.timeout(settings.semantic_timeout_seconds):
                        semantic["hypothesis_ids"] = await learn_semantics(
                            semantic_client,
                            knowledge,
                            adapter.scope,
                            accounting=semantic,
                        )
                except Exception as exc:  # noqa: BLE001 -- preserve failed consolidation cost without SDK secrets
                    semantic["error_type"] = type(exc).__name__
                review_accounting = {}
                semantic["review"] = review_accounting
                try:
                    async with asyncio.timeout(settings.semantic_timeout_seconds):
                        semantic["reviewed_ids"] = await review_semantics(
                            semantic_client,
                            knowledge,
                            adapter.scope,
                            accounting=review_accounting,
                        )
                except Exception as exc:  # noqa: BLE001 -- preserve failed review cost without SDK secrets
                    review_accounting["error_type"] = type(exc).__name__
                ledger.adaptation_seconds += semantic.get("semantic_learning_seconds", 0)
                ledger.adaptation_seconds += semantic.get("review", {}).get("review_seconds", 0)
                semantic["model_call_costs"] = semantic_client.calls[semantic_call_start:]
                semantic["model_settings"] = semantic_settings.model_dump()
                result["semantic_learning"] = semantic
            if learner is not None:
                from .contrastive import learn_contrastive_preconditions

                try:
                    contrastive_ids = learn_contrastive_preconditions(
                        knowledge,
                        adapter.scope,
                    )
                    if contrastive_ids:
                        result["contrastive_precondition_ids"] = contrastive_ids
                except Exception as exc:  # noqa: BLE001
                    result["contrastive_learning_error"] = type(exc).__name__
            # One non-overlapping call list includes failed calls and every learning stage.
            result["all_model_call_costs"] = [
                *({**call, "stage": "acting"} for call in wrapper.model_call_costs),
                *({**call, "stage": "learning"} for call in
                  (semantic_client.calls[learning_call_start:] if semantic_client else [])),
            ]
            # Score only after interaction. A missing rubric leaves the learned graph intact.
            attach_validation_score(result, adapter)
            result["ledger"] = asdict(ledger)
            result["retrievals"] = wrapper.retrievals
            if wrapper.gating_decision is not None:
                result["gating_decision"] = wrapper.gating_decision
            if settings.advisory:
                result["advisory"] = wrapper.advisory
            result["prediction_assessments"] = adapter.prediction_assessments
            if settings.operational:
                result["decision_audit"] = decision_audit
                result["goal_checks"] = adapter.goal_tracker.report()
                result["goal_verification_audit"] = adapter.goal_tracker.verification_audit
                result["goal_finish_audit"] = adapter.goal_tracker.finish_audit
                result["completion_evidence_status"] = getattr(
                    adapter.goal_tracker, "completion_status", "not_finished"
                )
                result["baseline_fallback"] = adapter.baseline_fallback.report()
                result["artifact_reuse"] = adapter.artifact_audit
                result["refresh_audit"] = refresh_audit
                result["lookup_audit"] = lookup_audit
                result["artifact_invocations"] = sum(
                    row["invoked"] and row["decision"]["action"]["tool"] == "run_procedure"
                    for row in decision_audit
                )
            result["information_audit"] = information_audit
            result["information_invocations"] = len(information_audit)
            result["information_invocations_by_tool"] = {
                name: sum(row["tool"] == name for row in information_audit)
                for name in ("inspect_contract", "investigate_hypothesis")
            }
            if adapter.shift is not None:
                result["controlled_shift"] = adapter.shift.report()
            results.append(result)
            if output_dir is not None:
                (task_output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                if result.get("grading_details"):
                    (task_output / "rubric.json").write_text(json.dumps(result["grading_details"], indent=2) + "\n")
                # The run-level evidence store accumulates every trial; keep only this one's.
                trial_evidence_path = task_output / "evidence.sqlite"
                trial_evidence_path.unlink(missing_ok=True)
                trial_evidence = KnowledgeStore(trial_evidence_path)
                try:
                    for episode in evidence.episodes(adapter.scope):
                        trial_evidence.append_episode(episode)
                finally:
                    trial_evidence.close()
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        scored = [r["task_completed_correctly"] for r in results if "task_completed_correctly" in r]
        report["mean_strict_success"] = sum(scored) / len(scored) if scored else None
        report["knowledge_claims"] = len(knowledge.claims())
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        knowledge.close()
        if frozen_source is not None and frozen_source is not knowledge:
            frozen_source.close()
        evidence.close()
        if checkpoint_source is not None:
            from .checkpoint import checkpoint_hash

            if checkpoint_hash(checkpoint_source) != checkpoint_metadata["sha256"]:
                raise ValueError("Measurement modified its source checkpoint")
