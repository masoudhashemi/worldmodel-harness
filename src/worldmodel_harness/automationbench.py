"""Upstream API-tool adapter. The evaluator is never exposed as an agent tool.

This uses upstream world initialization helpers, API functions, task prompts, and
rubric unchanged; Stirrup replaces only the upstream model/execution loop.
"""

import copy
import inspect
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, get_type_hints
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, create_model
from stirrup.core.models import Tool, ToolResult

from .behavior import BehaviorLearner, predict
from .budget import Ledger
from .composition import CompositionLearner
from .information import CURRENT_INFORMATION
from .interactions import CURRENT_INTERACTION
from .knowledge import Episode, LiveState, Scope, response_coverage
from .learning import Learner
from .observable_predictions import CURRENT_EXPECTATIONS, compare_expectations
from .procedures import CURRENT_ARTIFACT
from .shift import ResponseShift, ShiftController
from .store import KnowledgeStore

UPSTREAM_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    domain: str
    prompt: tuple[dict[str, str], ...]


def dataset(domain: str):
    from automationbench.domains import PUBLIC_DOMAINS, get_combined_dataset, get_domain_dataset

    if domain == "public":
        return get_combined_dataset(PUBLIC_DOMAINS)
    return get_domain_dataset(domain)


class AutomationBenchAdapter:
    def __init__(
        self,
        row: dict,
        domain: str,
        evidence: KnowledgeStore,
        ledger: Ledger,
        learner: Learner | None = None,
        knowledge: KnowledgeStore | None = None,
        purpose_override: str | None = None,
        retain_interactions: bool = False,
        shift: ResponseShift | None = None,
    ):
        from automationbench.runner import compute_allowed_services, strip_none_values
        from automationbench.schema.world import WorldState

        info = row["info"]
        if isinstance(info, str):
            info = json.loads(info)
        info = copy.deepcopy(info)
        initial = strip_none_values(info["initial_state"])
        info["assertions"] = [strip_none_values(a) for a in info["assertions"]]
        world = WorldState(**initial)
        world.meta.allowed_services = compute_allowed_services(
            initial, info["assertions"], info.get("zapier_tools", [])
        )
        # Private evaluator state is intentionally not part of PublicTask.
        self._state = {"world": world, "initial_state": copy.deepcopy(initial), "info": info}
        self.task = PublicTask(
            task_id=info["task_name"], domain=domain, prompt=tuple(copy.deepcopy(row["prompt"]))
        )
        self.scope = Scope(
            environment="automationbench",
            version=UPSTREAM_COMMIT,
            tenant=self.task.task_id,
            task_id=uuid4().hex,
        )
        self.live = LiveState(self.scope.task_id)
        self.evidence = evidence
        self.ledger = ledger
        self.learner = learner
        self.knowledge = knowledge or (learner.store if learner else evidence)
        self.prediction_assessments: list[dict] = []
        self.purpose_override = purpose_override
        self.retain_interactions = retain_interactions
        self.shift = ShiftController(shift) if shift is not None else None

    def tools(self) -> list[Tool]:
        from automationbench.tools.api import API_TOOLS

        result = []
        for function in API_TOOLS:
            annotations = get_type_hints(function)
            fields = {}
            for name, parameter in inspect.signature(function).parameters.items():
                if name == "world":
                    continue
                default = ... if parameter.default is inspect.Parameter.empty else parameter.default
                fields[name] = (annotations.get(name, Any), default)
            parameters = create_model(
                function.__name__ + "Input", __config__=ConfigDict(extra="forbid"), **fields
            )

            async def execute(params: BaseModel, fn=function):
                return await self.execute(fn, params.model_dump())

            result.append(
                Tool(
                    name=function.__name__,
                    description=inspect.getdoc(function) or "",
                    parameters=parameters,
                    executor=execute,
                )
            )
        return result

    async def execute(self, function, arguments: dict, purpose: str = "task") -> ToolResult:
        operation = function.__name__
        purpose = self.purpose_override or (
            "discovery" if operation == "api_search" and purpose == "task" else purpose
        )
        # Match the upstream empty-object-as-omitted convention.
        arguments = {k: v for k, v in arguments.items() if v != {} or not isinstance(v, dict)}
        if operation == "base64_encode":
            self.ledger.utility_calls += 1
        else:
            self.ledger.charge_environment(purpose)
        started = perf_counter()
        predictions = (
            predict(self.knowledge, self.scope, arguments) if operation == "api_fetch" else ()
        )
        proposed = CURRENT_EXPECTATIONS.get()
        expectations = proposed[1] if proposed is not None and proposed[0] == operation else ()
        pre_evidence = tuple(dict.fromkeys(f.episode_id for f in self.live.facts()))
        effect, effect_claim = "unknown", None
        if operation == "api_fetch":
            from .operation_semantics import interpret_effect

            effect, effect_claim = await interpret_effect(self, arguments)
        if operation == "api_fetch" and effect != "read":
            from .resource_invalidation import invalidation_keys

            # Include queries/collections and known cross-service dependents;
            # unknown write identity falls back to broad invalidation.
            self.live.invalidate(
                invalidation_keys(self, arguments),
                "pending-write",
            )
        kwargs = dict(arguments)
        if operation == "api_fetch" and "url" in kwargs and "?" in str(kwargs["url"]):
            import urllib.parse

            parsed = urllib.parse.urlsplit(str(kwargs["url"]))
            clean_url = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, "", "")
            )
            kwargs["url"] = clean_url
            if parsed.query:
                qs_dict = {
                    k: v[0] if len(v) == 1 else v
                    for k, v in urllib.parse.parse_qs(parsed.query).items()
                }
                existing_params = kwargs.get("params")
                if not existing_params:
                    kwargs["params"] = json.dumps(qs_dict)
                elif isinstance(existing_params, str):
                    try:
                        p_dict = json.loads(existing_params)
                        if isinstance(p_dict, dict):
                            kwargs["params"] = json.dumps({**qs_dict, **p_dict})
                    except Exception:
                        pass
                elif isinstance(existing_params, dict):
                    kwargs["params"] = json.dumps({**qs_dict, **existing_params})
        if "world" in inspect.signature(function).parameters:
            kwargs["world"] = self._state["world"]
        self.ledger.compute_seconds += perf_counter() - started
        tool_started = perf_counter()
        try:
            raw = function(**kwargs)
        except (TimeoutError, ConnectionError):
            # An exception does not establish whether a mutation happened.
            raw = json.dumps({"error": "transport_failure", "outcome": "unknown"})
            outcome = "unknown"
        else:
            outcome = "success"
        tool_elapsed = perf_counter() - tool_started
        if operation == "base64_encode":
            self.ledger.compute_seconds += tool_elapsed
        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            response = raw
        if isinstance(response, dict) and "error" in response and outcome != "unknown":
            outcome = "error"
        shifted = False
        if self.shift is not None:
            response, shifted = self.shift.apply(operation, arguments, response, outcome)
            if shifted:
                raw = json.dumps(response)
        episode = Episode(
            effect_kind=effect,
            effect_source_claim_id=effect_claim,
            interaction_id=CURRENT_INTERACTION.get(),
            environment_called=True,
            task_id=self.live.task_id,
            scope=self.scope,
            operation=operation,
            arguments=arguments,
            response=response,
            outcome=outcome,
            coverage=response_coverage(response),
            purpose=purpose,
            elapsed_seconds=tool_elapsed,
            predictions=predictions,
            output_expectations=expectations,
            pre_evidence=pre_evidence,
            information_request=CURRENT_INFORMATION.get(),
            **(CURRENT_ARTIFACT.get() or {}),
        )
        self.evidence.append_episode(episode)
        if expectations:
            self.prediction_assessments.append(
                {
                    "episode_id": episode.episode_id,
                    "proposed_output_assessments": compare_expectations(episode),
                    "interpretation": "Observed response checks, not causal or task-success proof",
                }
            )
        if (
            self.retain_interactions
            and self.learner is None
            and self.knowledge is not self.evidence
        ):
            self.knowledge.append_episode(episode)
        self.live.observe(episode)
        assessments = {}
        if self.learner:
            learning_started = perf_counter()
            if self.learner.store is not self.evidence:
                self.learner.store.append_episode(episode)
            assessments = self.learner.compare(episode)
            BehaviorLearner(self.learner).observe(episode)
            CompositionLearner(self.learner).observe(episode)
            from .rule_tests import observe_rules

            tested_rules = observe_rules(self.knowledge, episode)
            if tested_rules:
                assessments["tested_response_rules"] = tested_rules
            self.prediction_assessments.append(
                {"episode_id": episode.episode_id, "assessments": assessments}
            )
            self.ledger.compute_seconds += perf_counter() - learning_started
        elif predictions:
            from .learning import assess_prediction

            assessments = {p.claim_id: assess_prediction(p, episode) for p in predictions}
            self.prediction_assessments.append(
                {
                    "episode_id": episode.episode_id,
                    "assessments": assessments,
                }
            )
        documenter = self.learner or getattr(self, "documentation_learner", None)
        if documenter and operation == "api_search" and outcome == "success":
            learning_started = perf_counter()
            if self.learner is None and documenter.store is not self.evidence:
                documenter.store.append_episode(episode)
            reusable = Scope(environment=self.scope.environment, version=self.scope.version)
            for endpoint in response.get("results", []):
                documenter.document(
                    episode,
                    relation="endpoint_contract",
                    participants=(endpoint["id"], endpoint["url"], endpoint["method"]),
                    statement=endpoint.get("description", "API endpoint"),
                    payload=endpoint,
                    scope=reusable,
                )
            self.ledger.compute_seconds += perf_counter() - learning_started
        if self.shift is not None:
            self.shift.record(
                episode,
                applied=shifted,
                call_index=self.ledger.environment_calls,
                assessments=assessments,
            )
        return ToolResult(content=raw, success=outcome == "success")

    def score(self) -> dict[str, float]:
        from automationbench.rubric import partial_credit, task_completed_correctly

        partial_credit(self._state)
        return {
            "partial_credit": self._state["partial_credit"],
            "task_completed_correctly": task_completed_correctly(self._state),
        }

    def grading_details(self) -> list[dict]:
        """Controller-only diagnostics, available only after post-task scoring."""
        if "_assertion_results" not in self._state:
            raise RuntimeError("Score the completed rollout before requesting diagnostics")
        return copy.deepcopy(self._state["_assertion_results"])
