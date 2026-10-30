"""Upstream API-tool adapter. The evaluator is never exposed as an agent tool.

This uses upstream world initialization helpers, API functions, task prompts, and
rubric unchanged; Stirrup replaces only the upstream model/execution loop.
"""

import copy
import inspect
import json
import urllib.parse
from dataclasses import dataclass
from time import perf_counter
from typing import Any, get_type_hints
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, create_model
from stirrup.core.models import Tool, ToolResult

from .behavior import BehaviorLearner
from .budget import Ledger
from .composition import CompositionLearner
from .knowledge import Episode, Scope, response_coverage
from .learning import Learner
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
        return list(get_combined_dataset(PUBLIC_DOMAINS))
    return list(get_domain_dataset(domain))


def task_row(task_name: str) -> dict:
    domain = task_name.split(".", 1)[0]
    for row in dataset(domain):
        if json.loads(row["info"])["task_name"] == task_name:
            return row
    raise ValueError(f"Unknown task name: {task_name}")


def _split_query(kwargs: dict) -> None:
    """Move a query string in the URL into params, as the upstream routes expect."""
    parsed = urllib.parse.urlsplit(str(kwargs["url"]))
    kwargs["url"] = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    if not parsed.query:
        return
    query = {k: v[0] if len(v) == 1 else v for k, v in urllib.parse.parse_qs(parsed.query).items()}
    existing = kwargs.get("params")
    if not existing:
        kwargs["params"] = json.dumps(query)
    elif isinstance(existing, str):
        try:
            loaded = json.loads(existing)
        except ValueError:
            return
        if isinstance(loaded, dict):
            kwargs["params"] = json.dumps({**query, **loaded})
    elif isinstance(existing, dict):
        kwargs["params"] = json.dumps({**query, **existing})


class AutomationBenchAdapter:
    """Runs upstream tools on a private world. With a store, every call is learned from."""

    def __init__(self, row: dict, domain: str, ledger: Ledger, store: KnowledgeStore | None = None):
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
        self.ledger = ledger
        self.store = store
        self.learner = Learner(store) if store is not None else None
        self.episodes: list[Episode] = []

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

    async def execute(self, function, arguments: dict) -> ToolResult:
        operation = function.__name__
        purpose = "discovery" if operation == "api_search" else "task"
        # Match the upstream empty-object-as-omitted convention.
        arguments = {k: v for k, v in arguments.items() if v != {} or not isinstance(v, dict)}
        if operation == "base64_encode":
            self.ledger.utility_calls += 1
        else:
            self.ledger.charge_environment(purpose)
        kwargs = dict(arguments)
        if operation == "api_fetch" and "?" in str(kwargs.get("url", "")):
            _split_query(kwargs)
        if "world" in inspect.signature(function).parameters:
            kwargs["world"] = self._state["world"]
        started = perf_counter()
        try:
            raw = function(**kwargs)
            outcome = "success"
        except (TimeoutError, ConnectionError):
            # An exception does not establish whether a mutation happened.
            raw = json.dumps({"error": "transport_failure", "outcome": "unknown"})
            outcome = "unknown"
        except Exception as exc:  # noqa: BLE001 -- upstream ToolEnv returns tool errors to the model
            # Match upstream: the message is the tool result and the episode continues.
            raw = str(exc)
            outcome = "error"
        elapsed = perf_counter() - started
        if operation == "base64_encode":
            self.ledger.compute_seconds += elapsed
        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            response = raw
        if isinstance(response, dict) and "error" in response and outcome == "success":
            outcome = "error"
        if self.store is not None:
            learning_started = perf_counter()
            self._learn(operation, arguments, response, outcome, purpose, elapsed)
            self.ledger.compute_seconds += perf_counter() - learning_started
        return ToolResult(content=raw, success=outcome == "success")

    def _learn(self, operation, arguments, response, outcome, purpose, elapsed) -> None:
        # The latest successful read of each URL; a repeated read would make values ambiguous.
        latest = {
            str(e.arguments.get("url")): e.episode_id
            for e in self.episodes
            if e.operation == "api_fetch" and e.outcome == "success"
        }
        episode = Episode(
            task_id=self.scope.task_id,
            scope=self.scope,
            operation=operation,
            arguments=arguments,
            response=response,
            outcome=outcome,
            coverage=response_coverage(response),
            pre_evidence=tuple(latest.values()),
            purpose=purpose,
            elapsed_seconds=elapsed,
        )
        self.store.append_episode(episode)
        self.episodes.append(episode)
        if operation == "api_search" and outcome == "success":
            reusable = Scope(environment=self.scope.environment, version=self.scope.version)
            for endpoint in response.get("results", []):
                self.learner.document(
                    episode,
                    relation="endpoint_contract",
                    participants=(endpoint["id"], endpoint["url"], endpoint["method"]),
                    statement=endpoint.get("description", "API endpoint"),
                    payload=endpoint,
                    scope=reusable,
                )
        BehaviorLearner(self.learner).observe(episode)
        CompositionLearner(self.learner).observe(episode)

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
