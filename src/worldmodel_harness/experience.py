"""Offline, prefix-only experience memory. No rewards, gates, or actor integration.

Structural matches are not semantic state equivalence. Responses describe observed
knowledge, not a reconstructed world state or proof that a task was completed.
"""

import hashlib
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .behavior import value_kind
from .knowledge import Episode, Record, Scope


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class SymbolicSnapshot(Record):
    structure: Any
    truncated: bool = False


class Symbolizer:
    """Alpha-rename scalar instances while preserving equality and array positions.

    Field names are schema, not instance values. This is not a PII redactor for
    arbitrary dynamic dictionary keys. Literal values remain in source evidence.
    """

    def __init__(self, max_nodes: int = 512):
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.max_nodes = max_nodes
        self.bindings: dict[tuple[str, str], str] = {}

    def snapshot(self, value: Any) -> SymbolicSnapshot:
        remaining = self.max_nodes
        truncated = False

        def walk(item, field=""):
            nonlocal remaining, truncated
            if remaining <= 0:
                truncated = True
                return {"unobserved": "budget"}
            remaining -= 1
            if isinstance(item, dict):
                result = {}
                for key in sorted(item):
                    if remaining <= 0:
                        truncated = True
                        break
                    result[key] = walk(item[key], key)
                return {"object": result}
            if isinstance(item, list):
                result = []
                for child in item:
                    if remaining <= 0:
                        truncated = True
                        break
                    result.append(walk(child, field))
                return {"array": result, "length": len(item)}
            if item is None:
                return {"kind": "null"}
            key = (type(item).__name__, json.dumps(item, sort_keys=True))
            if key not in self.bindings:
                self.bindings[key] = f"v{len(self.bindings) + 1}"
            return {"kind": value_kind(item, field), "variable": self.bindings[key]}

        return SymbolicSnapshot(structure=walk(value), truncated=truncated)


class Capability(Record):
    operation: str
    method: str = ""
    route: str = ""
    contract_episode: str | None = None

    @property
    def key(self) -> str:
        return fingerprint((self.operation, self.method, self.route))


class Observation(Record):
    episode_id: str
    capability_key: str
    outcome: str
    response: SymbolicSnapshot


class ExperienceQuery(Record):
    scope: Scope
    task_id: str
    capability: Capability
    before: tuple[Observation, ...]
    action: SymbolicSnapshot
    history_omitted: int = 0

    def state_signature(self) -> str:
        # Recanonicalize variable names across the whole selected context/action.
        # This removes irrelevant numbering differences caused by older history.
        values = [
            {
                "capability": o.capability_key,
                "outcome": o.outcome,
                "response": o.response.model_dump(),
            }
            for o in self.before
        ]
        payload = {"before": values, "action": self.action.model_dump()}
        variables = {}

        def rename(value):
            if isinstance(value, dict):
                return {
                    k: (
                        variables.setdefault(v, f"v{len(variables) + 1}")
                        if k == "variable"
                        else rename(v)
                    )
                    for k, v in value.items()
                }
            if isinstance(value, list):
                return [rename(v) for v in value]
            return value

        return fingerprint(rename(payload))


class Transition(Record):
    event_id: str
    episode_id: str
    query: ExperienceQuery
    outcome: str
    observed_response: SymbolicSnapshot
    # Intentionally no inferred world effect or task success field.


class Match(Record):
    episode_id: str
    event_id: str
    task_id: str
    match_kind: str
    differences: tuple[str, ...]
    observed_outcome: str
    observed_response: SymbolicSnapshot


class PredictionReport(Record):
    matches: tuple[Match, ...]
    outcome_counts: dict[str, int]
    independent_tasks: int
    uncertainty: tuple[str, ...]


class ExperienceMemory:
    def __init__(self):
        self.transitions: list[Transition] = []
        self.events: set[tuple[str, ...]] = set()

    def append(self, transition: Transition) -> None:
        scope = transition.query.scope
        key = (
            scope.environment,
            scope.version,
            scope.role,
            scope.tenant or "",
            transition.event_id,
        )
        if key not in self.events:
            self.events.add(key)
            self.transitions.append(transition)

    def retrieve(self, query: ExperienceQuery, limit: int = 5) -> PredictionReport:
        if limit < 1:
            raise ValueError("limit must be positive")
        candidates = []
        for transition in self.transitions:
            previous = transition.query
            if previous.task_id == query.task_id:
                continue
            if previous.scope.model_dump(exclude={"task_id"}) != query.scope.model_dump(
                exclude={"task_id"}
            ):
                continue
            if previous.capability.key != query.capability.key:
                continue
            same = previous.state_signature() == query.state_signature()
            differences = () if same else ("Observed context or action structure differs",)
            candidates.append((same, transition, differences))
        # Do not mix different conditions into the exact-structure outcome counts.
        if any(same for same, _, _ in candidates):
            candidates = [item for item in candidates if item[0]]
        # Surface distinct outcomes before repeated examples of the same outcome.
        exemplars, remaining, outcomes = [], [], set()
        for item in candidates:
            outcome = item[1].outcome
            if outcome in outcomes:
                remaining.append(item)
            else:
                exemplars.append(item)
                outcomes.add(outcome)
        selected = (exemplars + remaining)[:limit]
        matches = tuple(
            Match(
                episode_id=t.episode_id,
                event_id=t.event_id,
                task_id=t.query.task_id,
                match_kind="same_observed_structure" if same else "same_capability",
                differences=diff,
                observed_outcome=t.outcome,
                observed_response=t.observed_response,
            )
            for same, t, diff in selected
        )
        uncertainty = [
            "Structural similarity does not establish semantic state equivalence",
            "Observed API outcomes do not establish task completion or causality",
        ]
        if not matches:
            uncertainty.append("No comparable historical evidence")
        if any(not same for same, _, _ in candidates):
            uncertainty.append("Some examples have different observed conditions")
        if (
            query.history_omitted
            or any(o.response.truncated for o in query.before)
            or query.action.truncated
        ):
            uncertainty.append("Current context is incomplete or truncated")
        counts = dict(Counter(t.outcome for _, t, _ in candidates))
        if any(
            t.query.history_omitted
            or t.query.action.truncated
            or t.observed_response.truncated
            or any(o.response.truncated for o in t.query.before)
            for _, t, _ in candidates
        ):
            uncertainty.append("Historical context or responses are incomplete or truncated")
        if len(counts) > 1:
            uncertainty.append("Historical outcomes disagree; missing conditions may explain this")
        return PredictionReport(
            matches=matches,
            outcome_counts=counts,
            independent_tasks=len({t.query.task_id for _, t, _ in candidates}),
            uncertainty=tuple(uncertainty),
        )


class PrefixExperience:
    """Build queries before observing the next response. One instance per task.

    Endpoint templates are learned only from preceding successful discovery
    responses. Unresolved routes are skipped, never generalized by ID guessing.
    """

    def __init__(self, scope: Scope, task_id: str, history_limit: int = 6):
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.scope, self.task_id = scope, task_id
        self.history_limit = history_limit
        self.symbolizer = Symbolizer()
        self.history: list[Observation] = []
        self.contracts: dict[tuple[str, str], str] = {}

    def query(self, operation: str, arguments: dict) -> ExperienceQuery | None:
        action = dict(arguments)
        capability = Capability(operation=operation)
        if operation == "api_fetch":
            method = str(arguments.get("method", "GET")).upper()
            url = str(arguments.get("url", ""))
            target = url.split("?", 1)[0].split("#", 1)[0]
            matches = []
            for (verb, route), source in self.contracts.items():
                parts = re.split(r"(\{[^{}]+\})", route)
                pattern = "".join("([^/?]+)" if p.startswith("{") else re.escape(p) for p in parts)
                match = re.fullmatch(pattern, target)
                if verb == method and match:
                    matches.append((route, source, match.groups()))
            if len(matches) != 1:
                return None
            route, source, bindings = matches[0]
            capability = Capability(
                operation=operation, method=method, route=route, contract_episode=source
            )
            action.pop("url", None)
            action.pop("method", None)
            action["path_bindings"] = list(bindings)
            action["url_query"] = [list(pair) for pair in sorted(parse_qsl(urlsplit(url).query))]
        # Normalize JSON containers, but do not infer encodings or business predicates.
        for key in ("body", "params"):
            if isinstance(action.get(key), str):
                try:
                    decoded = json.loads(action[key])
                    if isinstance(decoded, (dict, list)):
                        action[key] = decoded
                except ValueError:
                    pass
        return ExperienceQuery(
            scope=self.scope,
            task_id=self.task_id,
            capability=capability,
            before=tuple(self.history[-self.history_limit :]),
            history_omitted=max(0, len(self.history) - self.history_limit),
            action=self.symbolizer.snapshot(action),
        )

    def observe(self, episode: Episode, query: ExperienceQuery | None) -> Transition | None:
        if episode.task_id != self.task_id or episode.scope != self.scope:
            raise ValueError("Cannot mix task state or environment scopes")
        if episode.origin != "environment" or episode.environment_called is False:
            return None
        if query is not None and (query.task_id != self.task_id or query.scope != self.scope):
            raise ValueError("Query belongs to another task or scope")
        if query is not None and query != self.query(episode.operation, episode.arguments):
            raise ValueError("Query is stale or describes a different action")
        if episode.operation == "api_search" and episode.outcome == "success":
            response = episode.response
            if isinstance(response, dict):
                for endpoint in response.get("results", []):
                    if isinstance(endpoint, dict) and all(
                        isinstance(endpoint.get(k), str) for k in ("method", "url")
                    ):
                        self.contracts[(endpoint["method"].upper(), endpoint["url"])] = (
                            episode.episode_id
                        )
        if query is None:
            return None
        response = self.symbolizer.snapshot(episode.response)
        self.history.append(
            Observation(
                episode_id=episode.episode_id,
                capability_key=query.capability.key,
                outcome=episode.outcome,
                response=response,
            )
        )
        return Transition(
            event_id=episode.event_id,
            episode_id=episode.episode_id,
            query=query,
            outcome=episode.outcome,
            observed_response=response,
        )
