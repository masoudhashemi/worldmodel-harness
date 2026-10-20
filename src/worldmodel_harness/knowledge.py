"""Typed records shared by learning, retrieval, and checked execution.

Layers describe information lifetimes. Functional models consume these records;
they do not each own an independent graph.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def fields_overlap(required: tuple[str, ...], changed: set[str]) -> bool:
    return (
        not required
        or not changed
        or any(
            a == b or a.startswith(b + ".") or b.startswith(a + ".")
            for a in required
            for b in changed
        )
    )


def changed_payload_fields(before: dict, after: dict, prefix: str = "") -> set[str]:
    """Dictionary paths are granular; arrays and replacements are atomic changes."""
    changed = set()
    for key in before.keys() | after.keys():
        path = f"{prefix}.{key}" if prefix else key
        if key not in before or key not in after:
            changed.add(path)
        elif isinstance(before[key], dict) and isinstance(after[key], dict):
            changed.update(changed_payload_fields(before[key], after[key], path))
        elif type(before[key]) is not type(after[key]) or before[key] != after[key]:
            changed.add(path)
    return changed


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Layer(StrEnum):
    INTERFACE = "interface"
    CONFIGURATION = "configuration"
    BEHAVIOR = "behavior"
    PROCEDURES = "procedures"
    LIVE = "live"
    EVIDENCE = "evidence"


class Scope(Record):
    environment: str
    version: str
    role: str = "agent"
    tenant: str | None = None
    task_id: str | None = None

    def applies_to(self, target: "Scope") -> bool:
        return (
            self.environment == target.environment
            and self.version == target.version
            and self.role == target.role
            and (self.tenant is None or self.tenant == target.tenant)
            and (self.task_id is None or self.task_id == target.task_id)
        )


class Prediction(Record):
    claim_id: str
    claim_version: int = Field(ge=1)
    path: str
    expected: Any
    predicate: Literal["equals", "present", "type"] = "equals"


class InformationRequest(Record):
    """Declared intent, never evidence that uncertainty was actually resolved."""

    uncertainty: str = Field(min_length=1, max_length=1000)
    distinguishing_outcomes: tuple[str, ...] = Field(min_length=2, max_length=8)
    intended_use: Literal["current_task", "future_tasks"] = "current_task"
    target_claim_id: str | None = None


class OutputExpectation(Record):
    """A proposed observable prediction, not a supported world claim."""

    path: tuple[str | Annotated[int, Field(strict=True, ge=0)], ...] = ()
    predicate: Literal["equals", "present", "type", "format"] = "equals"
    expected: Any = None

    @model_validator(mode="before")
    @classmethod
    def normalize_expectation(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "path" in data and isinstance(data["path"], (str, int)):
                data["path"] = (data["path"],)
            if "type" in data:
                type_val = data.pop("type")
                if "predicate" not in data or data["predicate"] == "equals":
                    data["predicate"] = "type"
                if "expected" not in data or data["expected"] is None:
                    data["expected"] = type_val
        return data


class Episode(Record):
    episode_id: str = Field(default_factory=lambda: uuid4().hex)
    # Repeated reads/duplicate logging of one event do not supply independent support.
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    scope: Scope
    operation: str
    arguments: dict[str, Any]
    response: Any
    outcome: Literal["success", "error", "unknown", "blocked"]
    coverage: tuple[str, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    output_expectations: tuple[OutputExpectation, ...] = ()
    pre_evidence: tuple[str, ...] = ()
    information_request: InformationRequest | None = None
    artifact_id: str | None = None
    artifact_version: int | None = None
    artifact_step: str | None = None
    artifact_execution_id: str | None = None
    purpose: Literal["task", "refresh", "probe", "validation", "discovery"] = "task"
    elapsed_seconds: float = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=utc_now)


class Dependency(Record):
    claim_id: str
    version: int = Field(ge=1)
    fields: tuple[str, ...] = ()
    mode: Literal["requires", "monitors"] = "requires"


class Claim(Record):
    claim_id: str = Field(default_factory=lambda: uuid4().hex)
    version: int = Field(default=1, ge=1)
    layer: Literal[Layer.INTERFACE, Layer.CONFIGURATION, Layer.BEHAVIOR]
    relation: str
    participants: tuple[str, ...] = Field(min_length=1)
    statement: str
    scope: Scope
    conditions: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    status: Literal["hypothesis", "supported", "disputed", "retired"] = "hypothesis"
    origin: Literal["documented", "observed", "inferred"] = "inferred"
    supporting_episodes: tuple[str, ...] = ()
    contradicting_episodes: tuple[str, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    observed_at: datetime = Field(default_factory=utc_now)
    last_verified_at: datetime | None = None
    valid_until: datetime | None = None
    invalidation_events: tuple[str, ...] = ()
    needs_review: bool = False


class LiveFact(Record):
    key: str
    value: Any
    episode_id: str
    observed_at: datetime
    valid_until: datetime | None = None
    invalidated_by: str | None = None


class WorkingGraph(Record):
    scope: Scope
    claims: tuple[Claim, ...] = ()
    facts: tuple[LiveFact, ...] = ()
    gaps: tuple[str, ...] = ()
    # True means a missing mandatory dependency prevented a usable selection.
    blocked: bool = False


class LiveState:
    """An explicitly task-local overlay; it is never serialized as durable claims."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._facts: dict[str, LiveFact] = {}
        self.pending: dict[str, str] = {}

    def observe(self, episode: Episode) -> None:
        if episode.task_id != self.task_id:
            raise ValueError("Cannot import another task's live state")
        if episode.outcome == "unknown":
            self.pending[episode.event_id] = episode.operation
        # Different API resources must not share one "api_fetch:id" binding.
        namespace = episode.operation
        if "url" in episode.arguments:
            namespace += ":" + str(episode.arguments["url"])
        if isinstance(episode.response, dict):
            for key in episode.coverage:
                if key in episode.response:
                    self._facts[f"{namespace}:{key}"] = LiveFact(
                        key=f"{namespace}:{key}",
                        value=episode.response[key],
                        episode_id=episode.episode_id,
                        observed_at=episode.timestamp,
                    )

    def invalidate(self, keys: set[str], event_id: str) -> None:
        for key in keys & self._facts.keys():
            self._facts[key] = self._facts[key].model_copy(update={"invalidated_by": event_id})

    def facts(self, now: datetime | None = None) -> tuple[LiveFact, ...]:
        now = now or utc_now()
        return tuple(
            f
            for f in self._facts.values()
            if f.invalidated_by is None and (f.valid_until is None or f.valid_until > now)
        )
