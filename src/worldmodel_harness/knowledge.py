"""Typed records for observed episodes and the claims learned from them."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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


class Episode(Record):
    # Stores written by earlier harness versions carry fields this version ignores.
    model_config = ConfigDict(extra="ignore", frozen=True)

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
    # Earlier task-local episodes whose values a call may reuse.
    pre_evidence: tuple[str, ...] = ()
    purpose: Literal["task", "refresh", "probe", "validation", "discovery"] = "task"
    elapsed_seconds: float = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=utc_now)


def response_coverage(response: Any) -> tuple[str, ...]:
    """Covered root keys; array indexes use decimal strings in stored coverage."""
    if isinstance(response, dict):
        return tuple(response)
    if isinstance(response, list):
        return tuple(str(index) for index in range(len(response)))
    return ()


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
