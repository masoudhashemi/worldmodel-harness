"""Controller-owned response perturbations for explicitly nonstandard shift tests.

The upstream world and scorer stay unchanged. Only the declared API observation
changes; the agent and logger receive the same changed observation. Shift plans
and ground-truth change events are never supplied as agent knowledge or notices.
"""

from typing import Literal

from pydantic import Field

from .knowledge import Record


class ResponseShift(Record):
    shift_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    method: Literal["GET"] = "GET"
    field: str = Field(min_length=1)
    transformation: Literal["integer_to_string"] = "integer_to_string"
    start_task_index: int = Field(default=0, ge=0)
    after_matching_calls: int = Field(default=0, ge=0)


class ShiftController:
    def __init__(self, spec: ResponseShift):
        self.spec = spec
        self.matching_calls = 0
        self.events: list[dict] = []

    def apply(self, operation: str, arguments: dict, response, outcome: str):
        if (
            operation != "api_fetch"
            or arguments.get("url") != self.spec.url
            or arguments.get("method", "GET").upper() != self.spec.method
        ):
            return response, False
        self.matching_calls += 1
        if (
            self.matching_calls <= self.spec.after_matching_calls
            or outcome != "success"
            or not isinstance(response, dict)
            or type(response.get(self.spec.field)) is not int
        ):
            return response, False
        return {**response, self.spec.field: str(response[self.spec.field])}, True

    def record(self, episode, *, applied: bool, call_index: int, assessments: dict):
        if applied:
            detected = [
                p.claim_id
                for p in episode.predictions
                if p.path == self.spec.field and assessments.get(p.claim_id) == "contradicted"
            ]
            self.events.append(
                {
                    "shift_id": self.spec.shift_id,
                    "episode_id": episode.episode_id,
                    "environment_call_index": call_index,
                    "field": self.spec.field,
                    "detected_claim_ids": detected,
                }
            )

    def report(self) -> dict:
        first = self.events[0]["environment_call_index"] if self.events else None
        detected = next(
            (e["environment_call_index"] for e in self.events if e["detected_claim_ids"]), None
        )
        return {
            "spec": self.spec.model_dump(),
            "events": self.events,
            "first_exposed_call": first,
            "first_detected_call": detected,
            "detection_delay_calls": None if detected is None else detected - first,
            "notice_supplied": False,
            "repair_evaluated": False,
        }
