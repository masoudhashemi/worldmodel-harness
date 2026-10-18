"""Compare proposed observations, never infer hidden effects from API success."""

import json
from contextvars import ContextVar

from .knowledge import OutputExpectation

CURRENT_EXPECTATIONS: ContextVar[tuple[str, tuple[OutputExpectation, ...]] | None] = ContextVar(
    "current_output_expectations", default=None
)


def inspect_path(episode, path):
    """Inspect shape before coverage/freshness; never infer missing values."""
    value = episode.response
    for index, key in enumerate(path):
        if isinstance(value, dict) and isinstance(key, str) and key in value:
            value = value[key]
        elif (
            isinstance(value, list)
            and (type(key) is int or (isinstance(key, str) and key.isascii() and key.isdigit()))
            and 0 <= int(key) < len(value)
        ):
            value = value[int(key)]
        else:
            return {
                "result": "invalid_evidence_path",
                "failed_at": index,
                "path": list(path),
                "container_type": type(value).__name__,
                "available_keys": list(value) if isinstance(value, dict) else None,
                "array_length": len(value) if isinstance(value, list) else None,
                "next": "Correct the path using the cited response shape; a path error is not evidence that a write failed.",
            }
    if path and path[0] not in episode.coverage:
        return {"result": "uncovered_evidence_path", "path": list(path)}
    return {"result": "observed", "value": value}


def compare_expectations(episode):
    assessments = []
    types = {
        "null": type(None),
        "boolean": bool,
        "integer": int,
        "number": float,
        "string": str,
        "array": list,
        "object": dict,
    }
    for prediction in episode.output_expectations:
        row = {
            "prediction": prediction.model_dump(mode="json"),
            "result": "unobserved",
            "episode_id": episode.episode_id,
        }
        observed = episode.outcome not in {"unknown", "blocked"}
        value = episode.response
        if prediction.path and prediction.path[0] not in episode.coverage:
            observed = False
        for key in prediction.path:
            if isinstance(value, dict) and isinstance(key, str) and key in value:
                value = value[key]
            elif (
                isinstance(value, list)
                and (type(key) is int or (isinstance(key, str) and key.isascii() and key.isdigit()))
                and 0 <= int(key) < len(value)
            ):
                value = value[int(key)]
            else:
                observed = False
                break
        if observed:
            if prediction.predicate == "present":
                matches = True
            elif prediction.predicate == "format":
                from .behavior import value_format

                field = prediction.path[-1] if prediction.path else None
                matches = value_format(value, field) == prediction.expected
            elif prediction.predicate == "type":
                expected_type = (
                    types.get(prediction.expected) if isinstance(prediction.expected, str) else None
                )
                if expected_type is None:
                    row["result"] = "invalid_prediction"
                    assessments.append(row)
                    continue
                matches = type(value) is expected_type or (
                    prediction.expected == "number" and type(value) is int
                )
            elif isinstance(prediction.expected, dict) and set(prediction.expected) == {"$kind"}:
                from .behavior import value_kind

                field = prediction.path[-1] if prediction.path else None
                matches = value_kind(value, field) == prediction.expected["$kind"]
            else:
                matches = type(value) is type(prediction.expected) and json.dumps(
                    value, sort_keys=True
                ) == json.dumps(prediction.expected, sort_keys=True)
            row.update(result="matched" if matches else "contradicted", observed=value)
        assessments.append(row)
    return assessments


def prediction_feedback(assessments, *, max_chars=None):
    """Complete check-status index; full response values stay in source evidence."""
    from .context_compression import MandatoryContextOverflow, encode_records

    prefix = (
        "Observed checks of your prior output predictions (not task-success proof). "
        "Unobserved means inconclusive, not false. Investigate consequential contradictions; "
        "do not automatically retry uncertain writes.\n"
    )
    selected = []
    for episode in assessments:
        for assessment in episode.get("proposed_output_assessments", []):
            record = {
                "episode_id": assessment["episode_id"],
                "path": assessment["prediction"]["path"],
                "predicate": assessment["prediction"]["predicate"],
                "result": assessment["result"],
            }
            # The original tool response remains available; avoid duplicating
            # potentially large/private values in another prompt record.
            selected.append(record)
    content = prefix + encode_records(selected) if selected else ""
    if max_chars is not None and len(content) > max_chars:
        raise MandatoryContextOverflow("Complete prediction-check index exceeds context allowance")
    return content
