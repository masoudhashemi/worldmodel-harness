"""Prospective tests promote response predictions, never untested causal prose."""

import json
import re
from copy import deepcopy
from typing import Literal

from pydantic import Field

from .behavior import request_target, value_kind
from .knowledge import Claim, OutputExpectation, Record
from .learning import claim_identity
from .observable_predictions import compare_expectations


class ValueRole(Record):
    """Explicit interpretation of a rule leaf; undeclared leaves stay literal."""

    path: tuple[str | int, ...]
    role: Literal["example", "parameter", "constraint", "unknown"]


class ResponseRule(Record):
    operation: str = Field(min_length=1, max_length=120)
    argument_conditions: dict = Field(min_length=1, max_length=12)
    prediction: OutputExpectation
    value_roles: tuple[ValueRole, ...] = ()


_DECLARED_KINDS = frozenset(
    {"email", "phone", "date", "url", "identifier", "name", "integer", "number", "boolean", "string"}
)
_HTTP_OP = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD)\s+(\S+)$", re.IGNORECASE)


def normalized_arguments(arguments):
    """Normalize only declared JSON tool fields; raw episode evidence stays intact."""
    normalized = {}
    for raw_key, raw_val in arguments.items():
        if isinstance(raw_key, str) and "." in raw_key:
            head, tail = raw_key.split(".", 1)
            if head in ("params", "body", "headers"):
                bucket = normalized.get(head)
                if not isinstance(bucket, dict):
                    bucket = {}
                    normalized[head] = bucket
                bucket[tail] = raw_val
                continue
        normalized[raw_key] = raw_val
    for key in ("params", "body"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, (dict, list)):
                normalized[key] = decoded
    return normalized


def _explicit_kind_label(value) -> str | None:
    if not isinstance(value, str):
        return None
    folded = value.strip().casefold()
    if folded in _DECLARED_KINDS:
        return folded
    for kind in ("email", "phone", "date", "url", "identifier", "integer", "boolean"):
        if f"{kind} format" in folded or f"({kind})" in folded:
            return kind
    return None


def _template_url_id_segments(url: str) -> str:
    if "{" in url:
        return url
    parts = url.split("/")
    out = []
    for idx, seg in enumerate(parts):
        if (
            idx >= 3
            and seg
            and any(ch.isdigit() for ch in seg)
            and not re.fullmatch(r"v?\d+(?:\.\d+)+|v\d+", seg, re.IGNORECASE)
        ):
            out.append("{id}")
        else:
            out.append(seg)
    return "/".join(out)


def abstract_value(value, *, field=None, abstract=False):
    """Only explicitly requested example/parameter abstraction may replace a literal."""
    if isinstance(value, dict):
        if set(value) == {"$kind"}:
            return value
        return {key: abstract_value(item, field=key, abstract=abstract) for key, item in value.items()}
    if isinstance(value, list):
        return [abstract_value(item, abstract=abstract) for item in value]
    if abstract and field == "url" and isinstance(value, str):
        return _template_url_id_segments(value)
    if (
        abstract
        and isinstance(value, (str, int, float, bool))
        and field not in {"url", "method", "operation"}
        and not (isinstance(value, str) and "{" in value)
    ):
        explicit = _explicit_kind_label(value)
        return {"$kind": explicit or value_kind(value, field)}
    return value


def _expand_role_path(path: tuple[str | int, ...]) -> tuple[str | int, ...]:
    if (
        len(path) >= 2
        and path[0] == "argument_conditions"
        and isinstance(path[1], str)
        and "." in path[1]
    ):
        head, tail = path[1].split(".", 1)
        if head in ("params", "body", "headers"):
            return ("argument_conditions", head, tail, *path[2:])
    return path


def normalize_rule(rule):
    """Normalize JSON representation, never infer whether a literal is a constant.

    Paths start with argument_conditions or prediction/expected. Only scalar
    example/parameter leaves can become kind predicates. This loses equality
    intentionally: it is a candidate type regularity, not a value-copy rule.
    """
    data = rule.model_dump(mode="json")
    data["argument_conditions"] = deepcopy(normalized_arguments(rule.argument_conditions))
    http_match = _HTTP_OP.match(data["operation"].strip())
    if http_match:
        method, url_target = http_match.groups()
        data["operation"] = "api_fetch"
        data["argument_conditions"].setdefault("method", method.upper())
        data["argument_conditions"].setdefault("url", url_target)
    seen = set()
    normalized_roles = []
    for entry in rule.value_roles:
        path = _expand_role_path(entry.path)
        if path in seen:
            raise ValueError("Duplicate value role path")
        seen.add(path)
        if not path or not (
            path[0] == "argument_conditions" and len(path) > 1
            or path[:2] == ("prediction", "expected")
        ):
            raise ValueError("Value role must locate a condition or expected-value leaf")
        try:
            parent = data
            for key in path[:-1]:
                parent = parent[key]
            value = parent[path[-1]]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Value role path does not exist") from error
        if entry.role in ("example", "parameter"):
            if isinstance(value, (dict, list)) and not (
                isinstance(value, dict) and set(value) == {"$kind"}
            ):
                raise ValueError("Abstraction roles must identify scalar leaves")
            leaf_field = (
                str(rule.prediction.path[-1])
                if path == ("prediction", "expected") and rule.prediction.path
                else str(path[-1])
            )
            parent[path[-1]] = abstract_value(
                value, field=leaf_field, abstract=True
            )
        normalized_roles.append({"path": list(path), "role": entry.role})
    data["value_roles"] = sorted(normalized_roles, key=lambda item: json.dumps(item["path"]))
    return ResponseRule.model_validate(data)


def _has_template_or_kind(value) -> bool:
    if isinstance(value, dict):
        if set(value) == {"$kind"}:
            return True
        return any(_has_template_or_kind(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_template_or_kind(item) for item in value)
    return isinstance(value, str) and "{" in value and "}" in value


def is_parameterized_rule(rule: ResponseRule) -> bool:
    """True when a rule abstracts task-specific values via value roles or URL templates."""
    normalized = normalize_rule(rule)
    return (
        any(entry.role in ("example", "parameter") for entry in normalized.value_roles)
        or _has_template_or_kind(normalized.argument_conditions)
        or _has_template_or_kind(normalized.prediction.expected)
    )


def values_compatible(expected, actual, *, field=None) -> bool:
    if isinstance(expected, dict) and set(expected) == {"$kind"}:
        return value_kind(actual, field) == expected["$kind"]
    if isinstance(expected, str) and isinstance(actual, str) and (
        ("{" in expected and "}" in expected)
        or (field == "url" and expected.startswith("/") and actual.startswith("http"))
    ):
        parts = re.split(r"(\{[^{}]+\})", expected)
        pattern = "".join("[^/?]+" if part.startswith("{") else re.escape(part) for part in parts)
        target = request_target(actual)
        if (
            re.fullmatch(pattern, target)
            or re.fullmatch(pattern, actual)
            or (expected.startswith("/") and re.search(pattern + r"$", target))
        ):
            return True
    if isinstance(expected, dict) and isinstance(actual, dict):
        return (
            bool(expected) == bool(actual)
            and expected.keys() <= actual.keys()
            and all(
                values_compatible(expected[key], actual[key], field=str(key)) for key in expected
            )
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            values_compatible(item, other, field=field)
            for item, other in zip(expected, actual, strict=True)
        )
    return type(expected) is type(actual) and expected == actual


def applicable(rule, episode):
    rule = normalize_rule(rule)
    if episode.operation != rule.operation:
        return False
    args = normalized_arguments(episode.arguments)
    conditions = normalized_arguments(rule.argument_conditions)
    return all(
        key in args and values_compatible(value, args[key], field=str(key))
        for key, value in conditions.items()
    )


def validate_testability(rule, episodes):
    """A cited applicable call must cover the prediction; matching isn't proof.

    Counterevidence remains admissible: an executable but contradicted hypothesis
    is different from an expression the checker can never observe.
    """
    for episode in episodes:
        if applicable(rule, episode):
            result = compare_expectations(
                episode.model_copy(update={"output_expectations": (rule.prediction,)})
            )[0]["result"]
            if result in ("matched", "contradicted"):
                return
    raise ValueError(
        "Response rule is not testable on an applicable cited observation: check operation, argument field names/JSON conditions and covered response path; omit response_rule for an unobserved narrative hypothesis."
    )


def rule_signature(rule):
    data = normalize_rule(rule).model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def observe_rules(store, episode):
    """Only subsequent, covered observations count. Frozen stores never change."""
    if store.frozen or episode.purpose == "validation":
        return []
    if store.episode(episode.episode_id) != episode or episode.task_id != episode.scope.task_id:
        raise ValueError("Rule tests require the exact retained task observation")
    changed = []
    for claim in store.claims(episode.scope):
        spec = claim.payload.get("response_rule")
        if not spec or not claim.relation.startswith("possible_") or claim.status != "hypothesis":
            continue
        rule = ResponseRule.model_validate(spec)
        # Re-logging the generating event with a newer timestamp is not a test.
        generating_events = {
            store.episode(eid).event_id for eid in claim.supporting_episodes
        }
        if episode.event_id in generating_events:
            continue
        if episode.timestamp <= claim.observed_at or not applicable(rule, episode):
            continue
        rows = list(claim.payload.get("prospective_tests", []))
        if any(r["event_id"] == episode.event_id for r in rows):
            continue
        assessment = compare_expectations(
            episode.model_copy(
                update={
                    "output_expectations": (rule.prediction,),
                }
            )
        )[0]
        rows.append(
            {
                "event_id": episode.event_id,
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "result": assessment["result"],
            }
        )
        payload = {**claim.payload, "prospective_tests": rows}
        matched = [r for r in rows if r["result"] == "matched"]
        contradicted = [r for r in rows if r["result"] == "contradicted"]
        identity = claim_identity(claim.scope, "tested_response_rule", (rule_signature(rule),))
        try:
            previous = store.claim(identity)
        except KeyError:
            previous = None
        if previous is not None and previous.status == "retired":
            continue
        prior_support = previous.supporting_episodes if previous else ()
        prior_contradictions = previous.contradicting_episodes if previous else ()
        disputed = bool(contradicted or prior_contradictions)
        # Retain all contradictions; repair requires a new, narrower rule signature.
        if previous is not None or (len({r["task_id"] for r in matched}) >= 2 and not contradicted):
            prediction_claim = Claim(
                claim_id=identity,
                version=previous.version + 1 if previous else 1,
                layer="behavior",
                relation="tested_response_rule",
                scope=claim.scope,
                participants=claim.participants,
                statement="Observed conditional response prediction: " + rule_signature(rule),
                payload={
                    "response_rule": spec,
                    "source_hypothesis": claim.claim_id,
                    "interpretation": "Empirical response regularity, not necessity, causality or task success",
                },
                status="disputed" if disputed else "supported",
                needs_review=disputed,
                origin="observed",
                supporting_episodes=tuple(
                    dict.fromkeys((*prior_support, *(r["episode_id"] for r in matched)))
                ),
                contradicting_episodes=tuple(
                    dict.fromkeys((*prior_contradictions, *(r["episode_id"] for r in contradicted)))
                ),
                last_verified_at=episode.timestamp
                if assessment["result"] == "matched"
                else previous.last_verified_at
                if previous
                else None,
            )
            store.put_claim(prediction_claim)
            if disputed:
                from .learning import Learner

                Learner(store).reconsider(identity, set())
            payload["tested_claim_id"] = identity
            changed.append(identity)
        store.put_claim(claim.model_copy(update={"version": claim.version + 1, "payload": payload}))
    return changed
