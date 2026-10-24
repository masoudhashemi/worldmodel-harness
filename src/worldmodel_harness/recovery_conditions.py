"""Offline, evidence-linked request conditions learned from environment recovery.

Parsers describe representations; they do not encode which representations an
endpoint accepts. Acceptance associations come only from recorded responses.
"""

import base64
import binascii
import hashlib
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qsl

from .experience import Capability, PrefixExperience
from .knowledge import Episode, Record, Scope


def request_conditions(arguments: dict) -> dict[str, Any]:
    """Literal-free parser observations, retaining format and presence differences.

    No endpoint names, benchmark entities, or business rules are used. Field/header
    names are schema; dynamic dictionary keys are not a privacy-safe input format.
    """
    result = {}

    def text_features(text, path):
        result[path + "/empty"] = not text.strip()
        try:
            decoded = json.loads(text)
        except (ValueError, TypeError):
            decoded = None
            result[path + "/json_parseable"] = False
        else:
            result[path + "/json_parseable"] = True
            walk(decoded, path + "/json", decode=False)
        if "&" in text and "=" in text:
            try:
                pairs = parse_qsl(text, strict_parsing=True, keep_blank_values=True)
            except ValueError:
                pairs = []
            if pairs and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", k) for k, _ in pairs):
                result[path + "/form_keys"] = tuple(sorted({key for key, _ in pairs}))
        raw = text.strip()
        # Syntax check only: valid encoding is not proof of a valid message.
        try:
            if not raw or len(raw) > 100000 or not re.fullmatch(r"[A-Za-z0-9_+/=-]+", raw):
                raise ValueError("Not an encoding candidate")
            binary = base64.b64decode(raw + "=" * (-len(raw) % 4), altchars=b"-_", validate=True)
            decoded_text = binary.decode("utf-8")
        except (ValueError, binascii.Error, UnicodeError):
            result[path + "/base64_utf8"] = False
        else:
            result[path + "/base64_utf8"] = True
            headers(decoded_text, path + "/decoded")
        headers(text, path)

    def headers(text, path):
        # Generic header-block syntax; do not prescribe particular required headers.
        parts = re.split(r"\r?\n\r?\n", text, maxsplit=1)
        if len(parts) != 2:
            return
        names = []
        valid = True
        for line in parts[0].splitlines():
            if line.startswith((" ", "\t")) and names:
                continue
            match = re.match(r"^([A-Za-z0-9-]+):", line)
            if not match:
                valid = False
            else:
                names.append(match[1].casefold())
        result[path + "/header_block_valid"] = valid and bool(names)
        result[path + "/header_names"] = tuple(sorted(set(names)))

    def walk(value, path, decode=True):
        result[path + "/type"] = type(value).__name__
        if isinstance(value, dict):
            result[path + "/keys"] = tuple(sorted(value))
            for key, item in sorted(value.items()):
                walk(item, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            # Keep field presence across members without retaining concrete values.
            result[path + "/empty"] = not value
            result[path + "/member_types"] = tuple(sorted({type(v).__name__ for v in value}))
        elif isinstance(value, str) and decode:
            text_features(value, path)
        elif value is None:
            result[path + "/null"] = True

    for name in ("body", "params", "headers"):
        result[f"/{name}/present"] = name in arguments
        if name in arguments:
            walk(arguments[name], "/" + name)
    return result


class ConditionEvidence(Record):
    scope: Scope
    task_id: str
    episode_id: str
    event_id: str
    capability: Capability
    conditions: dict[str, Any]
    outcome: str


class RecoveryContrast(Record):
    scope: Scope
    capability: Capability
    failed_episode: str
    accepted_episode: str
    task_id: str
    failed_conditions: dict[str, Any]
    accepted_conditions: dict[str, Any]
    changed_paths: tuple[str, ...]
    intervening_episodes: tuple[str, ...]
    confounded: bool
    status: str = "hypothesis"


def compatible(left: Scope, right: Scope) -> bool:
    return left.model_dump(exclude={"task_id"}) == right.model_dump(exclude={"task_id"})


def signature(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class RecoveryMemory:
    def __init__(self):
        self.observations: list[ConditionEvidence] = []
        self.contrasts: list[RecoveryContrast] = []
        self.sources: dict[str, Episode] = {}

    def learn(self, episodes: list[Episode], window=6, *, seed_evidence=()):
        prefixes, history = {}, {}
        seen = {(o.scope.environment, o.scope.version, o.event_id) for o in self.observations}
        for episode in episodes:
            if episode.origin != "environment" or episode.environment_called is False:
                continue
            event = (episode.scope.environment, episode.scope.version, episode.event_id)
            if event in seen:
                continue
            seen.add(event)
            key = (episode.scope.model_dump_json(), episode.task_id)
            if key not in prefixes:
                prefix = PrefixExperience(episode.scope, episode.task_id)
                for prior in seed_evidence:
                    if compatible(prior.scope, episode.scope) and prior.task_id != episode.task_id:
                        cap = prior.capability
                        if cap.route and cap.contract_episode:
                            prefix.contracts[(cap.method, cap.route)] = cap.contract_episode
                prefixes[key] = prefix
            prefix = prefixes[key]
            query = prefix.query(episode.operation, episode.arguments)
            previous = history.setdefault(key, [])
            if query and episode.operation == "api_fetch":
                observation = ConditionEvidence(
                    scope=episode.scope,
                    task_id=episode.task_id,
                    episode_id=episode.episode_id,
                    event_id=episode.event_id,
                    capability=query.capability,
                    conditions=request_conditions(episode.arguments),
                    outcome=episode.outcome,
                )
                self.observations.append(observation)
                self.sources[episode.episode_id] = episode
                if episode.outcome == "success":
                    # Only the nearest same-resource attempt can be a recovery candidate.
                    for offset, (prior, prior_observation) in enumerate(
                        reversed(previous[-window:])
                    ):
                        if (
                            prior.operation,
                            prior.arguments.get("method"),
                            prior.arguments.get("url"),
                        ) != (
                            episode.operation,
                            episode.arguments.get("method"),
                            episode.arguments.get("url"),
                        ):
                            continue
                        if prior.outcome != "error" or prior_observation is None:
                            break
                        before, after = prior_observation.conditions, observation.conditions
                        changed = tuple(
                            sorted(
                                k
                                for k in before.keys() | after.keys()
                                if signature(before.get(k)) != signature(after.get(k))
                            )
                        )
                        intervening = [e for e, _ in previous[-offset:]] if offset else []
                        changed_other = any(
                            str(e.arguments.get("method", "")).upper()
                            not in ("", "GET", "HEAD", "OPTIONS")
                            for e in intervening
                        )
                        self.contrasts.append(
                            RecoveryContrast(
                                scope=episode.scope,
                                capability=query.capability,
                                failed_episode=prior.episode_id,
                                accepted_episode=episode.episode_id,
                                task_id=episode.task_id,
                                failed_conditions=before,
                                accepted_conditions=after,
                                changed_paths=changed,
                                intervening_episodes=tuple(e.episode_id for e in intervening),
                                # Representation changes can hide changed literals and world state.
                                confounded=changed_other or len(changed) != 1,
                            )
                        )
                        break
            else:
                observation = None
            previous.append((episode, observation))
            prefix.observe(episode, query)

    def retrieve(self, capability, scope, limit=4):
        eligible = [
            c
            for c in self.contrasts
            if c.capability.key == capability.key
            and compatible(c.scope, scope)
            and c.task_id != scope.task_id
        ]
        selected, seen = [], set()
        for contrast in eligible:
            pattern = signature((contrast.failed_conditions, contrast.accepted_conditions))
            if pattern not in seen and contrast.changed_paths:
                seen.add(pattern)
                selected.append(contrast)
        return selected[:limit]

    def assess(self, capability, scope, arguments):
        features = request_conditions(arguments)
        evidence = [
            o
            for o in self.observations
            if o.capability.key == capability.key
            and compatible(o.scope, scope)
            and o.task_id != scope.task_id
        ]
        matches = [o for o in evidence if signature(o.conditions) == signature(features)]
        return {
            "conditions": features,
            "observed_outcomes": dict(Counter(o.outcome for o in matches)),
            "source_episodes": [o.episode_id for o in matches],
            "independent_tasks": len({o.task_id for o in matches}),
            "uncertainty": "Associations, not causal guarantees; missing conditions may matter",
        }


def pair_fingerprint(first: Episode, second: Episode):
    """For excluding verbatim repeated action pairs across train and test."""
    return hashlib.sha256(
        signature([first.operation, first.arguments, second.operation, second.arguments]).encode()
    ).hexdigest()
