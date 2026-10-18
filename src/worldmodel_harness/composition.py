"""Conservative observational output-to-input connections, not causal bindings.

Only covered, task-local pre-action evidence may supply a value. Reusable claims
retain endpoints and structural paths, never the matching record values. No SQL
substring extraction, guessed transforms, or automatic executable adapters.
"""

import json
import re
from itertools import islice
from urllib.parse import unquote

from .behavior import endpoint_contract, request_shape, value_kind
from .knowledge import Claim, Dependency, Episode, Layer, Scope
from .learning import Learner, claim_identity


def leaves(value, path=(), *, depth=0):
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, item in list(value.items())[:64]:
            yield from leaves(item, (*path, str(key)), depth=depth + 1)
    elif isinstance(value, list):
        for item in value[:64]:
            yield from leaves(item, (*path, "*"), depth=depth + 1)
    elif isinstance(value, str) and len(value) >= 4:
        # Booleans, nulls, short constants and numeric coincidences are too weak.
        yield ".".join(path), value


def inputs(episode: Episode, contract: Claim):
    for section in ("body", "params"):
        value = episode.arguments.get(section, {})
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                continue
        yield from leaves(value, (section,))
    template = contract.payload.get("url", "")
    parts = re.split(r"(\{[^{}]+\})", template)
    pattern = "".join("([^/?]+)" if p.startswith("{") else re.escape(p) for p in parts)
    match = re.fullmatch(pattern, episode.arguments.get("url", ""))
    if match:
        names = [p[1:-1] for p in parts if p.startswith("{")]
        for name, value in zip(names, match.groups(), strict=True):
            value = unquote(value)
            if len(value) >= 4:
                yield "path." + name, value


class CompositionLearner:
    def __init__(self, learner: Learner):
        self.store = learner.store

    def observe(self, target: Episode) -> list[Claim]:
        if target.operation != "api_fetch" or target.outcome != "success":
            return []
        target_contract = endpoint_contract(self.store, target.scope, target.arguments)
        if target_contract is None:
            return []
        sources = []
        for episode_id in target.pre_evidence[-32:]:
            try:
                source = self.store.episode(episode_id)
            except KeyError:
                continue
            if (
                source.task_id != target.task_id
                or source.scope != target.scope
                or source.timestamp >= target.timestamp
                or source.operation != "api_fetch"
                or source.outcome != "success"
                or not isinstance(source.response, dict)
            ):
                continue
            contract = endpoint_contract(self.store, target.scope, source.arguments)
            if contract is None:
                continue
            covered = {k: v for k, v in source.response.items() if k in source.coverage}
            for path, value in islice(leaves(covered), 256):
                sources.append((source, contract, path, value))
        claims = []
        for input_path, input_value in islice(inputs(target, target_contract), 256):
            matches = [row for row in sources if row[3] == input_value]
            # Ambiguous source attribution is not resolved by arbitrary ordering.
            if len(matches) != 1:
                continue
            source, source_contract, output_path, output_value = matches[0]
            scope = Scope(
                environment=target.scope.environment,
                version=target.scope.version,
                role=target.scope.role,
            )
            payload = {
                "source_endpoint": source_contract.payload["id"],
                "target_endpoint": target_contract.payload["id"],
                "output_path": output_path,
                "input_path": input_path,
                "source_request_shape": request_shape(source.arguments),
                "target_request_shape": request_shape(target.arguments),
                "output_kind": value_kind(output_value, output_path),
                "input_kind": value_kind(input_value, input_path),
                "transformation": "observed_equal_value",
                "interpretation": "Observed reuse of a value kind between paths; not proof of identity, causation, or a safe general binding. The literal value is not stored.",
            }
            participants = (
                source_contract.payload["id"],
                target_contract.payload["id"],
                json.dumps(payload, sort_keys=True),
            )
            identity = claim_identity(scope, "observed_value_flow", participants)
            try:
                old = self.store.claim(identity)
            except KeyError:
                old = None
            evidence = tuple(
                dict.fromkeys(
                    (
                        *(old.supporting_episodes if old else ()),
                        source.episode_id,
                        target.episode_id,
                    )
                )
            )
            if old and evidence == old.supporting_episodes:
                continue
            # Two calls in one pair are not independent confirmations. Count
            # task/tenant pairs and require distinct observed target values.
            targets = [self.store.episode(eid) for eid in evidence]
            confirmations = [
                e
                for e in targets
                if e.pre_evidence
                and endpoint_contract(self.store, e.scope, e.arguments) == target_contract
            ]
            values = {
                value
                for e in confirmations
                for path, value in inputs(e, target_contract)
                if path == input_path
            }
            tenants = {e.scope.tenant for e in confirmations if e.scope.tenant is not None}
            tasks = {e.task_id for e in confirmations}
            status = (
                "supported"
                if len(tenants) >= 2 and len(tasks) >= 2 and len(values) >= 2
                else "hypothesis"
            )
            if old and old.status in ("disputed", "retired"):
                status = old.status
            dependencies = tuple(
                {
                    c.claim_id: Dependency(claim_id=c.claim_id, version=c.version)
                    for c in (source_contract, target_contract)
                }.values()
            )
            claim = Claim(
                claim_id=identity,
                version=old.version + 1 if old else 1,
                layer=Layer.BEHAVIOR,
                relation="observed_value_flow",
                participants=participants,
                statement=f"Observed {source_contract.payload['id']} {output_path} reused as "
                f"{target_contract.payload['id']} {input_path}; not an executable binding rule.",
                scope=scope,
                payload=payload,
                status=status,
                origin="inferred",
                supporting_episodes=evidence,
                dependencies=dependencies,
                contradicting_episodes=old.contradicting_episodes if old else (),
                needs_review=old.needs_review if old else False,
                last_verified_at=target.timestamp,
            )
            self.store.put_claim(claim)
            claims.append(claim)
        return claims
