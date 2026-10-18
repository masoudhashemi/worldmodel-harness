"""Conservative evidence gates and observable prediction assessment."""

import hashlib
import json
from typing import Literal

from .knowledge import (
    Claim,
    Episode,
    Layer,
    Prediction,
    Scope,
    changed_payload_fields,
    fields_overlap,
)
from .store import KnowledgeStore


def assess_prediction(
    prediction: Prediction, episode: Episode
) -> Literal["supported", "contradicted", "unobserved"]:
    if episode.outcome in ("unknown", "blocked"):
        return "unobserved"
    # Coverage enumerates observed fields, not a closed-world assertion about absent fields.
    if prediction.path not in episode.coverage or not isinstance(episode.response, dict):
        return "unobserved"
    if prediction.path not in episode.response:
        return "unobserved"
    if prediction.predicate == "present":
        return "supported"
    if prediction.predicate == "type":
        from .behavior import SEMANTIC_KINDS, value_kind, value_type

        observed = episode.response[prediction.path]
        if prediction.expected in SEMANTIC_KINDS:
            matches = value_kind(observed, prediction.path) == prediction.expected
        else:
            matches = value_type(observed) == prediction.expected
        return "supported" if matches else "contradicted"
    return (
        "supported" if episode.response[prediction.path] == prediction.expected else "contradicted"
    )


def claim_identity(scope: Scope, relation: str, participants: tuple[str, ...]) -> str:
    data = [scope.model_dump(), relation, participants]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


class Learner:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def document(
        self,
        episode: Episode,
        *,
        relation: str,
        participants: tuple[str, ...],
        statement: str,
        payload: dict,
        scope: Scope,
    ) -> Claim:
        """Only call for permitted schema/description observations, not model guesses."""
        if episode.operation not in ("api_search", "tool_schema") or episode.outcome != "success":
            raise ValueError("Documentation requires a successful schema observation")
        if not scope.applies_to(episode.scope):
            raise ValueError("Documentation scope mismatch")
        claim_id = claim_identity(scope, relation, participants)
        try:
            previous = self.store.claim(claim_id)
        except KeyError:
            previous = None
        if relation == "endpoint_contract":
            from .behavior import contract_signature, route_key

            for claim in self.store.claims(scope):
                if (
                    claim.relation == "endpoint_contract"
                    and claim.scope == scope
                    and route_key(claim.payload) == route_key(payload)
                    and contract_signature(claim.payload) == contract_signature(payload)
                    and claim.claim_id != claim_id
                ):
                    # Only compatible observations in exactly the same scope merge.
                    previous = claim
                    claim_id = claim.claim_id
                    if {k: v for k, v in payload.items() if k != "id"} == {
                        k: v for k, v in previous.payload.items() if k != "id"
                    }:
                        payload = previous.payload
                    break
        evidence = tuple(
            dict.fromkeys((*(previous.supporting_episodes if previous else ()), episode.episode_id))
        )
        claim = Claim(
            claim_id=claim_id,
            version=previous.version + 1 if previous else 1,
            layer=Layer.INTERFACE,
            relation=relation,
            participants=participants,
            statement=statement,
            scope=scope,
            payload=payload,
            status="supported",
            origin="documented",
            supporting_episodes=evidence,
            last_verified_at=episode.timestamp,
        )
        if (
            previous
            and previous.payload == payload
            and previous.status == "supported"
            and not previous.needs_review
            and evidence == previous.supporting_episodes
        ):
            # No gratuitous version churn for an identical contract.
            return previous
        self.store.put_claim(claim)
        if previous:
            changed = changed_payload_fields(previous.payload, payload)
            if changed:
                self.reconsider(claim_id, changed)
        return claim

    def reconsider(self, changed_id: str, fields: set[str]) -> set[str]:
        """Cycle-safe dependency propagation. Monitoring uncertain claims remains useful."""
        affected: set[str] = set()
        pending = [(changed_id, fields)]
        visited = {changed_id}
        while pending:
            target, changed_fields = pending.pop()
            self.store.suspend_dependents(target, changed_fields)
            for claim in self.store.claims():
                if claim.claim_id in visited:
                    continue
                matches = [
                    d
                    for d in claim.dependencies
                    if d.claim_id == target
                    and d.mode == "requires"
                    and fields_overlap(d.fields, changed_fields)
                ]
                if not matches:
                    continue
                visited.add(claim.claim_id)
                affected.add(claim.claim_id)
                self.store.put_claim(
                    claim.model_copy(
                        update={
                            "version": claim.version + 1,
                            "needs_review": True,
                            "invalidation_events": (*claim.invalidation_events, changed_id),
                        }
                    )
                )
                pending.append((claim.claim_id, set()))
        return affected

    def compare(self, episode: Episode) -> dict[str, str]:
        assessments = {}
        for prediction in episode.predictions:
            result = assess_prediction(prediction, episode)
            assessments[prediction.claim_id] = result
            if result == "unobserved":
                continue
            claim = self.store.claim(prediction.claim_id)
            if claim.version != prediction.claim_version:
                continue
            field = "supporting_episodes" if result == "supported" else "contradicting_episodes"
            if episode.episode_id in getattr(claim, field):
                continue
            ids = tuple(dict.fromkeys((*getattr(claim, field), episode.episode_id)))
            updates = {field: ids, "version": claim.version + 1}
            if result == "contradicted":
                updates["status"] = "disputed"
                updates["needs_review"] = True
            else:
                updates["last_verified_at"] = episode.timestamp
            self.store.put_claim(claim.model_copy(update=updates))
            if result == "contradicted":
                self.reconsider(claim.claim_id, {prediction.path})
        return assessments
