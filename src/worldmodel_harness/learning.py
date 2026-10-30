"""Documented endpoint contracts and change propagation to dependent claims."""

import hashlib
import json

from .knowledge import Claim, Episode, Layer, Scope, changed_payload_fields, fields_overlap
from .store import KnowledgeStore


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
        if episode.operation != "api_search" or episode.outcome != "success":
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
        """Mark claims that require a changed claim for review, transitively and cycle-safe."""
        affected: set[str] = set()
        pending = [(changed_id, fields)]
        visited = {changed_id}
        while pending:
            target, changed_fields = pending.pop()
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
