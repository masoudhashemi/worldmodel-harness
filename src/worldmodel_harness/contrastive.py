"""Mine observed contrasts for model analysis; correlation does not establish prerequisites."""

import hashlib
import re
from typing import Literal

from .knowledge import Claim, Episode, Layer, Record, Scope
from .learning import claim_identity
from .store import KnowledgeStore


class ContrastivePair(Record):
    pair_id: str
    target_tool: str
    target_route: str | None = None
    negative_episode_id: str
    positive_episode_id: str
    failure_kind: Literal["explicit_error", "silent_failure", "empty_result", "blocked"]
    divergence_summary: str


def _endpoint_route(episode: Episode) -> str | None:
    if episode.operation == "api_fetch" and isinstance(episode.arguments, dict):
        method = str(episode.arguments.get("method", "GET")).upper()
        url = str(episode.arguments.get("url", ""))
        clean_url = url.split("?")[0].split("#")[0]
        # Replace UUIDs or numeric IDs with placeholders to match templates
        clean_url = re.sub(r"/[0-9a-fA-F-]{16,36}\b", "/{id}", clean_url)
        clean_url = re.sub(r"/\d+\b", "/{id}", clean_url)
        return f"{method} {clean_url}"
    return episode.operation


def is_unproductive_episode(episode: Episode) -> tuple[bool, str]:
    """Detects explicit errors as well as silent/semantic failures with HTTP 200."""
    if episode.outcome in ("error", "blocked"):
        return True, "explicit_error"

    resp = episode.response
    if isinstance(resp, dict):
        # Application-level error flags despite 200 OK
        if resp.get("ok") is False or resp.get("success") is False:
            return True, "silent_failure"
        if resp.get("error") or resp.get("errors") or resp.get("error_code"):
            return True, "silent_failure"
        # Zero-update indicators
        if resp.get("updated") == 0 or resp.get("modified_count") == 0 or resp.get("affected_rows") == 0:
            return True, "silent_failure"
        # Empty collection returns when searching
        for list_key in ("records", "results", "items", "data", "rows", "values"):
            if (list_key in resp and isinstance(resp[list_key], list)
                and len(resp[list_key]) == 0 and episode.arguments):
                # An empty result nominates a contrast, not a missing prerequisite.
                return True, "empty_result"
    elif isinstance(resp, list) and len(resp) == 0 and episode.arguments:
        return True, "empty_result"

    return False, "none"


def is_productive_episode(episode: Episode) -> bool:
    """Select a contrasting observation; this does not verify task success."""
    if episode.outcome != "success":
        return False
    unproductive, _ = is_unproductive_episode(episode)
    return not unproductive


def mine_contrastive_pairs(store: KnowledgeStore, scope: Scope) -> list[ContrastivePair]:
    """Pair earlier candidate failures with later outcomes in the same observation channel."""
    episodes = store.episodes(scope)
    by_target: dict[tuple[str, str], list[Episode]] = {}
    for ep in episodes:
        if ep.purpose == "validation" or ep.scope.tenant != scope.tenant:
            continue
        route = _endpoint_route(ep)
        key = (ep.origin, route or ep.operation)
        by_target.setdefault(key, []).append(ep)

    pairs = []
    for group in by_target.values():
        negatives = []
        positives = []
        for ep in group:
            unproductive, kind = is_unproductive_episode(ep)
            if unproductive:
                negatives.append((ep, kind))
            elif is_productive_episode(ep):
                positives.append(ep)

        # Pair each negative with the most relevant positive episode
        for neg_ep, kind in negatives:
            for pos_ep in positives:
                if neg_ep.episode_id == pos_ep.episode_id or pos_ep.timestamp <= neg_ep.timestamp:
                    continue
                pair_id = hashlib.sha256(
                    f"{neg_ep.episode_id}:{pos_ep.episode_id}".encode()
                ).hexdigest()[:16]
                
                # Check divergence
                diff_keys = set(pos_ep.arguments.keys()) - set(neg_ep.arguments.keys())
                divergence = (
                    f"Positive trajectory included arguments {sorted(diff_keys)}"
                    if diff_keys
                    else "Diverged in execution sequence or prior environment state"
                )
                
                pairs.append(
                    ContrastivePair(
                        pair_id=pair_id,
                        target_tool=neg_ep.operation,
                        target_route=_endpoint_route(neg_ep),
                        negative_episode_id=neg_ep.episode_id,
                        positive_episode_id=pos_ep.episode_id,
                        failure_kind=kind,  # type: ignore[arg-type]
                        divergence_summary=divergence,
                    )
                )
                break  # One high-quality counterfactual counterpart is sufficient per negative

    return pairs


def induce_contrastive_precondition(
    pair: ContrastivePair,
    store: KnowledgeStore,
    scope: Scope,
) -> Claim:
    """Retain an unexplained contrast, never infer a necessary precondition.

    The legacy function name is kept for callers. Model consolidation consumes
    the original observations and proposes explanations separately.
    """
    negative = store.episode(pair.negative_episode_id)
    positive = store.episode(pair.positive_episode_id)
    if (
        negative.scope.tenant != positive.scope.tenant
        or negative.scope.tenant != scope.tenant
        or negative.origin != positive.origin
        or negative.scope.environment != scope.environment
        or positive.scope.environment != scope.environment
        or negative.scope.version != scope.version
        or positive.scope.version != scope.version
        or negative.scope.role != scope.role
        or positive.scope.role != scope.role
    ):
        raise ValueError("Contrast evidence must share environment, tenant and observation origin")
    target = pair.target_route or pair.target_tool
    identity = claim_identity(scope, "contrastive_observation", (pair.pair_id,))
    return Claim(
        claim_id=identity,
        layer=Layer.BEHAVIOR,
        relation="contrastive_observation",
        participants=(target,),
        statement="Different observed outcomes for " + target + "; cause unestablished.",
        scope=scope,
        payload={
            "negative_evidence": negative.episode_id,
            "positive_evidence": positive.episode_id,
            "contrastive_pair_id": pair.pair_id,
            "difference": pair.divergence_summary,
            "interpretation": "Candidate evidence for model analysis, not a prerequisite or causal rule",
        },
        origin="observed",
        status="hypothesis",
        needs_review=True,
        supporting_episodes=(negative.episode_id, positive.episode_id),
        observed_at=positive.timestamp,
    )


def learn_contrastive_preconditions(store: KnowledgeStore, scope: Scope) -> list[str]:
    """Retain deduplicated contrasts without promoting or broadening their scope."""
    if store.frozen:
        return []
    created_ids = []
    for pair in mine_contrastive_pairs(store, scope):
        claim = induce_contrastive_precondition(pair, store, scope)
        try:
            store.claim(claim.claim_id)
        except KeyError:
            store.put_claim(claim)
            created_ids.append(claim.claim_id)
    return created_ids
