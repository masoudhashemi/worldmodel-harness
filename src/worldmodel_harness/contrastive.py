"""Cross-task contrastive semantic analysis and precondition induction.

Mines counterfactual episode pairs (failures/silent errors vs verified successes)
using multi-modal evidence (arguments, raw responses, state diffs, verifier remarks)
and induces actionable precondition invariants stored directly in the Knowledge Graph.
"""

import hashlib
import json
import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from .knowledge import Claim, Episode, Layer, Record, Scope, utc_now
from .learning import claim_identity
from .retrieval import terms
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
            if list_key in resp and isinstance(resp[list_key], list) and len(resp[list_key]) == 0:
                # If arguments contained specific search terms, an empty list suggests a missed query/tab prerequisite
                if episode.arguments:
                    return True, "empty_result"
    elif isinstance(resp, list) and len(resp) == 0 and episode.arguments:
        return True, "empty_result"

    return False, "none"


def is_productive_episode(episode: Episode) -> bool:
    """Verifies that an episode produced non-empty, error-free results."""
    if episode.outcome != "success":
        return False
    unproductive, _ = is_unproductive_episode(episode)
    return not unproductive


def mine_contrastive_pairs(store: KnowledgeStore, scope: Scope) -> list[ContrastivePair]:
    """Scans historical episodes to find counterfactual pairs for the same tool or route."""
    episodes = store.episodes(scope)
    by_target: dict[str, list[Episode]] = {}
    for ep in episodes:
        if ep.purpose in ("validation", "probe"):
            continue
        route = _endpoint_route(ep)
        key = route or ep.operation
        by_target.setdefault(key, []).append(ep)

    pairs = []
    for key, group in by_target.items():
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
                if neg_ep.episode_id == pos_ep.episode_id:
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
    *,
    client: Any | None = None,
) -> Claim:
    """Extracts a causal invariant claim from a contrastive episode pair."""
    neg_ep = store.episode(pair.negative_episode_id)
    pos_ep = store.episode(pair.positive_episode_id)

    target_name = pair.target_route or pair.target_tool

    # Heuristic / deterministic difference extraction
    pos_args = pos_ep.arguments if isinstance(pos_ep.arguments, dict) else {}
    neg_args = neg_ep.arguments if isinstance(neg_ep.arguments, dict) else {}
    missing_args = sorted(set(pos_args.keys()) - set(neg_args.keys()))

    # Check trajectory history for prerequisite operations
    all_episodes = store.episodes(scope)
    pos_history = [
        e for e in all_episodes
        if e.task_id == pos_ep.task_id
        and e.timestamp < pos_ep.timestamp
        and e.outcome == "success"
    ]
    neg_history = [
        e for e in all_episodes
        if e.task_id == neg_ep.task_id
        and e.timestamp < neg_ep.timestamp
        and e.outcome == "success"
    ]
    pos_prior_ops = {e.operation for e in pos_history}
    neg_prior_ops = {e.operation for e in neg_history}
    # An action cannot be a prerequisite for itself, and discovery operations are not prerequisites
    diff_ops = sorted((pos_prior_ops - neg_prior_ops) - {pair.target_tool, "api_search", "lookup_knowledge"})

    # Discovery operations (api_search, lookup_knowledge) do not have workflow prerequisite actions
    if pair.target_tool in ("api_search", "lookup_knowledge"):
        prereq_action = None
    else:
        prereq_action = diff_ops[-1] if diff_ops else None
    required_arg = missing_args[0] if missing_args else None

    statement_parts = [f"Calling {target_name}"]
    if pair.failure_kind == "silent_failure":
        statement_parts.append("previously resulted in a silent failure/no-op")
    elif pair.failure_kind == "empty_result":
        statement_parts.append("returned empty unhelpful data")
    else:
        statement_parts.append("failed with an error")

    if prereq_action:
        statement_parts.append(f"unless preceded by {prereq_action}")
    if required_arg:
        statement_parts.append(f"with parameter '{required_arg}'")
    statement_parts.append("(derived from contrastive trajectory analysis).")

    # Identify canonical target operation from endpoint contract if present
    target_contract = None
    if neg_ep.operation == "api_fetch" and isinstance(neg_ep.arguments, dict):
        from .behavior import endpoint_contract
        target_contract = endpoint_contract(store, scope, neg_ep.arguments)

    target_operation = (
        target_contract.payload.get("id")
        if target_contract and isinstance(target_contract.payload, dict) and "id" in target_contract.payload
        else (pair.target_route or pair.target_tool)
    )

    statement = " ".join(statement_parts)
    payload = {
        "target_operation": target_operation,
        "target_contract_id": target_contract.claim_id if target_contract else None,
        "target_tool": pair.target_tool,
        "target_route": pair.target_route,
        "failure_kind": pair.failure_kind,
        "required_preceding_action": prereq_action,
        "required_argument": required_arg,
        "negative_evidence": pair.negative_episode_id,
        "positive_evidence": pair.positive_episode_id,
        "contrastive_pair_id": pair.pair_id,
    }

    env_scope = Scope(environment=scope.environment, version=scope.version, role=scope.role)
    participants = (target_operation, prereq_action or required_arg or "precondition")
    signature = (target_name, pair.failure_kind, prereq_action or "", required_arg or "")
    identity = claim_identity(env_scope, "contrastive_precondition", signature)

    return Claim(
        claim_id=identity,
        version=1,
        layer=Layer.BEHAVIOR,
        relation="contrastive_precondition",
        participants=participants,
        statement=statement,
        scope=env_scope,
        payload=payload,
        origin="inferred",
        status="supported",
        needs_review=False,
        supporting_episodes=(pair.positive_episode_id,),
        contradicting_episodes=(pair.negative_episode_id,),
        observed_at=pos_ep.timestamp,
        last_verified_at=pos_ep.timestamp,
    )


def learn_contrastive_preconditions(
    store: KnowledgeStore,
    scope: Scope,
    *,
    client: Any | None = None,
) -> list[str]:
    """Mines counterfactual pairs, induces contrastive precondition claims, and stores them."""
    if store.frozen:
        return []
    pairs = mine_contrastive_pairs(store, scope)
    created_ids = []
    for pair in pairs:
        claim = induce_contrastive_precondition(pair, store, scope, client=client)
        store.put_claim(claim)
        created_ids.append(claim.claim_id)
    return created_ids
