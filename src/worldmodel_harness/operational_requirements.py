"""AutomationBench boundary requirements derived from observed API descriptions.

No endpoint names or task workflows are prescribed here. Contract conditions
and dependencies must already have provenance in the graph. This does not yet
infer semantic business prerequisites from free-form documentation.
"""

from .behavior import matching_contracts, preferred_contract, request_target, route_key
from .decisions import ActionRequirements, ProposedAction, UnresolvedRequirements
from .knowledge import Dependency, Scope
from .store import KnowledgeStore


VALID_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


def api_requirements(
    store: KnowledgeStore, scope: Scope, action: ProposedAction
) -> ActionRequirements:
    if action.tool != "api_fetch":
        return ActionRequirements(operation_id=action.tool)
    raw_method = action.arguments.get("method")
    method = (raw_method or "GET").upper() if isinstance(raw_method, str) else "GET"
    url = action.arguments.get("url", "")
    # Query/fragment text does not change endpoint identity. Preserve the actual
    # request unchanged; this normalization is only for matching its contract.
    endpoint_url = request_target(url)

    # 1. Distinguish malformed HTTP method field (e.g. method="api_fetch") from missing route contract
    if method not in VALID_HTTP_METHODS:
        all_contracts = [
            c for c in store.claims(scope)
            if c.relation == "endpoint_contract" and c.status == "supported"
        ]
        supported_methods = sorted({
            c.payload.get("method", "").upper()
            for c in all_contracts
            if c.payload.get("method") and (
                request_target(c.payload.get("url", "")) == endpoint_url
                or (url and request_target(url) == request_target(c.payload.get("url", "")))
            )
        })
        methods_str = ", ".join(supported_methods) if supported_methods else "GET, POST, PUT, PATCH, DELETE"
        raise UnresolvedRequirements(
            f"Malformed HTTP method field '{raw_method}'. No environment action occurred.",
            details={
                "reason": "malformed_http_method",
                "method": raw_method,
                "url": endpoint_url,
                "valid_methods": supported_methods or sorted(VALID_HTTP_METHODS),
                "discovery_action": None,
                "instruction": (
                    f"No environment action occurred. '{raw_method}' is the tool name, not an HTTP method. "
                    f"The endpoint '{endpoint_url}' accepts: {methods_str}. "
                    f"Correct the action using a valid HTTP method; do not run api_search."
                ),
            },
        )

    matches = matching_contracts(store, scope, method, url)
    claim = preferred_contract(matches)
    if claim is None and matches:
        # If there is an exact literal template match for the requested endpoint URL,
        # prefer it over generic wildcards (e.g. /meta/bases vs /{baseId}/{tableId}).
        exact_matches = [
            c for c in matches
            if request_target(c.payload.get("url", "")) == endpoint_url
        ]
        if exact_matches:
            claim = preferred_contract(exact_matches)
    # Do not drop stale/disputed matches: the gate must reject their use, rather
    # than turning obsolete knowledge into absence of a requirement. Distinct URL
    # templates stay ambiguous. Identical templates are one route.
    if claim is None:
        templates = sorted({route_key(item.payload)[1] for item in matches})
        # Check if the URL exists under a different HTTP method
        all_contracts = [
            c for c in store.claims(scope)
            if c.relation == "endpoint_contract" and c.status == "supported"
        ]
        other_methods = sorted({
            c.payload.get("method", "").upper()
            for c in all_contracts
            if c.payload.get("method") and (
                request_target(c.payload.get("url", "")) == endpoint_url
                or (url and request_target(url) == request_target(c.payload.get("url", "")))
            )
        })
        if other_methods and method not in other_methods:
            raise UnresolvedRequirements(
                f"Unsupported HTTP method '{method}' for endpoint '{endpoint_url}'.",
                details={
                    "reason": "unsupported_method",
                    "method": method,
                    "url": endpoint_url,
                    "supported_methods": other_methods,
                    "matching_claim_ids": [item.claim_id for item in matches],
                    "discovery_action": None,
                    "instruction": (
                        f"No environment action occurred. Endpoint '{endpoint_url}' supports: {', '.join(other_methods)}, but not '{method}'. "
                        f"Use a supported method or identify an alternative mutation route; do not repeat this call unchanged."
                    ),
                },
            )
        raise UnresolvedRequirements(
            "Endpoint identity unresolved; discover or disambiguate its contract",
            details={
                "reason": "missing_contract" if not matches else "ambiguous_contract",
                "method": method,
                "url": endpoint_url,
                "matching_claim_ids": [item.claim_id for item in matches],
                "matching_templates": templates,
                "matching_contracts": [
                    {"claim_id": item.claim_id, "scope": item.scope.model_dump(),
                     "payload": item.payload} for item in matches
                ],
                "discovery_action": "api_search",
                "discovery_query": f"{method} {endpoint_url}",
                "instruction": (
                    "Conflicting applicable contracts remain unresolved. Compare the supplied "
                    "schemas and scopes; obtain evidence of the applicable version/tenant/role. "
                    "Do not choose arbitrarily or repeat discovery unchanged."
                    if matches
                    else "Discover the matching method/route contract or use a known applicable read route; do not repeat the blocked call unchanged."
                ),
            },
        )
    return ActionRequirements(
        claims=(Dependency(claim_id=claim.claim_id, version=claim.version),),
        operation_id=claim.payload.get("id") or claim.claim_id,
    )
