"""AutomationBench boundary requirements derived from observed API descriptions.

No endpoint names or task workflows are prescribed here. Contract conditions
and dependencies must already have provenance in the graph. This does not yet
infer semantic business prerequisites from free-form documentation.
"""

from .behavior import matching_contracts, preferred_contract, request_target, route_key
from .decisions import ActionRequirements, ProposedAction, UnresolvedRequirements
from .knowledge import Dependency, Scope
from .store import KnowledgeStore


def api_requirements(
    store: KnowledgeStore, scope: Scope, action: ProposedAction
) -> ActionRequirements:
    if action.tool != "api_fetch":
        return ActionRequirements(operation_id=action.tool)
    method = action.arguments.get("method", "GET").upper()
    url = action.arguments.get("url", "")
    # Query/fragment text does not change endpoint identity. Preserve the actual
    # request unchanged; this normalization is only for matching its contract.
    endpoint_url = request_target(url)
    matches = matching_contracts(store, scope, method, url)
    claim = preferred_contract(matches)
    # Do not drop stale/disputed matches: the gate must reject their use, rather
    # than turning obsolete knowledge into absence of a requirement. Distinct URL
    # templates stay ambiguous. Identical templates are one route.
    if claim is None:
        templates = sorted({route_key(item.payload)[1] for item in matches})
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
