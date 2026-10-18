"""Conservative declared service boundary plus observed cross-service dependencies.

Not a learned causal effect model. Queries/collections share the origin boundary.
Unresolved write identity falls back to invalidating every API observation.
"""

from urllib.parse import urlsplit

from .behavior import endpoint_contract


def origin(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            return parsed.scheme.lower(), parsed.hostname.lower(), port
    except (ValueError, TypeError):
        pass
    return None


def invalidation_keys(adapter, arguments):
    facts = [f for f in adapter.live.facts() if f.key.startswith("api_fetch:")]
    write_origin = origin(arguments.get("url", ""))
    contract = endpoint_contract(adapter.knowledge, adapter.scope, arguments)
    if write_origin is None or contract is None:
        return {f.key for f in facts}
    claims = adapter.knowledge.claims(adapter.scope)
    endpoints = {
        c.payload.get("id"): origin(c.payload.get("url", ""))
        for c in claims
        if c.relation == "endpoint_contract"
    }
    affected = {write_origin}
    # Even hypothesized value flows conservatively enlarge invalidation, not authority.
    edges = [
        (
            endpoints.get(c.payload.get("source_endpoint")),
            endpoints.get(c.payload.get("target_endpoint")),
        )
        for c in claims
        if c.relation == "observed_value_flow" and c.status != "retired"
    ]
    while True:
        dependents = {target for source, target in edges if source in affected}
        if None in dependents:
            return {f.key for f in facts}
        expanded = affected | dependents
        if expanded == affected:
            break
        affected = expanded
    keys = set()
    for fact in facts:
        try:
            observation = adapter.evidence.episode(fact.episode_id)
            read_origin = origin(observation.arguments.get("url", ""))
        except KeyError:
            read_origin = None
        if read_origin is None or read_origin in affected:
            keys.add(fact.key)
    return keys
