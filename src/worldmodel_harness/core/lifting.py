"""Lifting: literals replaced by where their values came from, so experience transfers.

A task query becomes a template plus typed bindings. A literal a call carries is
traced to its source:
- Query(type,k): the k-th literal of that type in the query;
- Resp(field|anchor) <- action: that field of the record, in an earlier successful
  response of `action`, picked out by an anchor (another field equal to a query literal,
  `key=Query(...)`, or a value the query names, `key@query`; `?` leaves the choice to the
  agent); Resp(field) when the response is not a list of records;
- Free: no source found: the agent computed or chose it.
A written value's origin is coarser: the action whose response held it anywhere
(also inside text), the query, or the agent. Terms never hold task values.
"""

import re

from .trace import Domain, Step, leaves, parse

LITERALS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("url", re.compile(r"https?://\S+")),
    ("channel", re.compile(r"#[\w-]+")),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("money", re.compile(r"\$\d[\d,]*(?:\.\d+)?")),
    ("id", re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z_])[\w-]{3,}\b")),
    ("number", re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")),
]
MAX_LITERAL_CHARS = 60


def literals(text: str) -> list[tuple[str, str]]:
    """(type, value) for the literals in a text, in the order of LITERALS."""
    return lift_query(text)[1]


def kind_of(value: str) -> str:
    for kind, pattern in LITERALS:
        if pattern.fullmatch(value):
            return kind
    return "text"


def lift_query(text: str) -> tuple[str, list[tuple[str, str]]]:
    """The query with literals replaced by {type}, and the (type, value) bindings."""
    bindings, template = [], text
    for kind, pattern in LITERALS:
        bindings += [(kind, match.rstrip(".,;:")) for match in pattern.findall(template)]
        template = pattern.sub("{" + kind + "}", template)
    return template, bindings


def traceable(value: str) -> bool:
    return 0 < len(value) <= MAX_LITERAL_CHARS and value.count(" ") <= 3


def origin(value: str, earlier: list[Step], query: str, domain: Domain) -> str:
    """Where a written value was first seen: "Query", the action of the latest earlier
    successful read whose response holds it, or "Agent"."""
    needle = value.lower()
    if needle in query.lower():
        return "Query"
    for step in reversed(earlier):
        if step.success and domain.reads(step) and needle in step.response.lower():
            return domain.action(step)
    return "Agent"


def response_fields(step: Step) -> dict[str, str]:
    """Value -> the last field name that holds it, for a successful response."""
    if not step.success:
        return {}
    return {value: path.rsplit(".", 1)[-1].replace("[]", "") for path, value in leaves(parse(step.response), "")}


def records(node):
    """Dicts with scalar fields anywhere in a JSON response: the records it lists."""
    if isinstance(node, dict):
        flat = {k: str(v) for k, v in node.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        if len(flat) >= 2:
            yield flat
        for v in node.values():
            yield from records(v)
    elif isinstance(node, list):
        for v in node:
            yield from records(v)


def mentioned(value: str, query: str) -> bool:
    """Whether a name-like value appears as whole words in the query."""
    if len(value) < 3 or not re.search(r"[A-Za-z]", value) or len(value) > MAX_LITERAL_CHARS:
        return False
    return re.search(r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)", query.lower()) is not None


def _anchor(record: dict, field: str, bindings, listed: list[dict], query: str) -> str:
    for key, value in record.items():
        if key != field and (term := query_term(value, bindings)):
            return f"{key}={term}"
    for key, value in record.items():
        if key != field and mentioned(value, query) and sum(mentioned(r.get(key, ""), query) for r in listed) == 1:
            return f"{key}@query"
    return "?"


def query_term(value: str, bindings) -> str | None:
    for kind in dict.fromkeys(k for k, _ in bindings):
        same = [v for k, v in bindings if k == kind]
        if value in same:
            return f"Query({kind},{same.index(value)})"
    return None


def source(value: str, bindings, earlier: list[Step], query: str, domain: Domain) -> str:
    """Where a literal came from: Query(type,k), Resp(field[|anchor]) <- action, or Free."""
    if term := query_term(value, bindings):
        return term
    for step in reversed(earlier):
        if not step.success:
            continue
        listed = list(records(parse(step.response)))
        for record in listed:
            for name, held in record.items():
                if held == value:
                    anchor = _anchor(record, name, bindings, listed, query) if len(listed) > 1 else None
                    return (f"Resp({name}|{anchor})" if anchor else f"Resp({name})") + f" <- {domain.action(step)}"
        fields = response_fields(step)
        if value in fields:
            return f"Resp({fields[value]}) <- {domain.action(step)}"
    return "Free"


def fill(term: str, bindings, earlier: list[Step], query: str, domain: Domain) -> str | None:
    """The value a term gives in this run, or None if it cannot be filled."""
    if term.startswith("Query("):
        kind, k = term[6:-1].rsplit(",", 1)
        same = [v for b, v in bindings if b == kind]
        return same[int(k)] if int(k) < len(same) else None
    if term.startswith("Const("):
        return term[6:-1]
    if not term.startswith("Resp("):
        return None
    term, _, held_by = term.partition(" <- ")
    name, _, anchor = term[5:-1].partition("|")
    if anchor == "?":
        return None
    by_mention = anchor.endswith("@query")
    key, _, anchor_term = anchor.removesuffix("@query").partition("=")
    wanted = fill(anchor_term, bindings, [], query, domain) if anchor and not by_mention else None
    for step in reversed(earlier):
        if not step.success or (held_by and domain.action(step) != held_by):
            continue
        listed = [r for r in records(parse(step.response)) if name in r]
        if by_mention:
            named = [r for r in listed if mentioned(r.get(key, ""), query)]
            if len(named) == 1:
                return named[0][name]
            continue
        for record in listed:
            if not anchor or record.get(key) == wanted:
                return record[name]
        if not anchor:
            for value, holder in response_fields(step).items():
                if holder == name:
                    return value
    return None
