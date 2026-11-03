"""Lifted traces: tool calls with each literal replaced by where its value came from.

Built by code from tool I/O, never from grades. A task query is split into a template
and typed bindings (emails, ids, numbers, channels). In each api_fetch call, every
short literal argument (a URL segment, a query parameter, a body field) is traced back
to its source:
- Query(type, k): the k-th literal of that type in the task query;
- Resp(field|anchor): that field of the record, in an earlier successful response, that
  the anchor picks out: one whose other field is a query literal, or whose name the query
  mentions (`name@query`), or, failing both, one with a unique name (`name=Const(...)`);
- Resp(field): the value of that field in an earlier successful response, latest first;
- Const(value): the same value in every task (set when a model is learned);
- Free: no source found, so the model computed or invented it.
Free text (messages, notes) is not traced. Two calls are the same abstract action when
their endpoint and request shape agree, so a lifted step can be filled in for a new
task from that task's own query and responses.
"""

import base64
import json
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlsplit

from worldmodel_harness.fixes import endpoint

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


@dataclass
class Step:
    tool: str
    arguments: dict
    response: str
    success: bool

    @property
    def action(self) -> str:
        return endpoint(self.arguments) if self.tool == "api_fetch" else self.tool


@dataclass
class Episode:
    task: str
    query: str
    steps: list[Step] = field(default_factory=list)


def lift_query(text: str) -> tuple[str, list[tuple[str, str]]]:
    """The query with literals replaced by {type}, and the (type, value) bindings.

    Patterns apply in order (an email is not also read as an id); within a type,
    bindings keep their order of appearance.
    """
    bindings, template = [], text
    for kind, pattern in LITERALS:
        bindings += [(kind, match.rstrip(".,;:")) for match in pattern.findall(template)]
        template = pattern.sub("{" + kind + "}", template)
    return template, bindings


def episode(path, task: str) -> Episode | None:
    """Every tool call of a saved trajectory with its result, in order."""
    try:
        groups = json.loads(path.read_text())
    except ValueError:
        return None
    if not isinstance(groups, list) or not groups or not isinstance(groups[0], list):
        return None
    messages = [m for group in groups for m in group if isinstance(m, dict)]
    query = next((str(m.get("content")) for m in messages if m.get("role") == "user"), "")
    found, calls = Episode(task, query), {}
    for message in messages:
        if message.get("role") == "assistant":
            for block in message.get("blocks") or []:
                if block.get("kind") == "tool_call":
                    calls[block.get("tool_call_id")] = block
        elif message.get("role") == "tool":
            call = calls.get(message.get("tool_call_id"))
            try:
                arguments = json.loads(call["arguments"]) if call else None
            except ValueError:
                arguments = None
            if isinstance(arguments, dict):
                content = str(message.get("content")).split("\n\n[Fix memory]", 1)[0].split("\n\n[Experience]", 1)[0]
                success = str(message.get("success")) == "True"
                found.steps.append(Step(str(message.get("name")), arguments, content, success))
    return found


def _json(text):
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _leaves(node, path: str):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for item in node:
            yield from _leaves(item, path + "[]")
    elif node is not None and not isinstance(node, bool):
        yield path, str(node)


def _traceable(value: str) -> bool:
    return 0 < len(value) <= MAX_LITERAL_CHARS and value.count(" ") <= 3


def slots(arguments: dict) -> list[tuple[str, str]]:
    """(slot, value) for each short literal of an api_fetch call."""
    url = urlsplit(str(arguments.get("url", "")))
    found = [
        (f"path[{i}]", segment)
        for i, segment in enumerate(url.path.split("/"))
        if "{}" in endpoint({"url": "x://h/" + segment}).split("/", 1)[-1]
    ]
    found += [(f"query.{k}", v) for k, v in parse_qsl(url.query)]
    for part in ("params", "body"):
        found += [(f"{part}.{p}", v) for p, v in _leaves(_json(arguments.get(part)), "")]
    found += _raw_recipients(_json(arguments.get("body")))
    return [(slot, value) for slot, value in found if _traceable(value)]


def _raw_recipients(body) -> list[tuple[str, str]]:
    """Recipients in the headers of a Gmail base64url 'raw' message, as body.raw.to slots."""
    raw = body.get("raw") if isinstance(body, dict) else None
    if not isinstance(raw, str):
        return []
    try:
        text = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", "ignore")
    except ValueError:
        return []
    headers = text.replace("\r\n", "\n").split("\n\n", 1)[0]
    found = []
    for name in ("To", "Cc", "Bcc"):
        for line in re.findall(rf"^{name}:(.*)$", headers, re.MULTILINE | re.IGNORECASE):
            found += [(f"body.raw.{name.lower()}", email) for email in LITERALS[0][1].findall(line)]
    return found


MIN_ORIGIN_CHARS = 4


def origin(value: str, earlier: list[Step], query: str = "") -> str:
    """Where a written value was first seen: "Query", the latest earlier successful read
    whose response contains it (anywhere, also inside text), or "Agent". A value shorter
    than MIN_ORIGIN_CHARS is "Short": it matches by chance (E033: the constant "doc")."""
    needle = value.lower()
    if len(needle.strip()) < MIN_ORIGIN_CHARS:
        return "Short"
    if needle in query.lower():
        return "Query"
    for step in reversed(earlier):
        read = str(step.arguments.get("method") or "GET").upper() == "GET"
        if step.success and step.tool == "api_fetch" and read and needle in step.response.lower():
            return step.action
    return "Agent"


def response_fields(step: Step) -> dict[str, str]:
    """Value -> the last field name that holds it, for a successful response."""
    if not step.success:
        return {}
    return {value: path.rsplit(".", 1)[-1].replace("[]", "") for path, value in _leaves(_json(step.response), "")}


ANCHOR_FIELDS = ("name", "title", "subject", "email", "label", "display_name", "full_name", "real_name", "channel")


def _records(node):
    """Dicts with scalar fields anywhere in a JSON response: the records it lists."""
    if isinstance(node, dict):
        flat = {k: str(v) for k, v in node.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        if len(flat) >= 2:
            yield flat
        for v in node.values():
            yield from _records(v)
    elif isinstance(node, list):
        for v in node:
            yield from _records(v)


def _mentioned(value: str, query: str) -> bool:
    """Whether a name-like value appears as whole words in the query."""
    if len(value) < 3 or not re.search(r"[A-Za-z]", value) or len(value) > MAX_LITERAL_CHARS:
        return False
    return re.search(r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)", query.lower()) is not None


def _anchor(record: dict, field: str, bindings, records: list[dict], query: str = "") -> str:
    """Another field that picks out this record: a query literal, or a name the query
    mentions. `?` if neither does: choosing the record is then the agent's selection."""
    for key, value in record.items():
        if key != field and (term := _query_term(value, bindings)):
            return f"{key}={term}"
    for key, value in record.items():
        if key != field and _mentioned(value, query) and sum(_mentioned(r.get(key, ""), query) for r in records) == 1:
            return f"{key}@query"
    return "?"


def _query_term(value: str, bindings) -> str | None:
    for kind in dict.fromkeys(k for k, _ in bindings):
        same = [v for k, v in bindings if k == kind]
        if value in same:
            return f"Query({kind},{same.index(value)})"
    return None


def source(value: str, bindings: list[tuple[str, str]], earlier: list[Step], query: str = "") -> str:
    """Where a literal came from: Query(type,k), Resp(field[|anchor]) <- action, or Free.

    A response value names the action whose response held it. When that response lists
    several records, the anchor names the record: `Resp(id|name@query) <- GET .../files`
    is the id of the listed file whose name the query mentions; `Resp(id|?)` leaves the
    choice of record to the agent. Terms hold no task values.
    """
    if term := _query_term(value, bindings):
        return term
    for step in reversed(earlier):
        if not step.success:
            continue
        records = list(_records(_json(step.response)))
        for record in records:
            for name, held in record.items():
                if held == value:
                    anchor = _anchor(record, name, bindings, records, query) if len(records) > 1 else None
                    return (f"Resp({name}|{anchor})" if anchor else f"Resp({name})") + f" <- {step.action}"
        fields = response_fields(step)
        if value in fields:
            return f"Resp({fields[value]}) <- {step.action}"
    return "Free"


def fill(term: str, bindings: list[tuple[str, str]], earlier: list[Step], query: str = "") -> str | None:
    """The value a term gives in another task, or None if it cannot be filled."""
    if term.startswith("Query("):
        kind, k = term[6:-1].rsplit(",", 1)
        same = [v for b, v in bindings if b == kind]
        return same[int(k)] if int(k) < len(same) else None
    if term.startswith("Const("):
        return term[6:-1]
    if not term.startswith("Resp("):
        return None
    term, _, origin = term.partition(" <- ")
    name, _, anchor = term[5:-1].partition("|")
    if anchor == "?":
        return None
    mention = anchor.endswith("@query")
    key, _, anchor_term = anchor.removesuffix("@query").partition("=")
    wanted = fill(anchor_term, bindings, []) if anchor and not mention else None
    for step in reversed(earlier):
        if not step.success or (origin and step.action != origin):
            continue
        records = [r for r in _records(_json(step.response)) if name in r]
        if mention:
            # As in `source`: the query must pick out one record, not the first of several.
            named = [r for r in records if _mentioned(r.get(key, ""), query)]
            if len(named) == 1:
                return named[0][name]
            continue
        for record in records:
            if not anchor or record.get(key) == wanted:
                return record[name]
        if not anchor:
            for value, holder in response_fields(step).items():
                if holder == name:
                    return value
    return None
