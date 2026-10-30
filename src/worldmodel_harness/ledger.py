"""Entity ledger: the state of the environment instance as observed in one episode.

Built by code from tool I/O, never by the model. Reads add entities (rows, records,
contacts) with their attributes; successful writes add effects, attached to every
entity whose key appears in the write. At the first finish the agent sees one
compact table of entities, their deciding attributes, and what was done to each.
This module uses generic parsing only; a learned observation model can replace
`entities_in` and `write_text` per endpoint.
"""

import base64
import json
import re
from collections import Counter
from dataclasses import dataclass, field

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
KEY_FIELDS = ("id", "email", "emailaddress", "key", "sid", "number", "name", "title", "subject")
READ_VERBS = ("search", "find", "query", "lookup")
MAX_ATTRIBUTE_CHARS = 40
MAX_ATTRIBUTES = 6
MAX_ROWS = 40


@dataclass
class Entity:
    kind: str
    key: str
    attributes: dict[str, str]
    keys: set[str] = field(default_factory=set)  # every identifying literal
    effects: list[int] = field(default_factory=list)
    row: int | None = None  # sheet row number, for range writes


@dataclass
class Write:
    index: int
    method: str
    target: str
    text: str


@dataclass
class EntityLedger:
    entities: dict[tuple[str, str], Entity] = field(default_factory=dict)
    writes: list[Write] = field(default_factory=list)


def _scalar(value) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _norm(text: str) -> str:
    return re.sub(r"[\s,$]", "", text).lower()


SHEET = re.compile(r"spreadsheets/([^/?]+)(?:/values/([^?:]+(?::[^?/:]+)?))?")
CELL_ROW = re.compile(r"[A-Za-z]+(\d+)")


def _row_span(a1_range: str) -> tuple[int, int] | None:
    """First and last row of an A1 range such as 'Tab!E2:E8'."""
    rows = [int(r) for r in CELL_ROW.findall(a1_range.rsplit("!", 1)[-1])]
    return (rows[0], rows[-1]) if rows else None


def _rows(values: list, first_row: int) -> list[tuple[int, dict[str, str]]]:
    """A sheet's value grid as (row number, header-keyed cells)."""
    header = [str(h) for h in values[0]]
    rows = []
    for offset, row in enumerate(values[1:], start=1):
        cells = {h: str(v) for h, v in zip(header, row) if _scalar(v)}
        if cells:
            rows.append((first_row + offset, cells))
    return rows


def _objects(node, kind: str):
    """(kind, flat object, sheet row) for every list of objects in a JSON response."""
    if isinstance(node, dict):
        if isinstance(node.get("values"), list) and node["values"] and isinstance(node["values"][0], list):
            span = _row_span(str(node.get("range", ""))) or (1, 1)
            yield from ((kind, cells, row) for row, cells in _rows(node["values"], span[0]))
            return
        for name, value in node.items():
            yield from _objects(value, name)
    elif isinstance(node, list) and node and all(isinstance(item, dict) for item in node):
        for item in node:
            flat = {k: s for k, v in item.items() if (s := _scalar(v)) is not None}
            if flat:
                yield kind, flat, None
            for name, value in item.items():
                if isinstance(value, (list, dict)):
                    yield from _objects(value, name)


def entities_in(response: str, url: str = "") -> list[tuple[str, dict[str, str], int | None]]:
    try:
        data = json.loads(response)
    except (TypeError, ValueError):
        return []
    sheet = SHEET.search(url)
    return list(_objects(data, f"{sheet.group(1)} rows" if sheet else "items"))


def _identity(attributes: dict[str, str]) -> tuple[str, set[str]]:
    # The first field names the item (a sheet row's first cell, a record's id);
    # ids, names and email addresses also identify it inside a write.
    primary = next(iter(attributes.values()))
    keys = {primary} | {v for v in attributes.values() if EMAIL.fullmatch(v)}
    for name, value in attributes.items():
        if name.lower().replace("_", "") in KEY_FIELDS or name.lower().endswith(("id", "email")):
            keys.add(value)
    return primary, {k for k in keys if len(_norm(k)) >= 3}


def _decode_raw(raw: str) -> str:
    try:
        return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", "ignore")
    except ValueError:
        return ""


def write_text(arguments: dict) -> str:
    """Everything a write sends, as searchable text (Gmail raw decoded)."""
    body = arguments.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            pass
    parts = [str(arguments.get("url", "")), json.dumps(body) if not isinstance(body, str) else body]
    if isinstance(body, dict) and isinstance(body.get("raw"), str):
        parts.append(_decode_raw(body["raw"]))
    return "\n".join(parts)


def is_write(arguments: dict) -> bool:
    method = str(arguments.get("method") or "GET").upper()
    path = str(arguments.get("url", "")).split("?", 1)[0].lower()
    return method != "GET" and not any(verb in path.rsplit("/", 1)[-1] for verb in READ_VERBS)


def observe(ledger: "EntityLedger", arguments: dict, response: str, success: bool) -> None:
    """Fold one api_fetch call and its response into the ledger."""
    if not success:
        return
    if not is_write(arguments):
        for kind, attributes, row in entities_in(response, str(arguments.get("url", ""))):
            primary, keys = _identity(attributes)
            entity = ledger.entities.setdefault(
                (kind, primary), Entity(kind=kind, key=primary, attributes={}, keys=set())
            )
            entity.attributes.update(attributes)
            entity.keys |= keys
            entity.row = row if row is not None else entity.row
        return
    text = write_text(arguments)
    method = str(arguments.get("method")).upper()
    target = str(arguments.get("url", "")).split("?", 1)[0].split("//", 1)[-1]
    write = Write(index=len(ledger.writes), method=method, target=target, text=text)
    ledger.writes.append(write)
    normalized = _norm(text)
    sheet = SHEET.search(str(arguments.get("url", "")))
    span = _row_span(sheet.group(2)) if sheet and sheet.group(2) else None
    for entity in ledger.entities.values():
        in_range = span and entity.kind == f"{sheet.group(1)} rows" and entity.row and span[0] <= entity.row <= span[1]
        if in_range or any(_norm(k) in normalized for k in entity.keys):
            entity.effects.append(write.index)


def _deciding(entities: list[Entity]) -> list[str]:
    """Attributes worth showing: short, varying, and not identity.

    Rules select items by categories (a status, a type, a flag), so fields with few
    distinct values come first; the naming field and ids are already in the key.
    """
    counts: dict[str, Counter] = {}
    for entity in entities:
        first = next(iter(entity.attributes), None)
        for name, value in entity.attributes.items():
            identity = name == first or name.lower().endswith("id")
            if len(value) <= MAX_ATTRIBUTE_CHARS and not identity:
                counts.setdefault(name, Counter())[value] += 1
    ranked = sorted(
        (name for name, c in counts.items() if len(c) > 1 or len(entities) == 1),
        key=lambda name: (len(counts[name]) / sum(counts[name].values()), -sum(counts[name].values()), name),
    )
    return ranked[:MAX_ATTRIBUTES]


def _summary(write: Write) -> str:
    recipients = sorted(set(EMAIL.findall(write.text)))[:2]
    service = write.target.split("/", 1)[0]
    return f"#{write.index + 1} {write.method} {service}" + (f" to {', '.join(recipients)}" if recipients else "")


def render(ledger: "EntityLedger", max_rows: int = MAX_ROWS) -> str:
    """The finish review: entities that writes could be about, grouped by kind."""
    if not ledger.writes:
        return ""
    by_kind: dict[str, list[Entity]] = {}
    for entity in ledger.entities.values():
        by_kind.setdefault(entity.kind, []).append(entity)
    # Kinds that some write touched: the task acted on that collection.
    kinds = [k for k, es in by_kind.items() if any(e.effects for e in es)]
    lines = ["Before finishing, check each item below against the task's rules."]
    rows = 0
    for kind in sorted(kinds, key=lambda k: -len(by_kind[k])):
        entities = by_kind[kind]
        shown = _deciding(entities)
        lines.append(f"{kind} ({len(entities)} read, {sum(bool(e.effects) for e in entities)} acted on):")
        for entity in entities:
            if rows == max_rows:
                lines.append("  ...")
                break
            attrs = ", ".join(f"{n}={entity.attributes[n]}" for n in shown if n in entity.attributes)
            done = "; ".join(_summary(ledger.writes[i]) for i in entity.effects[:3]) or "nothing written"
            lines.append(f"  {entity.key[:40]} [{attrs}] -> {done}")
            rows += 1
    return "\n".join(lines) if rows else ""
