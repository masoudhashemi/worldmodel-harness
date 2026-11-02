"""Error -> fix memory: a recovery graph from stuck points to what moved earlier runs on.

Built by code from tool I/O, never by the model, and never from grades. A node is a
failed api_fetch: its endpoint, normalized error message and request shape. Edges say
what came next in earlier episodes, and "moved on" means a later successful call on
the same service:
- shape: the same endpoint succeeded with a differently shaped request;
- route: the endpoint never succeeded, but another endpoint of the service did;
- dead end: no call to the service succeeded after this error, in any episode.
Shapes keep keys and value formats (base64url, form-encoded text, a 2D array) and
routes keep URL templates, never literal values, so nothing learned on one task
carries data into another.

During a task the advisor appends to a failed api_fetch result, in this order:
- the last request that worked on this endpoint in this task, if any;
- otherwise a remembered shape fix, else a route, else a dead-end notice;
- after LOOP_REPEATS identical failures, a plain statement that repeating will fail.
Successful calls are never changed. Separately, SearchRepeats replaces the result of an
api_search that repeats an earlier one word for word with a pointer to that result.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LOOP_REPEATS = 3
MIN_SUPPORT = 2  # a remembered fix must have worked in at least this many episodes
MIN_DEAD_ENDS = 3  # episodes in which a service never recovered from an error
MAX_EXAMPLE_CHARS = 80

BASE64URL = re.compile(r"[A-Za-z0-9_-]{16,}={0,2}")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
FORM = re.compile(r"[^\s=&{}]+=[^&]*(?:&[^\s=&{}]+=[^&]*)+")
NOT_CONNECTED = re.compile(r"no \w+ account is connected", re.IGNORECASE)


def endpoint(arguments: dict) -> str:
    """Method and URL path, with ids, ranges and versions replaced by {}."""
    path = str(arguments.get("url", "")).split("?", 1)[0].split("//", 1)[-1]
    segments = [
        "{}" if re.search(r"\d|_|^\{", s) or len(s) > 40 else s for s in path.split("/")
    ]
    return f"{str(arguments.get('method') or 'GET').upper()} {'/'.join(segments)}"


def service(point: str) -> str:
    """The host of an endpoint key such as 'GET sheets.googleapis.com/{}/spreadsheets'."""
    return point.split(" ", 1)[-1].split("/", 1)[0]


def template(arguments: dict) -> str:
    """Method and URL with ids, names and ranges replaced by {id}; versions are kept."""
    url = str(arguments.get("url", "")).split("?", 1)[0]
    scheme, _, path = url.rpartition("//")
    segments = []
    for s in path.split("/"):
        version = re.fullmatch(r"v\d+(\.\d+)?|\d{4}-\d{2}-\d{2}|\d+\.\d+", s)
        idlike = "_" in s or "!" in s or len(s) > 40 or (re.search(r"\d", s) and not version)
        segments.append("{id}" if idlike and not s.startswith("{") else s)
    return f"{str(arguments.get('method') or 'GET').upper()} {scheme + '//' if scheme else ''}{'/'.join(segments)}"


def signature(raw: str) -> str:
    """The error message with quoted values and numbers removed."""
    text = raw
    try:
        data = json.loads(raw)
        error = data.get("error", data) if isinstance(data, dict) else data
        text = error.get("message") or json.dumps(error) if isinstance(error, dict) else str(error)
    except (TypeError, ValueError):
        pass
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "_", str(text).lower())
    return re.sub(r"\d+", "#", text).strip()[:160]


def _format(value):
    if isinstance(value, dict):
        return {k: _format(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_format(value[0]), "..."] if len(value) > 1 else [_format(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return "<number>"
    text = str(value)
    if not text:
        return "<empty>"
    if EMAIL.fullmatch(text):
        return "<email>"
    if BASE64URL.fullmatch(text) and not text.isalpha():
        return "<base64url>"
    if FORM.fullmatch(text):
        return "<form-encoded text>"
    return "<text>"


def _parsed(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def shape(arguments: dict) -> str:
    """Keys and value formats of a request's params and body, without literal values."""
    parts = {}
    for name in ("params", "body"):
        if arguments.get(name) not in (None, "", {}):
            value = _parsed(arguments[name])
            parts[name] = _format(value) if not isinstance(value, str) else f"<non-JSON {_format(value)[1:-1]}>"
    return json.dumps(parts, sort_keys=True)


def _example(arguments: dict) -> str:
    """A request from this task, with long values shortened."""

    def short(value):
        if isinstance(value, dict):
            return {k: short(v) for k, v in value.items()}
        if isinstance(value, list):
            return [short(v) for v in value[:2]] + (["..."] if len(value) > 2 else [])
        if isinstance(value, str) and len(value) > MAX_EXAMPLE_CHARS:
            return value[: MAX_EXAMPLE_CHARS // 2] + f"...({len(value)} chars)"
        return value

    request = {k: short(_parsed(v)) for k, v in arguments.items() if v not in (None, "", {})}
    return json.dumps(request)


def _phrase(fmt) -> str:
    return "base64url text (base64_encode produces it)" if fmt == "<base64url>" else str(fmt).strip("<>")


def describe(current: str, working: str) -> list[str]:
    """What differs between a failed request's shape and a working one, in plain words.

    In probes Gemma ignored abstract shape comparisons, and also this note without the
    words "set to a JSON string"; with them it re-sent every call with a body.
    """
    now, good = json.loads(current), json.loads(working)
    sentences = []
    for part, wanted in good.items():
        have = now.get(part)
        if have == wanted:
            continue
        if have is None:
            sentences.append(f"Your api_fetch call had no `{part}` argument, so nothing was sent.")
        elif isinstance(wanted, dict) and not isinstance(have, dict):
            sentences.append(f"`{part}` was {_phrase(have)}, not a JSON object.")
        elif isinstance(wanted, dict):
            missing = [k for k in wanted if k not in have]
            if missing:
                sentences.append(f"`{part}` is missing {', '.join(f'`{k}`' for k in missing)}.")
            for key in wanted:
                if key in have and isinstance(wanted[key], str) and have[key] != wanted[key]:
                    sentences.append(f"`{part}.{key}` was {_phrase(have[key])}; it must be {_phrase(wanted[key])}.")
        sentences.append(
            f"Call api_fetch again with {part} set to a JSON string, for example "
            f"{part}='{json.dumps(wanted)}', filled with this task's values."
        )
    return sentences


@dataclass
class FixMemory:
    """The recovery graph, with one edge table per kind of edge."""

    # (endpoint, error, failed shape) -> Counter of shapes that then worked there
    fixes: dict[tuple[str, str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    # (endpoint, error) -> Counter of URL templates on the same service that then worked
    routes: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    # (service, error) -> [episodes where the service never recovered, episodes where it did]
    outcomes: dict[tuple[str, str], list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    def learn(self, calls: list[tuple[dict, str, bool]]) -> None:
        """One episode's api_fetch calls: (arguments, raw response, success)."""
        # Each node counts once per episode, at its first occurrence.
        nodes, errors, services = set(), set(), set()
        for i, (arguments, raw, success) in enumerate(calls):
            point, error, failed = endpoint(arguments), signature(raw), shape(arguments)
            if success or (point, error, failed) in nodes:
                continue
            nodes.add((point, error, failed))
            later = [a for a, _, ok in calls[i + 1 :] if ok and service(endpoint(a)) == service(point)]
            if (service(point), error) not in services:
                services.add((service(point), error))
                self.outcomes[(service(point), error)][bool(later)] += 1
            here = next((shape(a) for a in later if endpoint(a) == point), None)
            if here is not None:
                if here != failed:
                    self.fixes[(point, error, failed)][here] += 1
                continue
            method = point.split(" ", 1)[0]
            other = next((a for a in later if endpoint(a).split(" ", 1)[0] == method), None)
            if other is not None and (point, error) not in errors:
                errors.add((point, error))
                self.routes[(point, error)][template(other)] += 1

    def lookup(self, arguments: dict, raw: str) -> str | None:
        """The most supported working shape for this error, different from this request's."""
        point, error, current = endpoint(arguments), signature(raw), shape(arguments)
        exact = self.fixes.get((point, error, current))
        candidates = exact or sum(
            (c for (p, e, _), c in self.fixes.items() if p == point and e == error), Counter()
        )
        for working, support in candidates.most_common():
            if support >= MIN_SUPPORT and working != current:
                return working
        return None

    def route(self, arguments: dict, raw: str) -> str | None:
        """The URL template most earlier episodes moved on to, if it is a clear majority."""
        edges = self.routes.get((endpoint(arguments), signature(raw)))
        if not edges:
            return None
        target, support = edges.most_common(1)[0]
        return target if support >= MIN_SUPPORT and 2 * support > sum(edges.values()) else None

    def dead_end(self, arguments: dict, raw: str) -> bool:
        """True if the service never recovered from this error in earlier episodes."""
        host = service(endpoint(arguments))
        never, recovered = self.outcomes.get((host, signature(raw)), (0, 0))
        return bool(host) and never >= MIN_DEAD_ENDS and recovered == 0

    def save(self, path: Path) -> None:
        rows = [
            {"kind": "shape", "endpoint": p, "error": e, "failed": f, "working": dict(c)}
            for (p, e, f), c in sorted(self.fixes.items())
        ]
        rows += [
            {"kind": "route", "endpoint": p, "error": e, "working": dict(c)}
            for (p, e), c in sorted(self.routes.items())
        ]
        rows += [
            {"kind": "outcome", "service": s, "error": e, "never": n, "recovered": r}
            for (s, e), (n, r) in sorted(self.outcomes.items())
        ]
        path.write_text(json.dumps(rows, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "FixMemory":
        memory = cls()
        for row in json.loads(path.read_text()):
            kind = row.get("kind", "shape")
            if kind == "shape":
                memory.fixes[(row["endpoint"], row["error"], row["failed"])].update(row["working"])
            elif kind == "route":
                memory.routes[(row["endpoint"], row["error"])].update(row["working"])
            else:
                memory.outcomes[(row["service"], row["error"])] = [row["never"], row["recovered"]]
        return memory


class FixAdvisor:
    """Annotates failed api_fetch results within one task."""

    def __init__(self, memory: FixMemory | None = None, unavailable: bool = True):
        self.memory = memory or FixMemory()
        self.name_unavailable = unavailable
        self.worked: dict[str, dict] = {}  # endpoint -> last successful arguments in this task
        self.failures: Counter = Counter()
        self.unavailable: set[str] = set()  # services that said no account is connected
        self.log: list[dict] = []

    def annotate(self, arguments: dict, raw: str, success: bool) -> str:
        point = endpoint(arguments)
        if success:
            self.worked[point] = arguments
            return raw
        error, current = signature(raw), shape(arguments)
        self.failures[(point, error, current)] += 1
        repeats = self.failures[(point, error, current)]
        notes, kinds = [], []
        if point in self.worked and shape(self.worked[point]) != current:
            kinds.append("same_task")
            notes += describe(current, shape(self.worked[point]))
            notes.append("Your earlier call that worked here was: " + _example(self.worked[point]))
        elif (working := self.memory.lookup(arguments, raw)) is not None:
            kinds.append("memory")
            notes += describe(current, working) or [
                f"In earlier tasks this error was fixed by sending a request shaped like {working}."
            ]
        elif (target := self.memory.route(arguments, raw)) is not None:
            kinds.append("route")
            method, url = target.split(" ", 1)
            notes.append(
                "Earlier runs that got this error moved on by calling a different endpoint: "
                f"call api_fetch with method='{method}' and url='{url}', replacing each {{id}} "
                "with this task's ids."
            )
        elif self.memory.dead_end(arguments, raw):
            kinds.append("dead_end")
            notes.append(
                f"In earlier runs no call to {service(point)} ever succeeded after this error. "
                f"Do not call {service(point)} again. Continue with the remaining steps of the task."
            )
        host = service(point)
        if self.name_unavailable and NOT_CONNECTED.search(raw):
            # The error already says so; speak only once the agent comes back to the service.
            if host in self.unavailable and "dead_end" not in kinds:
                kinds.append("unavailable")
                notes.append(
                    f"{host} already answered that no account is connected. Do not call it "
                    "again. Continue with the remaining steps of the task."
                )
            self.unavailable.add(host)
        if repeats >= LOOP_REPEATS:
            kinds.append("loop")
            notes.append(
                f"This exact request has now failed {repeats} times with the same error. "
                "Sending it again will fail again: change the request, or skip this write."
            )
        self.log.append({"endpoint": point, "kinds": kinds, "repeats": repeats})
        return raw + ("\n\n[Fix memory] " + " ".join(notes) if notes else "")


class SearchRepeats:
    """Within one task, answers a word-for-word repeat of an api_search with a pointer.

    The catalog is fixed, so the results would be the same ones the agent already has.
    Replacing them (E024) changes the context that, at temperature 0, sends the same
    searches again and again until the turn budget runs out.
    """

    def __init__(self):
        self.seen: dict[str, int] = {}  # normalized query -> number of the search that first sent it
        self.searches = 0
        self.log: list[dict] = []

    def annotate(self, arguments: dict, raw: str) -> str:
        self.searches += 1
        key = json.dumps({k: " ".join(str(v).lower().split()) for k, v in sorted(arguments.items())})
        if key not in self.seen:
            self.seen[key] = self.searches
            return raw
        first = self.seen[key]
        self.log.append({"search": self.searches, "repeats": first})
        return json.dumps(
            {
                "results": [],
                "note": f"This is the same search as your search #{first} in this task. The API "
                "catalog does not change, so its results are the ones already returned there.",
            }
        )
