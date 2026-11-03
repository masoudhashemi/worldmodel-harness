"""The AutomationBench side of the core's Domain: HTTP calls through api_fetch.

An action is the method and URL path with id-like segments replaced by {}. Slots are
the id-like path segments, query parameters, and the leaves of params and body.
"""

import json
import re
from urllib.parse import parse_qsl, urlsplit

from ..core.lifting import traceable
from ..core.trace import Step, leaves, parse, structure

WRITES = ("POST", "PUT", "PATCH", "DELETE")


def method(arguments: dict) -> str:
    return str(arguments.get("method") or "GET").upper()


def _idlike(segment: str) -> bool:
    return bool(re.search(r"\d|_|^\{", segment)) or len(segment) > 40


def endpoint(arguments: dict) -> str:
    """Method and URL path, with id-like segments replaced by {}."""
    path = str(arguments.get("url", "")).split("?", 1)[0].split("//", 1)[-1]
    return f"{method(arguments)} {'/'.join('{}' if _idlike(s) else s for s in path.split('/'))}"


class HttpDomain:
    def action(self, step: Step) -> str:
        return endpoint(step.arguments) if step.tool == "api_fetch" else step.tool

    def slots(self, step: Step) -> list[tuple[str, str]]:
        if step.tool != "api_fetch":
            found = list(leaves(step.arguments, ""))
        else:
            url = urlsplit(str(step.arguments.get("url", "")))
            segments = url.path.split("/")
            found = [(f"path[{i}]", s) for i, s in enumerate(segments) if _idlike(s)]
            found += [(f"query.{k}", v) for k, v in parse_qsl(url.query)]
            for part in ("params", "body"):
                found += [(f"{part}.{p}", v) for p, v in leaves(parse(step.arguments.get(part)), "")]
        return [(slot, value) for slot, value in found if traceable(value)]

    def writes(self, step: Step) -> bool:
        return step.tool == "api_fetch" and method(step.arguments) in WRITES

    def reads(self, step: Step) -> bool:
        return step.tool == "api_fetch" and method(step.arguments) == "GET"

    def render(self, action: str, values: dict[str, str]) -> str:
        if " " not in action:
            return f"{action}({', '.join(f'{k}={v}' for k, v in values.items())})"
        verb, _, path = action.partition(" ")
        segments, query, body = path.split("/"), [], []
        for slot, value in values.items():
            if slot.startswith("path[") and (i := int(slot[5:-1])) < len(segments):
                segments[i] = value
            elif slot.startswith("query."):
                query.append(f"{slot[6:]}={value}")
            elif slot.startswith(("body.", "params.")) and not value.startswith("<"):
                body.append(f"{slot}={value}")
        url = "https://" + "/".join("<id>" if s == "{}" else s for s in segments)
        call = f"{verb} {url}" + ("?" + "&".join(query) if query else "")
        return call + (f" with {', '.join(body)}" if body else "")

    def shape(self, step: Step) -> str:
        parts = {p: structure(parse(step.arguments[p]) or step.arguments[p])
                 for p in ("params", "body") if step.arguments.get(p) not in (None, "", {})}
        return json.dumps(parts, sort_keys=True)
