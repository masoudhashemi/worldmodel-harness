"""Traces: what an agent did, in a form the graph can learn from, for any tool harness.

The core never looks inside tools. A Domain, supplied by the harness adapter, says what
an action is (so that two calls can be the same abstract action), which argument
values a call carries (its slots), whether a call changes state, and how to show a
call. Everything else is learned from episodes.
"""

import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Step:
    tool: str
    arguments: dict
    response: str
    success: bool


@dataclass
class Episode:
    task: str
    query: str
    steps: list[Step] = field(default_factory=list)


class Domain(Protocol):
    def action(self, step: Step) -> str:
        """The abstract action of a call: equal for calls that differ only in their values."""

    def slots(self, step: Step) -> list[tuple[str, str]]:
        """(slot, value) for each literal value the call carries."""

    def writes(self, step: Step) -> bool:
        """Whether the call changes the environment (its values are outputs of the run)."""

    def reads(self, step: Step) -> bool:
        """Whether the call's response is environment data (where written values can come from)."""

    def render(self, action: str, values: dict[str, str]) -> str:
        """The action as a call to show the agent; `values` fills its slots."""

    def shape(self, step: Step) -> str:
        """The call's argument structure without literal values."""


class ToolDomain:
    """A Domain for any tools with JSON arguments: the action is the tool and its argument
    names; slots are the leaves of the arguments."""

    def action(self, step: Step) -> str:
        return step.tool + "(" + ",".join(sorted(step.arguments)) + ")"

    def slots(self, step: Step) -> list[tuple[str, str]]:
        return [(path, value) for path, value in leaves(step.arguments, "")]

    def writes(self, step: Step) -> bool:
        return False

    def reads(self, step: Step) -> bool:
        return not self.writes(step)

    def render(self, action: str, values: dict[str, str]) -> str:
        tool = action.split("(", 1)[0]
        return f"{tool}({', '.join(f'{k}={v}' for k, v in values.items())})"

    def shape(self, step: Step) -> str:
        return json.dumps(structure(step.arguments), sort_keys=True)


def parse(text):
    """JSON in a string, else None."""
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def decoded(text: str) -> str | None:
    """The text inside a base64 or base64url string, if it is one."""
    if len(text) < 16 or " " in text:
        return None
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        out = raw.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    return out if out.isprintable() or "\n" in out else None


def leaves(node, path: str):
    """(path, value) for every scalar in a JSON value. Strings holding JSON or base64 are
    opened, so values packed inside another argument stay traceable."""
    if isinstance(node, str) and (inner := parse(node)) is not None and isinstance(inner, (dict, list)):
        node = inner
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for item in node:
            yield from leaves(item, path + "[]")
    elif isinstance(node, str) and (text := decoded(node)) is not None:
        from .lifting import literals

        for kind, value in literals(text):
            yield f"{path}.decoded.{kind}", value
    elif node is not None and not isinstance(node, bool):
        yield path, str(node)


def structure(node):
    """A JSON value with literals replaced by their kind, lists by their first item."""
    if isinstance(node, str) and (inner := parse(node)) is not None and isinstance(inner, (dict, list)):
        return structure(inner)
    if isinstance(node, dict):
        return {k: structure(v) for k, v in node.items()}
    if isinstance(node, list):
        return [structure(node[0])] if node else []
    if isinstance(node, bool) or node is None:
        return node
    if isinstance(node, (int, float)):
        return "<number>"
    if decoded(str(node)) is not None:
        return "<encoded text>"
    from .lifting import kind_of

    return f"<{kind_of(str(node))}>"
