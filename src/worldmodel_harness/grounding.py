"""Check model-supplied data bindings and bounded derivations against observations.

This checks provenance, identity joins and arithmetic. It does not prove that a
model's selected formula or selection predicate fully captures the user's task.
"""

import ast
import math
import operator
import re
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .knowledge import Record
from .observable_predictions import inspect_path
from .operation_semantics import is_observation_read


class EvidenceValue(Record):
    episode_id: str
    path: tuple[str | int, ...] = Field(min_length=1)


class IdentityJoin(Record):
    source_name: str
    target_path: tuple[str | int, ...] = ()
    other_source_name: str | None = None

    @model_validator(mode="after")
    def one_destination(self):
        if bool(self.target_path) == (self.other_source_name is not None):
            raise ValueError("Join must name exactly one target path or other source input")
        return self


class GroundedDerivation(Record):
    inputs: dict[str, EvidenceValue] = Field(min_length=1, max_length=32)
    expression: str = Field(min_length=1, max_length=1000)
    selection: str | None = Field(default=None, max_length=1000)
    joins: tuple[IdentityJoin, ...] = Field(default=(), max_length=8)
    task_anchor_path: tuple[str | int, ...] = ()
    write_episode_id: str


def evaluate_expression(expression, values):
    """Evaluate a small expression language; never Python eval or attribute access."""
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression allowance exceeded")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if not names & values.keys():
        raise ValueError("Expression must depend on observed input")

    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Require a bounded finite number")
        return value

    def run(node):
        if isinstance(node, ast.Expression):
            return run(node.body)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool):
            return node.value
        if isinstance(node, ast.Subscript):
            container, key = run(node.value), run(node.slice)
            if isinstance(container, dict) and isinstance(key, str):
                return container[key]
            if isinstance(container, (list, tuple)) and type(key) is int and 0 <= key < len(container):
                return container[key]
            raise ValueError("Invalid source subscript")
        binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv}
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return number(binary[type(node.op)](number(run(node.left)), number(run(node.right))))
        if isinstance(node, ast.UnaryOp):
            value = run(node.operand)
            if isinstance(node.op, ast.USub):
                return -number(value)
            if isinstance(node.op, ast.Not) and type(value) is bool:
                return not value
        comparisons = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
        if isinstance(node, ast.Compare) and all(type(op) in comparisons for op in node.ops):
            items = [run(node.left), *(run(n) for n in node.comparators)]
            return all(comparisons[type(op)](items[i], items[i + 1]) for i, op in enumerate(node.ops))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            items = [run(item) for item in node.values]
            if not all(type(item) is bool for item in items):
                raise ValueError("Selection operands must be booleans")
            return all(items) if isinstance(node.op, ast.And) else any(items)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [run(item) for item in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [run(arg) for arg in node.args]
            name = node.func.id
            if name in {"sum", "min", "max"} and len(args) == 1 and isinstance(args[0], (list, tuple)):
                if not 1 <= len(args[0]) <= 1000:
                    raise ValueError("Collection allowance exceeded")
                nums = [number(value) for value in args[0]]
                return number({"sum": sum, "min": min, "max": max}[name](nums))
            if name == "round" and len(args) in (1, 2):
                precision = args[1] if len(args) == 2 else 0
                if type(precision) is not int or not -8 <= precision <= 8:
                    raise ValueError("Invalid precision")
                return round(number(args[0]), precision) if len(args) == 2 else round(number(args[0]))
            if (name == "format" and len(args) == 2 and isinstance(args[1], str)
                and re.fullmatch(r",?\.\d{1,2}[fg%]", args[1])):
                return format(number(args[0]), args[1])
            if name == "concat" and all(type(value) in (str, int, float, bool) for value in args):
                text = "".join(str(value) for value in args)
                if len(text) <= 16000:
                    return text
        raise ValueError("Unsupported expression")

    return run(tree)


def verify_derivation(adapter, target, path, expected, task_quote, anchor_text, derivation):
    aliases = getattr(adapter, "evidence_aliases", {})

    def episode(identity):
        return adapter.evidence.episode(aliases.get(identity, identity))

    task = " ".join(m["content"] for m in adapter.task.prompt if m.get("role") != "system")
    if task_quote not in task:
        return "invalid_task_quote"
    try:
        write = episode(derivation.write_episode_id)
        if (
            write.scope != adapter.scope or write.origin != "environment"
            or write.operation != "api_fetch" or write.outcome != "success"
            or is_observation_read(adapter, write)
            or write.timestamp >= target.timestamp
            or urlsplit(str(write.arguments.get("url", ""))).netloc
            != urlsplit(str(target.arguments.get("url", ""))).netloc
        ):
            return "invalid_derivation_write"
        values, records = {}, {}
        for name, reference in derivation.inputs.items():
            source = episode(reference.episode_id)
            if (source.scope != adapter.scope or not is_observation_read(adapter, source)
                or source.timestamp >= write.timestamp or source.episode_id == target.episode_id):
                return "incompatible_derivation_source"
            observed = inspect_path(source, reference.path)
            if observed["result"] != "observed":
                return "invalid_derivation_source_path"
            values[name] = observed["value"]
            # Bind scalar fields explicitly. A container can contain multiple records;
            # joining one of them must never certify every row in that container.
            if type(values[name]) not in (str, int, float, bool):
                return "non_scalar_derivation_input"
            records[name] = (source.episode_id, tuple(str(p) for p in reference.path[:-1]))
        anchored = False
        if derivation.task_anchor_path:
            anchor = inspect_path(target, derivation.task_anchor_path)
            anchored = (bool(anchor_text) and anchor_text in task and anchor["result"] == "observed"
                        and anchor["value"] == anchor_text
                        and derivation.task_anchor_path[:-1] == tuple(path)[:-1])
        if not derivation.joins and not anchored:
            return "missing_derivation_identity"
        target_record = (target.episode_id, tuple(str(p) for p in path[:-1]))
        edges = {}
        for join in derivation.joins:
            if join.other_source_name is not None:
                if join.other_source_name not in values:
                    return "derivation_identity_mismatch"
                observed = {"result": "observed", "value": values[join.other_source_name]}
                destination = records[join.other_source_name]
            else:
                observed = inspect_path(target, join.target_path)
                if tuple(str(p) for p in join.target_path[:-1]) != target_record[1]:
                    return "derivation_identity_mismatch"
                destination = target_record
            value = values.get(join.source_name)
            if (join.source_name not in values or type(value) not in (str, int)
                or value == "" or observed["result"] != "observed"
                or type(value) is not type(observed["value"]) or value != observed["value"]):
                return "derivation_identity_mismatch"
            origin = records[join.source_name]
            edges.setdefault(origin, set()).add(destination)
            edges.setdefault(destination, set()).add(origin)
        connected, pending = set(), [target_record]
        while pending:
            record = pending.pop()
            if record not in connected:
                connected.add(record)
                pending.extend(edges.get(record, ()) - connected if record in edges else ())
        used = set()
        for expression in (derivation.expression, derivation.selection):
            if expression is not None:
                used.update(n.id for n in ast.walk(ast.parse(expression, mode="eval"))
                            if isinstance(n, ast.Name) and n.id in values)
        if any(records[name] not in connected for name in used):
            return "disconnected_derivation_input"
        if derivation.selection is not None and evaluate_expression(derivation.selection, values) is not True:
            return "derivation_selection_unmatched"
        calculated = evaluate_expression(derivation.expression, values)
        if type(calculated) is not type(expected) or calculated != expected:
            return "derivation_value_mismatch"
    except (KeyError, ValueError, TypeError, SyntaxError, ArithmeticError):
        return "invalid_derivation"
    return "grounded"
