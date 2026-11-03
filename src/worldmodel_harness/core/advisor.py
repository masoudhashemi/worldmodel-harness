"""The advisor speaks for the graph during one run.

After every tool call it turns what the graph knows about the run's state into
candidate messages, each of one kind:
- plan: the likeliest next successful calls of this workflow;
- sources: reads that earlier runs took values from that they then wrote, not made yet;
- unmade: calls most earlier runs made that this run has not;
- branch: the call just made is one after which earlier runs usually made up a value
  they wrote, and where runs that found it took it from;
- recovery: after a failed call, the call that earlier runs made successfully next.
Calls are filled in from this run's query and responses where the graph knows the
source; the rest are placeholders. A Policy decides which candidates are shown. Every
candidate offered and whether it was shown is logged with the step it followed, so the
outcome of showing it can be measured from the run's later calls.
"""

from typing import Protocol

from .graph import Graph
from .lifting import fill, lift_query
from .trace import Step

HEADER = "[Experience] Earlier runs of this same task, with other names and ids:"


class Policy(Protocol):
    def show(self, kind: str, template: str, state: str, text: str) -> bool: ...


class ShowNew:
    """Shows a candidate when its text differs from the last one shown of its kind. A
    placeholder with no tuning: which candidates help is to be learned (see Policy)."""

    def __init__(self):
        self.last: dict[str, str] = {}

    def show(self, kind: str, template: str, state: str, text: str) -> bool:
        if self.last.get(kind) == text:
            return False
        self.last[kind] = text
        return True


class Advisor:
    def __init__(self, graph: Graph, query: str, policy: Policy | None = None):
        self.graph = graph
        self.domain = graph.domain
        self.query = query
        self.template, self.bindings = lift_query(query)
        self.policy = policy or ShowNew()
        self.steps: list[Step] = []
        self.state = "start"
        self.visited: set[str] = set()
        self.log: list[dict] = []

    def after(self, step: Step) -> str:
        """Called with every tool call and its result; returns text to add to the result."""
        state = self.state
        first_visit = state not in self.visited
        self.visited.add(state)
        self.steps.append(step)
        if step.success:
            self.state = self.graph.action(step)
        candidates = self.candidates(step, state if first_visit else None)
        shown = []
        for kind, lines in candidates:
            text = "\n".join(lines)
            ok = self.policy.show(kind, self.template, self.state, text)
            self.log.append({"step": len(self.steps), "kind": kind, "shown": ok, "lines": lines})
            if ok:
                shown.append(text)
        return "\n\n" + HEADER + "\n" + "\n".join(shown) if shown else ""

    def candidates(self, step: Step, branched_from: str | None) -> list[tuple[str, list[str]]]:
        done = [self.graph.action(s) for s in self.steps if s.success]
        found = []
        if not step.success and (r := self.graph.recovery(step)):
            action, shape, hits, total = r
            call = self._call(action, done, [])
            found.append(("recovery", [f"- after this error, runs made {call} with arguments shaped like {shape} ({hits}/{total} runs)"]))
        if branched_from is not None:
            lines = []
            for write, slot, f, runs in self.graph.bad_branch(self.template, branched_from, self.graph.branch(step)):
                reads = self.graph.sources(self.template, set(done), only=(write, slot))
                if reads:
                    _, _, read, hits, total = reads[0]
                    lines.append(
                        f"- runs that made this call at this point wrote a {slot} of {write} found in no "
                        f"response in {runs - f}/{runs} runs; runs that found it took it from "
                        f"{self._chain(read, done)} ({hits}/{total} runs)"
                    )
            if lines:
                found.append(("branch", lines))
        plan = self.graph.plan(self.template, done)
        if plan:
            found.append(("plan", [
                f"- next {self._call(action, done, plan[: n - 1])} ({hits}/{total} runs)"
                for n, (action, hits, total) in enumerate(plan, 1)
            ]))
        sources = self.graph.sources(self.template, set(done))
        if sources:
            found.append(("sources", [
                f"- the {slot} of {write} came from {self._chain(read, done)} ({hits}/{total} runs)"
                for write, slot, read, hits, total in sources
            ]))
        unmade = self.graph.unmade(self.template, set(done))
        if unmade:
            found.append(("unmade", [f"- runs made {self._chain(a, done)} ({h}/{t} runs)" for a, h, t in unmade]))
        return found

    def _resolve(self, action: str, slot: str, nth: int):
        return self.graph.resolve(
            action, slot, lambda t: fill(t, self.bindings, self.steps, self.query, self.domain), self.template, nth
        )

    def _chain(self, action: str, done: list[str]) -> str:
        """The call, preceded by the call its missing value came from in earlier runs when
        this run has not made that one yet."""
        nth = done.count(action)
        for slot in self.graph.slots_of(self.template, action):
            term, value = self._resolve(action, slot, nth)
            first = term.partition(" <- ")[2] if term.startswith("Resp(") else ""
            if value is None and first and first not in done and first != action:
                return f"{self._call(first, done, [])} and then {self._call(action, done, [(first, 0, 0)])}"
        return self._call(action, done, [])

    def _call(self, action: str, done: list[str], before: list[tuple]) -> str:
        earlier = [a for a, _, _ in before]
        nth = [*done, *earlier].count(action)
        values = {}
        for slot in self.graph.slots_of(self.template, action):
            term, value = self._resolve(action, slot, nth)
            values[slot] = value if value is not None else placeholder(term, earlier)
        return self.domain.render(action, values)


def placeholder(term: str, earlier: list[str]) -> str:
    if term.startswith("Resp("):
        field = term[5:].split("|", 1)[0].split(")", 1)[0]
        held_by = term.partition(" <- ")[2]
        if held_by in earlier:
            return f"<{field} from the call before>"
        return f"<{field} from an earlier response>"
    return "<your choice>"
