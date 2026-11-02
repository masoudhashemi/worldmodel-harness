"""The experience graph of world-model-v3.md: a transition graph learned across runs.

Nodes are lifted states (the last actions and whether they succeeded); edges are next
actions with the number of episodes that took them. On each edge, every argument of an
api_fetch has a source (lifting.source): the query, the response of an earlier action,
or the agent. The graph holds no task values; a run's own query and responses (its
state) resolve sources to values. Learned only from tool calls and results, never
from grades.
"""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from .lifting import Episode, Step, episode, fill, lift_query, origin, slots, source

# A literal is part of an action's shape (users/me, valueInputOption=RAW) only if it
# fills the slot in this many different workflows and in most of the slot's uses.
MIN_CONST_WORKFLOWS = 2
MIN_CONST_SHARE = 0.8
Z = 1.2816  # one-sided 90%: the lower bound is exceeded 90% of the time
# A plan step is shown when its lower bound reaches this: one earlier run of the
# workflow that took it (0.38) is enough, one of two (0.12) is not.
PLAN_THETA = 0.35
PLAN_STEPS = 3
VERSION = re.compile(r"v\d+(\.\d+)?")
WRITES = ("POST", "PUT", "PATCH", "DELETE")
SOURCE_HINTS = 2


def task_of(path: Path, run: Path) -> str:
    parts = path.relative_to(run).parts
    return next((p for p in parts if "." in p and not p.endswith(".json")), "")


def load(runs: list[Path], keep) -> list[Episode]:
    found = []
    for run in runs:
        for path in sorted(run.rglob("trajectory.json")):
            task = task_of(path, run)
            if task and keep(task) and (e := episode(path, task)) and e.steps:
                found.append(e)
    return found


def states(steps, i):
    """Backoff keys for predicting step i: two steps of history, one, none."""
    one = (steps[i - 1].action, steps[i - 1].success) if i else ("start", True)
    two = (steps[i - 2].action, steps[i - 2].success) if i > 1 else ("start", True)
    return [(two, one), (one,), ()]


def workflow(task: str) -> str:
    """The workflow a task instance belongs to: variants share their source's."""
    return task.split("~", 1)[0]


class Model:
    """The transition graph: next actions and, on each edge, where each argument comes from.

    It holds no task values. An argument the agent chose (no source in the query or an
    earlier response) is recorded as `Agent` and never filled in. The only literals kept
    are those that are part of an action's shape across workflows (`shape`).
    """

    def __init__(self, episodes: list[Episode] = ()):
        self.next = defaultdict(Counter)  # state -> next action -> episodes
        # (template or None, action, slot[, nth]) -> term -> support; the template keys
        # hold what is learned from recurrences of one workflow.
        self.terms = defaultdict(Counter)
        self.free = defaultdict(lambda: defaultdict(set))  # (action, slot) -> literal -> workflows
        self.flows = defaultdict(set)  # (action, slot) -> workflows that filled the slot
        # The workflow level of local plans: (template, last two successful fetches) ->
        # next successful fetch ("end" if none) -> episodes, and the episodes that got there.
        self.fetches = defaultdict(Counter)
        self.visits = Counter()
        self.arguments = defaultdict(Counter)  # (template, action) -> slot -> episodes
        self.versions = {}  # (action, path index) -> API version segment, such as v3
        # Where written values were first seen: (template, write action, slot) -> the action
        # whose response held the value (or Query, Agent) -> episodes, and the episodes
        # that wrote the slot.
        self.origins = defaultdict(Counter)
        self.written = Counter()
        self.traced = Counter()  # ... and the episodes in which some written value was seen
        self.runs = Counter()  # template -> episodes
        self.made = Counter()  # (template, action) -> episodes with a successful call of it
        for e in episodes:
            self.add(e)

    def add(self, e: Episode):
        seen = set()
        template, bindings = lift_query(e.query)
        history = [s.action for s in e.steps if s.tool == "api_fetch" and s.success]
        self.runs[template] += 1
        for action in set(history):
            self.made[(template, action)] += 1
        for n, action in enumerate([*history, "end"]):
            for key in fetch_keys(template, history[:n]):
                if (key, "visit") not in seen:
                    seen.add((key, "visit"))
                    self.visits[key] += 1
                if (key, action) not in seen:
                    seen.add((key, action))
                    self.fetches[key][action] += 1
        for i, step in enumerate(e.steps):
            for state in states(e.steps, i):
                if (state, step.action) not in seen:
                    seen.add((state, step.action))
                    self.next[state][step.action] += 1
            if step.tool == "api_fetch" and step.success:
                nth = occurrence(e.steps, i)
                path = urlsplit(str(step.arguments.get("url", ""))).path.split("/")
                for index, segment in enumerate(path):
                    if VERSION.fullmatch(segment):
                        self.versions[(step.action, index)] = segment
                for slot in dict.fromkeys(slot for slot, _ in slots(step.arguments)):
                    if (template, step.action, slot, "arg") not in seen:
                        seen.add((template, step.action, slot, "arg"))
                        self.arguments[(template, step.action)][slot] += 1
                if str(step.arguments.get("method", "GET")).upper() in WRITES:
                    for slot, value in slots(step.arguments):
                        key = (template, step.action, slot)
                        found = origin(value, e.steps[:i], e.query)
                        if (key, "written") not in seen:
                            seen.add((key, "written"))
                            self.written[key] += 1
                        if (key, found) not in seen:
                            seen.add((key, found))
                            self.origins[key][found] += 1
                        if found != "Agent" and (key, "traced") not in seen:
                            seen.add((key, "traced"))
                            self.traced[key] += 1
                for slot, value in slots(step.arguments):
                    term = source(value, bindings, e.steps[:i], e.query)
                    if term == "Free":
                        self.free[(step.action, slot)][value].add(workflow(e.task))
                        term = "Agent"
                    self.flows[(step.action, slot)].add(workflow(e.task))
                    for key in ((None, step.action, slot), (template, step.action, slot), (template, step.action, slot, nth)):
                        self.terms[key][term] += 1

    def shape(self, action, slot) -> str | None:
        """The literal that is part of this action's shape, if there is one."""
        everywhere = len(self.flows.get((action, slot), ()))
        for value, flows in self.free.get((action, slot), {}).items():
            if len(flows) >= MIN_CONST_WORKFLOWS and len(flows) >= MIN_CONST_SHARE * everywhere:
                return f"Const({value})"
        return None

    def predict(self, steps, i):
        """Ranked next actions and the counts behind the top one, from the deepest known state."""
        for state in states(steps, i):
            if self.next[state]:
                ranked = self.next[state].most_common(3)
                return [a for a, _ in ranked], ranked[0][1], sum(self.next[state].values())
        return [], 0, 0

    def term(self, action, slot, template=None, nth=None):
        """The best-supported source: for this call of the workflow, the workflow, then any task.

        `Agent` (the agent decides) unless the literal is part of the action's shape.
        """
        for key in ((template, action, slot, nth), (template, action, slot), (None, action, slot)):
            if terms := self.terms.get(key):
                best = terms.most_common(1)[0][0]
                return (self.shape(action, slot) or best) if best == "Agent" else best
        return self.shape(action, slot) or "Agent"


    def resolve(self, action, slot, resolver, template=None, nth=None):
        """(term, value): the best-supported source that the current state can resolve.

        Levels go from this call of the workflow to any task. At the first level that
        knows the slot, `Agent` on top leaves the argument to the agent; otherwise the
        sources are tried by support, and a level none of whose sources resolve passes
        to the next. If nothing resolves, the best source is returned with no value, so
        a plan can say where the value will come from.
        """
        best = None
        for key in ((template, action, slot, nth), (template, action, slot), (None, action, slot)):
            ranked = [t for t, _ in self.terms.get(key, Counter()).most_common()]
            if ranked and ranked[0] == "Agent":
                shape = self.shape(action, slot)
                return (shape, resolver(shape)) if shape else (best or "Agent", None)
            for term in ranked:
                if term != "Agent" and (value := resolver(term)) is not None:
                    return term, value
            best = best or next((t for t in ranked if t != "Agent"), None)
        shape = self.shape(action, slot)
        return (shape, resolver(shape)) if shape else (best or "Agent", None)

    def plan(self, template: str, history: list[str], limit: int = PLAN_STEPS) -> list[tuple[str, int, int]]:
        """The likeliest next successful fetches of this workflow after `history`.

        Each step is (action, episodes that took it, episodes that got there); repeats
        of an endpoint are one step. The plan stops where the lower confidence bound
        drops below PLAN_THETA or runs end.
        """
        found, history = [], list(history)
        for _ in range(4 * limit):
            key = next((k for k in fetch_keys(template, history) if self.visits[k]), None)
            if key is None or len(found) == limit:
                break
            action, hits = self.fetches[key].most_common(1)[0]
            if action == "end" or lcb(hits, self.visits[key]) < PLAN_THETA:
                break
            # Repeated calls of one endpoint (paging, retries, a search refined) are one step.
            if not found or found[-1][0] != action:
                found.append((action, hits, self.visits[key]))
            history.append(action)
        return found

    def sources(self, template: str, called: set[str]) -> list[tuple[str, str, str, int, int]]:
        """Reads that earlier runs of this workflow took written values from, not yet called.

        Each is (write action, slot, read action, episodes, episodes in which a written
        value was seen somewhere), best first, gated like plan steps. A value the agent made
        up (found in no response or query) is no evidence about where it could have been
        found, so episodes with only such values are left out; values from the query are
        skipped.
        """
        found = []
        for (t, action, slot), counts in self.origins.items():
            if t != template:
                continue
            total = self.traced[(t, action, slot)]
            for read, hits in counts.items():
                if read not in ("Query", "Agent") and read not in called and lcb(hits, total) >= PLAN_THETA:
                    found.append((action, slot, read, hits, total))
        found.sort(key=lambda s: -lcb(s[3], s[4]))
        unique, reads = [], set()
        for s in found:
            if s[2] not in reads:
                reads.add(s[2])
                unique.append(s)
        return unique[:SOURCE_HINTS]

    def unmade(self, template: str, called: set[str]) -> list[tuple[str, int, int]]:
        """Calls most earlier runs of this workflow made successfully, not yet made in this
        run: (action, episodes that made it, episodes), best first, gated like plan steps."""
        total = self.runs[template]
        found = [
            (action, hits, total)
            for (t, action), hits in self.made.items()
            if t == template and action not in called and lcb(hits, total) >= PLAN_THETA
        ]
        return sorted(found, key=lambda s: -s[1])[:SOURCE_HINTS + 1]

    def slots_of(self, template: str, action: str) -> list[str]:
        """The arguments this workflow's calls of an action usually fill, versions aside."""
        counts = self.arguments.get((template, action), Counter())
        most = max(counts.values(), default=0)
        return [
            slot
            for slot, n in counts.items()
            if 2 * n >= most and not (slot.startswith("path[") and (action, int(slot[5:-1])) in self.versions)
        ]


def fetch_keys(template: str, history: list[str]) -> list[tuple]:
    """Plan keys after a run's successful fetches: the last two, then the last one."""
    return [(template, tuple(history[-2:])), (template, tuple(history[-1:]))]


def occurrence(steps, i) -> int:
    """How many earlier successful calls of this episode went to the same endpoint."""
    return sum(s.success and s.action == steps[i].action for s in steps[:i])


def lcb(hits: int, total: int) -> float:
    """Wilson lower bound of hits / total."""
    if not total:
        return 0.0
    p = hits / total
    centre = p + Z * Z / (2 * total)
    spread = Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total))
    return (centre - spread) / (1 + Z * Z / total)


PLAN_HEADER = (
    "[Experience] Earlier runs of this same task, with other names and ids, made these "
    "calls next (runs that made the call / runs that reached this point):"
)
SOURCE_HEADER = (
    "[Experience] In earlier runs of this same task, with other names and ids, values "
    "that were written came from reads this run has not made yet:"
)
UNMADE_HEADER = (
    "[Experience] Earlier runs of this same task, with other names and ids, made these "
    "calls successfully; this run has not made them yet:"
)
PLAN_FOOTER = (
    "<...> marks a value to take from this task. Check each call against the task "
    "before you make it."
)


class ExperienceAdvisor:
    """Speaks for the graph during one run: a local plan after each api_search.

    The run's successful fetches and their responses are its state. The first plan step
    is filled in from that state where the graph knows the source; later steps name
    where their values will come from. Nothing is shown below PLAN_THETA, and the same
    plan is not repeated.
    """

    def __init__(self, graph: Model, query: str):
        self.graph = graph
        self.query = query
        self.template, self.bindings = lift_query(query)
        self.steps: list[Step] = []
        self.log: list[dict] = []
        self.last = None

    def observe(self, arguments: dict, raw: str, success: bool):
        self.steps.append(Step("api_fetch", arguments, raw, success))

    def annotate_search(self, raw: str) -> str:
        history = [s.action for s in self.steps if s.success]
        steps = self.graph.plan(self.template, history)
        lines = [
            f"{n}. {self._call(action, history, steps[:n - 1])} ({hits}/{total})"
            for n, (action, hits, total) in enumerate(steps, 1)
        ]
        if not lines or lines == self.last:
            return raw
        self.last = lines
        self.log.append({"after_fetches": len(history), "plan": lines})
        return "\n\n".join([raw, PLAN_HEADER + "\n" + "\n".join(lines), PLAN_FOOTER])

    def source_hint(self) -> str:
        """Where earlier runs of this task found values they wrote, for reads this run
        has not made yet; empty if the graph knows none."""
        history = [s.action for s in self.steps if s.success]
        lines = [
            f"- the {slot.rsplit('.', 1)[-1]} of {write} came from the response of "
            f"{self._chain(read, history)} ({hits}/{total} runs)"
            for write, slot, read, hits, total in self.graph.sources(self.template, set(history))
        ]
        if lines:
            self.log.append({"after_fetches": len(history), "sources": lines})
            return SOURCE_HEADER + "\n" + "\n".join(lines)
        # No value source left to read: the calls earlier runs made that this one has not.
        lines = [
            f"- {self._chain(action, history)} ({hits}/{total} runs)"
            for action, hits, total in self.graph.unmade(self.template, set(history))
        ]
        if lines:
            self.log.append({"after_fetches": len(history), "unmade": lines})
            return UNMADE_HEADER + "\n" + "\n".join(lines)
        return ""

    def _chain(self, action: str, history: list[str]) -> str:
        """An action as a call, preceded by the read that earlier runs took its missing
        ids from when this run has not made that read yet (E026: a call with an id whose
        source is not named was ignored)."""
        nth = sum(a == action for a in history)
        for slot in self.graph.slots_of(self.template, action):
            term, value = self.graph.resolve(
                action, slot, lambda t: fill(t, self.bindings, self.steps, self.query), self.template, nth
            )
            first = term.partition(" <- ")[2] if term.startswith("Resp(") else ""
            if value is None and first.startswith("GET ") and first not in history and first != action:
                before = [(first, 0, 0)]
                then = self._call(action, history, before).replace(" from call 1>", " from the first call>")
                return f"{self._call(first, history, [])} and then {then}"
        return self._call(action, history, [])

    def _call(self, action: str, history: list[str], before: list[tuple]) -> str:
        """An action as a call: values from this run's state, placeholders otherwise."""
        method, _, path = action.partition(" ")
        segments, query, body = path.split("/"), [], []
        for i in range(len(segments)):
            segments[i] = self.graph.versions.get((action, i), segments[i])
        earlier = [a for a, _, _ in before]
        nth = sum(a == action for a in [*history, *earlier])
        for slot in self.graph.slots_of(self.template, action):
            term, value = self.graph.resolve(
                action, slot, lambda t: fill(t, self.bindings, self.steps, self.query), self.template, nth
            )
            text = value if value is not None else self._placeholder(term, earlier)
            if slot.startswith("path[") and (i := int(slot[5:-1])) < len(segments):
                segments[i] = text
            elif slot.startswith("query."):
                query.append(f"{slot[6:]}={text}")
            elif slot.startswith("body.") and not slot.startswith("body.raw.") and value is not None:
                body.append(f"{slot[5:]}={text}")
        url = "https://" + "/".join("<id>" if s == "{}" else s for s in segments)
        call = f"{method} {url}" + ("?" + "&".join(query) if query else "")
        return call + (f" with body fields {', '.join(body)}" if body else "")

    def _placeholder(self, term: str, earlier: list[str]) -> str:
        if term.startswith("Resp("):
            field = term[5:].split("|", 1)[0].split(")", 1)[0]
            origin = term.partition(" <- ")[2]
            if origin in earlier:
                return f"<{field} from call {earlier.index(origin) + 1}>"
            return f"<{field} from an earlier response>"
        return "<your choice>"
