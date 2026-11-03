"""The experience graph of world-model-v3.md: a transition graph learned across runs.

Nodes are lifted states (the last actions and whether they succeeded); edges are next
actions with the number of episodes that took them. On each edge, every argument of an
api_fetch has a source (lifting.source): the query, the response of an earlier action,
or the agent. The graph holds no task values; a run's own query and responses (its
state) resolve sources to values. Learned only from tool calls and results, never
from grades.
"""

import json
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
# A source hint waits until the run has repeated this many searches with no successful
# fetch in between (E027: hints after one accidental repeat could only hurt).
STALL = 2
# A branch has a bad record for a write slot when, over at least BRANCH_RUNS earlier runs
# that took it from this state, the upper bound of the slot's found rate is below
# BAD_BRANCH, and another branch from the state has a lower bound of at least PLAN_THETA.
BRANCH_RUNS = 3
# A written address within this edit distance of one the run has read is a slip in
# copying it (E034: Gmail raw messages Gemma base64-encodes by hand).
NEAR_COPY = 2
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
BAD_BRANCH = 0.35


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
        # Branches (E031): (template, state) -> (write action, slot) -> branch -> [found, made up].
        # A state is the last successful fetch; a branch is the next action of any kind.
        self.branches = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
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
                        if found not in ("Agent", "Short") and (key, "traced") not in seen:
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
        self._add_branches(template, e)

    def _add_branches(self, template: str, e: Episode):
        """At the first visit of each state, the branch taken and, for every write slot
        filled later, whether its values were found in a read or made up."""
        outcome = {}
        for i, step in enumerate(e.steps):
            if step.tool == "api_fetch" and step.success and str(step.arguments.get("method", "GET")).upper() in WRITES:
                for slot, value in slots(step.arguments):
                    found = origin(value, e.steps[:i], e.query)
                    if found not in ("Query", "Short"):
                        ok, first = outcome.get((step.action, slot), (True, i))
                        outcome[(step.action, slot)] = (ok and found != "Agent", first)
        state, visited = "start", set()
        for i, step in enumerate(e.steps):
            taken = branch_label(step)
            if taken and state not in visited:
                visited.add(state)
                for key, (ok, first) in outcome.items():
                    if first > i:
                        self.branches[(template, state)][key][taken][0 if ok else 1] += 1
            if step.tool == "api_fetch" and step.success:
                state = step.action

    def bad_branch(self, template: str, state: str, taken: str) -> list[tuple[str, str, int, int]]:
        """Write slots that earlier runs taking this branch from this state usually made up,
        while another branch did find them: (write action, slot, found, runs)."""
        found = []
        for (action, slot), branches in self.branches.get((template, state), {}).items():
            f, made_up = branches.get(taken, (0, 0))
            if f + made_up < BRANCH_RUNS or ucb(f, f + made_up) >= BAD_BRANCH:
                continue
            if any(b != taken and sum(v) >= BRANCH_RUNS and lcb(v[0], sum(v)) >= PLAN_THETA for b, v in branches.items()):
                found.append((action, slot, f, f + made_up))
        return found

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

    def sources(self, template: str, called: set[str], only=None) -> list[tuple[str, str, str, int, int]]:
        """Reads that earlier runs of this workflow took written values from, not yet called.

        Each is (write action, slot, read action, episodes, episodes in which a written
        value was seen somewhere), best first, gated like plan steps. A value the agent made
        up (found in no response or query) is no evidence about where it could have been
        found, so episodes with only such values are left out; values from the query are
        skipped.
        """
        found = []
        for (t, action, slot), counts in self.origins.items():
            if t != template or (only and (action, slot) != only):
                continue
            total = self.traced[(t, action, slot)]
            for read, hits in counts.items():
                if read not in ("Query", "Agent", "Short") and read not in called and lcb(hits, total) >= PLAN_THETA:
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


def branch_label(step: Step) -> str | None:
    """The branch a step takes: any search, a failed call, or a successful call's endpoint."""
    if step.tool == "api_search":
        return "search"
    if step.tool == "api_fetch":
        return step.action if step.success else "fail " + step.action
    return None


def distance(a: str, b: str, cap: int = NEAR_COPY + 1) -> int:
    """Edit distance, counted up to cap."""
    if abs(len(a) - len(b)) >= cap:
        return cap
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        if min(current) >= cap:
            return cap
        previous = current
    return min(previous[-1], cap)


def ucb(hits: int, total: int) -> float:
    """Wilson upper bound of hits / total."""
    return 1 - lcb(total - hits, total)


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
BRANCH_HEADER = "[Experience] In earlier runs of this same task, with other names and ids:"
CHECK_HEADER = "Not sent. An address in this call appears in no response you have read and not in the task, but one that does is nearly the same:"
CHECK_FOOTER = "If you meant that address, check how you wrote or encoded it and send again. If the address you wrote is right, send the same call again."
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
        self.stall = 0  # repeated searches since the last successful fetch
        self.shown: set[str] = set()
        self.state, self.visited, self.warned = "start", set(), set()
        self.stopped: set[str] = set()

    def observe(self, arguments: dict, raw: str, success: bool):
        self.steps.append(Step("api_fetch", arguments, raw, success))
        if success:
            self.stall = 0

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
        """Called on each repeated search. Once the run has stalled (STALL repeats with no
        successful fetch), where earlier runs of this task found values they wrote, for
        reads this run has not made yet. Each distinct hint is shown once: in E027 a hint
        repeated on every search kept an ignoring run in its loop."""
        self.stall += 1
        if self.stall < STALL:
            return ""
        text = self._source_hint()
        if not text or text in self.shown:
            return ""
        self.shown.add(text)
        return text

    def warn_branch(self, tool: str, arguments: dict, raw: str, success: bool) -> str:
        """Called after every search and fetch. At the first action from a state, if earlier
        runs that took the same branch from it usually made up a value they later wrote, say
        so and where runs that found it took it from (E031). Each slot is warned about once."""
        step = Step(tool, arguments, raw, success)
        taken, text = branch_label(step), ""
        if taken and self.state not in self.visited:
            self.visited.add(self.state)
            text = self._branch_text(taken)
        if tool == "api_fetch" and success:
            self.state = step.action
        return raw + ("\n\n" + text if text else "")

    def _branch_text(self, taken: str) -> str:
        history = [s.action for s in self.steps if s.success]
        lines = []
        for action, slot, found, runs in self.graph.bad_branch(self.template, self.state, taken):
            if (action, slot) in self.warned:
                continue
            reads = self.graph.sources(self.template, set(history), only=(action, slot))
            if not reads:
                continue
            self.warned.add((action, slot))
            _, _, read, hits, total = reads[0]
            went = "searched again" if taken == "search" else f"called {taken.removeprefix('fail ')}"
            lines.append(
                f"- the {slot.rsplit('.', 1)[-1]} of {action}: earlier runs that {went} at this point wrote a "
                f"value found in no response they had read in {runs - found}/{runs} runs. Runs that found it "
                f"took it from {self._chain(read, history)} ({hits}/{total} runs)"
            )
        if not lines:
            return ""
        self.log.append({"after_fetches": len(history), "branch": taken, "warnings": lines})
        return BRANCH_HEADER + "\n" + "\n".join(lines)

    def _source_hint(self) -> str:
        history = [s.action for s in self.steps if s.success]
        lines = [
            f"- the {slot.rsplit('.', 1)[-1]} of {write} came from the response of "
            f"{self._chain(read, history)} ({hits}/{total} runs)"
            for write, slot, read, hits, total in self.graph.sources(self.template, set(history))
        ]
        if lines:
            self.log.append({"after_fetches": len(history), "sources": lines})
            return SOURCE_HEADER + "\n" + "\n".join(lines)
        return ""

    def check_write(self, arguments: dict) -> str | None:
        """Called before a write is sent. An email address found in no response or the
        query that is a near copy of one found there is usually a slip in copying it (E034).
        Where earlier runs of this task mostly found this slot's values in reads, return an
        error naming the address that was read instead of sending. Each value is stopped once."""
        if str(arguments.get("method") or "GET").upper() not in WRITES:
            return None
        action, read, lines = Step("api_fetch", arguments, "", True).action, None, []
        for slot, value in slots(arguments):
            key = (self.template, action, slot)
            if not EMAIL.fullmatch(value) or value.lower() in self.stopped or origin(value, self.steps, self.query) != "Agent":
                continue
            if lcb(self.graph.traced[key], self.graph.written[key]) < PLAN_THETA:
                continue
            if read is None:
                read = self._read_addresses()
            close = sorted((d, a, where) for a, where in read.items() if 0 < (d := distance(value.lower(), a)) <= NEAR_COPY)
            if not close or (len(close) > 1 and close[1][0] == close[0][0]):
                continue
            self.stopped.add(value.lower())
            lines.append(f"- {slot.rsplit('.', 1)[-1]}: you wrote {value}; {close[0][2]} has {close[0][1]}")
        if not lines:
            return None
        self.log.append({"after_fetches": sum(s.success for s in self.steps), "stopped": lines})
        return json.dumps({"error": "not_sent", "message": "\n".join([CHECK_HEADER, *lines, CHECK_FOOTER])})

    def _read_addresses(self) -> dict[str, str]:
        """Address -> where it was seen: the task, or the latest read whose response has it."""
        found = {a.lower(): "the task" for a in EMAIL.findall(self.query)}
        for step in self.steps:
            if step.success and str(step.arguments.get("method") or "GET").upper() == "GET":
                for a in EMAIL.findall(step.response):
                    if a.lower() not in found or found[a.lower()] != "the task":
                        found[a.lower()] = f"the response of {step.arguments.get('url')}"
        return found

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
