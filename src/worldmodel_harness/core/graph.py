"""The experience graph: what earlier runs did, learned from tool calls and results only.

States are lifted: a query template (the task with its literals replaced by types) and
the run's last successful actions. The graph holds no task values; a run's own query and
responses resolve what it knows into values. Grades are never used. What it learns:
- plans: from a state, the next successful action and how many runs took it;
- argument sources: for each slot of an action, where its value came from (lifting.source);
- value origins: for each written slot, the read whose response held the value;
- branches: at the first visit of a state (until its next successful call), which calls
  led to written values that were found in a read rather than made up;
- recoveries: after a failed call, the next call that succeeded.
Counts are per episode, and every rule is gated by a Wilson lower bound.
"""

import json
import math
import re
from collections import Counter, defaultdict

from .lifting import lift_query, origin, source
from .trace import Domain, Episode, Step

Z = 1.2816  # one-sided 90%
THETA = 0.35  # a rule is used when its lower bound reaches this
MIN_CONST_WORKFLOWS = 2  # a literal is part of an action's shape if it fills the slot in
MIN_CONST_SHARE = 0.8  # this many workflows, in most of them, and in most task instances
# (surface variants) of each: a value the task's own literals determine changes with them
BRANCH_RUNS = 3
TOP = 3


def lcb(hits: int, total: int) -> float:
    """Wilson lower bound of hits / total."""
    if not total:
        return 0.0
    p = hits / total
    centre = p + Z * Z / (2 * total)
    spread = Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total))
    return (centre - spread) / (1 + Z * Z / total)


def ucb(hits: int, total: int) -> float:
    """Wilson upper bound of hits / total."""
    return 1 - lcb(total - hits, total)


def workflow(task: str) -> str:
    """The workflow of a task instance: surface variants (task~vN) share their source's."""
    return task.split("~", 1)[0]


def signature(response: str) -> str:
    """An error message with quoted values and numbers removed."""
    data = None
    try:
        data = json.loads(response)
    except (TypeError, ValueError):
        pass
    error = data.get("error", data) if isinstance(data, dict) else data
    text = (error.get("message") or json.dumps(error)) if isinstance(error, dict) else str(error if data is not None else response)
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "_", text.lower())
    return re.sub(r"\d+", "#", text).strip()[:160]


class Graph:
    def __init__(self, domain: Domain, episodes: list[Episode] = ()):
        self.domain = domain
        self.runs = Counter()  # template -> episodes
        self.made = Counter()  # (template, action) -> episodes with a successful call of it
        # Plans: (template, last two successful actions) -> next successful action ("end").
        self.after = defaultdict(Counter)
        self.visits = Counter()
        # Argument sources: (template or None, action, slot[, nth]) -> term -> episodes.
        self.terms = defaultdict(Counter)
        self.slot_runs = Counter()  # ... -> episodes that filled the slot
        self.free = defaultdict(lambda: defaultdict(set))  # (action, slot) -> literal -> tasks
        self.uses = defaultdict(set)  # (action, slot) -> tasks that filled it
        self.arguments = defaultdict(Counter)  # (template, action) -> slot -> episodes
        # Value origins: (template, write action, slot) -> read action | Query | Agent -> episodes.
        self.origins = defaultdict(Counter)
        self.traced = Counter()  # ... episodes in which some written value was seen
        # Branches: (template, state) -> (write action, slot) -> next action -> [found, made up].
        self.branches = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
        # Recoveries: (action, error) -> (next successful action, its shape) -> episodes,
        # and the episodes that hit the error.
        self.recoveries = defaultdict(Counter)
        self.failures = Counter()
        for e in episodes:
            self.add(e)

    def action(self, step: Step) -> str:
        return self.domain.action(step)

    def add(self, e: Episode):
        once = set()

        def first(key) -> bool:
            if key in once:
                return False
            once.add(key)
            return True

        template, bindings = lift_query(e.query)
        done = [self.action(s) for s in e.steps if s.success]
        self.runs[template] += 1
        for action in set(done):
            self.made[(template, action)] += 1
        for n, action in enumerate([*done, "end"]):
            for key in plan_keys(template, done[:n]):
                if first((key, "visit")):
                    self.visits[key] += 1
                if first((key, action)):
                    self.after[key][action] += 1
        for i, step in enumerate(e.steps):
            action = self.action(step)
            if not step.success:
                self._add_failure(e.steps, i, first)
                continue
            nth = sum(s.success and self.action(s) == action for s in e.steps[:i])
            values = self.domain.slots(step)
            for slot in dict.fromkeys(s for s, _ in values):
                if first((action, slot, "arg")):
                    self.arguments[(template, action)][slot] += 1
            for slot, value in values:
                term = source(value, bindings, e.steps[:i], e.query, self.domain)
                if term == "Free":
                    self.free[(action, slot)][value].add(e.task)
                    term = "Agent"
                self.uses[(action, slot)].add(e.task)
                for key in ((None, action, slot), (template, action, slot), (template, action, slot, nth)):
                    if first((key, "runs")):
                        self.slot_runs[key] += 1
                    if first((key, term)):
                        self.terms[key][term] += 1
                if self.domain.writes(step):
                    found = origin(value, e.steps[:i], e.query, self.domain)
                    key = (template, action, slot)
                    if first((key, found)):
                        self.origins[key][found] += 1
                    if found != "Agent" and first((key, "traced")):
                        self.traced[key] += 1
        self._add_branches(template, e)

    def _add_failure(self, steps: list[Step], i: int, first):
        action = self.action(steps[i])
        key = (action, signature(steps[i].response))
        if not first(("failure", key)):
            return
        self.failures[key] += 1
        later = next((s for s in steps[i + 1 :] if s.success and s.tool == steps[i].tool), None)
        if later is not None:
            self.recoveries[key][(self.action(later), self.domain.shape(later))] += 1

    def _add_branches(self, template: str, e: Episode):
        outcome = {}
        for i, step in enumerate(e.steps):
            if step.success and self.domain.writes(step):
                for slot, value in self.domain.slots(step):
                    found = origin(value, e.steps[:i], e.query, self.domain)
                    if found != "Query":
                        key = (self.action(step), slot)
                        ok, at = outcome.get(key, (True, i))
                        outcome[key] = (ok and found != "Agent", at)
        # The first visit of a state lasts until the next successful call, so failed calls
        # made from it and the call that then succeeded are all branches taken from it.
        state, seen, first, taken_here = "start", {"start"}, True, set()
        for i, step in enumerate(e.steps):
            taken = self.branch(step)
            if first and taken not in taken_here:
                taken_here.add(taken)
                for key, (ok, at) in outcome.items():
                    if at > i:
                        self.branches[(template, state)][key][taken][0 if ok else 1] += 1
            if step.success:
                state = self.action(step)
                first, taken_here = state not in seen, set()
                seen.add(state)

    def branch(self, step: Step) -> str:
        return self.action(step) if step.success else "fail " + self.action(step)

    # What the graph knows, for one run's template and state.

    def plan(self, template: str, done: list[str], limit: int = TOP) -> list[tuple[str, int, int]]:
        """The likeliest next successful actions after `done`: (action, runs that took it,
        runs that got there), while the lower bound stays at THETA."""
        found, done = [], list(done)
        for _ in range(4 * limit):
            key = next((k for k in plan_keys(template, done) if self.visits[k]), None)
            if key is None or len(found) == limit:
                break
            action, hits = self.after[key].most_common(1)[0]
            if action == "end" or lcb(hits, self.visits[key]) < THETA:
                break
            if not found or found[-1][0] != action:
                found.append((action, hits, self.visits[key]))
            done.append(action)
        return found

    def sources(self, template: str, done: set[str], only=None) -> list[tuple[str, str, str, int, int]]:
        """Reads that earlier runs took written values from, not made in this run yet:
        (write action, slot, read action, runs, runs in which a written value was found)."""
        found = []
        for (t, action, slot), counts in self.origins.items():
            if t != template or (only and (action, slot) != only):
                continue
            total = self.traced[(t, action, slot)]
            for read, hits in counts.items():
                if read not in ("Query", "Agent") and read not in done and lcb(hits, total) >= THETA:
                    found.append((action, slot, read, hits, total))
        found.sort(key=lambda s: -lcb(s[3], s[4]))
        unique, reads = [], set()
        for s in found:
            if s[2] not in reads:
                reads.add(s[2])
                unique.append(s)
        return unique[:TOP]

    def unmade(self, template: str, done: set[str]) -> list[tuple[str, int, int]]:
        """Actions most earlier runs made successfully that this run has not."""
        total = self.runs[template]
        found = [
            (action, hits, total)
            for (t, action), hits in self.made.items()
            if t == template and action not in done and lcb(hits, total) >= THETA
        ]
        return sorted(found, key=lambda s: -s[1])[:TOP]

    def bad_branch(self, template: str, state: str, taken: str) -> list[tuple[str, str, int, int]]:
        """Write slots that runs taking this branch from this state usually made up, while
        another branch found them: (write action, slot, found, runs)."""
        found = []
        for (action, slot), branches in self.branches.get((template, state), {}).items():
            f, made_up = branches.get(taken, (0, 0))
            if f + made_up < BRANCH_RUNS or ucb(f, f + made_up) >= THETA:
                continue
            if any(b != taken and sum(v) >= BRANCH_RUNS and lcb(v[0], sum(v)) >= THETA for b, v in branches.items()):
                found.append((action, slot, f, f + made_up))
        return found

    def recovery(self, step: Step) -> tuple[str, str, int, int] | None:
        """After this failure, the call that earlier runs most often made successfully next:
        (action, shape, runs, runs with this failure)."""
        key = (self.action(step), signature(step.response))
        total = self.failures[key]
        for (action, shape), hits in self.recoveries.get(key, Counter()).most_common(1):
            if lcb(hits, total) >= THETA and (action, shape) != (key[0], self.domain.shape(step)):
                return action, shape, hits, total
        return None

    def shape_const(self, action: str, slot: str) -> str | None:
        """The literal that is part of this action's shape, if any: it fills the slot across
        workflows and stays the same across the task instances of each."""
        uses = self.uses.get((action, slot), set())
        everywhere = {workflow(t) for t in uses}
        for value, tasks in self.free.get((action, slot), {}).items():
            flows = {workflow(t) for t in tasks}
            if len(flows) < MIN_CONST_WORKFLOWS or len(flows) < MIN_CONST_SHARE * len(everywhere):
                continue
            if all(
                sum(workflow(t) == w for t in tasks) >= MIN_CONST_SHARE * sum(workflow(t) == w for t in uses)
                for w in flows
            ):
                return f"Const({value})"
        return None

    def resolve(self, action, slot, resolver, template=None, nth=None):
        """(term, value): the best-supported confident source of a slot that this run can
        fill, from this call of the workflow, the workflow, then any task. `Agent` on top of the first
        level that knows the slot leaves it to the agent."""
        best = None
        for key in ((template, action, slot, nth), (template, action, slot), (None, action, slot)):
            counts = self.terms.get(key, Counter())
            ranked = [t for t, _ in counts.most_common()]
            if ranked and ranked[0] == "Agent":
                const = self.shape_const(action, slot)
                return (const, resolver(const)) if const else (best or "Agent", None)
            # Like every rule of the graph, a source is used only above the confidence level.
            confident = [t for t in ranked if t != "Agent" and lcb(counts[t], self.slot_runs[key]) >= THETA]
            for term in confident:
                if (value := resolver(term)) is not None:
                    return term, value
            best = best or next(iter(confident), None)
        const = self.shape_const(action, slot)
        return (const, resolver(const)) if const else (best or "Agent", None)

    def slots_of(self, template: str, action: str) -> list[str]:
        """The slots this workflow's calls of an action usually fill."""
        counts = self.arguments.get((template, action), Counter())
        most = max(counts.values(), default=0)
        return [slot for slot, n in counts.items() if 2 * n >= most]


def plan_keys(template: str, done: list[str]) -> list[tuple]:
    """Plan states after a run's successful actions: the last two, then the last one."""
    return [(template, tuple(done[-2:])), (template, tuple(done[-1:]))]

