"""A policy that learns from runs which kinds of candidate messages to show.

Within a run, each kind (plan, sources, ...) is either shown or held back for the whole
run, decided at its first offer. The outcome is the number of calls the run still made
after that first offer, minus the calls a control run of the same task (one without the
advisor) still made after the same situation: the same last successful action, the same
call and result, and the same repeat flag, at the same occurrence (E035: this difference
tracked graded outcomes, from tool calls alone). Raw calls left across tasks mostly
measure how long each task is (E036). A first offer with no matched point in the control
run is not learned from. Shown and held-back runs are compared per (query template,
kind), falling back to the kind across templates. The decision is Thompson sampling: the
kind is shown with the probability that showing it leaves fewer calls, drawn from the
posterior of the difference of the two means (Student t, pooled variance), so it keeps
exploring while the evidence is weak and a single long run cannot lock it in (E036).
While either side has fewer than MIN_RUNS runs, it is shown with probability 1/2.
Draws are seeded by the task and kind, so a resumed run makes the same decisions. Within a
shown kind, a message is shown when its text differs from the last one shown (ShowNew).
"""

import hashlib
import json
import math
import random
import statistics as st
from collections import defaultdict

from .trace import Domain, Step

MIN_RUNS = 2  # the fewest runs per side for a variance


def situation(domain: Domain, steps: list[Step], i: int) -> tuple:
    """The state before call i, the call and its success, and whether it repeats an
    earlier call of the run."""
    state = next((domain.action(s) for s in reversed(steps[:i]) if s.success), "start")
    call = lambda s: (s.tool, json.dumps(s.arguments, sort_keys=True))
    repeat = any(call(s) == call(steps[i]) for s in steps[:i])
    return state, domain.action(steps[i]), steps[i].success, repeat


def matched(domain: Domain, control: list[Step], wanted: tuple, nth: int) -> int | None:
    """The control run's call in the same situation, at the same occurrence (or its last)."""
    found = [j for j in range(len(control)) if situation(domain, control, j) == wanted]
    return found[min(nth, len(found) - 1)] if found else None


class LearnedPolicy:
    def __init__(self, domain: Domain):
        self.domain = domain
        # key -> ([calls left vs control in runs that showed the kind], [... that held it back])
        self.records = defaultdict(lambda: ([], []))

    def learn(self, template: str, log: list[dict], steps: list[Step], control: list[Step]):
        """Add a finished run (its advisor log and tool calls) with a control run of the
        same task."""
        first = {}
        for entry in log:
            first.setdefault(entry["kind"], (entry["step"], entry["shown"]))
        for kind, (step, shown) in first.items():
            i = step - 1  # the call the kind was first offered after
            if i >= len(steps):
                continue
            here = situation(self.domain, steps, i)
            nth = sum(situation(self.domain, steps, k) == here for k in range(i))
            j = matched(self.domain, control, here, nth)
            if j is None:
                continue
            left = (len(steps) - i - 1) - (len(control) - j - 1)
            for key in ((template, kind), ("*", kind)):
                self.records[key][0 if shown else 1].append(left)

    def estimate(self, template: str, kind: str) -> tuple[float, float, float, int, int] | None:
        """(mean calls left vs control when shown, when held back, the standard error of
        their difference, runs shown, runs held back) from the most specific key with
        MIN_RUNS on both sides, or None."""
        for key in ((template, kind), ("*", kind)):
            shown, held = self.records.get(key, ([], []))
            if len(shown) >= MIN_RUNS and len(held) >= MIN_RUNS:
                spread = st.pstdev(shown + held) or 1.0
                se = spread * math.sqrt(1 / len(shown) + 1 / len(held))
                return st.mean(shown), st.mean(held), se, len(shown), len(held)
        return None

    def for_run(self, task: str) -> "RunPolicy":
        return RunPolicy(self, task)


class RunPolicy:
    def __init__(self, learned: LearnedPolicy, task: str):
        self.learned = learned
        self.task = task
        self.decisions: dict[str, dict] = {}
        self.last: dict[str, str] = {}

    def _random(self, kind: str) -> random.Random:
        return random.Random(hashlib.sha256(f"{self.task}|{kind}".encode()).digest())

    def show(self, kind: str, template: str, state: str, text: str) -> bool:
        if kind not in self.decisions:
            rng = self._random(kind)
            estimate = self.learned.estimate(template, kind)
            if estimate is None:
                show, reason = rng.random() < 0.5, "explore"
            else:
                shown, held, se, n_shown, n_held = estimate
                df = n_shown + n_held - 2
                t = rng.gauss(0, 1) / math.sqrt(rng.gammavariate(df / 2, 2) / df)
                show, reason = held - shown + se * t > 0, "sampled"
            self.decisions[kind] = {"show": show, "reason": reason, "estimate": estimate}
        if not self.decisions[kind]["show"] or self.last.get(kind) == text:
            return False
        self.last[kind] = text
        return True
