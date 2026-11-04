"""A policy that learns from runs which kinds of candidate messages to show.

Within a run, each kind (plan, sources, ...) is either shown or held back for the whole
run, decided at its first offer. Its outcome is measured against a control run of the
same task (one without the advisor), from the point where the control run was in the same
situation: the same last successful action, the same call and result, and the same repeat
flag, at the same occurrence. A first offer with no such point is not learned from. Raw
outcomes across tasks mostly measure how long each task is (E036). Two outcomes, both
from tool calls only:
- calls: the calls the run still made, minus the control's (E035: this tracked graded
  outcomes);
- writes: the distinct successful writes the run still made, minus the control's. Fewer
  calls are only a gain if the work is still done; without this, a message that made the
  agent stop early would look best.
Shown and held-back runs are compared per (query template, kind), falling back to the
kind across templates. The decision is Thompson sampling on calls: the kind is shown with
the probability that showing it leaves fewer calls, drawn from the posterior of the
difference of the two means (Student t, pooled variance), so it keeps exploring while the
evidence is weak and a single long run cannot lock it in (E036). It is held back when
showing it makes fewer writes beyond the graph's confidence level (the upper bound of the
difference is below 0), the same no-regression-plus-gain rule as the evaluation. While
either side has fewer than MIN_RUNS runs, it is shown with probability 1/2. Draws are
seeded by the task, the trial and the kind, so repeated trials of a task draw afresh and a
resumed run makes the same decisions. Within a shown kind, a message is shown when its
text differs from the last one shown (ShowNew).
"""

import hashlib
import json
import math
import random
import statistics as st
from collections import defaultdict
from dataclasses import dataclass

from .graph import Z
from .trace import Domain, Step

MIN_RUNS = 2  # the fewest runs per side for a variance


def situation(domain: Domain, steps: list[Step], i: int) -> tuple:
    """The state before call i, the call and its success, and whether it repeats an
    earlier call of the run."""
    state = next((domain.action(s) for s in reversed(steps[:i]) if s.success), "start")
    repeat = any(_call(s) == _call(steps[i]) for s in steps[:i])
    return state, domain.action(steps[i]), steps[i].success, repeat


def _call(step: Step) -> tuple[str, str]:
    return step.tool, json.dumps(step.arguments, sort_keys=True)


def matched(domain: Domain, control: list[Step], wanted: tuple, nth: int) -> int | None:
    """The control run's call in the same situation, at the same occurrence."""
    found = [j for j in range(len(control)) if situation(domain, control, j) == wanted]
    return found[nth] if nth < len(found) else None


def writes_after(domain: Domain, steps: list[Step], i: int) -> int:
    """Distinct successful writes after call i."""
    return len({_call(s) for s in steps[i + 1 :] if s.success and domain.writes(s)})


@dataclass
class Estimate:
    shown: float  # mean calls left vs control when shown
    held: float  # ... when held back
    se: float  # standard error of their difference
    writes: float  # mean writes vs control, shown minus held back
    writes_se: float
    n_shown: int
    n_held: int


def _difference(shown: list[float], held: list[float]) -> tuple[float, float]:
    spread = st.pstdev(shown + held) or 1.0
    return st.mean(shown) - st.mean(held), spread * math.sqrt(1 / len(shown) + 1 / len(held))


class LearnedPolicy:
    def __init__(self, domain: Domain):
        self.domain = domain
        # key -> ([(calls, writes) vs control in runs that showed the kind], [... held it back])
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
            calls = (len(steps) - i - 1) - (len(control) - j - 1)
            writes = writes_after(self.domain, steps, i) - writes_after(self.domain, control, j)
            for key in ((template, kind), ("*", kind)):
                self.records[key][0 if shown else 1].append((calls, writes))

    def estimate(self, template: str, kind: str) -> Estimate | None:
        """From the most specific key with MIN_RUNS on both sides, or None."""
        for key in ((template, kind), ("*", kind)):
            shown, held = self.records.get(key, ([], []))
            if len(shown) >= MIN_RUNS and len(held) >= MIN_RUNS:
                _, se = _difference([c for c, _ in shown], [c for c, _ in held])
                writes, writes_se = _difference([w for _, w in shown], [w for _, w in held])
                return Estimate(
                    st.mean(c for c, _ in shown), st.mean(c for c, _ in held), se, writes, writes_se, len(shown), len(held)
                )
        return None

    def for_run(self, task: str, trial: str = "") -> "RunPolicy":
        return RunPolicy(self, task, trial)


class RunPolicy:
    def __init__(self, learned: LearnedPolicy, task: str, trial: str = ""):
        self.learned = learned
        self.task = task
        self.trial = trial
        self.decisions: dict[str, dict] = {}
        self.last: dict[str, str] = {}

    def _random(self, kind: str) -> random.Random:
        return random.Random(hashlib.sha256(f"{self.task}|{self.trial}|{kind}".encode()).digest())

    def show(self, kind: str, template: str, state: str, text: str) -> bool:
        if kind not in self.decisions:
            rng = self._random(kind)
            estimate = self.learned.estimate(template, kind)
            if estimate is None:
                show, reason = rng.random() < 0.5, "explore"
            elif estimate.writes + Z * estimate.writes_se < 0:
                show, reason = False, "fewer writes"
            else:
                df = estimate.n_shown + estimate.n_held - 2
                t = rng.gauss(0, 1) / math.sqrt(rng.gammavariate(df / 2, 2) / df)
                show, reason = estimate.held - estimate.shown + estimate.se * t > 0, "sampled"
            self.decisions[kind] = {
                "show": show, "reason": reason, "estimate": estimate.__dict__ if estimate else None
            }
        if not self.decisions[kind]["show"] or self.last.get(kind) == text:
            return False
        self.last[kind] = text
        return True
