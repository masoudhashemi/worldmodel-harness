"""A policy that learns from runs which kinds of candidate messages to show.

Within a run, each kind (plan, sources, ...) is either shown or held back for the whole
run, decided at its first offer. The outcome is the number of calls the run still made
after that first offer (E035: fewer calls left than at a matched point without the
message tracked graded outcomes, from tool calls alone). Shown and held-back runs are
compared per (query template, kind), falling back to the kind across templates:
- while either side has fewer than MIN_RUNS runs, the decision is random (1/2);
- otherwise the side with fewer calls left is chosen, and the other one with
  probability EXPLORE, so the comparison keeps being updated.
Decisions are seeded by the task and kind, so a resumed run makes the same ones. Within a
shown kind, a message is shown when its text differs from the last one shown (ShowNew).
"""

import hashlib
import random
import statistics as st
from collections import defaultdict

MIN_RUNS = 3
EXPLORE = 0.1


class LearnedPolicy:
    def __init__(self):
        # key -> ([calls left in runs that showed the kind], [... that held it back])
        self.records = defaultdict(lambda: ([], []))

    def learn(self, template: str, log: list[dict], calls: int):
        """Add a finished run: its advisor log and its number of tool calls."""
        first = {}
        for entry in log:
            first.setdefault(entry["kind"], (entry["step"], entry["shown"]))
        for kind, (step, shown) in first.items():
            for key in ((template, kind), ("*", kind)):
                self.records[key][0 if shown else 1].append(calls - step)

    def estimate(self, template: str, kind: str) -> tuple[float, float, int, int] | None:
        """(mean calls left when shown, when held back, runs shown, runs held back) from the
        most specific key with MIN_RUNS on both sides, or None."""
        for key in ((template, kind), ("*", kind)):
            shown, held = self.records.get(key, ([], []))
            if len(shown) >= MIN_RUNS and len(held) >= MIN_RUNS:
                return st.mean(shown), st.mean(held), len(shown), len(held)
        return None

    def for_run(self, task: str) -> "RunPolicy":
        return RunPolicy(self, task)


class RunPolicy:
    def __init__(self, learned: LearnedPolicy, task: str):
        self.learned = learned
        self.task = task
        self.decisions: dict[str, dict] = {}
        self.last: dict[str, str] = {}

    def _random(self, kind: str) -> float:
        digest = hashlib.sha256(f"{self.task}|{kind}".encode()).digest()
        return random.Random(digest).random()

    def show(self, kind: str, template: str, state: str, text: str) -> bool:
        if kind not in self.decisions:
            u = self._random(kind)
            estimate = self.learned.estimate(template, kind)
            if estimate is None:
                show, reason = u < 0.5, "explore"
            else:
                better = estimate[0] < estimate[1]
                show = better if u >= EXPLORE else not better
                reason = "learned" if u >= EXPLORE else "explore"
            self.decisions[kind] = {"show": show, "reason": reason, "estimate": estimate}
        if not self.decisions[kind]["show"] or self.last.get(kind) == text:
            return False
        self.last[kind] = text
        return True
