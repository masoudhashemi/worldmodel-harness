"""E032 offline: how many turns would batching needed, independent reads remove?

All trials are replayed in the order they ran, and each is judged with what earlier runs
of its query template taught (no later runs):
- needed: a read that was a data-flow ancestor of a successful write (its response held a
  value a write used, any number of hops back) in at least NEEDED of the earlier runs;
- dependencies: the actions its argument values came from in most earlier runs;
- ready: all its dependencies have succeeded in this run.
A turn is removable when every call in it is the first call of a needed read that was
already ready before the previous turn, so all of it could have been made earlier (by Gemma in one batch, or
prefetched by the harness). Tool calls and results only.

Usage: e032_parallel_bound.py [run ...]   (default: E027 and E029)
"""

import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from worldmodel_harness.experience import task_of, workflow
from worldmodel_harness.lifting import episode, lift_query, slots

sys.path.insert(0, str(Path(__file__).parent))
from e030_branch_credit import found_at
from step_metrics import turns_of

NEEDED = 0.5
MIN_RUNS = 2
TURN_SECONDS = 4.5


def is_read(step) -> bool:
    return step.tool == "api_fetch" and str(step.arguments.get("method") or "GET").upper() == "GET"


def credit(steps, query):
    """(credited step indices, parents: step index -> indices its arguments came from)."""
    parents = {}
    for i, s in enumerate(steps):
        if s.tool == "api_fetch" and s.success:
            parents[i] = {p for _, v in slots(s.arguments) if isinstance(p := found_at(v, steps[:i], query), int)}
    writes = [i for i, s in enumerate(steps) if s.tool == "api_fetch" and s.success and not is_read(s)]
    done, stack = set(writes), list(writes)
    while stack:
        for p in parents.get(stack.pop(), ()):
            if p not in done:
                done.add(p)
                stack.append(p)
    return done, parents


class Knowledge:
    def __init__(self):
        self.runs = Counter()  # template -> runs
        self.needed = defaultdict(Counter)  # template -> read -> runs in which it was credited
        self.deps = defaultdict(lambda: defaultdict(Counter))  # template -> read -> parent action -> runs
        self.seen = defaultdict(Counter)  # template -> read -> runs with a credited success

    def learn(self, template, steps, query):
        credited, parents = credit(steps, query)
        self.runs[template] += 1
        reads = {}
        for i in sorted(credited):
            if is_read(steps[i]):
                reads.setdefault(steps[i].action, i)
        for action, i in reads.items():
            self.needed[template][action] += 1
            self.seen[template][action] += 1
            for p in {steps[j].action for j in parents.get(i, ())}:
                self.deps[template][action][p] += 1

    def frontier_rules(self, template):
        """Needed reads and their dependencies (parents in most of the runs that made them)."""
        n = self.runs[template]
        if n < MIN_RUNS:
            return {}
        return {
            a: {p for p, k in self.deps[template][a].items() if 2 * k > self.seen[template][a]}
            for a, k in self.needed[template].items()
            if k >= NEEDED * n
        }


def removable(turns, rules) -> int:
    """Turns whose every call is a needed read that was ready before the previous turn."""
    made_at = {}  # action -> turn of its first success
    count = 0
    for t, calls in enumerate(turns):
        # Re-reads of an action already made are loops, not batching: prefetch cannot remove them.
        ok = bool(calls) and all(is_read(c) and c.action in rules and c.action not in made_at for c in calls)
        if ok:
            for c in calls:
                deps = rules[c.action]
                if any(d not in made_at for d in deps):
                    ok = False
                    break
                ready = max((made_at[d] + 1 for d in deps), default=0)
                if ready > t - 1:
                    ok = False
                    break
        count += ok
        for c in calls:
            if c.tool == "api_fetch" and c.success:
                made_at.setdefault(c.action, t)
    return count


def main():
    targets = set(sys.argv[1:]) or {"e027-chained-sources", "e029-hint-once"}
    trials = []
    for run in (p for p in Path("runs").iterdir() if p.is_dir()):
        for path in run.rglob("trajectory.json"):
            task = task_of(path, run)
            if task:
                trials.append((path.stat().st_mtime, run.name, path, task))
    trials.sort()
    know = Knowledge()
    per = defaultdict(list)  # (run, arm) -> (removable, turns, workflow)
    for _, run, path, task in trials:
        e = episode(path, task)
        if not e or not e.steps:
            continue
        template = lift_query(e.query)[0]
        if run in targets:
            _, turns = turns_of(path)
            arm = path.parent.name.rsplit("-r", 1)[0]
            per[(run, arm)].append((removable(turns, know.frontier_rules(template)), len(turns), workflow(task)))
        know.learn(template, e.steps, e.query)

    for (run, arm), rows in sorted(per.items()):
        r = [x[0] for x in rows]
        t = [x[1] for x in rows]
        print(f"{run:22s} {arm:9s} n={len(rows):3d} removable turns/task mean={st.mean(r):.2f} median={st.median(r):.1f} "
              f"share of turns={sum(r) / max(sum(t), 1):.1%} ~{st.mean(r) * TURN_SECONDS:.1f}s/task; tasks with >= 2: {sum(x >= 2 for x in r)}")
    flows = Counter()
    for rows in per.values():
        for rem, _, f in rows:
            flows[f] += rem
    print("most removable turns by workflow: " + ", ".join(f"{f} {n}" for f, n in flows.most_common(8)))


if __name__ == "__main__":
    main()
