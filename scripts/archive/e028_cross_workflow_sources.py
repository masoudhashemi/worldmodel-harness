"""E028 offline: do value sources transfer across workflows (dynamic skills from sub-skills)?

Value sources are keyed by (write action, slot), not by the task's template, and each
workflow is held out in turn: what it gets comes only from other workflows. Support
counts workflows, not episodes, so one busy workflow cannot dominate.

1. Prediction: for every write slot whose value a trial first saw in a read R, is the
   held-out top-ranked read R? (Imitation of the agent's own sourcing, all runs.)
2. Loops: for the trials with a >= 3 repeated-search streak (E022, E024), at the streak's
   start, would a held-out hint name a read not yet made, and does that read's service
   hold a value the grader expected (diagnosis only; grades never reach the counts)?
   The write slots are those the trial itself attempted at any point, an optimistic
   stand-in for knowing which writes the task needs.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from worldmodel_harness.experience import WRITES, lcb, load, workflow
from worldmodel_harness.lifting import episode, origin, slots

sys.path.insert(0, str(Path(__file__).parent))
from e026_source_bound import RUNS, expected_values, first_repeat_calls, holds

THETA = 0.35


def writes(e):
    """(index, action, slot, origin) for every write slot of an episode."""
    for i, step in enumerate(e.steps):
        if step.tool == "api_fetch" and str(step.arguments.get("method", "GET")).upper() in WRITES:
            for slot, value in slots(step.arguments):
                yield i, step.action, slot, origin(value, e.steps[:i], e.query)


def main():
    episodes = load([p for p in Path("runs").iterdir() if p.is_dir()], lambda t: True)
    hits = defaultdict(lambda: defaultdict(Counter))  # workflow -> (action, slot) -> read -> 1
    traced = defaultdict(Counter)  # workflow -> (action, slot) -> 1 if some value was seen
    for e in episodes:
        w = workflow(e.task)
        for _, action, slot, found in writes(e):
            if found != "Agent":
                traced[w][(action, slot)] = 1
                if found != "Query":
                    hits[w][(action, slot)][found] = 1
    flows = list(traced)

    def ranked(held: str, key) -> list[tuple[str, int, int]]:
        total = sum(traced[w][key] for w in flows if w != held)
        counts = Counter()
        for w in flows:
            if w != held:
                counts.update(hits[w][key])
        return [(r, n, total) for r, n in counts.most_common() if lcb(n, total) >= THETA]

    # 1. Prediction of the read a written value came from.
    by_slot = defaultdict(Counter)
    for e in episodes:
        w = workflow(e.task)
        for _, action, slot, found in writes(e):
            if found in ("Agent", "Query"):
                continue
            kind = f"{action.split(' ', 1)[1].split('/', 1)[0]} {slot}"
            options = ranked(w, (action, slot))
            by_slot[kind]["n"] += 1
            by_slot[kind]["gated"] += bool(options)
            by_slot[kind]["right"] += bool(options) and options[0][0] == found
    print("1. Held-out prediction of the read each written value came from (slots with n >= 30):")
    total = Counter()
    for kind, c in sorted(by_slot.items(), key=lambda x: -x[1]["n"]):
        total.update(c)
        if c["n"] >= 30:
            print(f"   {kind[:70]:70s} n={c['n']:5d} hint={c['gated'] / c['n']:.0%} "
                  f"top-1 right when hinted={c['right'] / max(c['gated'], 1):.0%}")
    print(f"   ALL n={total['n']} hint={total['gated'] / total['n']:.0%} "
          f"top-1 right when hinted={total['right'] / max(total['gated'], 1):.0%}")

    # 2. Loops, with the workflow held out.
    stats, per_flow = Counter(), defaultdict(Counter)
    for run in RUNS:
        out = Path(run)
        for row in json.loads((out / "summary.json").read_text())["results"]:
            trial = out / "trajectories" / row["task"] / f"{row['arm']}-r{row['round']}"
            n = first_repeat_calls(trial / "trajectory.json")
            if n is None:
                continue
            e = episode(trial / "trajectory.json", row["task"])
            w = workflow(row["task"])
            called = {s.action for s in e.steps[:n] if s.success}
            reads = []
            for key in dict.fromkeys((a, s) for _, a, s, _ in writes(e)):
                reads += [r for r, _, _ in ranked(w, key)[:1] if r not in called and r not in reads]
            right = [r for r in reads if holds(row["task"], r, expected_values(json.loads((trial / "result.json").read_text())))]
            stats["loops"] += 1
            stats["hint"] += bool(reads)
            stats["right"] += bool(right)
            per_flow[w]["loops"] += 1
            per_flow[w]["hint"] += bool(reads)
            per_flow[w]["right"] += bool(right)
            per_flow[w]["reads"] = per_flow[w]["reads"] or 0
            print(f"   {row['task']:42s} {row['arm']:9s} reads={[r.split(' ', 1)[1][:50] for r in reads]} right={bool(right)}") if "-v" in sys.argv else None
    print("\n2. Looping trials, workflow held out (hint = a read not yet made; right = its service holds an expected value):")
    for w, c in sorted(per_flow.items()):
        print(f"   {w:40s} loops={c['loops']:2d} hint={c['hint']:2d} right={c['right']:2d}")
    print(f"   ALL {dict(stats)}")


if __name__ == "__main__":
    main()
