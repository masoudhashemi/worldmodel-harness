"""E030 offline: when runs branch at the same state, does popularity pick the branch that leads somewhere?

States are the plan's keys: (query template, last two successful fetches). At each state,
the next successful fetch of each episode is a branch. Two rankings of branches:
- popularity: episodes that took the branch (what Model.plan uses);
- data-flow credit: episodes in which the branch's call is an ancestor of a successful
  write, i.e. a value written later came from its response, or from a read whose
  arguments came from it (followed back any number of hops).
Credit uses tool calls and results only. For states where the two rankings disagree, the
branches are compared on what followed: whether a grounded write came (at least half of
its values were found in the query or a read), and how many tool calls it took. Partial
credit is shown for diagnosis only.
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from worldmodel_harness.experience import WRITES, task_of, workflow
from worldmodel_harness.lifting import episode, lift_query, slots

MIN_EPISODES = 3  # a state must be reached by this many episodes to compare branches
# --reads: compare read branches only. A write is always credited, so with writes in the
# comparison the credited branch is often "write now" against "read more".
READS_ONLY = "--reads" in sys.argv


def found_at(value: str, earlier, query: str):
    """Index of the latest earlier successful read whose response holds the value,
    "query" if the query does, None if neither (as lifting.origin, with the index)."""
    needle = value.lower()
    if needle in query.lower():
        return "query"
    for i in range(len(earlier) - 1, -1, -1):
        s = earlier[i]
        if s.success and s.tool == "api_fetch" and str(s.arguments.get("method") or "GET").upper() == "GET" and needle in s.response.lower():
            return i
    return None


def analyse(e):
    """Per step: credited (an ancestor of a successful write), and the grounded writes."""
    steps = e.steps
    parents = {}
    for i, s in enumerate(steps):
        if s.tool == "api_fetch" and s.success:
            parents[i] = {p for _, v in slots(s.arguments) if isinstance(p := found_at(v, steps[:i], e.query), int)}
    writes = [i for i, s in enumerate(steps) if s.tool == "api_fetch" and s.success and str(s.arguments.get("method") or "GET").upper() in WRITES]
    credited, stack = set(writes), list(writes)
    while stack:
        for p in parents.get(stack.pop(), ()):
            if p not in credited:
                credited.add(p)
                stack.append(p)
    grounded = []
    for i in writes:
        values = [v for _, v in slots(steps[i].arguments)]
        hits = [found_at(v, steps[:i], e.query) is not None for v in values]
        if values and sum(hits) >= len(hits) / 2:
            grounded.append(i)
    return credited, grounded


def main():
    branches = defaultdict(lambda: defaultdict(list))  # state -> next action -> outcomes per episode
    n_eps = 0
    for run in sorted(p for p in Path("runs").iterdir() if p.is_dir()):
        for path in sorted(run.rglob("trajectory.json")):
            task = task_of(path, run)
            e = episode(path, task) if task else None
            if not e or not e.steps:
                continue
            n_eps += 1
            result = path.parent / "result.json"
            partial = json.loads(result.read_text()).get("partial_credit") if result.exists() else None
            credited, grounded = analyse(e)
            template = lift_query(e.query)[0]
            fetches = [i for i, s in enumerate(e.steps) if s.tool == "api_fetch" and s.success]
            seen = set()
            for n, i in enumerate(fetches):
                history = tuple(e.steps[j].action for j in fetches[max(0, n - 2):n])
                state, action = (template, history), e.steps[i].action
                if (state, action) in seen:
                    continue
                seen.add((state, action))
                later = [g for g in grounded if g >= i]
                branches[state][action].append({
                    "credited": i in credited,
                    "grounded": bool(later),
                    "calls": (later[0] - i) if later else None,
                    "partial": partial,
                    "flow": workflow(task),
                })

    compared = disagree = 0
    rows = []
    for state, nexts in branches.items():
        if READS_ONLY:
            nexts = {a: o for a, o in nexts.items() if a.startswith("GET ")}
        if len(nexts) < 2 or sum(len(v) for v in nexts.values()) < MIN_EPISODES:
            continue
        compared += 1
        popular = max(nexts, key=lambda a: (len(nexts[a]), a))
        credit = max(nexts, key=lambda a: (sum(o["credited"] for o in nexts[a]), len(nexts[a]), a))
        if popular != credit:
            disagree += 1
            rows.append((state, popular, credit, nexts[popular], nexts[credit]))

    def summary(outs):
        g = [o["grounded"] for o in outs]
        c = [o["calls"] for o in outs if o["calls"] is not None]
        p = [o["partial"] for o in outs if o["partial"] is not None]
        return sum(g) / len(g), (st.median(c) if c else float("nan")), (st.mean(p) if p else float("nan"))

    print(("reads only; " if READS_ONLY else "") + f"episodes={n_eps}; states with >= {MIN_EPISODES} episodes and >= 2 branches: {compared}; "
          f"popular != most-credited: {disagree} ({disagree / max(compared, 1):.0%})")
    pop = [o for r in rows for o in r[3]]
    cre = [o for r in rows for o in r[4]]
    for name, outs in (("popular branch", pop), ("credited branch", cre)):
        g, c, p = summary(outs)
        print(f"  {name:16s} episodes={len(outs):5d} grounded write follows={g:.0%} median calls to it={c:.0f} mean partial={p:.2f}")
    better = sum(summary(r[4])[0] > summary(r[3])[0] for r in rows)
    worse = sum(summary(r[4])[0] < summary(r[3])[0] for r in rows)
    print(f"  per state: credited branch reaches a grounded write more often in {better}, less often in {worse}, tied in {len(rows) - better - worse}")
    flows = Counter(o["flow"] for r in rows for o in r[3] + r[4])
    print(f"  workflows involved: {len(flows)}; top: {flows.most_common(5)}")
    print("\nExamples (largest states):")
    for state, popular, credit, po, co in sorted(rows, key=lambda r: -(len(r[3]) + len(r[4])))[:8]:
        gp, _, pp = summary(po)
        gc, _, pc = summary(co)
        hist = " > ".join(a.split(" ", 1)[1].split("/", 1)[0] + "/" + a.rsplit("/", 1)[-1] for a in state[1]) or "start"
        print(f"  [{po[0]['flow']}] after {hist}")
        print(f"     popular  {popular[:70]:70s} n={len(po):3d} grounded={gp:.0%} partial={pp:.2f}")
        print(f"     credited {credit[:70]:70s} n={len(co):3d} grounded={gc:.0%} partial={pc:.2f}")


if __name__ == "__main__":
    main()
