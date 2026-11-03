"""E035 offline: can the graph learn which messages to show from what followed them?

Every message a past arm showed is found in its trajectory (the tool result it was added
to), with its kind. Its outcome is measured from tool calls and results only, over the
next WINDOW calls, against the same task's trial in the arm without that kind of message,
from the first point at which that trial was in the same situation: the same last
successful action before the call, the same call (action and success), and the call being
a repeat of an earlier one or not. (Matching on the state alone compared messages shown in
a loop with control points outside one.)
- new: successful actions not made before in the trial (progress);
- repeat: calls identical to an earlier call of the trial (going round);
- follow: share of the actions the message named that were then made successfully;
- found: share of the values written in the rest of the run that were in an earlier read
  rather than made up;
- left: calls in the rest of the run.
Each is reported as the difference against the matched control point.

Grades are used only to check the measure: does the self reward separate trials where
showing messages raised partial credit from trials where it lowered it? Part 2 replays the
messages in the order they ran and learns, per (query template, kind) with the kind as
fallback, the mean self reward of earlier messages only. A message is kept while its key
has fewer than MIN_SEEN earlier messages or a positive mean.

Usage: e035_policy_offline.py
"""

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from worldmodel_harness.automationbench.domain import HttpDomain, endpoint
from worldmodel_harness.automationbench.traces import episode
from worldmodel_harness.core.lifting import lift_query, origin

D = HttpDomain()
WINDOW = 10
MIN_SEEN = 3
# (run, arm that showed the kind, arm without it)
PAIRS = {
    "plan": [("e022-learning-stream", "learning", "fixes")],
    "fix": [("e022-learning-stream", "fixes", "baseline")],
    "pointer": [("e024-search-repeats", "repeats", "fixes")],
    "sources": [(r, "sources", "repeats") for r in ("e026-learned-sources", "e027-chained-sources", "e029-hint-once")],
    "unmade": [(r, "sources", "repeats") for r in ("e026-learned-sources", "e027-chained-sources", "e029-hint-once")],
}
MARKERS = {
    "plan": "<...> marks a value to take from this task",
    "fix": "[Fix memory]",
    "pointer": "same search as your search",
    "sources": "values that were written came from reads",
    "unmade": "made these calls successfully",
}
CALL = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE) (https://\S+)")


def shown(path: Path) -> list[tuple[int, str, str]]:
    """(index of the tool result, kind, text added) for every message in a trajectory."""
    found, i = [], -1
    for group in json.loads(path.read_text()):
        for m in group:
            if m.get("role") != "tool":
                continue
            i += 1
            text = str(m.get("content"))
            for kind, marker in MARKERS.items():
                if marker in text:
                    found.append((i, kind, text))
    return found


def named(text: str) -> set[str]:
    clean = re.sub(r"<[^>]*>", "id_1", text.replace("\\n", "\n"))
    return {endpoint({"method": m, "url": u.rstrip(".,)")}) for m, u in CALL.findall(clean)}


def situation(steps, i) -> tuple:
    """The state before call i, the call, and whether it repeats an earlier call."""
    state = next((D.action(s) for s in reversed(steps[:i]) if s.success), "start")
    key = json.dumps([steps[i].tool, steps[i].arguments], sort_keys=True)
    repeat = any(json.dumps([s.tool, s.arguments], sort_keys=True) == key for s in steps[:i])
    return state, D.action(steps[i]), steps[i].success, repeat


def outcome(e, i: int, names: set[str]) -> dict:
    before, after = e.steps[: i + 1], e.steps[i + 1 : i + 1 + WINDOW]
    done = {D.action(s) for s in before if s.success}
    calls = {json.dumps([s.tool, s.arguments], sort_keys=True) for s in before}
    new = repeat = 0
    for s in after:
        key = json.dumps([s.tool, s.arguments], sort_keys=True)
        repeat += key in calls
        calls.add(key)
        if s.success and D.action(s) not in done:
            new += 1
            done.add(D.action(s))
    made = {D.action(s) for s in after if s.success}
    values = []
    for k, s in enumerate(e.steps[i + 1 :]):
        if s.success and D.writes(s):
            values += [origin(v, e.steps[: i + 1 + k], e.query, D) != "Agent" for _, v in D.slots(s)]
    return {
        "new": new,
        "repeat": repeat,
        "follow": len(names & made) / len(names) if names else None,
        "found": sum(values) / len(values) if values else None,
        "left": len(e.steps) - i - 1,
    }


def matched(control, wanted: tuple, nth: int) -> int | None:
    """The control trial's call in the same situation, the same occurrence (or its last)."""
    found = [j for j in range(len(control.steps)) if situation(control.steps, j) == wanted]
    return found[min(nth, len(found) - 1)] if found else None


def partial(path: Path) -> float | None:
    result = path.parent / "result.json"
    return json.loads(result.read_text()).get("partial_credit") if result.exists() else None


def spearman(x, y) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        for n, k in enumerate(order):
            r[k] = n
        return r

    rx, ry = ranks(x), ranks(y)
    return st.correlation(rx, ry) if len(x) > 2 else float("nan")


def main():
    rows = []  # one per message with a matched control point
    for kind, pairs in PAIRS.items():
        for run, arm, control in pairs:
            for path in sorted(Path("runs", run, "trajectories").glob(f"*/{arm}-r0/trajectory.json")):
                task = path.parent.parent.name
                other = path.parent.parent / f"{control}-r0" / "trajectory.json"
                e, c = episode(path, task), episode(other, task) if other.exists() else None
                pa, pc = partial(path), partial(other)
                if not e or not c or pa is None or pc is None:
                    continue
                for i, k, text in shown(path):
                    if k != kind or i >= len(e.steps):
                        continue
                    here = situation(e.steps, i)
                    j = matched(c, here, sum(situation(e.steps, k) == here for k in range(i)))
                    if j is None:
                        continue
                    added = text.split("\n\n")[-2] if kind == "plan" else text.split(MARKERS[kind], 1)[-1]
                    names = named(added) if kind != "pointer" else set()
                    oa, oc = outcome(e, i, names), outcome(c, j, names)
                    d = {m: (oa[m] - oc[m]) if oa[m] is not None and oc[m] is not None else None for m in oa}
                    rows.append({
                        "kind": kind, "run": run, "task": task, "trial": str(path), "time": path.stat().st_mtime,
                        "template": lift_query(e.query)[0], "d": d,
                        "graded": pa - pc,
                    })

    print(f"messages with a matched control point: {len(rows)}")
    units = defaultdict(list)  # (trial, kind) -> its messages; each trial counts once, as in the graph
    for r in rows:
        units[(r["trial"], r["kind"])].append(r)

    def value(rs, reward):
        v = [reward(r["d"]) for r in rs]
        v = [x for x in v if x is not None]
        return st.mean(v) if v else None

    print("\n1. By kind: mean difference against the matched control point")
    print(f"   {'kind':8s} {'msgs':>5s} {'trials':>6s} {'new':>6s} {'repeat':>7s} {'follow':>7s} {'found':>6s} {'left':>6s}")
    for kind in PAIRS:
        rs = [r for r in rows if r["kind"] == kind]
        cells = []
        for m in ("new", "repeat", "follow", "found", "left"):
            v = value(rs, lambda d, m=m: d[m])
            cells.append(f"{v:+.2f}" if v is not None else "-")
        print(f"   {kind:8s} {len(rs):5d} {len({r['trial'] for r in rs}):6d} " + " ".join(f"{c:>6s}" for c in cells))

    print("\n2. Candidate self rewards: do they track the graded paired delta? (grades for this check only)")
    for name, reward in REWARDS.items():
        pts = [(value(rs, reward), rs[0]["graded"], rs[0]["kind"]) for rs in units.values()]
        pts = [p for p in pts if p[0] is not None]
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        pos = [g for v, g, _ in pts if v > 0]
        neg = [g for v, g, _ in pts if v < 0]
        per_kind = ", ".join(
            f"{k} {spearman([p[0] for p in pts if p[2] == k], [p[1] for p in pts if p[2] == k]):+.2f}"
            for k in PAIRS if sum(p[2] == k for p in pts) > 4
        )
        print(f"   {name:9s} n={len(pts):3d} rho={spearman(x, y):+.2f} | reward>0: graded {st.mean(pos):+.3f} "
              f"(up {sum(g > 0 for g in pos)}/down {sum(g < 0 for g in pos)}) | reward<0: graded "
              f"{st.mean(neg) if neg else float('nan'):+.3f} (up {sum(g > 0 for g in neg)}/down {sum(g < 0 for g in neg)}) | {per_kind}")

    print("\n3. Learned policy, earlier trials only: per (query template, kind), falling back to the kind;")
    print(f"   keep while fewer than {MIN_SEEN} earlier trials or their mean reward is > 0. Graded delta of what it keeps/suppresses:")
    for name, reward in REWARDS.items():
        stats, kept, cut, cut_tasks = defaultdict(list), [], [], []
        for (trial, kind), rs in sorted(units.items(), key=lambda u: u[1][0]["time"]):
            r = rs[0]
            exact, fallback = stats[(r["template"], kind)], stats[("*", kind)]
            seen = exact if len(exact) >= MIN_SEEN else fallback
            keep = len(seen) < MIN_SEEN or st.mean(seen) > 0
            (kept if keep else cut).append(r["graded"])
            if not keep:
                cut_tasks.append(f"{kind}:{r['task'].split('.', 1)[1]}{r['graded']:+.2f}")
            v = value(rs, reward)
            if v is not None:
                exact.append(v)
                fallback.append(v)
        line = f"   {name:9s} kept n={len(kept):3d} {st.mean(kept):+.3f} (up {sum(g > 0 for g in kept)}/down {sum(g < 0 for g in kept)})"
        if cut:
            line += f" | suppressed n={len(cut):3d} {st.mean(cut):+.3f} (up {sum(g > 0 for g in cut)}/down {sum(g < 0 for g in cut)})"
        print(line)
        if cut_tasks:
            print("      suppressed: " + ", ".join(cut_tasks[:14]))


REWARDS = {
    "progress": lambda d: d["new"] - d["repeat"],
    "found": lambda d: d["found"],
    "shorter": lambda d: -d["left"],
}


if __name__ == "__main__":
    main()
