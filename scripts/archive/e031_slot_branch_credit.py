"""E031 offline: does the branch taken at a state decide whether a later write slot is found or invented?

States: (query template, last successful fetch). At the first visit of a state in an
episode, the branch is the next action of any kind: "search" (any api_search), a failed
call ("fail <endpoint>"), or a successful fetch (its endpoint). For every write slot the
episode filled (e.g. the Gmail recipient), the slot is "found" when all of its values were
seen in an earlier read and "invented" when any was not; slots filled from the query are
skipped. A state separates on a slot when one branch's found rate has a Wilson lower bound
above another branch's upper bound. Tool calls and results only; partial credit and the
grader's expected values are diagnosis only.

Part 3 repeats part 2 with only earlier runs: trials in the order they ran, each judged
by the counts before it (no optimism from later runs).

Part 2 replays the trials of E022-E029 with leave-one-out counts: would a warning have fired
(the run took a branch with a bad record for a slot while a branch with a good record
existed), did the run then invent that slot, and does the good branch's service hold a
value the grader expected?
"""

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from worldmodel_harness.experience import WRITES, task_of, workflow
from worldmodel_harness.lifting import episode, lift_query, origin, slots

sys.path.insert(0, str(Path(__file__).parent))
from e026_source_bound import expected_values, holds

Z = 1.2816
GOOD, BAD = 0.35, 0.35  # good: found-rate lower bound >= GOOD; bad: upper bound < BAD
MIN_RUNS = 3
LIVE = ("e022-learning-stream", "e024-search-repeats", "e026-learned-sources", "e027-chained-sources", "e029-hint-once")


def bounds(k: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = k / n
    c = p + Z * Z / (2 * n)
    s = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    d = 1 + Z * Z / n
    return (c - s) / d, (c + s) / d


def label(step) -> str | None:
    if step.tool == "api_search":
        return "search"
    if step.tool == "api_fetch":
        return step.action if step.success else "fail " + step.action
    return None


def origins(e) -> dict:
    """Write slot -> set of reads its values came from (found values only)."""
    out = defaultdict(set)
    for i, s in enumerate(e.steps):
        if s.tool == "api_fetch" and s.success and str(s.arguments.get("method") or "GET").upper() in WRITES:
            for slot, value in slots(s.arguments):
                o = origin(value, e.steps[:i], e.query)
                if o not in ("Query", "Agent"):
                    out[(s.action, slot)].add(o)
    return out


def outcomes(e) -> dict:
    """Write slot -> (found?, index of the first write of the slot)."""
    found = {}
    for i, s in enumerate(e.steps):
        if s.tool != "api_fetch" or not s.success or str(s.arguments.get("method") or "GET").upper() not in WRITES:
            continue
        for slot, value in slots(s.arguments):
            o = origin(value, e.steps[:i], e.query)
            if o == "Query":
                continue
            key = (s.action, slot)
            ok, first = found.get(key, (True, i))
            found[key] = (ok and o != "Agent", first)
    return found


def visits(e) -> list[tuple[str, str, int]]:
    """(state's last fetch, branch, step index) at the first visit of each state."""
    seen, out, last = set(), [], "start"
    for i, s in enumerate(e.steps):
        if last not in seen:
            nxt = next((label(t) for t in e.steps[i:] if label(t)), None)
            if nxt:
                seen.add(last)
                out.append((last, nxt, i))
        if s.tool == "api_fetch" and s.success:
            last = s.action
    return out


def main():
    data = []  # (run, path, task, template, visits, outcomes)
    for run in sorted(p for p in Path("runs").iterdir() if p.is_dir()):
        for path in sorted(run.rglob("trajectory.json")):
            task = task_of(path, run)
            e = episode(path, task) if task else None
            if e and e.steps:
                data.append((run.name, path, task, lift_query(e.query)[0], visits(e), outcomes(e), origins(e)))

    # counts[(template, state, slot)][branch] = [found, invented]; hops per (template, state, slot)
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    hops = defaultdict(list)
    source = defaultdict(Counter)  # (template, slot) -> read -> episodes in which the slot came from it
    found_runs = Counter()  # (template, slot) -> episodes in which the slot was found
    for _, _, _, tpl, vs, outs, ors in data:
        for slot, reads in ors.items():
            found_runs[(tpl, slot)] += 1
            for r in reads:
                source[(tpl, slot)][r] += 1
        for state, branch, i in vs:
            for slot, (ok, first) in outs.items():
                if first > i:
                    counts[(tpl, state, slot)][branch][0 if ok else 1] += 1
                    hops[(tpl, state, slot)].append(first - i)

    # 1. Separation.
    keys = [k for k, b in counts.items() if len(b) >= 2 and sum(sum(v) for v in b.values()) >= MIN_RUNS]
    separating, rows = [], []
    for k in keys:
        b = counts[k]
        lo = {a: bounds(f, f + n)[0] for a, (f, n) in b.items()}
        hi = {a: bounds(f, f + n)[1] for a, (f, n) in b.items()}
        best = max(b, key=lambda a: lo[a])
        worst = min(b, key=lambda a: hi[a])
        if lo[best] > hi[worst]:
            popular = max(b, key=lambda a: sum(b[a]))
            separating.append(k)
            rows.append((k, best, worst, popular, b, st.median(hops[k])))
    flows = {workflow_of(tpl, data) for (tpl, _, _) in separating}
    print(f"1. (template, state, slot) with >= {MIN_RUNS} runs and >= 2 branches: {len(keys)}; separating: {len(separating)} "
          f"({len(separating) / max(len(keys), 1):.0%}) in {len(flows)} workflows")
    if rows:
        print(f"   median hops from the branch to the write: {st.median(r[5] for r in rows):.0f}; "
              f"popular branch is the worst-recorded one in {sum(r[3] == r[2] for r in rows)}, the best-recorded one in {sum(r[3] == r[1] for r in rows)}")
        kinds = Counter(("search" if r[2] == "search" else "fail" if r[2].startswith("fail") else "fetch") for r in rows)
        print(f"   worst-recorded branch kind: {dict(kinds)}")
        print("   largest separations:")
        for (tpl, state, slot), best, worst, popular, b, h in sorted(rows, key=lambda r: -sum(sum(v) for v in r[4].values()))[:10]:
            f = workflow_of(tpl, data)
            print(f"    [{f}] after {state.split(' ', 1)[-1][:45]} slot {slot[0].split(' ', 1)[1].split('/')[0]} {slot[1]} (hops~{h:.0f})")
            print(f"       best  {best[:60]:60s} found {b[best][0]}/{sum(b[best])}   worst {worst[:40]:40s} found {b[worst][0]}/{sum(b[worst])}   popular={'best' if popular == best else 'worst' if popular == worst else 'other'}")

    # 2. Warnings, leave-one-out, on the live trials.
    stats, per_flow = Counter(), defaultdict(Counter)
    for run, path, task, tpl, vs, outs, ors in data:
        if run not in LIVE:
            continue
        own = defaultdict(lambda: [0, 0])
        for state, branch, i in vs:
            for slot, (ok, first) in outs.items():
                if first > i:
                    own[(state, slot, branch)][0 if ok else 1] += 1
        warned = None
        for state, branch, i in vs:
            for slot in {k for (t, s, k) in counts if t == tpl and s == state}:
                b = counts[(tpl, state, slot)]
                adj = {a: [f - own[(state, slot, a)][0], n - own[(state, slot, a)][1]] for a, (f, n) in b.items()}
                taken = adj.get(branch, [0, 0])
                good = [a for a, (f, n) in adj.items() if a != branch and f + n >= MIN_RUNS and bounds(f, f + n)[0] >= GOOD]
                if sum(taken) >= MIN_RUNS and bounds(taken[0], sum(taken))[1] < BAD and good:
                    warned = (state, branch, slot, max(good, key=lambda a: bounds(*[adj[a][0], sum(adj[a])])[0]))
                    break
            if warned:
                break
        stats["trials"] += 1
        if not warned:
            continue
        state, branch, slot, alt = warned
        stats["warning"] += 1
        f = workflow(task)
        per_flow[f]["warning"] += 1
        ok = outs.get(slot, (None,))[0]
        stats["slot invented after warning"] += ok is False
        stats["slot found anyway"] += ok is True
        result = path.parent / "result.json"
        expected = expected_values(json.loads(result.read_text())) if result.exists() else set()
        if not alt.startswith(("search", "fail")):
            stats["best branch's service holds an expected value"] += holds(task, alt, expected)
            stats["best branch checkable"] += 1
        # The value-origin rule for the same slot, leave-one-out: which read held it in other runs.
        counts_src = Counter(source[(tpl, slot)])
        counts_src.subtract({r: 1 for r in ors.get(slot, ())})
        n_found = found_runs[(tpl, slot)] - (1 if ors.get(slot) else 0)
        reads = [r for r, k in counts_src.items() if k > 0 and bounds(k, n_found)[0] >= GOOD]
        if reads:
            read = max(reads, key=lambda r: counts_src[r])
            stats["value-origin read available"] += 1
            stats["value-origin read's service holds an expected value"] += holds(task, read, expected)
    print(f"\n2. Live trials (E022-E029, leave-one-out): {dict(stats)}")

    # 3. The same, with only earlier runs.
    later = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    src, nfound = defaultdict(Counter), Counter()
    stats, per_flow = Counter(), defaultdict(Counter)
    for run, path, task, tpl, vs, outs, ors in sorted(data, key=lambda d: d[1].stat().st_mtime):
        if run in LIVE:
            stats["trials"] += 1
            warned = None
            for state, branch, _ in vs:
                for slot in [k for (t, s, k) in later if t == tpl and s == state]:
                    b = later[(tpl, state, slot)]
                    taken = b.get(branch, [0, 0])
                    good = [a for a, (f, n) in b.items() if a != branch and f + n >= MIN_RUNS and bounds(f, f + n)[0] >= GOOD]
                    if sum(taken) >= MIN_RUNS and bounds(taken[0], sum(taken))[1] < BAD and good:
                        warned = slot
                        break
                if warned:
                    break
            if warned:
                stats["warning"] += 1
                per_flow[workflow(task)]["warning"] += 1
                ok = outs.get(warned, (None,))[0]
                stats["slot invented after warning"] += ok is False
                stats["slot found anyway"] += ok is True
                reads = [r for r, k in src[(tpl, warned)].items() if bounds(k, nfound[(tpl, warned)])[0] >= GOOD]
                if reads:
                    result = path.parent / "result.json"
                    expected = expected_values(json.loads(result.read_text())) if result.exists() else set()
                    stats["value-origin read available"] += 1
                    stats["value-origin read's service holds an expected value"] += holds(task, max(reads, key=lambda r: src[(tpl, warned)][r]), expected)
        for slot, reads in ors.items():
            nfound[(tpl, slot)] += 1
            for r in reads:
                src[(tpl, slot)][r] += 1
        for state, branch, i in vs:
            for slot, (ok, first) in outs.items():
                if first > i:
                    later[(tpl, state, slot)][branch][0 if ok else 1] += 1
    print(f"\n3. Live trials, earlier runs only: {dict(stats)}")
    print("   warnings by workflow: " + ", ".join(f"{f} {c['warning']}" for f, c in sorted(per_flow.items(), key=lambda x: -x[1]['warning'])[:12]))
    print("   warnings by workflow: " + ", ".join(f"{f} {c['warning']}" for f, c in sorted(per_flow.items(), key=lambda x: -x[1]['warning'])[:12]))


FLOWS: dict[str, str] = {}


def workflow_of(template, data):
    if template not in FLOWS:
        FLOWS[template] = next((workflow(t) for _, _, t, tpl, _, _, _ in data if tpl == template), "?")
    return FLOWS[template]


if __name__ == "__main__":
    main()
