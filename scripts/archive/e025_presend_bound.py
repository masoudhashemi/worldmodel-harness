"""E025 offline (handoff 6.4): would a pre-send check catch wrong write arguments?

For every write call (POST, PUT, PATCH, DELETE) in the E022 v2 and v3 trials, each short
argument is compared with the value the graph resolves from that run's own query and
earlier responses, when the graph is confident: the first level that knows the slot
has a non-Agent top source with at least MIN_SUPPORT episodes and MIN_SHARE of them.
The graph is the learning arm's, as it stood before the task's first trial. Grades are
used only to diagnose: does a disagreement name a value that a failed check wanted?
"""

import json
import sys
from collections import Counter
from pathlib import Path

from worldmodel_harness.experience import Model, load, occurrence, workflow
from worldmodel_harness.lifting import episode, fill, lift_query, slots

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e022-learning-stream")
MIN_SUPPORT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MIN_SHARE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
# "strict": only query literals and anchored response fields (Resp(x|key=..) or
# Resp(x|key@query)) from a different endpoint, never from search docs.
STRICT = len(sys.argv) > 4 and sys.argv[4] == "strict"


def specific(term: str, action: str) -> bool:
    if term.startswith("Query("):
        return True
    head, _, origin = term.partition(" <- ")
    anchor = head.partition("|")[2].rstrip(")")
    return term.startswith("Resp(") and anchor not in ("", "?") and origin not in (action, "api_search")
WRITES = ("POST", "PUT", "PATCH", "DELETE")


def confident(graph: Model, template, action, slot, nth):
    for key in ((template, action, slot, nth), (template, action, slot), (None, action, slot)):
        if terms := graph.terms.get(key):
            term, n = terms.most_common(1)[0]
            total = sum(terms.values())
            if term == "Agent" or n < MIN_SUPPORT or n < MIN_SHARE * total:
                return None
            if STRICT and not specific(term, action):
                return None
            return term
    return None


def check(graph: Model, path: Path, task: str, failed: list[str]) -> Counter:
    c = Counter()
    e = episode(path, task)
    if not e:
        return c
    template, bindings = lift_query(e.query)
    for i, step in enumerate(e.steps):
        if step.tool != "api_fetch" or str(step.arguments.get("method", "GET")).upper() not in WRITES:
            continue
        nth = occurrence(e.steps, i)
        for slot, value in slots(step.arguments):
            c["slots"] += 1
            term = confident(graph, template, step.action, slot, nth)
            if term is None:
                continue
            want = fill(term, bindings, e.steps[:i], e.query)
            if want is None:
                continue
            c["confident"] += 1
            if want == value:
                c["agree"] += 1
                continue
            c["disagree"] += 1
            if any(want in f and value not in f for f in failed):
                c["disagree_matches_failed_check"] += 1
            c[f"example:{step.action} {slot} {term[:50]} got={value[:30]} want={want[:30]}"] += 1
    return c


def main():
    rows = json.loads((OUT / "summary.json").read_text())["results"]
    tasks = json.loads((OUT / "tasks.json").read_text())["tasks"]
    flows = {workflow(t) for t in tasks}
    others = [p for p in Path("runs").iterdir() if p.is_dir() and p.resolve() != OUT.resolve()]
    graph = Model(load(others, lambda t: workflow(t) not in flows))
    totals = {True: Counter(), False: Counter()}  # strict pass -> counts
    done = set()
    for row in rows:
        task = row["task"]
        if task not in done and task.rsplit("~", 1)[-1] in ("v2", "v3"):
            done.add(task)
            for other in (r for r in rows if r["task"] == task):
                trial = OUT / "trajectories" / task / f"{other['arm']}-r{other['round']}"
                if not (trial / "result.json").exists():
                    continue
                result = json.loads((trial / "result.json").read_text())
                failed = [json.dumps(d.get("params")) for d in result.get("grading_details", []) if not d.get("passed")]
                totals[other["strict_eval"]].update(check(graph, trial / "trajectory.json", task, failed))
        if row["arm"] == "learning":
            path = OUT / "trajectories" / task / f"learning-r{row['round']}" / "trajectory.json"
            if (e := episode(path, task)) and e.steps:
                graph.add(e)

    print(f"confident = top source non-Agent, support >= {MIN_SUPPORT}, share >= {MIN_SHARE}, strict={STRICT}")
    for passed, c in totals.items():
        print(
            f"{'passed' if passed else 'failed'} trials: write slots={c['slots']} confident={c['confident']} "
            f"agree={c['agree']} disagree={c['disagree']} "
            f"(disagreements naming a value a failed check wanted: {c['disagree_matches_failed_check']})"
        )
        for k, n in sorted(((k, n) for k, n in c.items() if k.startswith("example:")), key=lambda x: -x[1])[:12]:
            print(f"    {n:3d} {k[8:]}")


if __name__ == "__main__":
    main()
