"""Learning curve of the transition graph on recurring workflows (world-model-v3.md, §9).

Usage: replay_curve.py <out.json> <eval-tasks.json> <variant-run> <original-run> [...]

The graph learns from background trajectories (other tasks) plus k earlier runs of the
same workflow, then replays the next run step by step. Settings:
- variants: earlier renamed variants v1..vj predict v(j+1) (nothing can be copied);
- renamed: k original runs predict renamed v1;
- same: k original runs predict original run k+1 (identical literals).
Per successful api_fetch, the graph's prediction is resolved against that run's own
query and earlier responses (the state). Reports:
- top1: the next endpoint is predicted;
- traceable: share of traceable arguments (from the query, or from a response record the
  query identifies) that the graph fills in right;
- coverage: share of calls the graph proposes in full (no argument left to the agent);
- precision: share of proposed calls that are exactly right.
No grades are read.
"""

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from worldmodel_harness.experience import Model, load, occurrence, workflow
from worldmodel_harness.lifting import fill, lift_query, slots, source


def score(model: Model, test, counts: Counter):
    template, bindings = lift_query(test.query)
    for i, step in enumerate(test.steps):
        if not (step.tool == "api_fetch" and step.success):
            continue
        counts["calls"] += 1
        ranked, _, _ = model.predict(test.steps, i)
        if not ranked or ranked[0] != step.action:
            continue
        counts["top1"] += 1
        proposed, right = True, True
        for slot, value in slots(step.arguments):
            _, got = model.resolve(
                step.action, slot, lambda t, earlier=test.steps[:i]: fill(t, bindings, earlier, test.query),
                template, occurrence(test.steps, i),
            )
            proposed &= got is not None
            right &= got == value
            own = source(value, bindings, test.steps[:i], test.query)
            if own != "Free" and "|?)" not in own:  # not chosen or selected by the agent
                counts["traceable_n"] += 1
                counts["traceable_right"] += got == value
        counts["proposed"] += proposed
        counts["proposed_right"] += proposed and right


def summary(counts: Counter) -> dict:
    def ratio(a, b):
        return round(counts[a] / counts[b], 3) if counts[b] else None

    return {
        "calls": counts["calls"],
        "top1": ratio("top1", "calls"),
        "traceable": ratio("traceable_right", "traceable_n"),
        "coverage": ratio("proposed", "calls"),
        "precision": ratio("proposed_right", "proposed"),
        "right_share": ratio("proposed_right", "calls"),
    }


def main():
    out, tasks = Path(sys.argv[1]), set(json.loads(Path(sys.argv[2]).read_text())["tasks"])
    variant_run, original_runs = Path(sys.argv[3]), [Path(p) for p in sys.argv[4:]]
    used = {variant_run, *original_runs}
    others = [p for p in Path("runs").iterdir() if p.is_dir() and p not in used]
    base = Model(load(others, lambda t: workflow(t) not in tasks))
    originals = defaultdict(list)
    for run in original_runs:  # in run order
        for e in load([run], lambda t: t in tasks):
            originals[e.task].append(e)
    variants = defaultdict(dict)
    for e in load([variant_run], lambda t: "~v" in t and workflow(t) in tasks):
        variants[workflow(e.task)][int(e.task.rsplit("~v", 1)[1])] = e

    def run(cases) -> dict:
        counts = Counter()
        for seen, test in cases:
            model = copy.deepcopy(base)
            for e in seen:
                model.add(e)
            score(model, test, counts)
        return summary(counts)

    result = {}
    for j in (0, 1, 2):
        cases = [([v[n] for n in range(1, j + 1)], v[j + 1]) for v in variants.values() if all(n in v for n in range(1, j + 2))]
        result[f"variants_seen_{j}"] = run(cases)
    for k in (0, 1, 4, 8):
        result[f"originals_{k}_renamed"] = run([(originals[t][:k], v[1]) for t, v in variants.items() if len(originals[t]) >= k and 1 in v])
        result[f"originals_{k}_same"] = run([(seq[:k], seq[k]) for seq in originals.values() if len(seq) > k])
    for name, row in result.items():
        print(f"{name:24s} {row}")
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
