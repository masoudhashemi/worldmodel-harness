"""Offline go/no-go for the lifted experience graph (world-model-v3.md, Step 1).

Usage: replay_lifted.py <out.json> <eval-tasks.json> <test-run-dir>[,...] <train-run-dir> [...]

Learns next-action counts and argument sources from training trajectories (eval tasks
excluded), then replays the eval tasks' trajectories from the test runs, predicting each
step from the steps before it. No grades are read. Reports:
- how similar each eval query is to its nearest training query;
- top-1/top-3 next-action accuracy, globally and conditioned on similar tasks;
- how often a correctly predicted api_fetch can be filled in for the new task;
- accuracy against confidence (Wilson lower bound), for gating;
- api_search calls before an endpoint the training data already knew.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from worldmodel_harness.experience import Model, lcb, load, occurrence
from worldmodel_harness.lifting import Episode, fill, lift_query, slots, source

NEIGHBOURS = 3
WORD = re.compile(r"[a-z{}]+")


def words(query: str) -> set[str]:
    return set(WORD.findall(lift_query(query)[0].lower()))


def evaluate(test: list[Episode], train: list[Episode]) -> dict:
    model = Model(train)
    train_words = {e.task: words(e.query) for e in train}
    by_task = defaultdict(list)
    for e in train:
        by_task[e.task].append(e)
    report = {"similarity": [], "next": Counter(), "fill": Counter(), "sources": Counter(), "gate": defaultdict(Counter)}
    searches = Counter()
    known = {a for counts in model.next.values() for a, n in counts.items() if n >= 2}
    for e in test:
        mine = words(e.query)
        near = sorted(train_words, key=lambda t: -len(mine & train_words[t]) / len(mine | train_words[t]))[:NEIGHBOURS]
        sim = len(mine & train_words[near[0]]) / len(mine | train_words[near[0]]) if near else 0.0
        report["similarity"].append((e.task, round(sim, 2), near[0] if near else ""))
        local = Model([x for t in near for x in by_task[t]])
        bindings = lift_query(e.query)[1]
        for i, step in enumerate(e.steps):
            kind = "fetch_ok" if step.tool == "api_fetch" and step.success else "other"
            for name, m in [("global", model), ("similar_tasks", local)]:
                ranked, hits, total = m.predict(e.steps, i)
                if name == "similar_tasks" and not ranked:
                    ranked, hits, total = model.predict(e.steps, i)
                report["next"][(name, kind, "n")] += 1
                report["next"][(name, kind, "top1")] += bool(ranked) and ranked[0] == step.action
                report["next"][(name, kind, "top3")] += step.action in ranked
                if name == "global":
                    bucket = min(int(lcb(hits, total) * 5), 4) / 5
                    report["gate"][bucket]["n"] += 1
                    report["gate"][bucket]["right"] += bool(ranked) and ranked[0] == step.action
            if step.tool == "api_search":
                later = next((s.action for s in e.steps[i + 1 :] if s.tool == "api_fetch"), None)
                searches["total"] += 1
                searches["before_known_endpoint"] += later in known
            if kind != "fetch_ok":
                continue
            for slot, value in slots(step.arguments):
                report["sources"][source(value, bindings, e.steps[:i], e.query).split("(")[0]] += 1
            ranked, _, _ = model.predict(e.steps, i)
            if not ranked or ranked[0] != step.action:
                continue
            all_right = True
            for slot, value in slots(step.arguments):
                term = model.term(step.action, slot, lift_query(e.query)[0], occurrence(e.steps, i))
                right = fill(term, bindings, e.steps[:i], e.query) == value
                report["fill"][(term.split("(")[0], "n")] += 1
                report["fill"][(term.split("(")[0], "right")] += right
                all_right &= right
            report["fill"][("call", "n")] += 1
            report["fill"][("call", "right")] += all_right
    sims = [s for _, s, _ in report["similarity"]]
    return {
        "episodes": {"train": len(train), "test": len(test)},
        "similarity": {
            "median": median(sims) if sims else 0,
            "at_least_0.8": sum(s >= 0.8 for s in sims),
            "at_least_0.6": sum(s >= 0.6 for s in sims),
            "per_episode": sorted(set(report["similarity"]), key=lambda x: -x[1]),
        },
        "next_action": {"|".join(map(str, k)): v for k, v in sorted(report["next"].items())},
        "fill": {"|".join(k): v for k, v in sorted(report["fill"].items())},
        "test_sources": dict(report["sources"]),
        "gate_by_lcb": {str(k): dict(v) for k, v in sorted(report["gate"].items())},
        "searches": dict(searches),
    }


def main():
    out, tasks = Path(sys.argv[1]), set(json.loads(Path(sys.argv[2]).read_text())["tasks"])
    test_runs = [Path(p) for p in sys.argv[3].split(",")]
    train_runs = [Path(p) for p in sys.argv[4:] if Path(p) not in test_runs]
    test = load(test_runs, lambda t: t in tasks)
    train = load(train_runs, lambda t: t not in tasks)
    result = evaluate(test, train)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "similarity"}, indent=1))
    print(json.dumps({k: v for k, v in result["similarity"].items() if k != "per_episode"}))


if __name__ == "__main__":
    main()
