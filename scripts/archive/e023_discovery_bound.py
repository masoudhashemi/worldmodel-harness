"""E023 go/no-go (handoff 6.1): how many search turns could docs on the first search remove?

Replays the E022 stream in its run order, with the learning arm's graph as it stood
before each trial. A search-only turn after a run's first search turn is avoidable when
every endpoint Gemma fetched next (from its next fetching turn up to the following
search turn) was already known to the graph for the task's template. Also reports how
many API docs a prefill would return per workflow, and their size, as a context check.
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from worldmodel_harness.experience import Model, load, workflow
from worldmodel_harness.fixes import endpoint
from worldmodel_harness.lifting import episode, lift_query

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e022-learning-stream")
TURN_SECONDS = 4.5


def known(graph, template: str) -> set[str]:
    """Endpoints this workflow's earlier runs fetched successfully."""
    found = {a for (t, a) in graph.arguments if t == template}
    for key, nexts in graph.fetches.items():
        if key[0] == template:
            found |= set(nexts) - {"end"}
    return found


def turns(path: Path) -> tuple[str, list[list[tuple[str, dict]]]]:
    groups = json.loads(path.read_text())
    messages = [m for g in groups for m in g if isinstance(m, dict)]
    query = next((str(m.get("content")) for m in messages if m.get("role") == "user"), "")
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        calls = []
        for b in m.get("blocks") or []:
            if b.get("kind") == "tool_call":
                try:
                    a = json.loads(b["arguments"])
                except (TypeError, ValueError):
                    a = {}
                calls.append((b["name"], a if isinstance(a, dict) else {}))
        out.append(calls)
    return query, out


def docs(path: Path) -> dict[str, int]:
    """Endpoint -> size in characters of its api_search result entry, from one trajectory."""
    groups = json.loads(path.read_text())
    found = {}
    for m in (m for g in groups for m in g if isinstance(m, dict)):
        if m.get("role") == "tool" and m.get("name") == "api_search":
            try:
                results = json.loads(str(m.get("content")).split("\n\n[", 1)[0])["results"]
            except (ValueError, KeyError, TypeError):
                continue
            for r in results:
                found[endpoint(r)] = len(json.dumps(r))
    return found


def score(calls_by_turn, knows: set[str]) -> dict:
    searching = [any(n == "api_search" for n, _ in c) for c in calls_by_turn]
    fetching = [[endpoint(a) for n, a in c if n == "api_fetch"] for c in calls_by_turn]
    first = next((i for i, s in enumerate(searching) if s), None)
    s = {"turns": len(calls_by_turn), "search_only": 0, "avoidable": 0, "unknown": []}
    if first is None:
        return s
    for i in range(first + 1, len(calls_by_turn)):
        if not calls_by_turn[i] or not all(n == "api_search" for n, _ in calls_by_turn[i]):
            continue
        s["search_only"] += 1
        j = next((j for j in range(i + 1, len(calls_by_turn)) if fetching[j]), None)
        if j is None:
            continue  # searched and never fetched again: no endpoint to credit
        after = []
        for k in range(j, len(calls_by_turn)):
            if k > j and searching[k]:
                break
            after += fetching[k]
        missing = [e for e in after if e not in knows]
        if missing:
            s["unknown"] += missing
        else:
            s["avoidable"] += 1
    return s


def main():
    rows = json.loads((OUT / "summary.json").read_text())["results"]
    tasks = json.loads((OUT / "tasks.json").read_text())["tasks"]
    flows = {workflow(t) for t in tasks}
    others = [p for p in Path("runs").iterdir() if p.is_dir() and p.resolve() != OUT.resolve()]
    graph = Model(load(others, lambda t: workflow(t) not in flows))  # as run_arm_eval.experience_graph
    per = defaultdict(list)  # (round, arm) -> scores
    doc_sizes = defaultdict(dict)  # workflow -> endpoint -> chars
    # Both arms of a task are judged against the graph as it stood just before that
    # task's learning trial: arm order rotates, so a later fixes trial must not see it.
    before = {}
    for row in rows:
        task, arm, rnd = row["task"], row["arm"], row["round"]
        traj = OUT / "trajectories" / task / f"{arm}-r{rnd}" / "trajectory.json"
        if not traj.exists():
            continue
        doc_sizes[workflow(task)].update(docs(traj))
        if arm == "learning":
            before[task] = known(graph, lift_query(turns(traj)[0])[0])
            if (e := episode(traj, task)) and e.steps:
                graph.add(e)
    for row in rows:
        task, arm, rnd = row["task"], row["arm"], row["round"]
        traj = OUT / "trajectories" / task / f"{arm}-r{rnd}" / "trajectory.json"
        variant = task.rsplit("~", 1)[-1]
        if arm in ("fixes", "learning") and variant in ("v2", "v3") and traj.exists():
            s = score(turns(traj)[1], before[task])
            s["task"], s["known"] = task, before[task]
            per[(variant, arm)].append(s)

    for (variant, arm), scores in sorted(per.items()):
        n = len(scores)
        av = [s["avoidable"] for s in scores]
        so = [s["search_only"] for s in scores]
        tu = [s["turns"] for s in scores]
        withk = sum(bool(s["known"]) for s in scores)
        print(
            f"{variant} {arm:8s} n={n} graph-known={withk}/{n} turns/task={st.mean(tu):.1f} "
            f"later search-only/task={st.mean(so):.2f} avoidable/task mean={st.mean(av):.2f} "
            f"median={st.median(av):.1f} share of turns={sum(av) / max(sum(tu), 1):.1%} "
            f"~{st.mean(av) * TURN_SECONDS:.1f}s/task; capped at 5/task mean={st.mean(min(a, 5) for a in av):.2f}"
        )
    print("\nAvoidable turns per task (fixes arm, v2+v3):")
    dist = defaultdict(int)
    for v in ("v2", "v3"):
        for s in per[(v, "fixes")]:
            dist[s["avoidable"]] += 1
    print("  " + ", ".join(f"{k}:{dist[k]}" for k in sorted(dist)))

    print("\nPrefill size per workflow (docs the graph knows at v3, chars):")
    sizes = []
    for s in per[("v3", "fixes")]:
        store = doc_sizes[workflow(s["task"])]
        have = [store[e] for e in s["known"] if e in store]
        sizes.append((len(s["known"]), len(have), sum(have)))
    if sizes:
        print(
            f"  endpoints known median={st.median(x[0] for x in sizes)} max={max(x[0] for x in sizes)}; "
            f"docs found={sum(x[1] for x in sizes)}/{sum(x[0] for x in sizes)}; "
            f"chars median={st.median(x[2] for x in sizes):.0f} max={max(x[2] for x in sizes)}"
        )


if __name__ == "__main__":
    main()
