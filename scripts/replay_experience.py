"""Evaluate offline experience retrieval using disjoint evidence databases.

Reads only episodes, never summaries, graders, or claims. Test transitions never
enter training memory. The query is built before the test response is observed.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from worldmodel_harness.experience import ExperienceMemory, PrefixExperience
from worldmodel_harness.knowledge import Episode


def load(path):
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        return [
            Episode.model_validate_json(row[0])
            for row in db.execute("SELECT body FROM episodes ORDER BY rowid")
        ]


def task_keys(episodes):
    return {
        (e.scope.environment, e.scope.version, e.scope.role, e.scope.tenant, e.task_id)
        for e in episodes
    }


def replay(episodes, memory, *, training):
    prefixes = {}
    seen = set()
    rows = []
    for episode in episodes:
        if episode.origin != "environment" or episode.environment_called is False:
            continue
        key = (episode.scope.model_dump_json(), episode.task_id)
        event_key = (key, episode.event_id)
        if event_key in seen:
            continue
        seen.add(event_key)
        prefix = prefixes.setdefault(key, PrefixExperience(episode.scope, episode.task_id))
        query = prefix.query(episode.operation, episode.arguments)
        report = memory.retrieve(query, limit=3) if query and not training else None
        # Response enters state only after retrieval has finished.
        transition = prefix.observe(episode, query)
        if training:
            if transition:
                memory.append(transition)
            continue
        row = {
            "episode_id": episode.episode_id,
            "operation": episode.operation,
            "query_available": query is not None,
            "observed_outcome": episode.outcome,
        }
        if report:
            row["retrieval"] = report.model_dump(mode="json")
            exact = [m for m in report.matches if m.match_kind == "same_observed_structure"]
            # A diagnostic about API outcome only, not semantic correctness or policy value.
            if exact and report.independent_tasks >= 2:
                labels = ("success", "error", "blocked", "unknown")
                probabilities = {
                    label: report.outcome_counts.get(label, 0) / sum(report.outcome_counts.values())
                    for label in labels
                }
                row["outcome_probabilities"] = probabilities
                row["brier_score"] = sum(
                    (p - int(episode.outcome == label)) ** 2 for label, p in probabilities.items()
                )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = args.train + args.test
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    train = [e for p in args.train for e in load(p)]
    test = [e for p in args.test for e in load(p)]
    if task_keys(train) & task_keys(test):
        parser.error("Training and test task instances must be disjoint")
    if {e.event_id for e in train} & {e.event_id for e in test}:
        parser.error("Training and test evidence events overlap")
    memory = ExperienceMemory()
    replay(train, memory, training=True)
    rows = replay(test, memory, training=False)
    scored = [r["brier_score"] for r in rows if "brier_score" in r]
    report = {
        "mode": "offline_structural_experience",
        "limitations": [
            "Not semantic state equivalence or task-quality evaluation",
            "No inferred business predicates or causal preconditions",
            "No test transitions added to training memory",
        ],
        "input_sha256": hashes,
        "training_transitions": len(memory.transitions),
        "test_calls": len(rows),
        "resolved_queries": sum(r["query_available"] for r in rows),
        "calls_with_history": sum(bool(r.get("retrieval", {}).get("matches")) for r in rows),
        "scored_calls": len(scored),
        "mean_outcome_brier": sum(scored) / len(scored) if scored else None,
        "results": rows,
    }
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.resolve())]:
            raise RuntimeError("Evidence changed during replay")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
