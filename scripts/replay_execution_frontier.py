"""Audit wrapper resolution on saved trajectories; no model or environment calls."""

import argparse
import json
from collections import Counter
from pathlib import Path

from worldmodel_harness.execution_state import advance_frontier, environment_actions
from worldmodel_harness.knowledge import Episode
from worldmodel_harness.store import KnowledgeStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    knowledge = KnowledgeStore(args.checkpoint, frozen=True)
    results = []
    for trial in json.loads(args.summary.read_text())["results"]:
        evidence = KnowledgeStore(Path("runs") / trial["run_id"] / "evidence.sqlite", frozen=True)
        episodes = [
            Episode.model_validate_json(row[0])
            for row in evidence.db.execute("SELECT body FROM episodes ORDER BY rowid")
        ]
        current, completed = None, set()
        counts = Counter()
        trace = []
        for episode in episodes:
            if not episode.operation.startswith("tool_call:"):
                continue
            before = current
            previous_completed = set(completed)
            actions = environment_actions(episode, evidence)
            current = advance_frontier(
                knowledge, episode.scope, evidence, episode, current, completed
            )
            assert current is None or not current.startswith("tool_call:")
            if not any(
                a.outcome == "success" and a.operation not in ("api_search", "lookup_knowledge")
                for a in actions
            ):
                assert current == before and completed == previous_completed
                counts["non_advancing_verified"] += 1
            counts["wrappers"] += 1
            counts["linked_environment_actions"] += len(actions)
            counts["frontier_changes"] += current != before
            trace.append(
                {
                    "wrapper": episode.episode_id,
                    "before": before,
                    "after": current,
                    "actions": [
                        {"id": a.episode_id, "operation": a.operation, "outcome": a.outcome}
                        for a in actions
                    ],
                }
            )
        results.append(
            {
                "run_id": trial["run_id"],
                "task": trial["task"],
                "iteration": trial["iteration"],
                "counts": dict(counts),
                "trace": trace,
            }
        )
        evidence.close()
    knowledge.close()
    args.output.write_text(
        json.dumps(
            {
                "note": "Uses final checkpoint contracts for identity resolution; "
                "not a reconstruction of historical retrieval or reward.",
                "results": results,
            },
            indent=2,
        )
    )
    print(dict(sum((Counter(r["counts"]) for r in results), Counter())))


if __name__ == "__main__":
    main()
