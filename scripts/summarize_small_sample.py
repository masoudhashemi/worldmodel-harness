"""Read-only diagnostic extraction; retain full original run artifacts."""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from worldmodel_harness.curve_report import runtime_diagnostics


def summarize(manifest_path):
    root = Path.cwd()
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for arm in manifest["arms"]:
        report = json.loads((root / "runs" / arm["run_id"] / "summary.json").read_text())
        connection = sqlite3.connect(f"file:{root / arm['snapshot']}?mode=ro", uri=True)
        claims = [
            json.loads(body)
            for (body,) in connection.execute(
                "SELECT c.body FROM claims c JOIN "
                "(SELECT id, MAX(version) v FROM claims GROUP BY id) latest "
                "ON c.id=latest.id AND c.version=latest.v"
            )
        ]
        connection.close()
        row = {
            **arm,
            "claims_by_relation_status": dict(
                Counter(c["relation"] + ":" + c["status"] for c in claims)
            ),
            "response_rule_hypotheses": sum(
                bool(c["payload"].get("response_rule"))
                for c in claims
                if c["relation"].startswith("possible_")
            ),
            "prospective_tests": dict(
                Counter(
                    t["result"] for c in claims for t in c["payload"].get("prospective_tests", [])
                )
            ),
            "tasks": [],
        }
        for task in report["results"]:
            audit = task.get("decision_audit", [])
            semantic = task.get("semantic_learning", {})
            retrievals = task.get("retrievals", [])
            row["tasks"].append(
                {
                    "task": task["task_name"],
                    "grade": task["task_completed_correctly"],
                    "finished": task["finished"],
                    "error": task.get("error_type"),
                    "completion": task.get("completion_evidence_status", "not_instrumented"),
                    "goal_checks": dict(Counter(c["result"] for c in task.get("goal_checks", []))),
                    "environment_calls": task["ledger"]["environment_calls"],
                    "runtime": runtime_diagnostics([task]),
                    "decision_invocations": sum(r["invoked"] for r in audit),
                    "decision_blocks": dict(
                        Counter(g for r in audit if not r["invoked"] for g in r["gate"]["gaps"])
                    ),
                    "injected_claims": sorted(
                        {c for r in retrievals for c in r.get("injected_claims", [])}
                    ),
                    "retrievals_with_claims": sum(
                        bool(r.get("injected_claims")) for r in retrievals
                    ),
                    "retrieval_count": len(retrievals),
                    "retrieval_gaps": dict(
                        Counter(g for r in retrievals for g in r.get("gaps", []))
                    ),
                    "hypothesis_context_calls": sum(
                        bool(r.get("hypothesis_chars")) for r in retrievals
                    ),
                    "semantic_hypotheses": len(semantic.get("hypothesis_ids", [])),
                    "semantic_reviewed": len(semantic.get("reviewed_ids", [])),
                    "semantic_error": semantic.get("error_type"),
                    "semantic_tokens": sum(semantic.get("token_usage", {}).values())
                    + sum(semantic.get("review", {}).get("token_usage", {}).values()),
                    "semantic_seconds": task["ledger"].get("adaptation_seconds", 0),
                    "artifact_invocations": task.get("artifact_invocations", 0),
                    "artifact_reuse": task.get("artifact_reuse", []),
                    "refresh_calls": len(task.get("refresh_audit", [])),
                    "information_calls": task.get("information_invocations", 0),
                }
            )
        rows.append(row)
    return {
        "complete": manifest["complete"],
        "arms": rows,
        "boundary_update": manifest.get("boundary_update"),
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
