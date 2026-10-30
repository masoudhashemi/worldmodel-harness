"""Learn the check-memory table from saved trajectories.

Reads <runs-dir>/trajectories/<task>/<arm>-*/trajectory.json for the chosen arms,
resolves each api_fetch to a documented endpoint using the contracts in --contracts,
and writes {"tasks": {task: {write_endpoint: {read_endpoint: distinct_targets}}}}.
Only tool calls are used; grades and results are never read.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from worldmodel_harness.automationbench import UPSTREAM_COMMIT
from worldmodel_harness.behavior import endpoint_contract
from worldmodel_harness.checks import learn_checks, trajectory_calls
from worldmodel_harness.knowledge import Scope
from worldmodel_harness.store import KnowledgeStore


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, action="append", required=True)
    parser.add_argument("--arms", nargs="+", default=["baseline"])
    parser.add_argument("--contracts", type=Path, required=True, help="Store holding endpoint contracts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    store = KnowledgeStore(args.contracts)
    scope = Scope(environment="automationbench", version=UPSTREAM_COMMIT)
    cache: dict[tuple[str, str], str | None] = {}

    def resolve(method: str, url: str) -> str | None:
        key = (method, url.split("?", 1)[0])
        if key not in cache:
            contract = endpoint_contract(store, scope, {"method": method, "url": key[1]})
            cache[key] = contract.payload.get("id") if contract else None
        return cache[key]

    trajectories: dict[str, list[Path]] = defaultdict(list)
    for runs_dir in args.runs_dir:
        for arm in args.arms:
            for path in sorted(runs_dir.glob(f"trajectories/*/{arm}-*/trajectory.json")):
                trajectories[path.parent.parent.name].append(path)
    table = learn_checks(trajectories, resolve)
    calls = [c for paths in trajectories.values() for p in paths for c in trajectory_calls(p)]
    resolved = sum(resolve(str(c.get("method") or "GET").upper(), str(c["url"])) is not None for c in calls)
    table["provenance"] = {
        "runs_dirs": [str(d) for d in args.runs_dir],
        "arms": args.arms,
        "trajectories": sum(len(p) for p in trajectories.values()),
        "api_fetch_calls": len(calls),
        "resolved_calls": resolved,
    }
    args.output.write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table["provenance"], indent=2))
    store.close()


if __name__ == "__main__":
    main()
