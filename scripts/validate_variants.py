"""Check that surface variants stay solvable and correctly graded.

Usage: validate_variants.py <out.json> <seed> <run-dir> [...]

For every saved trajectory whose source task passed, the trajectory's api_fetch and
base64_encode calls are replayed on variant <seed> of the task, with every literal
renamed by the variant's rename map (inside base64-encoded Gmail messages too). The
variant is valid if the renamed replay gets the same partial credit as the source run.
Grades are used only to validate the benchmark, never for learning.
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

from worldmodel_harness.automationbench import AutomationBenchAdapter, dataset, task_row
from worldmodel_harness.automationbench.traces import episode
from worldmodel_harness.budget import Ledger
from worldmodel_harness.variants import apply, known_first_names, make_variant


def _rename_raw(body, mapping):
    """Rename inside a Gmail body's base64url 'raw' message."""
    try:
        data = json.loads(body) if isinstance(body, str) else body
    except ValueError:
        return body
    if isinstance(data, dict) and isinstance(data.get("raw"), str):
        raw = data["raw"]
        try:
            text = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", "ignore")
        except ValueError:  # not valid base64: the environment rejects it either way
            return body
        data["raw"] = base64.urlsafe_b64encode(apply(text, mapping).encode()).decode().rstrip("=")
        return json.dumps(data)
    return body


async def replay(row: dict, steps, mapping: dict) -> float:
    adapter = AutomationBenchAdapter(row, row["info"] and json.loads(row["info"])["task_name"].split(".")[0], Ledger())
    tools = {t.name: t for t in adapter.tools()}
    for step in steps:
        if step.tool not in ("api_fetch", "base64_encode"):
            continue
        arguments = {k: apply(v, mapping) if isinstance(v, str) else v for k, v in step.arguments.items()}
        if step.tool == "api_fetch" and "body" in arguments:
            arguments["body"] = _rename_raw(arguments["body"], mapping)
        try:
            params = tools[step.tool].parameters(**arguments)
        except Exception:  # noqa: BLE001, S112 - a malformed saved call is skipped, as the agent's was
            continue
        await tools[step.tool].executor(params)
    return adapter.score()["partial_credit"]


def main():
    out, seed = Path(sys.argv[1]), int(sys.argv[2])
    names = known_first_names(dataset("public"))
    checked, seen = [], set()
    for run in map(Path, sys.argv[3:]):
        for result_path in sorted(run.rglob("result.json")):
            result = json.loads(result_path.read_text())
            task = result.get("task_name", "")
            if not result.get("task_completed_correctly") or task in seen or "~v" in task:
                continue
            seen.add(task)
            found = episode(result_path.parent / "trajectory.json", task)
            if not found:
                continue
            source = task_row(task)
            variant, mapping = make_variant(source, seed, names)
            same = asyncio.run(replay(source, found.steps, {}))
            renamed = asyncio.run(replay(variant, found.steps, mapping))
            checked.append({"task": task, "source_replay": same, "variant_replay": renamed, "renamed": len(mapping)})
            print(f"{task}: source {same:.2f} variant {renamed:.2f} ({len(mapping)} literals)")
    out.write_text(json.dumps(checked, indent=1) + "\n")
    valid = [c for c in checked if c["source_replay"] == 1.0]
    print(f"{len(valid)} passing replays; variant also passes in {sum(c['variant_replay'] == 1.0 for c in valid)}")


if __name__ == "__main__":
    main()
