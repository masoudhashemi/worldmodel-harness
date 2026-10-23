"""Audit saved runs without executing tools, modifying stores, or reading graders.

Example: .venv/bin/python scripts/audit_task_interpretation.py runs/<run-id> \
  --model-config configs/model-gemma.toml --output runs/shadow-audit
"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import tomllib
from pathlib import Path

from worldmodel_harness import shadow_interpretation
from worldmodel_harness.config import ModelSettings, load_api_key
from worldmodel_harness.knowledge import Episode
from worldmodel_harness.runtime import MeteredClient, build_client


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path):
    histories = list(path.glob("*/history.json"))
    if len(histories) != 1:
        raise ValueError("Supply one single-task run directory")
    history = json.loads(histories[0].read_text())
    messages = history[0] if history and isinstance(history[0], list) else history
    task = next(
        m["content"]
        for m in messages
        if m.get("role") == "user" and m.get("kind") != "turn_warning"
    )
    database = path / "evidence.sqlite"
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        episodes = [
            Episode.model_validate_json(row[0])
            for row in connection.execute("SELECT body FROM episodes ORDER BY rowid")
        ]
    scopes = {e.scope.model_dump_json() for e in episodes if e.origin == "environment"}
    if len(scopes) != 1:
        raise ValueError("Replay requires exactly one environment task scope")
    scope = next(e.scope for e in episodes if e.origin == "environment")
    return task, episodes, scope, histories[0]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--boundary", choices=["before_first_write", "final", "both"], default="both"
    )
    parser.add_argument("--max-evidence-chars", type=int, default=60000)
    args = parser.parse_args()
    if args.max_evidence_chars < 1000:
        parser.error("Evidence allowance must be at least 1000 characters")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = ModelSettings.model_validate(tomllib.loads(args.model_config.read_text())["model"])
    client = MeteredClient(build_client(settings, load_api_key(Path.cwd(), settings)))
    manifest = {
        "mode": "offline_shadow",
        "model_settings": settings.model_dump(),
        "source_sha256": digest(Path(shadow_interpretation.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "results": [],
        "complete": False,
    }
    boundaries = ["before_first_write", "final"] if args.boundary == "both" else [args.boundary]
    failed = False
    for run in args.runs:
        task, episodes, scope, history = load_run(run)
        hashes = {str(p.resolve()): digest(p) for p in (run / "evidence.sqlite", history)}
        for boundary in boundaries:
            print(f"AUDIT {run.name} {boundary}", flush=True)
            start = len(client.calls)
            accounting = {}
            try:
                report = await shadow_interpretation.interpret(
                    client,
                    task,
                    episodes,
                    scope,
                    max_chars=args.max_evidence_chars,
                    boundary=boundary,
                    accounting=accounting,
                )
            except Exception as exc:  # noqa: BLE001 -- record failure without provider secrets
                report = {
                    "mode": "offline_shadow",
                    "boundary": boundary,
                    "error_type": type(exc).__name__,
                    "accounting": accounting,
                }
                failed = True
            report.update(run_id=run.name, input_hashes=hashes, model_calls=client.calls[start:])
            output = args.output / f"{run.name}-{boundary}.json"
            output.write_text(json.dumps(report, indent=2))
            assert all(digest(Path(p)) == h for p, h in hashes.items()), (
                "Input artifact changed during audit"
            )
            manifest["results"].append(
                {
                    "report": output.name,
                    "run_id": run.name,
                    "boundary": boundary,
                    "error_type": report.get("error_type"),
                    "assessment": report.get("assessment"),
                    "accounting": report.get("accounting"),
                }
            )
            (args.output / "summary.json").write_text(json.dumps(manifest, indent=2))
    manifest["complete"] = True
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
