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


def extract_final_summary(messages):
    """Use the last finish declaration, including an unresolved/rejected attempt.

    Tool feedback is never used as summary text or as proof of completion.
    """
    summaries = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = [b for b in message.get("blocks", []) if b.get("kind") == "tool_call"]
        calls += [c.get("function", c) for c in message.get("tool_calls", [])]
        for call in calls:
            if call.get("name") != "finish":
                continue
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            if isinstance(args, dict) and isinstance(args.get("summary"), str):
                summaries.append(args["summary"])
    if summaries:
        return summaries[-1], "last_finish_attempt"
    if messages and messages[-1].get("role") == "assistant":
        last = messages[-1]
        if not last.get("tool_calls") and not any(
            b.get("kind") == "tool_call" for b in last.get("blocks", [])
        ):
            text = last.get("content") or "\n".join(
                b.get("text", "") for b in last.get("blocks", []) if b.get("kind") == "text"
            )
            if isinstance(text, str) and text:
                return text, "terminal_assistant_message"
    return None, "missing"


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
    return task, episodes, scope, histories[0], extract_final_summary(messages)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--boundary",
        choices=["before_first_write", "final", "both"],
        default="final",
        help="Final audits completion and summary; before_first_write retains the legacy interpretation audit.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=60000,
        help="Completion audits bound the serialized compact evidence catalog; schema and assessments are additional context.",
    )
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
        task, episodes, scope, history, (summary, summary_source) = load_run(run)
        hashes = {str(p.resolve()): digest(p) for p in (run / "evidence.sqlite", history)}
        for boundary in boundaries:
            print(f"AUDIT {run.name} {boundary}", flush=True)
            start = len(client.calls)
            accounting = {}
            try:
                if boundary == "final":
                    report = await shadow_interpretation.audit_completion(
                        client,
                        task,
                        summary,
                        episodes,
                        scope,
                        max_chars=args.max_evidence_chars,
                        accounting=accounting,
                    )
                else:
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
            if boundary == "final":
                report.update(
                    final_summary=summary,
                    summary_source=summary_source,
                )
                report.setdefault("audit_type", "staged_task_completion")
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
