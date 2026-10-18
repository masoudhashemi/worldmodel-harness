"""Offline diagnostic of one saved v3.2 failure; no agent recipe or model calls."""

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from worldmodel_harness.goal_checks import CheckedFinish, GoalTracker
from worldmodel_harness.knowledge import Episode, LiveState
from worldmodel_harness.store import KnowledgeStore
from worldmodel_harness.verification_tools import VerifyGoal, verification_tool


def main():
    run = Path("runs/25fdd5e341bc4d79a0fdf54c812034ac")
    result = json.loads((run / "summary.json").read_text())["results"][0]
    source_check, effect_check = result["goal_checks"]
    with closing(
        sqlite3.connect(f"file:{(run / 'evidence.sqlite').resolve()}?mode=ro", uri=True)
    ) as db:
        events = [
            Episode.model_validate_json(row[0]) for row in db.execute("SELECT body FROM episodes")
        ]
    with TemporaryDirectory(prefix="verification-replay-") as directory:
        store = KnowledgeStore(Path(directory) / "evidence.sqlite")
        try:
            adapter = SimpleNamespace(
                evidence=store,
                scope=events[0].scope,
                live=LiveState(events[0].task_id),
                evidence_aliases={},
                task=SimpleNamespace(prompt=[{"content": source_check["task_quote"]}]),
            )
            adapter.goal_tracker = GoalTracker(adapter)
            for event in events:
                store.append_episode(event)
                adapter.live.observe(event)
            tool = verification_tool(adapter)
            # Explicit fixture bindings from the saved public task and observed checks.
            # These are not installed into the runtime or supplied to the live model.
            request = VerifyGoal(
                requirement_id="source",
                task_quote=source_check["task_quote"],
                kind="source",
                episode_id=source_check["episode_id"],
                path=tuple(source_check["observation"]["path"]),
                anchor_text="Jordan Lee",
                anchor_path=("from",),
            )
            assert asyncio.run(tool.executor(request)).success
            desired = effect_check["observation"]["expected"]
            field = effect_check["observation"]["path"][0]
            read = next(
                e
                for e in reversed(events)
                if e.arguments.get("method") == "GET" and e.response.get(field) == desired
            )
            request = VerifyGoal(
                requirement_id="effect",
                task_quote=effect_check["task_quote"],
                kind="effect",
                episode_id=effect_check["episode_id"],
                path=(field,),
                expected=desired,
                source_requirement_id="source",
                identity_path=("Name",),
                resource_url=read.arguments["url"],
            )
            bad = asyncio.run(tool.executor(request))
            assert not bad.success and "invalid_evidence_path" in bad.content
            good = asyncio.run(
                tool.executor(request.model_copy(update={"episode_id": read.episode_id}))
            )
            assert good.success
            assert adapter.goal_tracker.finish(CheckedFinish(summary="offline replay")).success
            print(
                json.dumps(
                    {
                        "source": "matched",
                        "write_ack": json.loads(bad.content),
                        "readback": json.loads(good.content),
                        "model_calls": 0,
                        "environment_calls": 0,
                    }
                )
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
