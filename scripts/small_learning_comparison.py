"""Compare plain tools, structured harness, and retained/model learning on the same tasks.

Requires passed real-model acceptance checks. Each trial has isolated storage.
This is a diagnostic comparison, not a significance test or per-layer causal ablation.
"""
import argparse
import asyncio
import json
from pathlib import Path

from worldmodel_harness.checkpoint import copy_checkpoint
from worldmodel_harness.experiment import ExperimentSettings, run_automationbench
from worldmodel_harness.learning_curve import summarize_round
from worldmodel_harness.store import KnowledgeStore


async def main(args):
    acceptance = json.loads(args.acceptance.read_text())
    if not acceptance.get("complete") or not acceptance.get("passed"):
        raise ValueError("Real-model recovery and verification acceptance must pass first")
    if len(args.task) != len(set(args.task)) or len(args.task) > 3:
        raise ValueError("Use one to three distinct diagnostic tasks")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "acceptance": str(args.acceptance), "results": [],
              "interpretation": "Learning arm combines retained knowledge and runtime learning. "
                  "Harness arm still observes current documentation; it disables semantic learning."}
    path = args.output / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    arms = ("plain", "harness", "learning")
    for index, task in enumerate(args.task):
        for arm in arms[index % 3:] + arms[:index % 3]:
            store = args.output / f"{index}-{arm}.sqlite"
            if arm == "learning":
                copy_checkpoint(args.checkpoint, store)
            else:
                KnowledgeStore(store).close()
            settings = ExperimentSettings(domain=task.split('.', 1)[0], task_names=(task,),
                limit=1, phase="test", memory="none" if arm == "plain" else "graph",
                operational=arm != "plain", semantic_learning=arm == "learning",
                adaptive=False, max_turns=15, max_environment_calls=20, timeout_seconds=180,
                semantic_timeout_seconds=90, semantic_max_tokens=4096)
            run = await run_automationbench(args.root, settings, store)
            report["results"].append({"arm": arm, "task": task, "run_id": run["run_id"],
                                      "metrics": summarize_round(run)})
            path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    asyncio.run(main(parser.parse_args()))
