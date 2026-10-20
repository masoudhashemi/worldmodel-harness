"""Repeat one fixed task list and compare a fresh store with a carried store.

reset starts every task from an empty graph. learn keeps one graph across tasks
and iterations. Both arms may learn within a task. Validation scores are recorded
after the task and are not written into the graph.

The Qwen profile is required. There is no model fallback.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

MODEL = "openrouter/qwen/qwen3.8-27b"
ARMS = ("reset", "learn")
# AutomationBench task setup: 50 turns, no added wall-clock or environment-call cap.
DEFAULT_TURNS = 50
DEFAULT_PER_DOMAIN = 5
DEFAULT_ITERATIONS = 5


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def select_names(names, *, limit, seed):
    """Seeded order from task names only. Scores and assertions are not inputs."""
    ordered = sorted(set(names))
    if len(ordered) != len(list(names)):
        raise ValueError("Task names must be unique")
    if not 1 <= limit <= len(ordered):
        raise ValueError("Task limit is outside the available set")
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:limit]


def select_per_domain(pools, *, count, seed):
    """Take the same number of names from each domain, then interleave them."""
    if not pools:
        raise ValueError("No domains to sample")
    rng = random.Random(seed)
    chosen = {}
    for domain, names in sorted(pools.items()):
        ordered = sorted(set(names))
        if len(ordered) != len(list(names)):
            raise ValueError("Task names must be unique")
        if not 1 <= count <= len(ordered):
            raise ValueError("Task count is outside the available set")
        rng.shuffle(ordered)
        chosen[domain] = ordered[:count]
    return [chosen[domain][index] for index in range(count) for domain in sorted(chosen)]


def store_path(output, arm, iteration, task_index):
    if arm == "reset":
        return output / "stores" / f"iter-{iteration:02d}" / f"{task_index:02d}.sqlite"
    if arm == "learn":
        return output / "stores" / "learn.sqlite"
    raise ValueError("Unknown arm")


def task_metrics(result):
    ledger = result.get("ledger") or {}
    purposes = ledger.get("by_purpose") or {}
    partial = result.get("partial_credit")
    return {
        "task": result.get("task_name"),
        "strict_success": result.get("task_completed_correctly") == 1,
        "partial_credit": partial if isinstance(partial, (int, float)) else None,
        "validation_available": bool(result.get("validation_available")),
        "finished": bool(result.get("finished")),
        "checked": result.get("completion_evidence_status") == "checked",
        "error_type": result.get("error_type"),
        "environment_calls": ledger.get("environment_calls", 0),
        "discovery_calls": purposes.get("discovery", 0),
    }


def _mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else None


def iteration_summary(rows):
    if not rows:
        raise ValueError("An iteration needs at least one task")
    return {
        "tasks": len(rows),
        "strict_accuracy": sum(row["strict_success"] for row in rows) / len(rows),
        "mean_partial_credit": _mean(rows, "partial_credit"),
        "grades_missing": sum(not row["validation_available"] for row in rows),
        "discovery_calls": sum(row["discovery_calls"] for row in rows),
        "environment_calls": sum(row["environment_calls"] for row in rows),
        "errors": sum(bool(row["error_type"]) for row in rows),
    }


def _by_task(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row["strict_success"])
    return grouped


def consistency(rows):
    grouped = _by_task(rows)
    return {
        "tasks": len(grouped),
        "stable_outcome_tasks": sum(len(set(values)) == 1 for values in grouped.values()),
        "always_passed": sum(all(values) for values in grouped.values()),
    }


def verdict(records):
    """Accuracy is required. Fewer searches count only when accuracy holds."""
    arms = {}
    for arm in ARMS:
        iterations = sorted({row["iteration"] for row in records if row["arm"] == arm})
        if not iterations:
            raise ValueError(f"Missing {arm} records")
        arms[arm] = [
            iteration_summary([row for row in records if row["arm"] == arm and row["iteration"] == i])
            for i in iterations
        ]
        if [row["iteration"] for row in records if row["arm"] == arm] != sorted(
            row["iteration"] for row in records if row["arm"] == arm
        ):
            raise ValueError("Records must stay in iteration order")
    reset, learn = arms["reset"], arms["learn"]
    if len(reset) != len(learn):
        raise ValueError("Arms must complete the same iterations")
    last_reset, first_learn, last_learn = reset[-1], learn[0], learn[-1]
    accuracy_holds = last_learn["strict_accuracy"] >= last_reset["strict_accuracy"]
    accuracy_rose = last_learn["strict_accuracy"] > first_learn["strict_accuracy"]
    fewer_than_reset = last_learn["discovery_calls"] < last_reset["discovery_calls"]
    fewer_than_start = last_learn["discovery_calls"] < first_learn["discovery_calls"]
    return {
        "iterations": [
            {"iteration": index + 1, "reset": reset[index], "learn": learn[index]}
            for index in range(len(reset))
        ],
        "consistency": {arm: consistency([row for row in records if row["arm"] == arm]) for arm in ARMS},
        "accuracy_holds": accuracy_holds,
        "accuracy_rose": accuracy_rose,
        "fewer_searches_than_reset": fewer_than_reset,
        "fewer_searches_than_start": fewer_than_start,
        "efficiency_win": accuracy_holds and (fewer_than_reset or fewer_than_start),
        "goal_pattern": accuracy_holds and accuracy_rose and fewer_than_start,
    }


TRANSIENT_ERRORS = {
    "APIConnectionError",
    "APIError",
    "TimeoutError",
    "InternalServerError",
}
RATE_LIMIT_ERRORS = {"RateLimitError", "RetryError", "DailyRateLimit"}


def retry_delay(attempt):
    """Wait 60s, then 120s, 240s, and so on, never more than 15 minutes."""
    return min(900, 60 * (2 ** (attempt - 1)))


def error_name(exc):
    """Label a transport failure without retaining the provider payload."""
    if "free-models-per-day" in str(exc):
        return "DailyRateLimit"
    return type(exc).__name__


def summarize_directory(output):
    records = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    expected = len(manifest["tasks"]) * manifest["iterations"] * len(ARMS)
    summary = {
        "tasks": manifest["tasks"],
        "recorded": len(records),
        "complete": len(records) == expected and expected > 0,
        "by_arm": {},
    }
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        if not rows:
            continue
        summary["by_arm"][arm] = [
            {
                "iteration": iteration,
                **iteration_summary([row for row in rows if row["iteration"] == iteration]),
            }
            for iteration in sorted({row["iteration"] for row in rows})
        ]
    if summary["complete"]:
        summary.update(verdict(records))
    return summary


def task_pools(domain):
    from automationbench.domains import PUBLIC_DOMAINS

    from worldmodel_harness.automationbench import dataset

    domains = list(PUBLIC_DOMAINS) if domain == "public" else [domain]
    return {
        name: [json.loads(row["info"])["task_name"] for row in dataset(name)] for name in domains
    }


async def run_experiment(root, output, *, domain, names, per_domain, iterations, seed, budgets, semantic):
    from worldmodel_harness.config import load_settings
    from worldmodel_harness.store import KnowledgeStore

    settings = load_settings(root)
    model_name = settings.name
    # Stirrup retries a rate limit three times within ten seconds. Keep one request,
    # then wait here so the free endpoint is not burst.
    from stirrup.clients.litellm_client import LiteLLMClient

    generate = LiteLLMClient.generate
    if hasattr(generate, "__wrapped__"):
        LiteLLMClient.generate = generate.__wrapped__
    pools = task_pools(domain)
    available = {task for names_in_domain in pools.values() for task in names_in_domain}
    tasks = list(names) if names else select_per_domain(pools, count=per_domain, seed=seed)
    if any(task not in available for task in tasks) or len(tasks) != len(set(tasks)):
        raise ValueError("Task list must be unique names from the selected domains")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "domain": domain,
        "per_domain": per_domain,
        "tasks": tasks,
        "iterations": iterations,
        "seed": seed,
        "arms": list(ARMS),
        "model": model_name,
        "budgets": budgets,
        "semantic_learning": semantic,
        "protocol": (
            "Same tasks each iteration. reset uses an empty graph for every task. "
            "learn carries one graph across every domain, task, and iteration. "
            "Validation scores are report-only."
        ),
    }
    write_json(output / "manifest.json", manifest)
    results = output / "results.jsonl"
    if not results.exists():
        results.write_text("")
    completed = set()
    for line in results.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((row["iteration"], row["arm"], row["task"]))
    if not store_path(output, "learn", 1, 0).exists():
        KnowledgeStore(store_path(output, "learn", 1, 0)).close()
    for iteration in range(1, iterations + 1):
        for arm in ARMS:
            for index, task in enumerate(tasks):
                path = store_path(output, arm, iteration, index)
                if (iteration, arm, task) in completed:
                    continue
                if arm == "reset":
                    path.unlink(missing_ok=True)
                    KnowledgeStore(path).close()
                print(
                    json.dumps({"starting": {"iteration": iteration, "arm": arm, "task": task}}),
                    flush=True,
                )
                row = None
                attempt = 0
                while True:
                    attempt += 1
                    row = await run_one_task(root, task, path, budgets, semantic, iteration, arm)
                    limited = row["error_type"] in RATE_LIMIT_ERRORS and not row["environment_calls"]
                    transient = row["error_type"] in TRANSIENT_ERRORS and not row["environment_calls"]
                    if not limited and (not transient or attempt >= 4):
                        break
                    delay = retry_delay(attempt)
                    print(
                        json.dumps(
                            {
                                "retry": {
                                    "task": task,
                                    "attempt": attempt,
                                    "error_type": row["error_type"],
                                    "wait_seconds": delay,
                                }
                            }
                        ),
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                with results.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                write_json(output / "summary.json", summarize_directory(output))
                print(json.dumps({"finished": row}), flush=True)
    return summarize_directory(output)


async def run_one_task(root, task, path, budgets, semantic, iteration, arm):
    from worldmodel_harness.experiment import ExperimentSettings, run_automationbench

    try:
        report = await run_automationbench(
            root,
            ExperimentSettings(
                domain=task.split(".", 1)[0],
                task_names=(task,),
                limit=1,
                phase="test",
                memory="graph",
                operational=True,
                semantic_learning=semantic,
                **budgets,
            ),
            path,
        )
        result = report["results"][0]
        return {
            "iteration": iteration,
            "arm": arm,
            "run_id": report["run_id"],
            **task_metrics(result),
            "task": task,
        }
    except Exception as exc:  # noqa: BLE001 -- keep SDK text out of the curve
        return {
            "iteration": iteration,
            "arm": arm,
            "task": task,
            "run_id": None,
            "strict_success": False,
            "partial_credit": None,
            "validation_available": False,
            "finished": False,
            "checked": False,
            "error_type": error_name(exc),
            "environment_calls": 0,
            "discovery_calls": 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--domain", default="public")
    parser.add_argument("--per-domain", type=int, default=DEFAULT_PER_DOMAIN)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--max-turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--semantic-learning", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(
        run_experiment(
            args.root,
            args.output,
            domain=args.domain,
            names=tuple(args.task),
            per_domain=args.per_domain,
            iterations=args.iterations,
            seed=args.seed,
            budgets={
                "max_turns": args.max_turns,
                "max_environment_calls": None,
                "timeout_seconds": None,
            },
            semantic=args.semantic_learning,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
