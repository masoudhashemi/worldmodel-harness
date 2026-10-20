"""Explicit CLI: offline checks, opt-in free-model smoke, and configured experiments."""

import argparse
import asyncio
import json
from pathlib import Path

from .config import load_api_key, load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["check-config", "smoke", "automationbench", "adapt", "compare", "rounds"],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--domain")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--task-timeout-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument(
        "--phase",
        choices=["development", "frozen", "online", "checkpoint", "train", "test"],
        default="development",
    )
    parser.add_argument("--memory", choices=["none", "episodes", "flat", "graph"], default="graph")
    parser.add_argument("--knowledge", type=Path)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--operational", action="store_true")
    parser.add_argument("--semantic-learning", action="store_true")
    parser.add_argument("--information-policy", choices=["off", "task", "future"], default="off")
    parser.add_argument("--max-information-calls", type=int, default=4)
    parser.add_argument("--adaptation-config", type=Path)
    parser.add_argument("--shift-config", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "check-config":
            settings = load_settings(args.root)
            load_api_key(args.root, settings)
            print(json.dumps({"model": settings.name, "credential_available": True}))
        elif args.command == "smoke":
            from .smoke import run_smoke

            report = asyncio.run(run_smoke(args.root))
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise SystemExit(1)
        elif args.command == "rounds":
            from .learning_curve import LearningCurveSettings, run_learning_curve

            report = asyncio.run(
                run_learning_curve(
                    args.root,
                    LearningCurveSettings(
                        rounds=args.rounds,
                        task_count=args.task_count,
                        seed=args.seed,
                        domain=args.domain or "public",
                        operational=args.operational,
                        semantic_learning=args.semantic_learning,
                        timeout_seconds=args.task_timeout_seconds,
                    ),
                )
            )
            print(json.dumps(report, indent=2))
        elif args.command == "adapt":
            from .adapt_benchmark import AdaptationSpec, adapt_automationbench

            if args.knowledge is None or args.adaptation_config is None:
                raise ValueError("adapt requires --knowledge and --adaptation-config")
            spec = AdaptationSpec.model_validate_json(args.adaptation_config.read_text())
            report = asyncio.run(adapt_automationbench(args.root, args.knowledge, spec))
            print(json.dumps(report, indent=2))
            if not report["result"]["adopted"]:
                raise SystemExit(1)
        else:
            from .experiment import ExperimentSettings, run_automationbench
            from .shift import ResponseShift

            settings = ExperimentSettings(
                domain=args.domain or "simple",
                limit=args.limit,
                offset=args.offset,
                task_names=tuple(args.task),
                phase=args.phase,
                memory=args.memory,
                adaptive=args.adaptive,
                operational=args.operational,
                semantic_learning=args.semantic_learning,
                timeout_seconds=args.task_timeout_seconds,
                information_policy=args.information_policy,
                max_information_calls=args.max_information_calls,
                shift=ResponseShift.model_validate_json(args.shift_config.read_text())
                if args.shift_config
                else None,
            )
            if args.command == "compare":
                from .comparison import compare_automationbench

                if args.knowledge is None:
                    raise ValueError("compare requires --knowledge")
                report = asyncio.run(compare_automationbench(args.root, settings, args.knowledge))
            else:
                report = asyncio.run(run_automationbench(args.root, settings, args.knowledge))
            print(json.dumps(report, indent=2))
            if any("error_type" in result for result in report.get("results", [])):
                raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 -- public CLI must not expose SDK request secrets
        # SDK exception strings can contain request headers. Never print those.
        print(f"Failed ({type(exc).__name__}); exception details suppressed for credential safety.")
        raise SystemExit(1) from None
