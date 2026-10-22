"""Run an isolated learning comparison with explicit, disjoint public task names."""

import argparse
import asyncio
import json
from pathlib import Path

from worldmodel_harness.controlled_learning import run_controlled
from worldmodel_harness.experiment import ExperimentSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--domain", default="simple")
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--validate", action="append", required=True)
    parser.add_argument("--evaluate", action="append", required=True)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--semantic-timeout-seconds", type=int, default=60)
    parser.add_argument("--semantic-max-tokens", type=int, default=4096)
    args = parser.parse_args()
    report = asyncio.run(run_controlled(args.root, args.output,
        training_names=tuple(args.train), validation_names=tuple(args.validate),
        evaluation_names=tuple(args.evaluate), base=ExperimentSettings(
            domain=args.domain, operational=True, max_turns=args.max_turns,
            timeout_seconds=args.timeout_seconds,
            semantic_timeout_seconds=args.semantic_timeout_seconds,
            semantic_max_tokens=args.semantic_max_tokens)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
