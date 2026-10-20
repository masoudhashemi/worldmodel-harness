"""Print the accuracy, consistency, and search curve for an iterative evaluation."""

import argparse
import json
from pathlib import Path

from iterative_eval import summarize_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    summary = summarize_directory(parser.parse_args().output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
