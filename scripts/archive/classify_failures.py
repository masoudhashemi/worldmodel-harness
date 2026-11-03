"""Classify failed grader checks by where their expected literals appear.

Offline diagnosis only: grader output is read here to understand failures, never
used for learning. Usage: classify_failures.py <run-dir> [round]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TARGET_KEYS = {"to", "channel_name", "email", "to_number", "channel"}


def norm(text) -> str:
    return re.sub(r"[\s,$]", "", str(text)).lower()


def flat(value) -> list[str]:
    if isinstance(value, list):
        return [y for x in value for y in flat(x)]
    if isinstance(value, dict):
        return [y for x in value.values() for y in flat(x)]
    return [] if isinstance(value, bool) else [str(value)]


def messages(trajectory: Path) -> list[dict]:
    groups = json.loads(trajectory.read_text())
    return groups[0] if groups and isinstance(groups[0], list) else groups


def fetches(msgs: list[dict]) -> list[tuple[dict, str, bool]]:
    """(api_fetch arguments, response, success) in call order."""
    calls = {}
    for m in msgs:
        for b in m.get("blocks") or ():
            if b.get("kind") == "tool_call" and b.get("name") == "api_fetch":
                try:
                    calls[b["tool_call_id"]] = json.loads(b.get("arguments") or "{}")
                except ValueError:
                    calls[b["tool_call_id"]] = {}
    out = []
    for m in msgs:
        if m.get("role") == "tool" and m.get("name") == "api_fetch" and m.get("tool_call_id") in calls:
            out.append((calls[m["tool_call_id"]], str(m.get("content") or ""), bool(m.get("success"))))
    return out


def trial_text(msgs: list[dict]) -> tuple[str, str, str]:
    """Normalized task text, read responses and write arguments."""
    from worldmodel_harness.ledger import is_write, write_text

    task = norm(next(m["content"] for m in msgs if m["role"] == "user"))
    reads, writes = [], []
    for arguments, response, _ in fetches(msgs):
        (writes if is_write(arguments) else reads).append(write_text(arguments) if is_write(arguments) else response)
    for m in msgs:
        for b in m.get("blocks") or ():
            if b.get("kind") == "tool_call" and b.get("name") == "finish":
                writes.append(b.get("arguments") or "")
    return task, norm("".join(reads)), norm("".join(writes))


def classify(check: dict, task: str, reads: str, writes: str) -> str:
    params, kind = check["params"], check["type"]
    if "not" in kind.split("_"):
        return "A exclusion violated"

    def where(s):
        n = norm(s)
        return "write" if n in writes else "task" if n in task else "read" if n in reads else "unseen"

    targets = [str(v) for k, v in params.items() if k in TARGET_KEYS and isinstance(v, (str, int))]
    content = [s for k, v in params.items() if k not in TARGET_KEYS and not k.endswith("_id") and k != "action_key" for s in flat(v)]
    places = {where(s) for s in targets} - {"write"}
    if places:
        return "B missed target (" + "/".join(sorted(places)) + ")"
    if content:
        found = {where(s) for s in content}
        if "unseen" in found:
            return "C content not in any read (computed or unread source)"
        if found & {"read", "task"}:
            return "D content was read but not written"
        return "E all literals written, wrong place/form"
    return "F id/structure only"


def failed_checks(trial: Path):
    result = json.loads((trial / "result.json").read_text())
    if result["task_completed_correctly"]:
        return []
    return [c for c in result["grading_details"] if not c["passed"] and not c.get("excluded")]


def main():
    run_dir, rnd = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "0"
    arms = sorted({p.name.rsplit("-r", 1)[0] for p in (run_dir / "trajectories").glob("*/*")})
    for arm in arms:
        counts, tasks = Counter(), defaultdict(set)
        for trial in sorted((run_dir / "trajectories").glob(f"*/{arm}-r{rnd}")):
            checks = failed_checks(trial)
            if not checks:
                continue
            texts = trial_text(messages(trial / "trajectory.json"))
            for check in checks:
                label = classify(check, *texts)
                counts[label] += 1
                tasks[label].add(trial.parent.name)
        total = sum(counts.values())
        print(f"== {arm}: {total} failed checks")
        for label, n in sorted(counts.items()):
            print(f"  {n:4} {100 * n / total:4.0f}%  tasks={len(tasks[label]):2}  {label}")


if __name__ == "__main__":
    main()
