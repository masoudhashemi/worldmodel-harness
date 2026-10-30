"""Would the finish review have shown the error? Offline replay of saved trajectories.

For each trial, rebuild the entity ledger from its api_fetch calls and render the
review. For each failed check, test whether the rendered review makes the error
visible: an excluded item shown as acted on (A), a missed target shown with nothing
written (B), or the missing content shown beside its item (D, E). Also reports the
review's size on passing trials (noise). Grader output is used only for this
diagnosis. Usage: ledger_offline.py <run-dir> [arm] [round]
"""

import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify_failures import (
    TARGET_KEYS,
    classify,
    failed_checks,
    fetches,
    flat,
    messages,
    norm,
    trial_text,
)

from worldmodel_harness.ledger import Ledger, observe, render


def visible(check: dict, label: str, review: str) -> bool:
    rows = [norm(line) for line in review.splitlines()[1:]]
    literals = [norm(s) for s in flat(check["params"]) if len(norm(s)) >= 3]
    if not literals:
        return False
    hit = [r for r in rows if any(lit in r for lit in literals)]
    if label.startswith("A"):
        return any("nothingwritten" not in r for r in hit)
    if label.startswith("B"):
        # The item is listed and none of its writes went to the missed target.
        targets = [norm(v) for k, v in check["params"].items() if k in TARGET_KEYS and isinstance(v, str)]
        return any(all("to" + t not in r for t in targets) for r in hit)
    return bool(hit)


def main():
    run_dir = Path(sys.argv[1])
    arm = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    rnd = sys.argv[3] if len(sys.argv) > 3 else "0"
    seen, shown, sizes = Counter(), Counter(), {True: [], False: []}
    for trial in sorted((run_dir / "trajectories").glob(f"*/{arm}-r{rnd}")):
        msgs = messages(trial / "trajectory.json")
        ledger = Ledger()
        for arguments, response, success in fetches(msgs):
            observe(ledger, arguments, response, success)
        review = render(ledger)
        checks = failed_checks(trial)
        sizes[not checks].append((len(review.splitlines()), len(review)))
        texts = trial_text(msgs)
        for check in checks:
            label = classify(check, *texts)
            label = label[0] + (" " + label.split("(")[1].rstrip(")") if label.startswith("B") else "")
            seen[label] += 1
            shown[label] += visible(check, label, review)
    print(f"{arm} r{rnd}: failed checks the review makes visible")
    for label in sorted(seen):
        print(f"  {label:8} {shown[label]:3}/{seen[label]:3}  ({100 * shown[label] / seen[label]:.0f}%)")
    core = [k for k in seen if k[0] in "ADE" or k == "B read"]
    print(f"  core A/B/D/E: {sum(shown[k] for k in core)}/{sum(seen[k] for k in core)}")
    for ok in (True, False):
        lines = [n for n, _ in sizes[ok]]
        chars = [c for _, c in sizes[ok]]
        print(f"{'pass' if ok else 'fail'} trials={len(lines)} review lines median={statistics.median(lines)} "
              f"max={max(lines)} chars median={statistics.median(chars)} empty={sum(n == 0 for n in lines)}")


if __name__ == "__main__":
    main()
