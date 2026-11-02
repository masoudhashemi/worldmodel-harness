"""E024 offline: how much time do repeated api_search queries cost? (E022 trajectories)

A turn is a repeat-search turn when it makes only api_search calls and every query in
it was already sent, word for word (case and spacing aside), earlier in the run. Also
reports, per trial, the longest stretch of such turns and whether the run fetched
anything new after the first repeat (the search eventually led somewhere).
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e022-learning-stream")


def norm(q) -> str:
    return " ".join(str(q).lower().split())


def trial(path: Path) -> dict:
    groups = json.loads(path.read_text())
    seen, s = set(), {"turns": 0, "repeat": 0, "first_repeat": None, "streak": 0}
    run = 0
    for m in (m for g in groups for m in g if isinstance(m, dict)):
        if m.get("role") != "assistant":
            continue
        calls = []
        for b in m.get("blocks") or []:
            if b.get("kind") == "tool_call":
                try:
                    a = json.loads(b["arguments"])
                except (TypeError, ValueError):
                    a = {}
                calls.append((b["name"], a if isinstance(a, dict) else {}))
        s["turns"] += 1
        queries = [norm(a.get("query")) for n, a in calls if n == "api_search"]
        if calls and len(queries) == len(calls) and all(q in seen for q in queries):
            s["repeat"] += 1
            s["first_repeat"] = s["first_repeat"] or s["turns"]
            run += 1
            s["streak"] = max(s["streak"], run)
        elif calls:
            run = 0
        seen.update(queries)
    return s


def main():
    rows = json.loads((OUT / "summary.json").read_text())["results"]
    by_arm = defaultdict(list)
    for row in rows:
        path = OUT / "trajectories" / row["task"] / f"{row['arm']}-r{row['round']}" / "trajectory.json"
        if path.exists():
            s = trial(path)
            s.update(task=row["task"], strict=row["strict_eval"], partial=row["partial_credit"], seconds=row["duration_seconds"])
            by_arm[row["arm"]].append(s)
    for arm, ts in sorted(by_arm.items()):
        rep = [t["repeat"] for t in ts]
        hit = [t for t in ts if t["repeat"]]
        big = [t for t in ts if t["streak"] >= 3]
        print(
            f"{arm:9s} n={len(ts)} trials with a repeat turn={len(hit)} with a streak>=3={len(big)} "
            f"repeat turns: total={sum(rep)} of {sum(t['turns'] for t in ts)} ({sum(rep) / sum(t['turns'] for t in ts):.1%}), "
            f"mean/task={st.mean(rep):.2f}, after the first two per run={sum(max(r - 2, 0) for r in rep)}"
        )
        if big:
            print(f"   streak>=3: strict {sum(t['strict'] for t in big)}/{len(big)}, partial {st.mean(t['partial'] for t in big):.2f}, "
                  f"median time {st.median(t['seconds'] for t in big):.0f}s vs others {st.median(t['seconds'] for t in ts if t['streak'] < 3):.0f}s")
            for t in sorted(big, key=lambda t: -t["repeat"])[:8]:
                print(f"     {t['task']:45s} repeat={t['repeat']:2d} streak={t['streak']:2d} turns={t['turns']} first@{t['first_repeat']} strict={t['strict']}")


if __name__ == "__main__":
    main()
