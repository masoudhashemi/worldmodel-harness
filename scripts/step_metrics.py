"""Step-level evaluation (docs/evaluation-steps.md): no regression, gains on familiar steps.

Usage: step_metrics.py <run-dir> <control-arm> <arm> [aa] [--stream]
  aa: the A/A subset to use as the noise reference, one of
      v1        tasks ending in ~v1 (E022: no plans are possible in round v1)
      no-note   tasks where the arm's repeat pointer never fired (E024)
      no-hint   tasks where the arm showed no source hint (E026, E027)

  --stream  earlier runs of the run's own workflows are left out of the history, as the
            E022 learning stream's graph did (otherwise every step of a workflow run
            since E016 is "workflow seen").

A step is everything the agent did to reach one successful api_fetch, from the previous
one. Familiarity is judged against runs that started before this one plus, within this
run, trials of other tasks that ran before the task's first trial (so both arms of a task
see the same history). Never from grades; grades only give the task guardrails.
"""

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from worldmodel_harness.experience import load, workflow
from worldmodel_harness.lifting import Step, origin, slots

LEVELS = ("F0 unseen", "F1 endpoint seen", "F2 transition seen", "F3 workflow seen")
Z = 1.2816


@dataclass
class StepRecord:
    target: str
    previous: str
    occurrence: int
    turns: int = 0  # exploration turns: turns in the step minus the one that succeeded
    calls: int = 0  # tool calls other than the successful target call
    searches: int = 0
    first_try: bool = True
    traced: float | None = None  # write calls only: share of values seen in the query or a read
    level: int = 0


@dataclass
class Index:
    """What earlier runs did: per workflow, successful endpoints and transitions."""

    endpoints: dict = field(default_factory=lambda: defaultdict(set))
    transitions: dict = field(default_factory=lambda: defaultdict(set))

    def add(self, task: str, steps: list[Step]):
        w, previous = workflow(task), "start"
        for s in steps:
            if s.tool == "api_fetch" and s.success:
                self.endpoints[w].add(s.action)
                self.transitions[w].add((previous, s.action))
                previous = s.action

    def level(self, task: str, previous: str, target: str) -> int:
        w = workflow(task)
        if target in self.endpoints.get(w, ()):
            return 3
        others = [o for o in self.endpoints if o != w]
        if any((previous, target) in self.transitions[o] for o in others):
            return 2
        if any(target in self.endpoints[o] for o in others):
            return 1
        return 0


def turns_of(path: Path) -> tuple[str, list[list[Step]]]:
    """The query and, per assistant turn, its tool calls with their results."""
    groups = json.loads(path.read_text())
    messages = [m for g in groups for m in g if isinstance(m, dict)]
    query = next((str(m.get("content")) for m in messages if m.get("role") == "user"), "")
    results = {m.get("tool_call_id"): m for m in messages if m.get("role") == "tool"}
    turns = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        calls = []
        for b in m.get("blocks") or []:
            if b.get("kind") != "tool_call":
                continue
            try:
                arguments = json.loads(b["arguments"]) if b.get("arguments") else {}
            except ValueError:
                arguments = {}
            r = results.get(b.get("tool_call_id"), {})
            content = str(r.get("content", "")).split("\n\n[", 1)[0]
            calls.append(Step(b["name"], arguments if isinstance(arguments, dict) else {}, content, str(r.get("success")) == "True"))
        if calls:
            turns.append(calls)
    return query, turns


def steps_of(task: str, path: Path, index: Index) -> list[StepRecord]:
    query, turns = turns_of(path)
    found, history, done = [], [], Counter()
    pending_turns, pending_calls, pending_searches, attempts = 0, 0, 0, defaultdict(list)
    previous = "start"
    for calls in turns:
        pending_turns += 1
        first_in_turn = True
        for call in calls:
            if call.tool == "api_search":
                pending_searches += 1
            if call.tool == "api_fetch":
                attempts[call.action].append(call.success)
            if call.tool == "api_fetch" and call.success:
                write = str(call.arguments.get("method") or "GET").upper() != "GET"
                values = [v for _, v in slots(call.arguments)] if write else []
                traced = [origin(v, history, query) != "Agent" for v in values]
                rec = StepRecord(call.action, previous, done[call.action])
                rec.turns = pending_turns - 1 if first_in_turn else 0
                rec.calls = pending_calls if first_in_turn else 0
                rec.searches = pending_searches if first_in_turn else 0
                rec.first_try = attempts[call.action][0]
                rec.traced = sum(traced) / len(traced) if traced else None
                rec.level = index.level(task, previous, call.action)
                found.append(rec)
                done[call.action] += 1
                previous = call.action
                attempts = defaultdict(list)
                if first_in_turn:
                    pending_turns, pending_calls, pending_searches = 0, 0, 0
                first_in_turn = False
            elif call.tool != "finish":
                pending_calls += 1
            history.append(call)
    return found


def start_time(run: Path) -> float:
    times = [p.stat().st_mtime for p in run.rglob("trajectory.json")]
    return min(times) if times else math.inf


def wilson_upper(k: int, n: int) -> float:
    if not n:
        return 1.0
    p = k / n
    centre = p + Z * Z / (2 * n)
    spread = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (centre + spread) / (1 + Z * Z / n)


def regressions(rows: dict, tasks: list[str], control: str, arm: str) -> tuple[int, int, int, list[str]]:
    """(strict losses, control strict passes, partial drops >= 0.1, the tasks that regressed)."""
    lost = [t for t in tasks if rows[(t, control)]["strict_eval"] and not rows[(t, arm)]["strict_eval"]]
    passes = sum(rows[(t, control)]["strict_eval"] for t in tasks)
    drops = [t for t in tasks if rows[(t, arm)]["partial_credit"] <= rows[(t, control)]["partial_credit"] - 0.1]
    return len(lost), passes, len(drops), sorted(set(lost) | set(drops))


def aa_tasks(rows: dict, tasks: list[str], arm: str, mode: str | None) -> list[str]:
    if mode == "v1":
        return [t for t in tasks if t.endswith("~v1")]
    if mode == "no-note":
        return [t for t in tasks if not rows[(t, arm)].get("repeat_notes")]
    if mode == "no-hint":
        return [t for t in tasks if not rows[(t, arm)].get("source_hints")]
    return []


def main():
    stream = "--stream" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--stream"]
    run, control, arm = Path(args[0]), args[1], args[2]
    mode = args[3] if len(args) > 3 else None
    order = json.loads((run / "summary.json").read_text())["results"]
    rows = {(r["task"], r["arm"]): r for r in order if r["round"] == 0}
    broken = {r["task"] for r in order if r.get("error_type") and r["arm"] in (control, arm)}
    tasks = [t for t in dict.fromkeys(r["task"] for r in order) if (t, control) in rows and (t, arm) in rows and t not in broken]

    began = start_time(run)
    earlier = [p for p in Path("runs").iterdir() if p.is_dir() and p != run and start_time(p) < began]
    index = Index()
    flows = {workflow(t) for t in tasks}
    for e in load(earlier, (lambda t: workflow(t) not in flows) if stream else (lambda t: True)):
        index.add(e.task, e.steps)

    # Within the run: label both arms of a task before any of its trials joins the index.
    per_task = {}
    for r in order:
        t = r["task"]
        if t in tasks and t not in per_task:
            per_task[t] = {
                a: steps_of(t, run / "trajectories" / t / f"{a}-r0" / "trajectory.json", index) for a in (control, arm)
            }
        path = run / "trajectories" / t / f"{r['arm']}-r{r['round']}" / "trajectory.json"
        if path.exists():
            index.add(t, [s for calls in turns_of(path)[1] for s in calls])

    print(
        f"== {run.name}: {control} -> {arm}, {len(tasks)} tasks, history from {len(earlier)} earlier runs"
        + (" without these workflows" if stream else "")
        + (f" ({len(broken)} excluded: error)" if broken else "")
    )

    # 1. Task guardrails.
    def m(a, k, f=st.mean):
        return f(rows[(t, a)][k] for t in tasks)

    print(
        f"1. Task: strict {sum(rows[(t, control)]['strict_eval'] for t in tasks)} -> {sum(rows[(t, arm)]['strict_eval'] for t in tasks)}, "
        f"partial {m(control, 'partial_credit'):.2f} -> {m(arm, 'partial_credit'):.2f}, "
        f"turns {m(control, 'model_calls'):.1f} -> {m(arm, 'model_calls'):.1f}, "
        f"median time {m(control, 'duration_seconds', st.median):.0f} -> {m(arm, 'duration_seconds', st.median):.0f} s"
    )

    # 2. Regression against A/A.
    lost, passes, drops, regressed = regressions(rows, tasks, control, arm)
    line = f"2. Regression: strict lost {lost}/{passes} control passes; partial drop >= 0.1 on {drops}/{len(tasks)} tasks"
    aa = aa_tasks(rows, tasks, arm, mode)
    if aa:
        a_lost, a_passes, a_drops, _ = regressions(rows, aa, control, arm)
        line += (
            f" | A/A ({mode}, n={len(aa)}): strict lost {a_lost}/{a_passes}, partial drop {a_drops}/{len(aa)}, "
            f"90% upper bound of the drop rate {wilson_upper(a_drops, len(aa)):.0%} (arm: {drops / len(tasks):.0%})"
        )
    print(line)
    if regressed:
        print("   regressed: " + ", ".join(regressed))

    # 3. Steps per familiarity level, paired by task (sums per task) and by step.
    print("3. Steps by familiarity (control -> arm):")
    print("   level                 steps     expl.turns/step  expl.calls/step  searches/step  first-try  traced   paired-step d.turns (n)")
    for level, name in enumerate(LEVELS):
        cs = [s for t in tasks for s in per_task[t][control] if s.level == level]
        rs = [s for t in tasks for s in per_task[t][arm] if s.level == level]
        if not cs and not rs:
            continue
        pairs = []
        for t in tasks:
            mine = {(s.target, s.occurrence): s for s in per_task[t][arm]}
            for s in per_task[t][control]:
                if s.level == level and (s.target, s.occurrence) in mine:
                    pairs.append(mine[(s.target, s.occurrence)].turns - s.turns)

        def avg(xs, k):
            vals = [getattr(x, k) for x in xs if getattr(x, k) is not None]
            return st.mean(vals) if vals else float("nan")

        print(
            f"   {name:20s} {len(cs):4d}->{len(rs):<4d}  {avg(cs, 'turns'):5.2f}->{avg(rs, 'turns'):<5.2f}      "
            f"{avg(cs, 'calls'):5.2f}->{avg(rs, 'calls'):<5.2f}     {avg(cs, 'searches'):5.2f}->{avg(rs, 'searches'):<5.2f}   "
            f"{avg(cs, 'first_try'):.0%}->{avg(rs, 'first_try'):<4.0%}  {avg(cs, 'traced'):.0%}->{avg(rs, 'traced'):<4.0%} "
            f"{(st.mean(pairs) if pairs else float('nan')):+.2f} ({len(pairs)})"
        )


if __name__ == "__main__":
    main()
