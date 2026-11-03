"""E034 offline: where would the write check have stopped a send, and was it right?

All trials are replayed in the order they ran; each is judged by a graph of the earlier
runs only (Model.add after each). Before every write the trial made, the advisor's
check_write sees the run's state so far. A stop is counted as:
- a slip: the address it names (the near copy that was read) is one the grader expected
  in a check the trial failed (diagnosis only; grades are not used by the check);
- harmless: the trial passed no check with the written address and failed none with the named one;
- harmful: a check on the written address passed (the stop would have blocked a right send).
"""

import json
import re
from collections import Counter
from pathlib import Path

from worldmodel_harness.experience import WRITES, ExperienceAdvisor, Model, task_of, workflow
from worldmodel_harness.lifting import episode

ADDRESS = re.compile(r"(?:you wrote )(\S+); .* has (\S+)$")


def main():
    trials = []
    for run in (p for p in Path("runs").iterdir() if p.is_dir()):
        for path in run.rglob("trajectory.json"):
            task = task_of(path, run)
            if task:
                trials.append((path.stat().st_mtime, path, task))
    trials.sort()
    graph, stats, flows = Model([]), Counter(), Counter()
    for _, path, task in trials:
        e = episode(path, task)
        if not e or not e.steps:
            continue
        advisor = ExperienceAdvisor(graph, e.query)
        result = path.parent / "result.json"
        details = json.loads(result.read_text()).get("grading_details", []) if result.exists() else []
        stats["trials"] += 1
        for s in e.steps:
            if s.tool != "api_fetch":
                continue
            if str(s.arguments.get("method") or "GET").upper() in WRITES:
                stopped = advisor.check_write(s.arguments)
                if stopped:
                    stats["stops"] += 1
                    flows[workflow(task)] += 1
                    for line in json.loads(stopped)["message"].splitlines():
                        m = ADDRESS.search(line)
                        if not m:
                            continue
                        wrote, read = m.group(1).lower(), m.group(2).lower()
                        ok_wrote = any(d.get("passed") and wrote in json.dumps(d.get("params")).lower() for d in details)
                        missed_read = any(not d.get("passed") and read in json.dumps(d.get("params")).lower() for d in details)
                        stats["harmful" if ok_wrote else "slip" if missed_read else "harmless"] += 1
            advisor.observe(s.arguments, s.response, s.success)
        graph.add(e)
    print(dict(stats))
    print("stops by workflow:", flows.most_common(12))


if __name__ == "__main__":
    main()
