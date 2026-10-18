"""Per-task fixed-checkpoint reliability, never pooled across learning states."""

from math import comb


def reliability(outcomes, *, trials=3):
    if not outcomes or any(len(values) != trials for values in outcomes.values()):
        raise ValueError("Require the same complete number of trials for every task")
    if trials < 1 or any(type(v) is not bool for values in outcomes.values() for v in values):
        raise ValueError("Reliability outcomes must be explicit booleans")
    counts = {name: sum(values) for name, values in outcomes.items()}
    return {
        "task_successes": counts,
        "trials_per_task": trials,
        "pass_all": {
            str(k): sum(comb(c, k) / comb(trials, k) for c in counts.values()) / len(counts)
            for k in range(1, trials + 1)
        },
        "pass_any": {
            str(k): sum(1 - comb(trials - c, k) / comb(trials, k) for c in counts.values())
            / len(counts)
            for k in range(1, trials + 1)
        },
    }
