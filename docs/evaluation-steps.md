# Step-level evaluation: no regression, gains on familiar steps

Status: adopted 2026-11-02. It replaces "accuracy and time together, at task level" as the primary criterion. Task strict, partial and time stay as guardrails.

## 1. Why

- AutomationBench tasks are mostly unique, so large gains in task accuracy are not expected.
- The graph acts on individual steps that recur across tasks or runs, not on whole tasks. Task-level metrics average its effect with the many steps it never touches, and that hides it.
- E022 is the example. The learning arm made about 12% fewer API calls and read 17–27% fewer input tokens, with accuracy within noise. It was judged "not a win" at task level. Nobody checked whether the savings sat on the steps the graph knew.

## 2. Success criterion

An arm succeeds when both hold against a same-day control:

1. **No regression.** It does not hurt what the model already does (§5).
2. **Step gain.** On familiar steps (§3) it reduces exploration and raises step accuracy, while unfamiliar steps stay unchanged (§4).

Task success may or may not improve. It is reported but not required.

## 3. Steps and familiarity

**Step.** Everything the agent does to reach one successful `api_fetch`, counted from the previous successful `api_fetch`. That includes the searches, failed attempts and turns in between. A step's **target** is the normalised endpoint of the successful call (`fixes.endpoint`). The final `finish` is not a step.

**Familiarity** of a step is judged only against runs that came before the trial. It never uses grades or the trial itself.

| Level | Name | Definition | Knowledge it tests |
|---|---|---|---|
| F0 | unseen | The target endpoint never succeeded in any earlier run | none (control) |
| F1 | endpoint seen | The target succeeded in an earlier run of another workflow | API mechanics (E028: transfers) |
| F2 | transition seen | F1, and in that run it came right after the same previous successful endpoint | local procedure |
| F3 | workflow seen | The same target in an earlier run of the same workflow (another variant or round) | recurrence (E026) |

A step takes the highest level it meets. A stricter variant (F2+, F3+) also requires that the step's argument sources (`lifting.source`) match those of the earlier run. It is reported alongside the coarse levels.

## 4. Step metrics

Computed per step, then aggregated per familiarity level. Arms are paired by task (same task, same day).

| Metric | Definition | Uses grades? |
|---|---|---|
| Exploration turns | Assistant turns in the step, minus one | no |
| Exploration calls | Tool calls in the step other than the successful target call (searches, failed fetches) | no |
| Searches | `api_search` calls in the step | no |
| First-try success | The first `api_fetch` to the target endpoint in this step succeeded | no |
| Traced arguments | Share of the target call's short argument values that `lifting.origin` traces to the query or an earlier response (not `Agent`) | no |
| Source agreement | Share of arguments with a confident, specific graph source (E025 strict rules) whose value matches it | no |
| Write-step accuracy | For writes, whether the grader checks that refer to this call's resource pass | yes, diagnosis only |

**Expected signature of a working mechanism.**
- F1–F3: exploration down; first-try success and traced arguments up; source agreement not down.
- F0: unchanged within noise. This is the control inside the same tasks.
- If F0 moves as much as F1–F3, the effect is not the graph's: it is noise, or a global side effect such as shorter contexts.

**Step pairing.** Steps of two arms are matched by (task, target endpoint, occurrence number), for example the second successful call to `GET .../values/{}`. Steps present in only one arm are counted separately as "added" and "dropped".

## 5. No regression

- **Regression rate.** Among the tasks the control passes strict, the share the arm fails. Separately, among all tasks, the share where the arm's partial credit is lower by at least 0.1.
- **Reference.** The same rates between two identically configured arms (A/A): E021 (two identical arms) and the no-hint subsets of E024, E026 and E027. The current noise estimate is about 5 of 45 tasks strict.
- **Verdict.** Non-inferiority: the arm's regression rate is not above the A/A rate by more than the margin. The margin starts as the A/A rate's 90% upper bound (Wilson). Revise it as more A/A data comes in.
- **Per-step diagnosis.** For every regressed task, name the first step where the arms diverge and its familiarity level. A mechanism that hurts should show where, for example recruitment v1 in E026.

## 6. Report template (every live arm from now on)

1. Task guardrails: strict, partial, median and mean time, turns, against the control, paired.
2. Regression rate against A/A, with the regressed tasks and their divergence steps.
3. Step table: for F0, F1, F2 and F3, the number of steps, and control → arm for exploration turns, exploration calls, searches, first-try success, traced arguments and source agreement.
4. Added and dropped steps per level.
5. One line on the mechanism: where it fired, and at which levels.

## 7. Implementation

`scripts/step_metrics.py` (built 2026-11-02; usage `step_metrics.py <run> <control> <arm> [v1|no-note|no-hint] [--stream]`, test in `tests/test_step_metrics.py`):
- `steps(episode)`: split a trajectory into steps, keeping turn boundaries from the raw trajectory.
- `familiarity(step, prior)`: F0–F3 from an index built from runs earlier than the trial, in run order. Use `load()` and exclude the run's own trials, as `experience_graph` does.
- `pair(control, arm)`: match steps by (task, target, occurrence).
- `report(run_dir, control_arm, arm, aa_runs)`: print §6.
- Tests in `tests/test_step_metrics.py`, in the style of `tests/test_experience.py`.

## 8. Tracker

| Run | Control → arm | Task guardrails | Step report | Regression vs A/A | Notes |
|---|---|---|---|---|---|
| E022 `runs/e022-learning-stream` | fixes → learning (v1–v3) | done (journal) | done (`--stream`) | non-inferior: partial drop 10% vs A/A (v1) 9.5%, upper bound 17% | F3: exploration turns/step 0.34 → 0.29 (paired −0.07, n=1244), searches 0.49 → 0.43; F1 worse (+0.36, n=59); F0, F2 flat |
| E024 `runs/e024-search-repeats` | fixes → repeats | done | done | non-inferior: 14% vs A/A (no-note) 14%, upper bound 30% | F3: turns/step 0.65 → 0.44, searches 1.23 → 0.68; no F0 control (all steps F3) |
| E026 `runs/e026-learned-sources` | repeats → sources | done | done | non-inferior: 14% vs A/A (no-hint) 9%, upper bound 20% | F3: turns/step 0.47 → 0.38 (paired −0.10, n=435); written values found, not invented, 33% → 54% |
| E027 `runs/e027-chained-sources` | repeats → sources (chained) | done | done | non-inferior but close: 17% vs A/A (no-hint) 9%, upper bound 20% | F3: turns/step 0.51 → 0.47 (paired −0.11, n=446), searches 0.80 → 0.66, found values 33% → 42%; recruitment v1 harm replicated |
| E029 `runs/e029-hint-once` | repeats → sources (once, after a stall) | done | done | non-inferior: 14% vs A/A (no-hint) 7%, upper bound 17% | F3: turns/step 0.42 → 0.34 (paired −0.05, n=502), searches 0.70 → 0.56, found values 30% → 45%; headcount 3/3 again |
| E033 `runs/e033-branch-warnings` | sources → branch (warnings at the point of choice) | done | done | effectively A/A (2/42 warned): 21% vs A/A (no-warning) 9/40, upper bound 32% | Inert: the warning is suppressed because the value-origin read was already called; gorgias harm from the "calls not made yet" hint |
| A/A reference | E022 v1, no-note and no-hint subsets | — | — | done per run | Small n (14–42); a dedicated A/A run would tighten the band |

## 9. Open decisions

- **No F0 control outside the stream.** Every workflow has run since E016, so with the full history almost every step is F3 (E024, E026). The in-task F0 control exists only where the mechanism's history excludes the workflows (`--stream`, E022). Options: evaluate future arms on workflows held out of the graph, or label familiarity by the mechanism's own graph rather than all history.
- **Traced arguments** are computed for write calls only: read arguments (ranges, search strings) are legitimately chosen by the agent.

- **Matching for F1 and F2.** The default is the normalised endpoint plus the previous endpoint, the level E028 showed transfers. The stricter argument-source match is reported alongside.
- **Turn boundaries in a step.** A turn with calls to two endpoints counts toward the step of each endpoint that succeeds in it. Its exploration turns are not double-counted.
- **Margin for non-inferiority.** Fix it once there is enough A/A data; until then, report the rates and the band.
