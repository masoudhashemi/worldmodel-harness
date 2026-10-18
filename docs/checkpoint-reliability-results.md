# Repaired operational interface: fixed-checkpoint reliability

Status: completed, twelve executions. See the [predeclared plan](checkpoint-reliability-plan.md)
and [v3.1 measurement manifest](../runs/checkpoint-reliability-8c2197dd48c04115b1ce84ceed0be33c/manifest.json).
The [first manifest](../runs/checkpoint-reliability-741a102ba4844383934a78122622e7e8/manifest.json)
contains two completed executions and one interrupted execution, not a complete
reliability sample. The learned email trial finished checked but encountered seven
outer-schema rejections before dispatch, which still received generic errors. The
measurement was stopped to repair that boundary; its results will not be pooled
with the v3.1 restart. No run artifacts were removed.

## Implemented before measurement

| Repair | Implementation and safeguard |
| --- | --- |
| Actionable gate failures | Validation field paths, error types/messages and complete checked schema; invalid values are not echoed. Missing/ambiguous contracts report method, URL, candidate IDs and discovery guidance. No tool executes on rejection. |
| Array evidence paths | Typed integer indexes plus legacy numeric strings on arrays; object keys remain strings. Negative, absent and uncovered paths do not count as observations. |
| Evidence lifetimes | Goal checks explicitly choose current state or a historical read observation. Default is current state; a registered requirement cannot switch modes to evade freshness. Historical evidence never satisfies live prerequisites. |
| Write invalidation | Known writes invalidate same-origin reads including queries/collections, plus known cross-origin value-flow dependents. Unknown write identity/origin falls back to broad invalidation. This is an explicit conservative boundary policy, not learned causal effect precision. |
| Testable semantic rules | JSON params/body normalization for comparison, preserving raw episodes. Executable rules require an applicable cited observation covering the prediction; favorable review also enforces testability. Coverage and applicability do not establish truth or causal validity. |
| In-sample checkpoint measurement | Each single-task replicate gets a disposable copy of its source checkpoint. Within-task discovery/updates are local to that copy. Source hashes stay fixed; no replicate learning carries to another. Independent frozen-transfer overlap protection remains. |

Before restarted live calls: **250 tests passed**, including real Stirrup checkpoint isolation,
frozen-overlap protection, array paths, historical/current checks, argument repair,
cross-resource invalidation, normalized rule testability and per-task reliability arithmetic.
Outer-schema regressions exercise malformed nested action JSON, null actions and
extra fields through the actual SDK session, without executing rejected calls.
Entirely malformed top-level tool-call JSON can still fail in Stirrup's rendering
before dispatch; that provider/SDK boundary is not repaired by this change.

## Measurement

Twelve sequential executions: two tasks × two arms × three replicates. Nemotron free;
identical settings and seeded pairwise arm order. No procedures, semantic consolidation,
semantic hint injection or information-policy probes. Thus semantic-rule repairs are
tested offline, not evaluated for live learning benefit by this experiment.

Grader success, clean acting completion and matched requirement checks are distinct
outcomes. A correct final state may coexist with failed termination. Model-selected
requirement coverage is not independent task-success proof. Gate counts cover audited
dispatch rejections; failures before dispatch may require raw history inspection.
Returned tokens are lower bounds when calls fail without usage. Source graph construction
cost is outside the measurement, not zero.

## Results

Both source hashes remained unchanged. All twelve summaries report
`operational-learning-v3.1`; every task/arm has exactly three reset trials using
the declared free model. No paid fallback or 50-task restart occurred.

| Outcome / total over six trials | Empty checkpoint | Learned checkpoint |
| --- | ---: | ---: |
| Strict final-state success | 6/6 | 6/6 |
| Clean acting completion | 4/6 | 6/6 |
| Nominal checked completion | 4/6 | 6/6 |
| Finished with matched Phone/Email equality (post-hoc audit) | 4/6 | 5/6 |
| Environment calls | 55 | 49 |
| Discovery calls | 26 | 19 |
| Model calls | 100 | 91 |
| Returned tokens | 1,225,185 | 1,178,737 |
| Calls without usage | 1 | 0 |
| Model-call elapsed seconds | 1,076.9 | 988.1 |
| Inner gate rejections | 5 | 1 |
| Outer schema rejections | 2 | 4 |

Returned-token totals are observed lower bounds, not billing. The empty arm's
timeout includes one call without usage. Learned memory used 26.9% fewer discovery
calls and 10.9% fewer environment calls in aggregate. Observed returned tokens were
3.8% lower, but three of six paired trials used more tokens with learned memory.
Graph construction cost is excluded. These are descriptive pilot results, not a
statistically established or amortized efficiency advantage.

| Metric | Empty pass^1 / pass^2 / pass^3 | Learned pass^1 / pass^2 / pass^3 |
| --- | --- | --- |
| Strict grader success | 100% / 100% / 100% | 100% / 100% / 100% |
| Clean completion | 66.7% / 33.3% / 0% | 100% / 100% / 100% |
| Nominal checked completion | 66.7% / 33.3% / 0% | 100% / 100% / 100% |

Strict grader pass@1/@2/@3 is 100% in both arms. For clean and nominal checked
completion, empty pass@1/@2/@3 is 66.7%/100%/100%, and learned is 100% throughout.
Each estimator is computed within task, then averaged across the two tasks; no
pooling across graph states. Zero empirical empty pass^3 means neither task
succeeded cleanly on all three observed trials, not that its true probability is zero.

Raw clean/nominal-checked outcomes in replicate order:

| Task | Empty | Learned |
| --- | --- | --- |
| Phone update | pass, timeout, pass | pass, pass, pass (weak checks) |
| Email update | turn limit, pass, pass | pass, pass, pass |

All entries, including the timeout and turn limit, passed the final-state grader.
The grader therefore has a ceiling on this pair; it is not an invalid signal,
but it does not measure clean termination or evidence quality.

## What worked and what remains

- **Array checks worked live:** matched `records / 0 / Phone` and
  `records / 0 / Email` checks occurred in empty-arm successes. The earlier
  unsupported-array failure is repaired.
- **Evidence lifetimes worked in successful cases:** source email checks survived
  contact writes; empty/phone replicate 3 explicitly used historical source-email
  evidence and current-state Phone equality. This does not establish completeness
  of the declared service-boundary invalidation policy for arbitrary environments.
- **Repair feedback was exercised:** all six outer rejections were recorded with
  field/schema details. Runs containing rejections sometimes recovered, but
  empty/phone replicate 2 timed out with checks still missing. Rich errors do not
  guarantee that the model repairs its call in time.
- **Retained graph access showed a useful pilot signal:** fewer total discovery
  calls and more clean finishes. This comparison does not isolate which relation
  types caused that difference, nor measure learning across these twelve trials.
- **Nominal checks can be inadequate:** learned/phone replicate 3
  (`09d6268d31c44bdf9f4ed30ec4f6ccf9`) used `present` at the response root for both
  requirements. It checked no phone value. All other checked finishes included
  matched Phone/Email equality, though some source-email checks were presence-only.
  Equality coverage is a post-hoc diagnostic, not a replacement preregistered metric
  or independent proof of full task coverage.
- **Semantic-rule testability was verified offline only.** No semantic review,
  procedure proposal, refresh or hypothesis probe was enabled by this protocol.

The next targeted repair should address requirement-to-predicate adequacy and
separate missing-path feedback from stale-evidence feedback. A generic learned or
task-grounded check must establish the requested relation/value, not merely response
existence. Do not hard-code these benchmark answers or interpret grader success as
proof of an operational rule. Keep the two-task pilot for regressions; a larger,
predeclared task sample is needed before claiming reliable learning gains.

## Interpretation limits

Early v3.1 observation (empty/email, replicate 1): the final state passed, but the
model exhausted twenty turns. Its final update predicate used `[0, "Email"]` rather
than the response's root-field/array path, and the goal checker classified it as
`stale_or_nonlive_evidence`. Supporting array traversal does not ensure the model
selects the right root path; the combined freshness/path diagnostic remains too coarse.
No runtime change was made after the restart began.

These are two already-learned synthetic tasks, not an independent generalization test.
Three trials per task allow empirical pass^1/pass^2/pass^3, but are too few for strong
reliability conclusions. Same model temperature does not guarantee identical samples;
provider variation and trial ordering remain confounds. Replicates use identical initial
checkpoints and permit identical within-task update policies, not cross-trial adaptation.
