# Small-sample mechanism diagnostic — 2026-10-16

## Purpose and protocol

Written before execution. Diagnose the four implemented improvements without claiming
benchmark-wide gains: compact actions/evidence, focused graph retrieval, goal checks and
evidence-tested rule promotion, and trace-grounded procedure proposals/reuse.

Use only `configs/model-free-test.toml` (Nemotron free via OpenRouter), synthetic
AutomationBench `simple` examples, and fresh experiment-owned stores. Do not restart
the stopped 50-task run. Grader feedback is report-only; this is trajectory-only learning.
The same tasks are learning and test tasks; there is no held-out evaluation.

Select by workflow, before observing results:

- `simple.email_sf_contact_phone_update`
- `simple.email_sf_contact_email_update`

These related tasks exercise discovery, identity lookup, updates, and verification.
They are tool-use diagnostics, not the main AutomationBench score.

## Conditions (two tasks each, ten total executions)

| ID | Initial memory | Operational interface | Semantic learning | Adaptation | Question |
| --- | --- | --- | --- | --- | --- |
| A basic | Empty, no persistent memory access | Off | Off | Off | Can the model perform the tasks through ordinary tools? |
| B operational | Empty graph | On | Off | Off | Does the compact/gated/goal-check bundle run correctly? |
| C semantic | Empty graph | On | On | Experimental access enabled, initially no artifacts | Does trajectory consolidation produce usable, appropriately qualified claims? |
| D fresh repeat | Empty graph | On | On | Same access as C, initially no artifacts | How much does a fresh rerun vary without prior-task memory? |
| E learned repeat | C's persisted graph, after one bounded proposal attempt | On | On | At most one new experimental procedure, two uses/task | Are previous claims/procedures actually retrieved/reused, and do outcomes/costs improve? |

Run A, B, C, D, then construct at most one artifact from C's trajectories and run E.
Never pass A–D grades to the proposer. C and E share a store; other conditions do not.
All arms have identical task order, model, output allowance (8,192), 20 acting turns,
24 environment calls/task and a 240-second acting timeout. Semantic learning has an
additional 120-second/task limit and 16,384 output tokens; adaptation has a separate
120-second boundary limit. Record these extra costs rather than claiming equal total
budgets across arms. C/D/E have matched available mechanisms and allowances.
Provider retries remain the existing runtime policy; do not switch to paid routes.

## Evidence and acceptance criteria

| Mechanism | Inspect | Evidence of working | Failure / limitation |
| --- | --- | --- | --- |
| Compact decisions | Tool traces, decision audit, schema errors | Schema-valid dispatched calls, valid citations resolve | Invalid calls, missing real evidence, repeated rejected decisions |
| Focused retrieval | Retrieved/injected claim IDs and dependency/gap logs | Relevant prior claims injected on E, intact dependencies | Empty/irrelevant retrieval, gaps, context overflow; injection alone is not benefit |
| Goal verification | Goal-check ledger, completion status, grade | Observed checks support completion and agree with task outcome | Unresolved/stale checks, false completion, runtime error despite high grade |
| Rule promotion | Claim kinds/statuses, semantic review and prospective tests | Separate supported rule only after qualifying observations | Hypothesis count/review alone is not promotion; no triggering evidence means not exercised |
| Trace proposals | Boundary plan, cited traces, artifact version/dependencies | Valid bounded proposal derives from C evidence | Decline, error, or unsupported proposal; no forced recipe |
| Reuse | Invocation audit, successful subgoals, environment calls and returned tokens | Actual reuse with correct outcome and measured cost contrast | Unused procedure or estimated savings is not measured value |

Report grade and clean completion separately. Include failed tasks in denominators.
Returned tokens are a lower bound where provider attempts have missing usage; keep
acting, semantic, and adaptation accounting separate. Compare task-level A/B, B/C,
C/D/E outcomes and cost, with no statistical or causal claim from n=2. Fixed order,
shared model sampling, identical tenant/task repetition, and bundling are limitations.
Flat-vs-graph and individual-feature causal ablations are not established by this pilot.

Scope limitation identified during the preflight code review: each benchmark task name
is a distinct tenant, and semantic rules require two subsequent task executions in
their applicable scope. C-to-E supplies only one later execution per tenant. Therefore
this pilot can inspect prospective rule tests, but cannot expect a newly proposed
tenant-local rule to become supported. The deterministic promotion/contradiction tests
cover that transition; absence of live promotion here is not a failure.

## Offline checks, stopping, and artifacts

Run the existing test suite before live work; inspect focused tests for frozen/live
citations, refresh exception safety, complete goal context, rule promotion/contradiction,
dependency-aware retrieval, trace slices and async checked finish. Passing deterministic
tests is implementation evidence, not model performance evidence.

Save an incremental experiment manifest and original per-run summaries/traces under
`runs/`. Record results in a companion Markdown report with links. Do not tune prompts,
fix runtime logic, or silently alter budgets between conditions. Record failures for the
next implementation pass. Stop on repeated provider unavailability (two whole arms
with no environment interaction); preserve partial data. No expanded benchmark run.

Runner: `WORLDMODEL_MODEL_CONFIG=configs/model-free-test.toml uv run --extra automationbench python scripts/small_sample_experiments.py`

## User clarification during execution: sharing and reliability

The intended learner has one environment-level graph shared across task executions,
with scoped tenant/configuration claims and task-local live facts. Sharing a store is
not sufficient: retrieval eligibility and abstraction determine actual transfer.
Current endpoint contracts, structural output types and value-flow claims can cross
task boundaries. Semantic hypotheses retain tenant scope, and this adapter currently
uses the task name as tenant identity. This blocks semantic transfer between different
task names; it is an implementation limitation, not the intended architecture.
Do not fix it by globally exposing task-local record values. Broader semantic claims
need a parameterized scope, explicit applicability and additional testing.

Use checkpoint reliability as the primary subsequent evaluation, with pass@k as a
complementary measure. For checkpoint r, M tasks, n trials/task and c_ir successes:

$$
\widehat{\mathrm{pass}^{k}}_r = \frac{1}{M}\sum_i
\frac{\binom{c_{ir}}{k}}{\binom{n}{k}},\qquad
\widehat{\mathrm{pass@}k}_r = \frac{1}{M}\sum_i
\left(1-\frac{\binom{n-c_{ir}}{k}}{\binom{n}{k}}\right),\quad 1\leq k\leq n.
$$

Use binomial coefficient zero when its upper argument is below k. These are the
[τ-bench definitions](https://arxiv.org/html/2406.12045v1). Average per-task values;
do not raise aggregate pass@1 to the kth power. For fixed per-task success probability,
increasing k makes pass@k nondecreasing and pass^k nonincreasing. Across *learning
checkpoints at fixed k*, improvement of either is an empirical hypothesis, not a guarantee.

At each learning checkpoint, freeze persistent graph and procedure versions; reset
environment/live state and conversation for every replicate. Use the same tasks
and model/budgets, retain the learned graph, and keep trial evidence out of that
checkpoint's siblings. Resume shared learning between checkpoints. This can remain
entirely in-sample; freezing updates for measurement does not imply held-out validation.
Use at least three trials/task for a small pass^2/pass^3 pilot, and report the small
sample size. Record strict grader success and an additional clean-runtime success
criterion separately; unresolved self-verification remains a separate diagnostic.

The present ten-execution pilot cannot estimate standard pass^k: C-to-E changes
knowledge, and A/B/C differ in mechanism. Do not pool those outcomes into reliability
estimates. C/D are whole learning-stream repeats, not fixed-checkpoint independent
trials. Report online success, regressions and cost for this pilot, then run checkpoint
replicates as a separately declared experiment. The current `frozen` runner rejects
tasks already used for learning; an explicit in-sample checkpoint measurement mode
is needed, without weakening the independent-evaluation overlap guard.
