# Interaction learning changes

The learner derives routing, failure explanations and reusable behavior from public tool
schemas and observed interactions. No service-specific recovery recipe is added.

## Learning layers

1. **Evidence:** public schemas, local rejections, actual environment calls and verification
   feedback retain their origin and linked interaction IDs. Local feedback never updates
   live world state. Evaluation grades remain report-only.
2. **Interpretation:** the model proposes explanations, alternative explanations and
   distinguishing observations. Contrasts are observations, not mandatory preconditions.
   Nonstandard read operations can be interpreted from observed documentation, with quoted
   evidence and a separate review. This is documentation interpretation, not empirical proof
   of hidden effects. Unknown semantics remain unknown.
3. **Prospective testing:** existing response-rule tests use later observations, not the
   generating examples, and dispute contradicted claims. Repeated calls or failures can
   trigger bounded, metered reflection during the task. A reflection does not promote itself
   to a supported rule.
4. **Retrieval and use:** retrieval includes the current action and local failure context.
   Hypotheses are labeled unverified. Decisions retain the versions and evidence for claims
   the model declares it considered. This is an audit trail, not causal attribution.
5. **Procedures:** repeated linked failure-to-success interactions become investigation
   candidates alongside existing API recovery and composition candidates. Tool success does
   not establish that the sequence solved the task. A model must propose the parameterized
   program, dependencies, applicability and termination. The controlled runner requires
   independent development validation through the existing interpreter/adoption gates before
   reuse. Insufficient evidence or no proposal is a valid outcome.
6. **Evaluation:** development, artifact validation and evaluation task names are disjoint.
   Each evaluation task and arm receives an isolated copy of one checkpoint. Evaluation
   grades and sibling trial updates never enter training. Acting and learning calls are
   counted once, including failed calls with unknown token usage.

## Verification infrastructure

Root JSON arrays retain coverage by index. Read verification uses protocol semantics or
supported, current documentation interpretations rather than assuming all POSTs are writes.

For derived values, the model supplies source bindings, a bounded expression, an optional
selection predicate and explicit identity joins. Every used scalar input must belong to a
record connected to the target by checked equality joins. Source-to-source joins allow
transitive linkage across observations. Unrelated rows and container bindings are rejected;
collection calculations must enumerate scalar bindings (at most 32). Arithmetic and
provenance checks do not prove that a formula or selection predicate captures the entire
user request.

## Controlled comparison

Run `scripts/controlled_eval.py` with explicit `--train`, `--validate` (at least two), and
`--evaluate` task names. `WORLDMODEL_MODEL_CONFIG` selects the same model profile for acting
and learning. The output directory must be new.

Arms:

- `reset`: empty persistent knowledge; within-task learning enabled.
- `carried`: development checkpoint; disposable within-task learning enabled.
- `frozen`: checkpoint without knowledge updates; separate current-task evidence.
- `no_reflection`: carried checkpoint, no within-task reflection.
- `no_procedures`: carried checkpoint, no procedure execution.
- `no_semantics`: carried checkpoint, no model semantic updates or hypothesis hints.

These are runtime ablations. They do not remove the selected layer's contribution to the
training checkpoint. For a training-layer causal claim, train separate matched checkpoints.
Arm order rotates across evaluation tasks. The runner records raw run IDs, metrics, checkpoint
hashes and learning/validation overhead; it never infers improvement merely from completion.

The small run in `runs/controlled-learning-smoke` is a pipeline smoke test with four acting
turns and a 60-second task timeout, not an accuracy or generalization study. Use larger
matched task sets and adequate task budgets before drawing conclusions about learning.

## Validation result

At the earlier smoke checkpoint, the offline suite passed **391 tests**. The Gemma smoke run completed all six arms
(`runs/controlled-learning-smoke/summary.json`). All six scored zero strict success:
four hit the task timeout; `no_reflection` and `no_semantics` finished explicitly
unresolved. Training retained hypotheses but supplied insufficient repeated evidence for
an artifact proposal, so no procedure was adopted. Checkpoint hashes remained unchanged.

The run reported 161,917 tokens across training and evaluation, a lower bound because
some failed/cancelled calls had no reported usage. There is **no demonstrated learning
benefit** from this run. Its task budgets, single evaluation task and runtime-only
ablations do not support a performance or causal conclusion. The next substantive
experiment needs adequate per-task budgets and multiple matched evaluation tasks.

## Post-review acceptance work

The broad `controlled-gemma-20261021T040849Z` run was stopped with user approval;
its results and code snapshot are preserved. Frozen evaluation now uses a separate
checkpoint copy per task to retain current public documentation, without semantic
learning or carrying observations to sibling trials. Reflection can propose a
schema-checked distinguishing action and record its execution and predicted outcomes.
Procedure proposals now undergo a separate model review against the requested goal
before isolated execution validation. This review screens omissions; it is not proof.

`scripts/learning_acceptance.py` runs real-model recovery and verification fixtures.
The first run is recorded in `runs/learning-acceptance-20261021/summary.json`:

- Verification passed: GET, PATCH, GET, grounded verification and checked finish.
- Recovery acceptance failed: the model proposed the correct changed cursor and
  distinct predictions, but cited an observed `event_id` rather than its `episode_id`.
  Both internal identifiers were exposed in the proposal prompt. The generic repair
  response did not resolve this ambiguity. The actor subsequently read the record,
  but the rejected experiment prevents attributing this to experimental follow-through.

No corrective rerun was made at that stage after this failure, following the user's stop-and-diagnose
instruction. Proposed correction: expose one controller-owned evidence reference per
observation and resolve it internally; report invalid references precisely. Do not
hardcode cursor behavior or relax provenance checks.

`scripts/small_learning_comparison.py` prepares plain tools, the structured harness,
and retained/model learning on identical tasks. It refuses to run until both real-model
acceptance checks pass. This comparison has not yet been run.

## Scoped blocker acceptance (2026-10-21)

The evidence-reference correction passed both recovery and verification in
`runs/learning-acceptance-scoped-blockers-20261021/summary.json`. Recovery now sees
linked environment observations alongside wrapper observations, excluding unrelated
failures. Blocker hypotheses retain applicability, pause and reopening conditions;
they remain advisory hypotheses, not proof of permanent impossibility.

The offline suite passes 397 tests. The paired real-model check in
`runs/blocker-acceptance-20261021-linked-v3/summary.json` completed but failed both
acceptance criteria:

- Blocked: semantic learning attached a response rule to a blocker, violating a
  Python validator whose conditional constraint is not represented in the supplied
  JSON schema. Generic repair did not explain the violated constraint. The model
  also reversed the intended pause and reopening conditions. Recovery planning
  was not reached; the three requests were fixture seeds, not learned stopping.
- Recoverable: the model selected the returned cursor and the runner executed it
  successfully. Predictions used `ok` from the environment response, while outcome
  assessment used the wrapper response, where the path is `result.ok`. Both
  predictions were therefore unobserved and acceptance correctly failed.

The runner now converts executor ToolResult into framework ToolMessage for the
interaction recorder. Earlier runner startup failures are preserved separately.
No further production correction was made after the paired result. Next changes
need an explicit blocker schema contract with unambiguous retry/resume meanings,
specific bounded validation feedback, and an explicit prediction target distinguishing
wrapper and environment responses. These are interface corrections, not hardcoded
service-specific remedies. This check does not establish actor retry savings,
reopening behavior or transfer across tasks; the broader comparison remains unrun.

### Clarified contracts: paired acceptance passed

The approved correction exposes the blocker constraint in the model-facing JSON
schema (`possible_blocker` requires non-null conditions and a null response rule).
Field descriptions distinguish pausing retries from resuming them. Bounded repair
feedback includes controller-authored contract errors without echoing arbitrary
exception contents. Recovery predictions now explicitly select `tool_result` or
`harness_response`; the interpreter maps a tool-result path under `result` and never
guesses an alternative root after a mismatch. Existing proposals retain the wrapper
root by default.

All 399 offline tests passed. The completed Gemma run in
`runs/blocker-acceptance-20261021-contracts/summary.json` passed both cases without
model-output repair:

- Blocked: proposed an environment-grounded pause while the account remains
  suspended, reopening on evidence of administrator reactivation.
- Recoverable: proposed the observed current cursor; the runner executed the action
  successfully. Its tool-result prediction matched and the alternative was contradicted.

The blocked case has only its three seeded requests; this is not a measurement of
actor retry savings. The recoverable case has those seeds plus one executed model
proposal. Actual resumption after a condition change, cross-task reuse, and net
benefit on the broader benchmark remain untested by this paired fixture.

## Actor behavior evaluation

`scripts/blocker_behavior_eval.py` trains on two synthetic tasks, then compares
`no_knowledge`, `retained`, and `active` on three separate evaluation tasks:
recoverable cursor error, persistent account suspension, and suspension followed
by an explicit external reactivation notice. Training uses three actual seeded
failures per task; evaluation starts with no seeded failures. The actor selects
all evaluation tool calls and finishes. Each arm/scenario receives an isolated
knowledge store. Retained claims do not update; active learning uses the production
OnlineLearner and semantic hypothesis retrieval. This deliberately tests the
semantic learning layer, not all graph/procedure mechanisms or all benchmark tasks.

Metrics record successful reads, evidence-backed blocker reports, duplicate failed
calls, premature abandonment, resumption, false read reports, and model usage.
The resumption scenario runs a second actor session with an explicit public notice
and the same task knowledge store; it does not test autonomous detection of an
unannounced change. Actor phases have eight turns/180 seconds, with at most two
online reflections per active trial. Arm order rotates, the source checkpoint hash
is checked after every trial, and training costs are reported separately.

The initial run is `runs/blocker-behavior-20261021`; its source/config snapshot and
hash manifest are in `runs/launch-blocker-behavior-20261021`. Offline validation
passes 401 tests, including actual scripted actor sessions and grading controls.
The two abandoned acceptance startup directories (`linked` and `linked-v2`) were
archived in `runs/archive/startup-failures-20261021/runs.tar.gz` with an inventory.
Completed acceptance diagnostics and the earlier broad run were preserved.

### Actor behavior results

The nine trials completed with no errors. Each arm passed all three scenarios and
used six environment calls total, with zero duplicate failed calls, false read
reports, or premature abandonments. All three resumed successfully after the
external notice. The retained/active arms received hypotheses, but no evaluation
trial triggered online reflection; claims remained unchanged in all arms.

| Arm | Scenarios passed | Environment calls | Evaluation tokens |
| --- | ---: | ---: | ---: |
| No knowledge | 3/3 | 6 | 3,824 |
| Retained | 3/3 | 6 | 9,002 |
| Active | 3/3 | 6 | 9,002 |

Shared training used six model calls and 46,644 reported tokens, in addition to
six seeded environment requests. Checkpoint hashes remained unchanged and no model
calls lacked reported usage. `comparison.json` contains the aggregate results.

These fixtures demonstrate correct basic actor behavior and no observed harm from
retained blocker hints. They do not demonstrate a learning benefit: the baseline
already solved every case, hints increased evaluation tokens about 2.35-fold, and
active adaptation was not exercised. A useful next evaluation needs predeclared
ambiguous failures and held-out variants that require gathering new evidence,
including condition changes, before comparing benefit. Do not infer improved broad
benchmark performance from this small, explicit-feedback fixture.

## Opaque-error incident evaluation

`scripts/ambiguous_blocker_eval.py` extends the actor comparison with identical
initial `request_unavailable` errors for stale cursors and externally held access.
Following diagnostic references reveals current service state; successful diagnostic
reads do not count as reading the requested record. A blocker report requires an
observed held state. Resumption announces only a service update and requires a new
successful record read. Training task IDs and diagnostic references differ from
evaluation; task IDs exposed to the learner do not name the expected outcome.

Every trial starts with three executed failed requests, and every actor receives
the same public incident history. This deliberately exercises the active arm's
existing reflection trigger; it does not measure spontaneous triggering or claim
those seed calls were chosen by the actor. Actor calls and seed calls are separate.
Training uses the same actor tools and production reflection, plus post-task model
consolidation. Grading labels are never passed to the learner. Each evaluation arm
receives an isolated store, arm order rotates, and the checkpoint hash is checked.

The pre-run scenario/arm manifest, code and configuration are preserved in
`runs/launch-ambiguous-blocker-20261021`; results are written incrementally to
`runs/ambiguous-blocker-20261021`. The fixed design has two training tasks and nine
evaluation trials, twelve actor turns and 240 seconds per phase. No scenario or
prompt tuning is performed based on its results. Offline validation passes 403 tests.

### Opaque-error results

The first attempt was stopped after discovering an omitted required accounting
argument in the runner's post-task review call. That attempt and its source snapshot
are preserved. The corrected runner has an offline regression test; all 404 tests
pass. No scenario, prompt, or production learner changes were made for the rerun.

`runs/ambiguous-blocker-20261021-v2` completed all nine trials; its source snapshot is
`runs/launch-ambiguous-blocker-20261021-v2`. Checkpoint integrity held. All arms passed
recovery, evidence-backed blocking, and resumption. There were no premature actor
abandonments, duplicate failed actor calls, false read reports, or evaluation errors.

| Arm | Scenarios passed | Actor environment calls | Evaluation tokens | Reflections |
| --- | ---: | ---: | ---: | ---: |
| No knowledge | 3/3 | 11 | 11,577 | 0 |
| Retained | 3/3 | 14 | 28,307 | 0 |
| Active | 3/3 | 11 | 107,682 | 3 |

Each arm additionally received nine seeded failed calls across its three trials.
Shared training used 108,540 reported tokens and included a failed recovery
consolidation: the model mixed catalog references and episode IDs, causing citation
validation to reject the batch twice. No returned call lacked usage accounting.

Unlike the earlier explicit-feedback evaluation, active reflection ran in all
three active trials. It produced pause proposals in all three. In recovery, the
pause concerned repeating the obsolete cursor; the proposal itself listed lookup
and state diagnostics as remaining alternatives. Pausing that repeated action is
reasonable, but this supplied no concrete distinguishing-action proposal. The
actor continued diagnosing and succeeded; the evidence does not show premature
actor abandonment or establish that the pause caused its actions.

There is still no observed benefit: active learning matched baseline calls at
about 9.3 times its evaluation tokens, and retained knowledge used three extra
calls. This small synthetic run supports diagnosis, not general causal claims.
The retained checkpoint also reflects the failed recovery consolidation, so it is
not evidence about an ideally trained learner. Next implementation targets are a
consistent model-facing citation interface across consolidation/review and a
recovery decision that distinguishes pausing an ineffective action from pausing
all investigation while informative alternatives remain. No further implementation
changes or reruns were made in response to these results.

## Full iterative runner

`scripts/iterative_eval.py --full-learning` now enables operational graph memory,
semantic learning/reflection, and use of active procedures. After each task it calls
the existing trace-driven proposal and independent validation pipeline before any
new procedure can become active. Repeated tasks and procedure-validation tasks must
be disjoint, with at least two validation task names. Grader outputs remain report-only;
only the execution-cost allowlist enters planning. Candidate adoption is not guaranteed.

Both arms use the same mechanisms. Reset stores are fresh for every task; learn
keeps one store across tasks and iterations. `--reset-passes 1 --iterations 3` runs
one reset pass and three learning iterations. Atomic summaries include task metrics,
procedure-update/validation costs, missing-usage flags, artifact invocation/adoption
counts, errors and knowledge counts. A checkpoint is saved after each learning
iteration. Existing non-full mode remains available for compatibility.

Validation: 407 offline tests passed, including mocked full scheduling/persistence,
validation-task separation and accounting checks. The live Gemma run was launched
as PID 91667 with output `runs/full-iterative-gemma-20261021` and source/config snapshot
`runs/launch-full-iterative-gemma-20261021`. Startup was verified: the first reset task
was making real tool calls with operational/adaptive/semantic learning enabled.

Repeated tasks are `hr.job_posting_refresh`, `operations.contractor_badge_expiration`,
and `sales.zoom_meeting_followup` (a diagnostic selection from earlier runs, not a
random generalization sample). Six separate public tasks supply procedure validation.
There are twelve task trials, with fifty actor turns, forty environment calls and
900 seconds per task; semantic consolidation and procedure validation add overhead.
The process runs independently of the chat. One reset pass does not establish a
baseline variance estimate. This is an in-sample learning curve, not held-out transfer.
