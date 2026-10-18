# Small-sample mechanism diagnostic — results

Status: complete, ten task executions. See the [prospective plan](small-sample-experiment-plan.md)
and [incremental manifest](../runs/small-sample-bd095ff2395c485bbe451ecfe3be3881/manifest.json).

Model: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`. Two synthetic
AutomationBench `simple` tasks, ten planned executions. Trajectory-only learning;
grades are report-only. No prompt/runtime changes during this experiment.

## Offline implementation checks

Before live execution, `uv run pytest -q`: **229 passed**. Focused tests establish:

| Mechanism | Deterministic evidence | What it does not establish |
| --- | --- | --- |
| Compact action/citations | Serialized action objects still undergo schema validation; frozen decisions resolve current evidence from task store | A lower invalid-output rate for the live model |
| Focused retrieval | Endpoint-template anchors retrieve matching claims and required dependencies, with explicit budget gaps | A performance advantage over flat retrieval |
| Goal context | All registered checks survive lossless compression; mandatory overflow fails explicitly | Completeness of the model's registered task interpretation |
| Checked finish | Real Stirrup scripted test keeps SQLite access on its owning thread | Correct live-model evidence selection |
| Rule promotion | Two later qualifying executions support a separate response rule; contradictions dispute it and suspend dependent procedures | Proof of causal explanations or transfer beyond scope |
| Trace proposals | Relevant trace neighbors are retained, unrelated/validation episodes excluded, overflow explicit | Autonomous useful artifact construction and measured savings |
| Refresh | Exhausted budgets/pre-observation exceptions preserve existing facts; unknown routes cannot be invented | Learned check value or automatic discovery of new routes |

Tests: [learning improvements](../tests/test_learning_improvements.py),
[decisions](../tests/test_decisions.py), [refresh](../tests/test_live_refresh.py).

## Live results

All conditions scored 2/2 on the hidden final-state grader. Clean acting completion
and evidence-backed completion distinguish them; task score alone hides the failures.

| Condition | Grader passes | Clean acting finishes | Environment calls | Acting calls / missing usage | Returned acting tokens | Returned semantic tokens |
| --- | --- | --- | --- | --- | --- | --- |
| [A basic](../runs/d30000a5a0d94130976b81c94539c8bd/summary.json) | 2/2 | 2/2 | 16 | 18 / 0 | 72,565 | 0 |
| [B operational](../runs/954fcf405cfd48348ebbf893301d7131/summary.json) | 2/2 | 0/2 | 15 | 36 / 2 | 403,204 | 0 |
| [C semantic](../runs/9047d03075744ca9a81724205712a7d6/summary.json) | 2/2 | 0/2 | 20 | 35 / 1 | 437,143 | 33,450 |
| [D fresh repeat](../runs/c828c6859fb7452b8250c2d0953b2469/summary.json) | 2/2 | 0/2 | 22 | 37 / 1 | 497,254 | 31,999 |
| [E learned repeat](../runs/1db9381be12d4f4fbc8510abda0032a8/summary.json) | 2/2 | 2/2 | 19 | 28 / 0 | 324,397 | 33,414 |

E alone achieved `checked` completion on both tasks. C, D and E each had one
semantic-stage timeout; a clean acting finish is not a fully successful learning
pipeline. Four interrupted acting calls have unknown usage, and semantic timeouts
also leave some usage unavailable. Counts above are returned-token lower bounds,
not complete billing. Boundary adaptation made no model call. E's row excludes C's
prior construction cost, which must be included when assessing amortized net benefit.

Baseline has no task-evidence completion ledger, so its clean finishes are not
evidence-checked completion. Both B sessions timed out. The first updated the phone
but attempted the same undiscovered GET contract three times instead of discovering
it. The second attempted goal updates containing an output-only `result` field;
the gate rejected these as invalid arguments. These are useful gate protections,
but the model's response to them did not produce a clean completion within budget.
Writing Salesforce also made previous email observations stale under the current
broad invalidation policy, adding verification work unrelated to the written service.

C's phone task exhausted all 20 turns without finishing; the email task timed out.
The phone task repeated a goal-check update with numeric array index `0` five times,
rejected each time by the string-only path schema. The dispatcher returns only
`arguments:invalid`, not the offending field/type, so the model receives little repair
guidance. Changing the index to string would still hit the dictionary-only traversal
gap described below. The email task obtained one matched check but kept a stale email
check. C spent another 33,450 returned semantic tokens and about 219.7 seconds on
consolidation/review. It retained four hypotheses across the pair, reviewed three;
the second task's review reached the shared 120-second semantic limit.

D also exhausted turns on the phone task and timed out on the email task. It used
31,999 returned semantic tokens, retained/reviewed one hypothesis, and hit the
semantic time limit on the second task. Its graph did promote **seven structural
output-type claims** to supported using observations across the two task/tenant
scopes. This is real cross-task structural induction, distinct from the more limited
model-proposed semantic-rule pathway.

### Procedure adaptation boundary

After C, the deterministic opportunity selector returned no eligible opportunity.
No proposal model call occurred; no artifact was installed. Its current inputs are
claim-based repeated compositions/recoveries, environment errors and changed claims,
not the local gate/goal-verification failure traces themselves. The recurring
verification problems therefore did not automatically become adaptation opportunities.
The E replay tests graph reuse only; autonomous procedure construction and measured
procedure savings remain unexercised in this pilot. Do not force a hand-written
procedure to make that metric nonzero.

### Learned replay (E)

The phone replay finished with grade 1.0 and `completion_evidence_status=checked`.
It used eight environment calls, twelve acting calls and 120,445 returned acting
tokens, versus C's twelve / twenty / 292,778. Both registered requirements matched
current observations. It used a direct record read with top-level `Phone`, avoiding
the unsupported array path. The initial decision received six injected claims and
all twelve decisions received labeled hypothesis context from the learned store.

This is a descriptive improvement over C/D for this task, not a causal attribution:
neither memory-only nor same-checkpoint repeated trials isolate the effect. It still
cost more acting tokens than baseline A (36,420). Its semantic stage rejected an
invalid numeric-path proposal and timed out during bounded repair; no new semantic
hypothesis was retained for that task. Acting success therefore does not imply a
fully successful learning stage.

The email replay also finished with grade 1.0 and both checks matched: eleven
environment calls, sixteen acting calls and 203,952 returned acting tokens. It received
graph claims on every decision, but **no semantic hint context** (C's email hypothesis
had not completed review). Its semantic stage retained two hypotheses and reviewed one.
The two-task result is therefore not evidence that semantic hints caused improvement.

E's total acting tokens were 25.8% below C and 34.8% below D; clean acting completion
improved from 0/2 to 2/2. However, email acting tokens rose relative to C and D, while
phone cost fell; the aggregate improvement is not uniform. Baseline A still used
far fewer tokens and finished both tasks without the extra verification guarantees.
E's final graph contained 51 supported structural output-type claims and one supported
value-flow claim. No model-proposed semantic response rule was promoted or prospectively
tested. No refresh, information-policy probe or procedure was invoked in any arm.

## Findings and next implementation priorities

| Part | Verdict from this pilot | Next step |
| --- | --- | --- |
| Shared graph | Working for interface and structural knowledge; eight prior claims injected into B's second task, and multi-task structural promotion observed | Keep shared environment knowledge; parameterize semantic transfer instead of equating task name with lasting tenant boundaries |
| Compact action/evidence interface | Calls and aliases work, but invalid inner arguments caused repeated loops | Return structured validation field/type errors; distinguish writable inputs from output-only assessment fields |
| Focused retrieval | Prior claims reached E on all 28 decisions; lower aggregate replay cost is suggestive | Isolate memory/retrieval effects in matched checkpoint trials; do not equate injection with usefulness |
| Goal verification | Refused unsupported finish; async callback worked; E checked both tasks | Support typed list paths, improve repair guidance, and distinguish historical task evidence from facts requiring current freshness |
| Semantic learning/review | Produced cited hypotheses but retained untestable rule specifications and sometimes exceeded time budgets | Validate conditions against actual tool schemas/serialization and paths against checker capabilities before retention; separate agent strategy from environmental necessity |
| Procedure proposal/reuse | Not exercised: no eligible opportunity, no model proposal or artifact | Include recurring local validation/verification failures as typed learning evidence and test their abstraction into opportunities; do not force a recipe |
| Free endpoint | Baseline and E acted successfully; no recorded acting-provider exceptions before outer task deadlines | No basis here for attributing failures to endpoint availability; model capability and latency remain confounds, not ruled out |

The pilot does not justify a large benchmark run yet. Prioritize the concrete
verification and rule-testability gaps, then repeat the matched diagnostic and add
the [in-sample checkpoint reliability protocol](small-sample-experiment-plan.md#user-clarification-during-execution-sharing-and-reliability).
Current `frozen` mode rejects previously learned tasks, so implement an explicit
in-sample measurement mode without weakening independent-evaluation protections.
Use pass^k at fixed k across checkpoints as the primary reliability metric, alongside
pass@1, pass@k, clean completion, verification quality and cost. **No standard pass^k
or pass@k is estimated from these changing-condition executions.**

Failed acting sessions retain evidence, call telemetry and decision audits, but the
runner saves full `history.json` only after `session.run` returns normally. Outer
timeouts therefore lose part of the raw model/schema-error trace. Improving that
failure-path logging would make future diagnosis more complete.

### Cross-task sharing observed

B's second task began with eight injected claims already used in its first task.
Its snapshot contains 19 supported endpoint contracts, 46 hypothesized output-type
claims and three hypothesized value-flow claims. Thus the graph is genuinely shared
across different tasks, not reset for each task. This is not yet proof of a benefit:
all semantic hypotheses remain task-name/tenant-scoped in the current implementation,
and retrieval of a claim alone does not establish useful use.

### Semantic rule testability gap (C, first task)

C retained and reviewed three hypotheses (precondition, effect, identity mapping).
The proposed effect's `response_rule` conditions use `params` as a dictionary, while
the actual `api_fetch` episode arguments retain a JSON string. `observe_rules` requires
exact type/value agreement, so that condition cannot trigger on that argument form.
The prediction path `records / 0 / Phone` also crosses an array, but the current
expectation evaluator only traverses dictionaries. Even with matching arguments it
would be unobserved. Review labeled the narrative plausible without validating those
execution semantics. This is an end-to-end gap despite passing synthetic rule tests.

The rule also embeds a concrete post-update phone value without a represented
"update has occurred" condition; it is not a general effect model. The precondition
narrative describes the agent's chosen read-before-write sequence, not proof that the
environment requires that sequence. Hypothesis status correctly prevents automatic
promotion, but executable-rule validation and abstraction still need improvement.

## Interpretation limits

The task pair is deliberately small and related. Scores have a ceiling if baseline
already solves both; cost and verification quality matter. Fixed order and stochastic
generation prevent causal attribution. Semantic/adaptation work has additional explicit
allowances; cost comparisons must include it. Returned usage does not recover unknown
provider billing. This is in-sample reuse, not independent evaluation or generalization.

Because semantic hypotheses are tenant-scoped and new rule promotion requires two
later qualifying executions, one replay per tenant cannot establish live promotion.
Refresh and hypothesis investigation count as unexercised if the model does not call
them; information policy is off in this matched pilot.

Recompute diagnostics without model calls:

```sh
uv run python scripts/summarize_small_sample.py runs/small-sample-bd095ff2395c485bbe451ecfe3be3881/manifest.json
```
