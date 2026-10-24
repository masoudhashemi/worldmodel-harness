# Experiment journal

## E001 — Does state-conditioned experience improve offline prediction?

Status: completed. Started and completed 2026-10-23.

### Goal

Determine whether conditioning historical environment feedback on symbolic state
improves prediction over matching only the tool/endpoint. This is a test of the
experience representation, not task reward, actor performance, or causal inference.

### Hypotheses

1. Capability-matched history predicts API outcomes and response structure better
   than a predictor with no experience.
2. Matching the preceding symbolic observations and proposed action improves on
   capability-only matching, with enough coverage to be useful.
3. Missing matches, contradictory outcomes, and truncated observations remain
   explicit rather than becoming confident semantic conclusions.

### Method

Use saved `iterative-eval-post-shadow` episodes for Asana fire drill, contractor
badges, and after-hours incidents. Baseline and iteration 1 are training; iteration
3 is held out. Iteration 2 is unused. Test episodes never enter training memory.
This tests repeated task instances, not generalization to unseen task families.

For each held-out environment call, build a query from its preceding observations
and proposed action before inspecting its response. Compare:

- No experience: uniform distribution over the four recorded API outcome labels;
  abstain on response structure.
- Capability history: empirical outcome distribution and modal response structure
  from training calls to the same capability and environment scope.
- State history: use matching symbolic context/action when available; otherwise
  fall back to capability history. Report exact-state coverage and fallback counts.
- Global history (control): use the scoped training outcome base rate and modal
  structure, without matching capabilities. This guards against an easy all-success
  dataset making any experience look impressive against a uniform baseline.

These are deterministic empirical predictors, with no LLM calls or actor changes.
Report outcome accuracy and multiclass Brier score, response-field/type F1, exact
response-structure accuracy, example-context characters, and differences on paired
calls. Structural outcomes are observational proxies; they do not establish
business effects, policy applicability, or task completion. Exclude truncated
observed response structures from structure scoring. Empirical distributions are
not claimed to be calibrated. Correlated calls within tasks are not independent
statistical trials.

### Results

Artifact: [evaluation.json](../runs/offline-experience-e001/evaluation.json).
Runner: [evaluate_experience.py](../scripts/evaluate_experience.py). The artifact
contains input/code hashes, source episode references, per-call predictions,
scores, and per-task breakdowns. Run artifacts are local and may not exist in a
fresh checkout.

Training supplied 90 transitions. Of 55 test environment calls, 51 had resolvable
queries and are the common denominator for all four arms. Four unresolved routes
are reported separately, not scored as prediction failures. The test set contained
54 successes and only one error; outcome accuracy alone is therefore misleading.

| Arm | Outcome Brier ↓ | Response-field/type F1 ↑ | Exact response schema | Mean example characters |
|---|---:|---:|---:|---:|
| No experience (uniform; schema abstention) | 0.7500 | 0.0000 | 0/51 | 0 |
| Global history control | 0.0392 | 0.1720 | 6/51 | 7,151 |
| Capability history | 0.0539 | 0.7453 | 31/51 | 5,527 |
| State history with capability fallback | 0.0539 | 0.7933 | 35/51 | 5,142 |

All arms had 50/51 outcome classification accuracy. For the uniform baseline,
the fixed tie-breaking rule chooses success; this does not indicate useful
prediction. The global-history base rate beat the capability/state outcome Brier
score. There is no evidence here that state conditioning improves error prediction.

Only **7/51** comparable calls matched observed state structure; **44/51** fell
back to capability history (one had no historical capability evidence either).
On those seven calls, capability-only schema accuracy was 3/7 and state-conditioned
accuracy was 7/7. The four changed predictions all improved, comprising three
`api_search` calls and one `api_fetch` call. This small, selected subset is not a
generalization or statistical significance claim.

Response schema ignores literal values and array cardinality. Example-character
counts cover up to three historical responses, not a complete LLM prompt. Relevance
is assessed only indirectly through predictive utility. Business effects, policy
interpretation, and information usefulness were **not** evaluated. Existing unit
tests cover identity renaming, distinct objects sharing an owner, and conflicting
outcomes; these are integrity checks, not additional benchmark trials.

### Decision

Keep the implementation offline. Capability history predicts response structure
substantially better than a global template; conditioning on state provides a
small additional benefit with low coverage. The dominant gap is representing
outcome-relevant semantic conditions rather than exact recent-history structure.

Next experiment: test evidence-grounded semantic predicates on a broader,
failure-rich held-out set, keeping this evaluator and split as a reference. Include
policy/reference resolution and representation-dependent errors. Do not interpret
schema accuracy as evidence that the actor chooses correct records or completes
tasks. No online prompt, execution gate, or learned production store was changed.

## E002 — Fresh LLM forecasts with experience ablations

Status: completed. Started and completed 2026-10-23.

### Goal and hypothesis

Test whether retrieved experience helps fresh model generations predict API
outcomes and response structure. E001 tested deterministic historical matching;
this experiment uses Gemma generations. This is still a forecast experiment, not
fresh task execution or a task-reward evaluation.

### Method

Keep E001 training inputs (baseline and iteration 1). Use previously unused
iteration 2 as held-out evidence. For each of three tasks select the first,
middle, and last resolvable `api_fetch` call by position, without selecting on
outcome. Generate three independent forecasts per case: no experience,
capability history, and state-conditioned history with capability fallback.
Nine cases yield 27 planned forecasts. Rotate arm order across cases.

All arms receive the same proposed action and bounded symbolic prior observations.
Only historical examples differ. Current context is bounded at 12,000 characters,
historical examples at 6,000. Omission counts are explicit. The selected action's
response is withheld until generation finishes. No grader, final summary, or
benchmark score enters a prompt. The endpoint availability check returned HTTP 200.

Use `configs/model-gemma.toml`, temperature 0, 3,000 output tokens, 60-second
per-call timeout, and at most one schema repair. Record prompts, forecasts, usage,
latency, invalid outputs, input hashes, and paired metrics. Stop after three
consecutive failed forecasts. Narrative expectations/uncertainties are saved but
not automatically treated as correct labels. This small pilot has no repeated
sampling or consistency estimate.

### Results and decision

Initial pilot artifact: `runs/offline-experience-e002/summary.json`.
Completed corrected run: `runs/offline-experience-e002-validated/summary.json`.

### Protocol correction during E002

Stopped the initial pilot after six saved forecasts. The model emitted response
paths without leading `/`, and the evaluator's schema failed to reject them. Thus
schema scores were confounded by an unvalidated output convention. The original
artifact is preserved with a stop reason and is not used for conclusions.

Added JSON Pointer leading-slash validation and a neutral formatting example to
the prompt; rerunning all arms under the same corrected protocol in
`runs/offline-experience-e002-validated/summary.json`. This is a protocol revision,
not a repair of historical answers or a selective rescore. The same selected
cases, training inputs, model, and budgets are retained.

### E002 final results

The corrected run completed **27/27 valid forecasts** (9 paired cases), with no
schema repairs needed. All input evidence hashes remained unchanged. Model usage:
77,129 input tokens and 9,651 answer tokens; summed model-call latency was 265.9
seconds. This excludes the stopped six-forecast pilot. The full test suite passed
505 tests; focused tests cover withheld-response isolation, paired denominators,
and JSON Pointer validation.

| Fresh LLM arm | Outcome accuracy | Outcome Brier ↓ | Response-field/type F1 ↑ | Exact response schema | Mean input-payload characters |
|---|---:|---:|---:|---:|---:|
| No experience | 8/9 | 0.20049 | 0.6055 | 1/9 | 6,585 |
| Capability history | 8/9 | 0.19936 | 0.8510 | 5/9 | 7,986 |
| State history with fallback | 8/9 | 0.19936 | 0.8510 | 5/9 | 7,757 |

Input-payload characters exclude the common system instruction and output schema;
actual token counts are recorded in the artifact. Four schema predictions improved
with historical experience. The failure case's field F1 slightly worsened
(0.154 to 0.143), so the benefit was not universal.

Two cases had matching state structure; their scores did not improve over
capability history. The other seven cases used the same historical evidence as
capability history. Identical prompts sometimes produced different narrative
wording at temperature zero, but the scored metrics were the same. This run does
not demonstrate an incremental benefit from state matching or high consistency.

All arms missed a Twilio POST that actually returned HTTP 400 with an
`Invalid JSON` parsing error (episode `15c6211c00674f4c874b9f6e613dd781`). They
predicted successful message submission. Crucially, the training split contained
**97 environment successes and zero environment errors**; 90 successes were
represented as transitions. Thus it provided no observed failure/recovery example
to teach this formatting failure. This is a coverage limitation, not proof that
failure learning works or fails. The model could potentially infer the error from
the proposed action, but did not in this test.

### E002 conclusion and next experiment

Fresh generations provide preliminary evidence that historical responses help
predict API response structure. They do **not** yet demonstrate improved error
anticipation, semantic state understanding, action selection, or task reward.

Keep the implementation offline. Next, build a separate failure-rich training
set from actual environment errors and recovery attempts, and evaluate on disjoint
held-out episodes. Preserve the E002 test set as held-out evidence; do not promote
its known error into training for a claimed improvement on that same case. Compare
representation-sensitive state predicates against capability-only examples.
A subsequent fresh-execution benchmark requires advisory actor integration and
repeated matched trials; it has not been run in E002.

## E003 — Learning request conditions from failures and recoveries

Status: completed. Started and completed 2026-10-23.

### Goal

Learn evidence-linked representation conditions from actual environment failures
and accepted retries, then test whether fresh model generations can discriminate
between candidate actions. Do not prescribe endpoint-specific requirements.

### Implementation and hypothesis

`recovery_conditions.py` extracts generic request observations: field presence,
JSON parseability, form field names, base64-to-UTF8 decodability, and header-block
syntax/names. The parsers do not decide whether an API accepts a representation.
Associations with acceptance are learned from source episodes. A failed attempt
and a nearby accepted request to the same resource create a contrast hypothesis,
with changed conditions and source episode references. Multiple changes and
intervening mutations are flagged as confounded; even a single measured change is
not a proven cause. Conflicting outcomes remain counts rather than invented rules.

The hypothesis is that these conditions provide more useful discrimination than
an unconditioned endpoint history. Policy applicability and arbitrary semantic
reference interpretation are not implemented by these format predicates.

### Data and method

Training runs: `285966d82e344a0485b208679eee581c`,
`391bf5dcbb9c406ab831ee743de4cb13`, `6401e884bf1e4bb188193c6929fd4ec3`,
`d9622678f6c04108965d90bac75812fe`.

Held-out runs: `863beeb92d754fbabc67b7d1acb7be0a`,
`96d1a9f552d54b83a7a6f8c70c48be6d`, `e85519a346c34cbabd6daba1e3603438`,
`85b3adc461aa402b902dc626a86396bf`.

These runs are separate from E001/E002 inputs. This is an exploratory split:
source errors were inspected to identify available failure modes, so it is not
an untouched blind benchmark. Exclude verbatim duplicate failed/accepted action
pairs across training and test; two repeated Twilio pairs were identified for
exclusion before generation. Pair selection is retrospective and outcome-based.
It tests discrimination among supplied alternatives, not free-form planning or
causal counterfactual effects at an identical state.

Three arms receive identical candidate parser observations: no history, unpaired
capability observations (including failures), and learned contrast hypotheses.
Alternatives A/B are counterbalanced; arm order rotates. Candidate responses and
expected choices are withheld from prompts. Only training hypotheses are shown.
Each forecast returns a choice, acceptance estimates, reasoning, and uncertainty.
Score choice accuracy, abstentions, and acceptance Brier on paired valid cases.
Budgets: 18,000 prompt characters, 2,000 output tokens, 60-second timeout, at most
one schema repair. No environment writes or online integration.

### Results

Output: [summary.json](../runs/offline-experience-e003/summary.json); learned
contrasts in [learned-conditions.json](../runs/offline-experience-e003/learned-conditions.json).

The training inputs contain 57 accepted environment calls and 8 errors. Of those,
the learner resolved 32 REST calls to preceding documented endpoint templates:
27 successes and 5 errors, yielding 5 recovery contrasts. Other calls include
discovery/non-REST operations and unresolved routes; inherited endpoint contracts
are not reconstructed from future evidence. The failure coverage is incomplete
and is reported rather than silently treating unresolved calls as learned.

Five held-out contrasts were mined. Two were excluded as exact repetitions of
training action pairs, leaving three distinct Gmail cases. All nine fresh model
generations completed and validated without schema repair; input hashes remained
unchanged. Usage: 10,869 input tokens, 1,668 answer tokens; summed call latency
50.1 seconds. The full test suite passed **512 tests**.

| Arm | Correct choices | Abstentions | Mean acceptance Brier ↓ |
|---|---:|---:|---:|
| No experience | 2/3 | 0 | 0.2767 |
| Unpaired capability history | 2/3 | 0 | 0.2083 |
| Learned condition contrasts | 2/3 | 1 | 0.1667 |

There is **no improvement in correct-choice count**. In the difficult case, the
baseline and capability-history arms selected a malformed encoded message over
the structured payload that the environment accepted. The learned-condition arm
abstained instead of selecting the rejected request. This is a useful difference
in uncertainty behavior, but does not solve the case or establish calibration.
Its Brier improvement is descriptive on three cases, not statistically reliable.
The abstention also assigned zero acceptance probability to both alternatives,
including the one actually accepted. It is not evidence of well-calibrated
uncertainty; it avoided a wrong selection while retaining a wrong belief.

The generated explanations reveal reliance on a pretrained assumption that the
send operation must use an encoded `raw` field. The observed accepted alternative
used a structured `payload`. Training contrasts did not demonstrate that accepted
alternative, and current array features retain member types but do not capture
the full semantic relationships inside header lists. These are coverage and
representation limits; the model's claim about what an API requires is not a
verified rule and is not promoted into the world model.

### Decision and next step

Keep this offline. Request conditions now preserve distinctions that the previous
shape-only abstraction lost, and learned contrasts can expose uncertainty, but
this pilot has not demonstrated better action selection or task completion.

Next, preserve relations inside structured key/value arrays and gather independent
evidence of alternative accepted representations. Validate on fresh disjoint
cases; do not train on this held-out accepted payload and claim improvement on
the same case. Policy conditions and reference resolution still need a separate,
evidence-grounded semantic extractor; this implementation handles representation
conditions only. No endpoint-specific acceptance rule or online execution gate
was added.

## E004 — New-case choice evaluation and repeated-generation consistency

Status: completed. Started and completed 2026-10-23.

### Goal and setup

Run fresh action-choice generations on separate trajectories while keeping E003's
training data and condition learner fixed. Candidate test runs:
`4893ce2ea437432db041f8687fe274c8`, `a803d5d9295c4a61981f108d465137fd`,
`de2b0eb587f94a0a9bd226b3fb7d40a3`, `e9a886c565714eb08c4f27a84afad963`,
`f8efc5fc88fa421694bb07ce0bf3ea28`.

These runs yielded no recovery pairs under local-discovery-only route resolution.
Several inherited contracts that are absent from their local discovery histories.
Added optional frozen-training contract seeding for test replay; source episodes
are retained, scope must match, and no test outcomes enter training. All arms use
this same route resolution. The representation, hypothesis construction, model,
and generation budgets remain unchanged.

This resolves 51 REST observations but only one eligible recovery pair. Rather
than claim a broad new evaluation, run **five fresh repetitions of this one case
per arm** (15 planned generations). Swap A/B between repetitions and rotate arm
order. Report correct choices and abstentions over repetitions, explicitly not
five independent cases. Same three arms as E003: no experience, capability
history, learned conditions. No end-to-end task execution.

### Results

Output: `runs/offline-experience-e004/summary.json`. This remains an
exploratory, retrospective acceptance-discrimination test. Candidate records can
differ and nearby recovery does not establish a causal effect at identical state.

### E004 completed results

Initial run: 15 attempted generations, 14 valid responses, one provider
`InternalServerError`. Retried that exact prompt once with unchanged settings;
the retry succeeded. The initial result is preserved in `summary.json`, the
retry in `provider-retry.json`, and both original/effective metrics in
[analysis.json](../runs/offline-experience-e004/analysis.json). There were 16 model
call attempts in total, yielding 15 usable forecasts. All evidence hashes remained
unchanged. The full test suite passed **513 tests**.

| Arm | Correct choices across five repetitions | Abstentions | Mean acceptance Brier ↓ |
|---|---:|---:|---:|
| No experience | 5/5 | 0 | 0.0060 |
| Capability history | 5/5 | 0 | 0.0000 |
| Learned conditions | 5/5 | 0 | 0.0000 |

All arms consistently selected the accepted request even when A/B positions
swapped. There is **no choice-accuracy improvement** over the baseline. The small
probability-score difference reflects higher confidence on one already-correct
case, not demonstrated calibration or general reliability.

This is **one distinct case**, repeated five times, not five independent task
examples. It does not establish broader accuracy, policy/reference understanding,
or improved end-to-end task reward. The supply of unused recovery pairs in the
saved trajectories is nearly exhausted under this extraction method. A larger
meaningful evaluation requires collecting fresh environment interactions and/or
expanding the state representation, rather than repeatedly rescoring this case.
