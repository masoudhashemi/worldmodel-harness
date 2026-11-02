# World model v3: a lifted experience graph

Status: design proposal and plan (2026-10-31), branch `world-model-v2`. It follows E014–E020. The [v2 design](world-model-v2.md) and the [experiment journal](experiment-journal.md) hold the evidence. Per-paper equations are in [literature and formulation](literature-and-formulation.md).

## 1. The idea

Organize experience as a graph so that, at a given point in a task, code knows:

1. **Local plans:** from the current node, the next steps that most likely succeed, at least a few steps ahead.
2. **Reuse across surface changes:** when a query is the same as an earlier one except for the user, email, id or other literals, what the earlier run learned about how to do it.

Both should come from experience: tool calls and results, never grades. Both should reach the agent in a form Gemma-4-31B actually uses.

## 2. What the experiments constrain

| # | Lesson | Evidence |
|---|---|---|
| X1 | Knowledge given before the agent acts (routes, plans, catalogs) replaces exploration. The skipped reads held the deciding facts, so accuracy falls. | E014 (−17 points), E015 (−31% calls, lower accuracy) |
| X2 | Showing Gemma state and asking it to judge against the task has no effect. | E017: acted on in 4 of 34 trials |
| X3 | A note works when code finds a concrete defect in the last call and gives a call to copy, including the phrase "set to a JSON string, for example body='{...}'". | E018, replicated in E019 (faster on 34 of 45 tasks, 17% fewer calls, same accuracy) |
| X4 | Open-ended instructions ("get the information from another source") become a new search goal. A note shown at the wrong moment can start a loop even when it is neutral. | E019, E020 (`hr.recruitment_channel_roi`: 0/2 with a note, 2/2 without) |
| X5 | Accurate knowledge is not enough. Dead ends were always correct and still net negative. | E019–E020 |
| X6 | Run-to-run noise is about ±2 tasks on 45, even at temperature 0. Same-day arms and per-task sign tests are required. | E016–E020 |
| X7 | The 1200 s timeouts are reproducible runaway generations (one model call runs the full 600 s, with `max_tokens = 12000`), not random server stalls. | E020 |
| X8 | Most failed checks are selection and rule errors (wrong rows, missed exclusions, values in the wrong place). The graph does not model these. | v2 §1.2 |

Consequences for v3:
- The graph is consulted by **code**, which turns its knowledge into concrete calls. Gemma never reasons over it as text.
- Its realistic target is **cost** (calls, time, tokens) at equal accuracy. Accuracy gains are a hope, not the claim.
- Every delivery channel needs a confidence and match level that decides whether it speaks at all.

## 3. Related work

None of the components is new. The proposal is in how they combine and in a delivery policy grounded in measured agent behaviour.

| Source | What it contributes | What we change |
|---|---|---|
| Procedural Graphs ([2609.09153](../papers/2609.09153-procedural-graphs.pdf)), closest | Graph G=(V,R,E,Φ). It finds the agent's node from its last tool call and takes the h=2 neighbourhood. An LLM turns that neighbourhood into text guidance every step. Graph edits are kept offline only if a validation score holds. | Edges carry counted confidence, not text. Suggestions are filled-in calls. Learning uses tool outcomes, not task scores. Guidance is not given every step (X1, X4). |
| AWM (workflow memory; a baseline in Procedural Graphs) | Replaces example-specific values in workflows with variables. | Each variable records where its value comes from (query slot or earlier response field), so a stored step can be filled in for a new task. |
| AutoGuide (also a baseline there) | Guidelines tied to the current state. | The state is lifted and includes the effects achieved so far. |
| Continual discovery ([2609.19551](../papers/2609.19551-continual-enterprise-wm.pdf)) | Rules as (condition, effect), added, refined and retired. | Learning happens alongside task execution; edges decay. |
| E-Ledger / WorldAbduct ([2610.11552](../papers/2610.11552-eledger-worldabduct.pdf)) | Splits static world knowledge 𝒲_S from dynamic task state 𝒲_D. | The same split decides what may be reused: structure, yes; instance data, never (§4.6). |
| Harness2 ([2610.14224](../papers/2610.14224-harness2.pdf)) | Lessons stored as (cue, lesson, evidence, confidence), each judged helped / hurt / no effect / confounded. | Applied per edge: was the suggestion followed, and did it help (§4.3). |
| Harness-G ([2607.27652](../papers/2607.27652-harness-g.pdf)) | Credits each step by information gain relative to other options. | Grade-free progress signal for edges (§4.3). |
| RRSI ([2609.24972](../papers/2609.24972-rrsi.pdf)) | Accepts a change only beyond a noise band δ, with a cost ceiling. | δ ≈ 2 tasks on 45; report cost and accuracy together. |
| Offline RL notes ([offline-rl-for-noisy-trajectories.md](offline-rl-for-noisy-trajectories.md)) | Pessimism (CQL, MOPO); learning from achieved sub-goals (HER); turning literals into variables (`experience.py`, removed in 1a91442, recoverable from 98beb65). | Lower confidence bounds for ranking; failed runs still teach the sub-steps that worked. |
| v1 state-local retrieval ([state_local_world_model_retrieval.md](state_local_world_model_retrieval.md), [state_transition_exploration_and_freezing.md](state_transition_exploration_and_freezing.md)) | Already built local plans: shortest path from the current node, several paths with confidence, freezing confident paths. | That was pushed into the prompt every turn and failed E014. v3 keeps the plan in code (§4.5). |
| Daw, Niv & Dayan 2005; Stachenfeld et al. 2017 | Choose between a stored plan and fresh reasoning by uncertainty; represent states by predicted next states. | Confidence and match level gate reuse; the local plan is a short prediction of next steps. |

## 4. Formulation

### 4.1 Lifting: what "same query, surface changed" means

A query q is split into a template and bindings, λ(q) = (q̄, β). Literals the code can recognise become typed slots: emails, ids, numbers, dates, quoted names, URLs. For example, "email alice@x.com about deal 006A" becomes "email {email} about deal {id}" with β = {email: alice@x.com, id: 006A}.

A trajectory τ = (a₁, o₁, …, a_T, o_T) is lifted to τ̄ = (ā₁, ō₁, …). In each lifted action ā, every literal argument is replaced by a term that records its source:

```
term ::= Query(k)          value of slot k of the query
       | Resp(j, path)     the value at a JSON path in the response of step j
       | Const(c)          a value that is the same in every run of the template
       | Free              no source found (the model computed or invented it)
```

The sources are found by exact value matching backwards through the trajectory: the query first, then earlier responses, latest first. A value seen identically across runs of the same template is promoted from `Free` to `Const`.

Two queries **match** at one of three levels:
- **exact:** same template q̄, and their successful lifted traces agree on the abstract action sequence;
- **prefix:** the lifted traces agree on their first k actions;
- **endpoint:** they share only endpoints.

Surface differences are exactly differences in β and in `Resp` values. They change the filled-in calls, not the plan.

### 4.2 Nodes

A lifted state is

  s̄ₜ = (c(q̄), āₜ₋₁, ōₜ₋₁, Eₜ)

- c(q̄) is the template cluster. It is optional: with no cluster, the graph falls back to template-free statistics.
- ōₜ₋₁ is the outcome class: success, or the normalized error signature from `fixes.signature`.
- Eₜ is the set of abstract effects achieved so far, as (method, endpoint template) pairs of successful writes.

The fix memory is the special case s̄ = (∅, endpoint, error, request shape).

### 4.3 Edges and confidence

Each edge (s̄, ā) keeps a Beta(α, β) posterior over **progress**, measured from tool results only. A step makes progress if:
- the call succeeded, and
- it added information or effect: a new entity or field in a response, or an accepted write to a target not yet written, and
- it is not a repeat of a call that already failed (Harness-G-style information gain, without gold answers).

Confidence used for ranking is the lower bound: LCB(s̄, ā) = the 10% quantile of Beta(α, β). Counts decay with factor γ per learning batch, so a service that becomes connected is eventually forgotten as a dead end. Episodes count once per edge, as in `FixMemory.learn`.

Two further counts per edge, following Harness2:
- **shown:** times code suggested this edge;
- **followed and progressed / followed and not / ignored.**

Progress after a suggestion is credited to the edge's suggestion statistics, not its independent statistics. An edge reaches the highest delivery level only with independent support, observed without any suggestion. This prevents the graph from confirming its own advice.

From failed episodes, edges that made progress are kept (HER): a run that failed the task still shows how to read a sheet.

### 4.4 Local plans

  π_h(s̄ₜ) = top-k abstract paths (ā₁ … ā_h), h ≤ 3, ranked by ∏ LCB along the path,

using beam search over the edge table. Each step is **filled in** from the current bindings: `Query(k)` from β, `Resp(j, path)` from this task's responses. A step whose terms cannot be filled (a `Resp` from a step not yet taken, or a `Free` term) stays abstract.

### 4.5 Delivery policy

The graph speaks only through these channels, chosen by match level m and confidence:

| Situation | Channel | Form |
|---|---|---|
| Last call failed, and an edge from this failure has LCB ≥ θ_fix | **Reactive fix** (current fix memory, generalised) | The defect, plus one filled-in call to copy |
| The same failure repeated, or a search loop (the same `api_search` query family ≥ N times) | **Loop breaker** | A fact plus one filled-in call; never an open-ended goal (X4) |
| m = exact, LCB ≥ θ_reuse, and the next step is a discovery step (`api_search`, listing ids or tabs) | **Discovery prefill** | Code makes the call itself, or offers the filled-in call, so the search is skipped |
| m = exact, before a write, with a high-confidence argument shape | **Pre-send check** | Silent unless the agent's call differs in shape; then the same note as a reactive fix, before the call is sent |
| Anything else | Silent | — |

Never delivered: whole plans in the prompt, routes before a failure, state reviews, or advice that asks the agent to decide something open-ended (X1, X2, X4).

### 4.6 What may be reused

Following the E-Ledger split:
- **Structure (reusable):** endpoints, methods, argument shapes, tab and field names, `Const` values, id formats, and how values flow between steps.
- **Instance data (never reused):** row contents, recipients, amounts, statuses, and any `Query` or `Resp` value from another task.

The rule is: **skip discovery, never skip observation.** Prefill may remove searches and id lookups. Reads of the data the task's rules depend on always run (X1, E015).

### 4.7 Objective

For arm A against the fix-memory arm F on the same day:

  accept A ⇔ strict and partial accuracy not worse beyond δ (≈ 2 tasks) and per-task sign test, and median time, p90 time and mean environment calls lower.

Accuracy and time are judged together: fewer turns at higher latency do not count.

## 5. How v3 differs from v1 and from Procedural Graphs

1. **Filled-in, not textual.** Variables record their source, so a suggestion is an executable call. In v1 and Procedural Graphs it is text the model must interpret.
2. **Gated by confidence and outcome.** Pessimistic counts with suggestion attribution decide whether the graph speaks. Procedural Graphs gives guidance every step; v1 injected plans every turn.
3. **Grade-free learning.** Progress is measured from tool results. Procedural Graphs gates edits on validation task scores.
4. **A delivery policy grounded in measured behaviour.** The channels in §4.5 are the ones E014–E020 showed a 31B model can use without losing accuracy.

These claims need the experiments below. A negative result on AutomationBench would not rule out the design for repeated workflows (§7).

## 6. Plan

### Step 0: housekeeping (no graph)

- **0a. Cap runaway generations.** Lower `max_tokens` from 12000 to about 2048, after checking that no successful turn in E018–E020 used more. Then one same-day run of fix memory with the old cap vs. the new cap. This fixes X7 in every arm.
- **0b. Dead ends only on repeat.** Use the error text ("No X account is connected"), fire on the second call to the same service, and fold it into the fix memory. It is small and rare (about 40 wasted calls across 225 trials); it can share the 0a run.
- **0c.** Commit the E019/E020 write-ups and code after review.

### Step 1: offline replay (go / no-go), no live runs

Build `src/worldmodel_harness/lifting.py` (query templates, lifted traces, source terms; restore what applies from `experience.py` at 98beb65) and `scripts/replay_lifted.py`. Train on the 741 trajectories excluding the 45 eval tasks; test on the eval tasks' trajectories from E018–E020. Report:

| Metric | Question |
|---|---|
| Template match rate (exact / prefix / endpoint) | How often does a held-out query match experience? |
| Source coverage: share of literal arguments with a `Query`/`Resp`/`Const` term vs. `Free` | Can steps be filled in? |
| Next-action top-1 / top-3 accuracy of π_h, abstract and filled-in, on successful steps | Do local plans predict what worked? |
| LCB calibration (predicted vs. observed progress, Brier score) | Is confidence meaningful? |
| Prefillable discovery calls and the turns they cost | What is the cost ceiling of discovery prefill? |
| Replay of each channel in §4.5: how often it would fire, and on which tasks | How often would it speak, and where? |

Also run the replay on the isomorphic-variant runs (`runs/eval-isomorphic-perturbations`, `runs/eval-isomorphic-transfer`). These are the setting where surface-changed reuse should pay off most.

Go criteria for a live arm:
- exact or prefix match on at least 30% of held-out tasks;
- filled-in top-1 accuracy of at least 70% where the graph would speak;
- an estimated ceiling of at least 10% of median time.

If AutomationBench fails these but the isomorphic split passes, the result is that reuse needs repeated workflows. We then evaluate on a task stream with repeats (§7) instead of forcing it here.

### Step 2: live arms, one channel at a time

Each arm runs against the fix memory on the same day: 45 tasks, rotated order, sign tests, with the X6 noise band. Add one channel at a time, in order of expected value and risk:

1. **Pre-send check:** the E018 mechanism before the call is sent.
2. **Search-loop breaker.**
3. **Discovery prefill:** only if Step 1 passes.

Stop a channel at the first replicated harm (two runs), as with dead ends.

### Step 3: learning over time

Run E013-style iterations. After each batch, fold the new trajectories into the graph (tool outcomes only) and update the suggestion statistics. Report each iteration in order. Never learn from a test task before its last evaluated round.

## 7. Evaluation protocol and threats

- **Grades:** offline diagnosis only, never learning or edge confidence.
- **Leakage:** templates and edges come only from tasks outside the test set, or from earlier iterations in a stream. Instance values are never reused across tasks (§4.6).
- **Benchmark fit:** AutomationBench has many distinct tasks. If its tasks rarely share templates, reuse cannot be shown on it. Use the isomorphic splits, or a stream in which templates recur with new entities, which is closer to production.
- **Self-confirmation:** handled by attributing suggestion outcomes separately (§4.3).
- **Noise:** δ ≈ 2 of 45; a single run is never enough to accept or reject.
- **Scope:** the graph targets cost. Selection and rule failures (X8) need a different lever, such as code-checked rules or a stronger model.

## 8. Step 1 result: offline replay on AutomationBench (2026-10-31)

`src/worldmodel_harness/lifting.py`, `scripts/replay_lifted.py`; output in `runs/v3-replay/automationbench.json`.
- **Training:** 678 trajectories, excluding the 45 eval tasks.
- **Test:** 315 trajectories of the eval tasks from E018–E020.
- Each step is predicted from the steps before it. No grades are read.

| Measure | Result | Go criterion |
|---|---|---|
| Eval queries with a near-identical training query (template word Jaccard ≥ 0.8) | **0 / 315** (median similarity 0.36; 14 at ≥ 0.6) | ≥ 30% |
| Next successful api_fetch, top-1 / top-3, from the last two steps | 44% / 66% | — |
| The same, conditioned on the 3 most similar training tasks | 16% / 27% (worse than global) | — |
| Correctly predicted calls whose every short argument is filled in correctly | 20% (8.6% of all successful fetches) | ≥ 70% |
| Argument fill-in by source: Const / Resp / Query | 68% / 25% / 7 of 7 | — |
| Literal arguments in eval calls with a traceable source (Query or Resp) | 32%; the rest are constants, ranges or computed values | — |
| High-confidence predictions (lower bound ≥ 0.6) | 1013 of 12184 steps, 91% right, but almost all trivial: search after search, a first search, Gmail send after `base64_encode`, finish after a Slack post | — |
| Search-only model turns | **28% of all turns**; 11.3 searches per episode (median 7) | — |
| Searches followed by an endpoint already known from ≥ 2 training episodes | 82% | — |
| Search-to-endpoint index learned from past searches: top-1 / top-3 hit | 44% / 63% (the environment's own search result names it in about 44%) | — |

Conclusions:
- **Reuse across surface changes cannot be tested on AutomationBench.** No eval task is a surface variant of a training task. Conditioning on the most similar tasks makes prediction worse, so the nearest queries are not structurally similar. The only isomorphic-variant runs (`runs/eval-isomorphic-*`) cover 4 tasks in an old store format. Testing this half of the idea needs a stream in which workflows recur with new entities.
- **Local plans are not good enough to deliver.** The next endpoint is right 44% of the time and fully fillable 8.6% of the time. The confident predictions are ones Gemma already makes. A live local-plan arm would not pass §4.7.
- **The real cost is discovery.** 28% of turns are search-only, and most lead to endpoints the graph already knows. Matching past searches to the endpoints they led to recalls the needed endpoint in its top 3 for 63% of searches. That is the remaining lever: add learned endpoints, each with a filled-in example call, to the agent's own `api_search` results. It reacts to a request the agent made and replaces no observation. Its offline ceiling is modest; it is the next hypothesis to test.

## 9. Learning over time: a recurring-workflow stream

The graph must keep learning, not learn once and freeze. The question is therefore a **learning curve**: as the graph sees a workflow more often, do accuracy and cost improve relative to an agent that learns nothing? Step 1 measured a frozen graph on unseen tasks, which cannot show this.

### 9.1 Surface variants (`src/worldmodel_harness/variants.py`)

E005 built 4 isomorphic variants by hand; they were removed in 1a91442. The new generator makes variants of any of the 600 tasks. It renames every surface identifier consistently across the user prompt, initial state and grader assertions:
- ids (with a digit or underscore, plus codes such as `BDG-3001`);
- emails and phone numbers;
- person names, including first and last names on their own.

Titles, companies, amounts and dates are kept, so the task's rules are unchanged. `task_row("<task>~vN")` loads variant N.

Validation (`scripts/validate_variants.py`, `runs/v3-replay/variant-validation.json`):
- 31 earlier passing trajectories were replayed with every literal renamed, including inside base64 Gmail messages. All 31 variants pass, as their sources do.
- No leaks: no old literal remains in any of 1800 variants (3 per task).
- One bug was found this way and fixed: 3-letter first names (Bob, Eve) that assertions check were not renamed.

### 9.2 Protocol (E022)

- **Stream.** N workflows (the 45 eval tasks), each appearing K times as variants v1…vK. Round r contains variant r of every workflow, in a fixed shuffled order.
- **Arms, same day, interleaved per trial:**
  - **baseline:** learns nothing;
  - **fixes:** E018 fix memory, frozen;
  - **learning:** the graph is updated after every trial from tool calls and results only (never grades), and speaks only through the §4.5 channels. Exact template matches now exist from round 2, so discovery prefill and the pre-send check have something to use.
- **Primary measure.** Per round r: strict, partial credit, median time and environment calls, for learning minus baseline. A learning effect is a difference that grows with r. Sign tests are per round, and the slope is taken over rounds.
- **Model-level curve (offline, alongside):** on each round's trajectories, next-action and fill-in accuracy as a function of how many earlier variants of the workflow the graph has seen. This tells whether the graph's knowledge improves even before it reaches the agent.
- **Leakage.** A trial learns only from earlier trials in the stream. Instance values never carry over, because they change between variants by construction.
- **Size.** 45 × K = 3 × 3 arms = 405 trials, about 7.5 h. Phase A runs baseline only on v1–v3 (135 trials, about 2.5 h), which also provides data for the model-level curve, before the full stream.

### 9.3 First model-level curve: the same task again (offline)

Each of the 45 eval tasks has 14 saved runs (E016–E020, in run order). For each task, the graph learned from the 678 background trajectories plus the task's first k runs, then predicted run k+1 step by step. The literals are unchanged here, so this is the easiest form of recurrence.

| Runs of this task seen (k) | 0 | 1 | 2 | 4 | 8 |
|---|:---:|:---:|:---:|:---:|:---:|
| Next successful api_fetch, top-1 | 0.40 | 0.56 | 0.58 | 0.71 | 0.72 |
| Fully filled-in call | 0.07 | 0.10 | 0.12 | 0.09 | 0.09 |

- **The graph does learn.** Endpoint prediction rises from 40% to 72%, with most of the gain after a single run, and levels off by about 4. The later runs come from different arms, so the agent's own run-to-run variation probably sets the ceiling.
- **Filling in arguments does not learn, which is a defect of the model, not of the data.** Even with identical literals it stays near 10%, for two reasons:
  - a constant needs the same value in 2 *different* tasks, so repeats of one task never create one;
  - `Resp(field)` takes the latest value of a field name, which is ambiguous when a response lists many records.

  Fix before E022: learn constants per template, and choose `Resp` values by the record they belong to (the record whose other fields match the query or earlier calls).

After the fix (constants learned per template; response values tied to their record, for example `Resp(id|name=Const(Recruiting Performance))`):

| Runs of this task seen (k) | 0 | 1 | 2 | 4 | 8 |
|---|:---:|:---:|:---:|:---:|:---:|
| Next successful api_fetch, top-1 | 0.40 | 0.56 | 0.58 | 0.71 | 0.72 |
| Fully filled-in call (share of all successful fetches) | 0.07 | 0.13 | 0.15 | 0.25 | 0.33 |

Both measures now rise with experience. At k = 8, 46% of correctly predicted calls are filled in exactly (0.33 / 0.72). This is the model-level answer to "does it improve as it learns": yes, for knowing the next endpoint and its arguments. Whether that reaches task accuracy is what E022 tests.

With terms also indexed by occurrence (the n-th call of an endpoint in the workflow learns its own values):

| Runs of this task seen (k) | 0 | 1 | 2 | 4 | 8 |
|---|:---:|:---:|:---:|:---:|:---:|
| Fully filled-in call | 0.07 | 0.16 | 0.19 | 0.37 | 0.46 |

At k = 8, 64% of correctly predicted calls are filled in exactly (0.46 / 0.72). Most remaining failures are iterations over records (one call per organisation, ticket or row). Which records to act on is the task's selection logic. It is deliberately left unfilled: E014–E017 showed that the graph must not make that choice for the agent.

### 9.4 Phase A and the curve on renamed variants (offline)

Phase A (`runs/e022a-variant-baseline`, baseline only, 135 trials): v1 11/45 strict, partial 0.46; v2 13/45, 0.50; v3 9/45, 0.44. Strict is close to the originals (about 13/45).

Curve on variants, from Phase A trajectories (`scratchpad/vcurve.py`; background excludes the eval tasks):

| Graph has seen | predicts | next api_fetch top-1 | fully filled |
|---|---|:---:|:---:|
| 0 variants | v1 | 0.45 | 0.05 |
| v1 | v2 | 0.66 | 0.16 |
| v1, v2 | v3 | 0.59 | 0.26 |
| 1 / 4 / 8 original runs | renamed v1 | 0.65 / 0.72 / 0.66 | 0.10 / 0.15 / 0.15 |
| 1 / 4 / 8 original runs | original run k+1 | 0.56 / 0.71 / 0.72 | 0.16 / 0.37 / 0.46 |

- **Next endpoint transfers fully under renaming** (0.72 vs 0.71 after 4 runs).
- **Fill-in learns across variants** (0.05 → 0.16 → 0.26), where nothing can be copied.
- **Fill-in from original runs stops at 0.15 on variants.** Failure classes per correctly predicted call, after 4 original runs (a call counts under its worst failing slot):

| | renamed v1 | original run 5 |
|---|:---:|:---:|
| filled exactly | 0.21 | 0.52 |
| a slot holds a value the agent chose (Free in the test run) | 0.43 | 0.10 |
| a slot traceable to a response, wrong term | 0.35 | 0.39 |
| a stale old literal | 0.00 | — |

  The gap is not memorised literals (0%). It comes from values the agent chooses itself, such as Drive search strings and sheet ranges. At temperature 0 the agent repeats these on the identical task, so a per-template `Const` matches them. On a variant it explores differently. Same-task fill-in rates (§9.3) therefore overstate what transfers.
- **Consequences:**
  - only traceable slots are candidates for prefill, never agent-chosen values;
  - the model's main fixable weakness is the wrong `Resp` term (about 35% of calls in both settings);
  - the 11/45 workflows whose prompts name people get a different template after renaming, because person names are not lifted. Lifting them would let template-level knowledge carry across those variants.

### 9.5 Transition graph without values, resolved against the run's state

Design, agreed with the user on 2026-11-01:
- **Transition graph** (across runs): lifted state → next action with support. On each edge, every argument has a **source**:
  - `Query(type,k)`;
  - `Resp(field[|anchor]) <- action`, the response of the named earlier action;
  - `Agent`.
- **Anchors** pick a record only through a query literal (`key=Query(..)`) or a name the query mentions (`key@query`). Otherwise the anchor is `?`, and the record is the agent's selection.
- **No task values on edges.** The only literals kept are an action's shape: the same value in ≥2 workflows and ≥80% of the workflows using that slot.
- **State graph** (one per run): the query and the run's responses. It resolves a source to a value. The transition graph offers sources by support, from the most specific level (this call of this workflow) to any task. The first source the state can resolve is used. `Agent` on top leaves the argument to the agent.

Curve (`scripts/replay_curve.py`, `runs/v3-replay/curve.json`):

| Seen → predicted | top-1 | traceable args right | coverage | precision | calls fully right |
|---|:---:|:---:|:---:|:---:|:---:|
| 0 variants → v1 | 0.45 | 0.10 | 0.13 | 0.35 | 0.05 |
| v1 → v2 | 0.66 | 0.21 | 0.28 | 0.39 | 0.11 |
| v1, v2 → v3 | 0.59 | 0.32 | 0.25 | 0.67 | 0.17 |
| 8 original runs → renamed v1 | 0.66 | 0.23 | 0.11 | 0.61 | 0.07 |
| 8 original runs → original run 9 | 0.72 | 0.38 | 0.16 | 0.73 | 0.12 |

- **Precision and coverage trade off.**
  - Without fallback across levels, precision was higher (v1, v2 → v3: 0.76; same task, k = 8: 0.89), but coverage was lower (0.21 / 0.12).
  - With fallback, coverage rises and precision drops.
- **Calls fully right stay at about 17% of successful fetches after two variants**, the same share as the value-holding model had. So removing values cost nothing on renamed variants and removed the inflation on repeats.
- **Conclusion for the live arm.**
  - **The endpoint carries over reliably:** 0.66–0.72 top-1, and it holds under renaming. It feeds the discovery channel, replacing `api_search` turns, which are 28% of all turns.
  - **Argument prefill reaches 0.67–0.76 precision.** That is too low to apply without a gate. It may only be offered per edge, when the lower confidence bound of that edge's past resolutions is high.
