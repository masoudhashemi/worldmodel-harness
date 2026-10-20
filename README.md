# Operational world graphs for adaptive agent harnesses

An agent should learn its way around the environment it operates in: which tools access which tables, how outputs supply later inputs, what conditions actions require, what errors mean, and how outcomes can be verified or recovered.

It does that with several graphs, updated while the model interacts with the environment. They have different jobs and different lifetimes, so each one can make the model or the harness more effective without contaminating the others. The [architecture](docs/system-design.md#cooperating-graphs) specifies them:

- **Environment graph.** How the environment works: contracts, output shapes, value flow, errors, and possible causes. Versioned and reused across tasks, only as far as the evidence and scope allow.
- **Execution-tip graph.** How to carry out a recurring subgoal: a stored procedure and the claims it depends on. Reused until those assumptions change.
- **Feedback graph.** What this attempt showed: predictions, observations, and requirement checks. Mostly lives for the attempt. It steers the current task and tells the other graphs what to revise. A score is not, by itself, a new environment rule.

Live record values last only for the attempt. The evidence log is append-only and is what the graphs cite. On the next call the model sees a bounded working slice, not the whole store.

This project studies that design in a **multi-source learning architecture**. Six information layers organize scope and validity inside the graphs; evidence channels specify documentation, trajectories, probes and evaluator/teacher feedback; learning mechanisms transform permitted evidence into knowledge and execution support. At task time, the agent retrieves a working slice, refreshes consequential facts, and chooses actions for progress and useful learning. Accumulated knowledge can improve procedures, adapters, retrieval recipes and verification checks.

The connected evaluation/feedback view links requirements, attempts, trajectories, evaluator outputs, hypotheses and procedure versions. It does not replace the six layers. Task-success feedback is not automatic proof of an environment rule or cause. Once feedback influences learning or selection, subsequent influenced performance is **feedback-assisted**, not independent evaluation. The current runtime remains a grader-isolated, trajectory-only configuration of this broader design; feedback ingestion is proposed, not implemented. See [channel access equations](docs/formulations.md#evidence-channels-and-access-rules) and [evaluation-view connections](docs/system-design.md#evidence-channels-and-the-connected-evaluation-view).

Status: implementation in progress. Stirrup/LiteLLM runs the pinned upstream AutomationBench API toolset; scoped knowledge, retrieval, and a checked procedure interpreter have initial implementations. Full behavioral learning, adaptive artifact generation, and the controlled research evaluation remain incomplete. No research performance or novelty claim is established.

### Operational-learning v2 implementation

The four literature-driven mechanisms are connected to the operational execution path:

| Mechanism | Implemented behavior | Limit |
| --- | --- | --- |
| Compact action/evidence interface | Short task-local action/evidence references, bounded next-action schema, optional concise rationale; one costed schema/citation repair for semantic proposal/review | No automatic replay of a dispatched action or provider failure |
| Task-focused working graph | Current action/tool/endpoint and subgoal seed retrieval; endpoint-template matching; dependency bundles remain intact; redundant review histories stay in the store | Typed anchors plus lexical ranking, not a learned semantic retriever |
| Goal verification and rule testing | Public-task requirement ledger; finish checks current observations or explicitly reports unresolved work; prospective response tests produce separate supported predictions after two distinct task executions | Checklist coverage is model-proposed; response support is not proof of a causal explanation or hidden task success |
| Trace-grounded procedures and reuse | Adjacent failure/action/result slices inform proposals; stored triggers and stop conditions; bounded experimental deployment; invocation ID, steps, time, calls and completion recorded | Measured savings remain unknown until a matched control; no automatic promotion to independently validated status |

Mandatory task checks and prediction-check indices use **lossless column/row compression**: every registered requirement and check result is retained. No fixed character prefix or silent omission is used for these records. If compressed required state exceeds the declared shared context allowance, the run reports `MandatoryContextOverflow` with required/available sizes rather than executing with an incomplete ledger. Selected procedure trace neighborhoods also retain whole observations; explicit capacity checks fail rather than dropping records. Optional historical-knowledge retrieval remains a selection operation, with omissions/gaps reported.

The broader architecture permits documentation, trajectories, probes and grader feedback as distinct evidence channels. This implementation still uses the **trajectory-only configuration**, not an architectural prohibition on grader learning. Reports label `feedback_protocol` and `harness_revision`; comparisons reject mismatches. A connected grader-feedback learner is not implemented by these four changes.

The initial implementation was checked offline. Subsequent user-requested live integration tests use the free Nemotron profile below. The previously running 50-task curve remains stopped and its artifacts are intact. Small diagnostic tests are not evidence of learning-curve improvement; a fresh, consistently configured experiment is required for that claim.

### Value/binding safeguards (v3.4)

Rule literals now remain exact unless explicit example/parameter roles request
kind abstraction; constraints and unknown roles stay literal. Automatic source
binding requires one exact scalar-value source plus matching target identity and
records its provenance. Kind predicates cannot verify literal task outcomes.
Conflicting same-route alias schemas stay separate and are reported with scope.
See [design boundaries](docs/system-design.md#value-abstraction-and-binding-safeguards-v34).
Kind predicates cannot verify literal task outcomes. These checks are safety
constraints on what may be stored, not a measured performance improvement.

### Direct verification (v3.3 / v3.3.1)

The actor now has a flat typed `verify_goal` tool, separate from nested action
dispatch. Source checks use task-grounded identity plus observed content; effect
checks use a grounded desired value, current read, and same-record identity.
Failed updates retain pending requirements and a dedicated audit survives timeout.
No benchmark answers or grader feedback are installed in this mechanism.
Anchors can be repaired until they match, then they stay fixed. See
[design and limitations](docs/system-design.md#direct-sourceeffect-verification-v33).

### Operational-learning v3.1 repair and reliability pilot

Follow-up **v3.2** tightens verification: a checked finish requires non-root
equality against a fixed desired outcome, with the expected value grounded
in the public task or a distinct source read. API evidence must be a GET/HEAD of
the declared verification URL. Presence/type-only predicates, write acknowledgments,
self-grounding, and changed targets cannot establish completion. Invalid paths,
uncovered paths, and stale observations now have separate diagnostics. See
[verification scope and limits](docs/system-design.md#implemented-outcome-verification-v32).
Target paths can be repaired without changing the desired value/resource/grounding.
Offline tests cover the verifier. A past live pilot is not retained here and is not
evidence that the stronger checks produce a usable completion loop.

Checkpoint reliability repeats fixed tasks from an empty store and from a saved
store. Each trial uses a disposable working copy; the source checkpoint stays
unchanged. That protocol measures in-sample repeat agreement. It is not independent
frozen transfer or a learning curve.

The repairs add field/schema feedback at both validation boundaries, array-index
evidence paths, explicit historical/current goal evidence, conservative
service/dependency write invalidation, and executable semantic-rule testability
checks. These are generic interface and evidence repairs, not task-specific
solutions. Model-selected goal coverage and coarse freshness/path diagnostics remain
limitations; entirely malformed root tool-call JSON can still fail before dispatch
in the SDK renderer.

### Free model for live integration tests

The `smoke` command defaults to [model-free-test.toml](configs/model-free-test.toml). The default [model.toml](configs/model.toml) also uses free Nemotron. The explicit profile override keeps test selection unambiguous:

```sh
WORLDMODEL_MODEL_CONFIG=configs/model-free-test.toml uv run worldmodel smoke
WORLDMODEL_MODEL_CONFIG=configs/model-free-test.toml uv run --extra automationbench worldmodel automationbench --domain simple --limit 1 --operational --semantic-learning
```

The override is root-relative and applies to acting, semantic-learning/review and adaptation clients. A missing or invalid profile fails rather than falling back to Qwen. The exact route is `openrouter/nvidia/nemotron-3-super-120b-a12b:free`; there is no configured paid-model fallback. Unit tests remain offline. Record model settings and do not compare different models/context allowances as within-run learning gains.

OpenRouter lists this endpoint as free with tool/structured-output support, subject to availability/rate limits. Its endpoint notice prohibits confidential/personal data; use synthetic fixtures and appropriate public benchmark examples only. Credentials are loaded separately from `.env`, not included in prompts or reports. [Endpoint and data-use notice](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free).

The compact input parser decodes one extra JSON boundary around a nested action object, then applies the unchanged action schema and dispatch gate. Malformed objects remain rejected, and API argument strings are not recursively rewritten. The completion callback is async so it does not touch SQLite from Stirrup's worker thread.

Graph sharing works for environment-level contracts and structural claims; semantic
hypotheses are currently restricted by task-name-as-tenant scope. Reliability
measurement should use **pass^k at fixed graph/harness checkpoints**, with reset
replicates and an explicit in-sample protocol. Changing online graph states must
not be pooled as standard pass^k/pass@k trials.

### Current acceptance experiment: learn on the same 50 tasks

The requested first success test is repeated **in-sample** improvement, not a held-out split. Run:

```sh
uv run --extra automationbench worldmodel rounds --task-count 50 --rounds 3 --seed 1729 --operational --semantic-learning --task-timeout-seconds 900
```

This keeps the 40-environment-call, 20-turn, and 8,192-output-token limits, but
allows 900 seconds per task in every round. It starts a new empty store. The
completed `1a0673db473d49e7951040f37ad174a9` run scored 11/50, 11/50, and 8/50,
with 2, 12, and 14 timeout errors under a 300-second limit; both new-path flags
were off. Changing the execution path and timeout together creates a new
baseline, not a causal comparison against that run. Semantic explanations remain
hypotheses; prospective tests can now support a separate, narrower response-prediction
claim. Enabling the path does not complete the design.

New curve metrics and the read-only reporter include `runtime_diagnostics`:
returned per-call tokens (including calls in failed tasks), missing-usage call
counts, and strict passes grouped by execution error. Failed tasks remain in
the overall score denominator. These token totals are lower bounds, not complete
provider billing. Legacy task-level token totals remain for compatibility.

This selects 50 fixed tasks across the six public domains (not the `simple` diagnostics), starts with empty knowledge and no handcrafted artifacts, and resets each environment on every round. All 50 tasks are both learning and evaluation tasks. Only the scoped world model, evidence, and generated execution artifacts persist. There is no separate validation task set. Scores are recorded after execution and are not fed into the learner or proposal model.

The acting model now has an 8,192-token output allowance, fixed across the restarted experiment. The user-approved restart preserves pilot `25f5ef43781947e5aa37e28255c2e22a`: it stopped after nine completed tasks with one strict pass and five output-limit errors under the previous 2,048-token allowance. Its partial round is `024873337a3e4690abec7173322d2578`; data was not deleted. The replacement starts empty on the identical subset. Any difference between those two settings is not counted as learned improvement; comparisons within the replacement run use the same settings.

Between rounds, generic opportunity selection looks for repeated learned compositions/recoveries, failures, or disputed knowledge. The model may decline to change anything or derive a subgoal and parameter interface from those runs, then propose a bounded artifact. No benchmark-specific recipe configuration is used. Structurally checked proposals may enter the next round as **experimental**, not independently validated: one candidate per boundary, at most two experimental invocations per task. Estimated savings need not exceed construction cost to authorize this bounded experiment; they remain estimates. Normal dependency/budget checks and suspension after failed execution remain in place. Existing independently validated artifact paths remain separate.

`runs/<id>/learning_curve.json` records the fixed task names, per-round run IDs, success, calls, reported tokens, errors, artifact usage, and between-round learning decisions/costs. More claims or generated procedures is not success. The criterion is improved verified task performance and/or reduced total cost over repeated rounds; flat or worse results do not demonstrate the intended improvement. This protocol does not establish unseen-task generalization or isolate learning from stochastic variation. Failed-provider billing remains incomplete. The manually specified recipe configs elsewhere in this README are integration fixtures, **not evidence of autonomous improvement**.

For read-only paired reporting, use `uv run --extra automationbench python -m worldmodel_harness.curve_report runs/<id>/learning_curve.json`. It checks that completed rounds use the same tasks and controls, lists newly solved tasks and regressions, and reports call/token changes. Missing token usage remains unknown, not zero; no trend is claimed before two rounds finish. This reporting tool does not modify the learner or its evidence.

## Implementation decisions

### Opt-in operational execution

Future runs can add `--operational` to `automationbench` or `rounds`. This uses
structured decisions in the existing Stirrup loop, resolves known endpoint
requirements at dispatch, gates direct calls and procedure steps, and exposes a
costed task-local read refresh tool. Between rounds it can select a suspended
artifact for experimental repair from changed dependency evidence without a
manual repair specification. Missing or disputed dependency evidence defers
repair; repaired procedures are not labeled independently validated.

This mode was **not enabled** in the historical baseline
`2b714499b8904fc485e90774b47010f4`. Do not combine its results with that baseline
as one learning curve. Integration tests cover dispatch, discovery, refresh,
and shared budgeting; broad live-model performance of this mode remains unmeasured.
It still lacks semantic business-prerequisite induction, learned refresh-value
estimation, automatic evidence gathering for deferred repairs, and the full
controlled research evaluation. The presence of typed decision fields does not
establish that the model makes correct diagnoses or bindings.

Operational integration run `9809209211a34662904910e49945a366` used an empty
store and the configured Qwen model on `finance.audit_sample_selection`, a task
already in the selected 50. It passed with 10 environment calls (3 discovery,
7 task), 1 local utility call, and 127,962 input + 2,721 answer + 2,694 reasoning
tokens. Of 12 structured decisions, one fetch was blocked for unresolved
contract identity; subsequent discovery allowed progress. Seven decisions
checked retained contracts. No live-refresh or procedure invocation occurred.
This is integration evidence, not an in-sample learning curve or a causal
comparison with the baseline. Its gate timings are in the decision audit but
were not included in the aggregate compute ledger; subsequent runs charge
each gate once, including nested procedure-step gates.

`--operational --semantic-learning` additionally runs a bounded model-based
consolidation pass over task observations before scoring. It can retain scoped
configuration/behavior hypotheses about prerequisites, effects, identity,
routing, errors, and recoveries. Exact response citations are checked before
storage; the explanatory claims remain `hypothesis`. Optional typed response
rules are canonicalized without their prose, scoped, and tested on subsequent
observations. Two distinct task executions with matching covered predictions
and no contradictions support a separate `tested_response_rule`; unobserved
fields, duplicate events, and validation episodes cannot establish support.
Later counterevidence disputes the supported rule and propagates through dependencies. Alternatives
and distinguishing observations are retrieved as labeled diagnostic hints within
the existing context budget, never as permissions or mandatory rules. Model
usage and elapsed consolidation time are reported separately. This path has
deterministic tests but no live-model learning-benefit result yet. General causal,
precondition, identity and recovery verification remains beyond these narrow response tests.

First semantic integration run `bddae1fad1904749a6eb10f3210879b8` passed the same
finance task with 10 environment calls, but its consolidation failed with
`OutputTokenLimitError` after 138.16 seconds using the acting client's 8,192-token
allowance. No semantic hypotheses were retained; failed-provider token usage is
unavailable. Future semantic runs use a separately declared 16,384-token
consolidation allowance and a 180-second wall cap, recorded in experiment
settings. This change does not alter the running baseline or its acting budget.

Consolidation-only replay `026f5d2ab54c41b08979ba940f388975`, on a separate copy
of that run's evidence, hit the 180-second timeout at 16,384 output tokens and
retained no hypotheses. No environment calls were made. The public OpenRouter
model metadata then reported Qwen's default reasoning effort as `xhigh`, with
`low` supported. Future consolidation explicitly requests `low` effort while
acting-model settings remain unchanged. Reasoning shares the output allowance;
see [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
This is a prospective cost-control change, not a demonstrated fix or learning gain.

Low-effort replay `bc87bfb94aae4a08ae3ba42e01a69f69` returned after 71.35 seconds
(6,597 input, 4,118 answer, 255 reasoning tokens), but failed JSON parsing and
stored no semantic hypotheses. Future consolidation requests JSON-object mode
and retains malformed answer text for offline diagnosis. Citation validation is
unchanged. Faster response generation is not yet successful semantic learning.

JSON-mode replay `0b9bdaba773d49a4a3e34f6905734f36` returned in 30.90 seconds
(6,597 input, 1,574 answer, 255 reasoning tokens), but failed evidence validation
and retained no claims. Rejected parsed proposals are now retained as well as
malformed responses, and prompts specify the same canonical response encoding
used by the citation checker. No semantic-consolidation success is claimed.

Replay `5e3e758539ed4d0192fa3789bd2aff28` returned in 10.77 seconds but was also
rejected. Its retained proposal exposed a concrete mismatch: it cited whole
interaction paths (including arguments), while the schema required
response-relative paths. The learner now supplies a controller-generated
catalog of observed request/response nodes and accepts only reference IDs from
that catalog. The controller stores their exact source paths and excerpts;
fabricated references and altered source observations are rejected. Citation
validity still does not verify the hypothesis, and the revised catalog path
has not yet passed a live-model consolidation check.

Catalog replay `fc932769dd194f42a2181e8a0cb1d96f` retained three tenant-scoped
hypotheses in 24.96 seconds (8,132 input, 869 answer, 255 reasoning tokens), with
no environment calls. This verifies citation resolution and storage, **not
semantic correctness**. Inspection found a header-overwrite claim inconsistent
with the recorded post-append read, and an encoding claim with conflicting
requirements. All remain hypotheses. A semantic counterevidence/verification
stage is still required before treating this consolidation path as useful
operational learning; retained-claim count is not an acceptance result.

New semantic hypotheses now start with `needs_review=true`. A separate bounded
counterevidence review checks current and historical scoped observations, with
catalog-backed citations and version checks. Suspected contradictions, internal
inconsistencies, and unresolved claims remain withheld from diagnostic retrieval.
Only a `plausible` review permits a labeled hypothesis hint; no review promotes
it to a supported fact. Review tokens and time count toward consolidation cost.
This is model screening, not independent validation, causal inference, or proof;
its general effectiveness remains unestablished.

Review replay `94a6740ce0654f4b91664fe073dc5744` screened all three catalog-replay
hypotheses on a separate store copy, without supplied correctness labels or
new environment calls. It found counterevidence to the header-overwrite claim
and left the encoding and routing claims unresolved; all three remain withheld
from diagnostic retrieval. The review took 9.59 seconds and 9,897 input, 965
answer, and 1,248 reasoning tokens. Its bounded evidence selection omitted the
append observation needed to assess the routing claim fully. This is one
screening check, not demonstrated learning benefit, complete evidence retrieval,
or validation of a general operational rule.

Evidence selection now prioritizes hypotheses' source observations before recent
potential counterevidence, retaining whole observations within the same prompt
budget and reporting omitted source IDs. Replay `33d557ac21c041159a317c4d428067ae`
included every source observation: routing was screened as plausible, header
overwrite as contradicted, and encoding as unresolved. It used 9,664 input, 759
answer, and 1,371 reasoning tokens in 8.85 seconds, with no environment calls.
The routing claim remains only a hypothesis: successful parameterization does
not prove necessity. Retrieved hints now include the reviewer's caveat. This
replay tests evidence selection and screening, not task-performance improvement.

Future runs also record acting-model call latency and returned usage, preserving
completed-call usage even when a later call fails or is cancelled. Failed-call
billing remains unknown. Between-round planning receives an explicitly
allowlisted task-cost summary, never scores or verifier details. Tool execution
time is not total reasoning/discovery cost, and task-wide costs are not measured
savings attributable to a candidate. Missing measurements remain unknown. The
already-running 50-task curve retains its original code and cost inputs; these
additions do not retroactively improve its accounting or alter its decisions.

With `--operational --semantic-learning` and a non-off information policy,
`investigate_hypothesis` exposes a bounded investigation path for retained,
scoped semantic hypotheses. The model supplies a discovery query or GET/HEAD
request plus competing expected observations. Existing contract/prerequisite
checks and the shared environment budget still apply; mutations and automatic
remedies are excluded from this probe surface. Episodes retain the declared
uncertainty and actual observation, and attempted probes request renewed
screening rather than automatically promoting a claim. Unresolved hypotheses
can appear as explicitly labeled investigation targets, not operational advice;
claims screened as contradicted are withheld. This is covered by upstream-adapter
integration tests, but useful autonomous query selection and learning benefit
have not yet been demonstrated. Declaring two possible observations is not proof
that a query actually distinguishes them.

Live integration run `99abb4a5e1e74826a7463607faa6413d` repeated the finance audit
task in a fresh environment using a copy of store `33d557ac21c041159a317c4d428067ae`.
It passed with seven environment calls (six task, one discovery), one utility
call, 122,638 input, 3,484 answer, and 5,398 reasoning tokens. Eleven acting-model
calls took 94.49 seconds. Hypothesis context was injected at all eleven decisions,
but **no hypothesis probes were selected**. Consolidation retained four new
hypotheses; review then failed citation validation because the model returned
episode IDs instead of catalog references. The invalid review was not committed;
new hypotheses remain pending. Consolidation/review cost was 19.72 seconds. This
is an integration check, not a controlled learning-benefit result, and fewer calls
than earlier smoke runs do not isolate the effect of learned knowledge.

Repeated consolidation of an identical hypothesis from identical evidence now
leaves its version and review unchanged. Additional evidence preserves earlier
citations and screening history while requiring fresh review. This prevents
repetition from silently erasing counterevidence; it does not yet identify
semantically equivalent hypotheses expressed in different words.

Review prompts now enumerate valid catalog references in their citation schema.
Replay `f4279ddf868e42bd8b9b94f279093839` exposed a context-budget problem: no
observation fit the six-claim bundle, so no model review ran. Bounded-batch replay
`1520acc64c8346cda2ca2bd9234b8587` then produced valid citations for five reviews
(one plausible, one counterevidence, three unresolved; one claim deferred), using
8,760 input, 1,252 answer, and 1,195 reasoning tokens in 11.29 seconds. Missing
source observations still limited those judgments. Batch selection now reduces
the claim set further when needed to include its source observations, and reports
deferred claims or an impossible budget explicitly. Citation validation remains
strict; neither schema compliance nor a plausible review establishes truth.

Structured operational decisions can now supply up to eight `output_expectations`
before a primitive call. The evidence episode retains these model-proposed
predictions separately from predictions backed by existing claims. A deterministic
comparison records matched, contradicted, or unobserved results against actual
covered response fields; missing fields and unknown transport outcomes remain
unobserved. Nested field paths, presence, exact values, and JSON types are
supported. Checks do not establish hidden effects, causal relationships, or task
completion and do not automatically promote claims. Adapter integration tests
exercise the path; autonomous prediction quality and its task benefit remain
unmeasured. These checks concern primitive responses, not procedure-wide effects.
Before the next operational decision, losslessly compressed feedback reports all check
results with their evidence IDs and field paths. It shares the existing retrieval
character budget, never silently drops a check, does not duplicate response values, and explicitly distinguishes
an inconclusive observation from a contradiction. A test exercises the full path
from an AutomationBench operation through the stored check to the next model
input. The agent still decides whether to investigate or revise its plan; no
automatic retry or world-claim promotion follows from this feedback.

Review scheduling prioritizes unscreened hypotheses, then older reviews. A
complete-source review records a fingerprint of the available scoped evidence;
the same unchanged claim/evidence pair is not screened again on every task.
Additional scoped observations or a changed claim version make it eligible again.
Unresolved claims stay withheld, and other-tenant evidence does not trigger a
repeat review. Reviews with omitted source observations remain eligible. This
avoids redundant screening but does not guarantee every oversized claim can be
reviewed within the configured context budget.

An interim paired inspection of the first ten tasks in the unchanged 50-task
curve found three strict passes in round one versus one in round two. This is
not a completed learning curve. The tuition-reimbursement trace used 21 of 40
environment calls for discovery in round two (18 in round one), encountered a
malformed-JSON send request, and exhausted the budget after fewer successful
notification sends. Round one also exhausted its budget, but had already reached
the graded target state: task score and clean execution termination are distinct.
The customer-journey task fell from a strict pass to partial credit despite using
14 rather than 21 environment calls. Offline grading details identified missing
notification content and an extra notification, not transport failure. These
traces motivate testing discovery reuse and evidence-backed completion checks;
they do not establish that retained memory caused the regressions. Grading
details remain offline diagnostic evidence and are not supplied to learning,
retrieval, or adaptation. No task-specific fix was installed from these findings.

Operational mode now offers `lookup_knowledge` through `execute_decision` for
agent-directed local retrieval by capability or relationship. It reads current
scoped claim versions, applies freshness and dependency checks, and returns whole
claim bundles within the controller's context ceiling. It does not refresh live
records or replace an environment observation. A per-task allowance bounds local
queries; the ledger records their count and computation time. This supplements
automatic task-prompt retrieval and has no benchmark-specific routes. Tests cover
disputed updates, stale claims, cross-tenant exclusion, and ordinary upstream
discovery followed by local reuse. Live task benefit remains to be measured.

Targeted-lookup integration `326b5c7b89a540f4bfb600aa36d7633e` used a copied
learning-curve store on the tuition task. It selected four local lookups but
failed (zero partial credit), spending 22 environment calls, eight utility calls,
and all 20 model turns. The agent pursued speculative service locations based
on generic interface knowledge and mistook local prerequisite blocks for API
responses. The trace also exposed a contract-matching bug for URLs containing
query parameters. Matching now ignores query/fragment text without changing the
request or relaxing host/path checks. Blocked results explicitly identify the
local controller and state that no request was sent; memory instructions separate
interface contracts from tenant connections and record locations. These fixes
pass tests but were not present in that failed run. Local lookup use alone is
not evidence of better discovery or task performance.

Use **Stirrup** as the existing benchmark execution harness, with its **LiteLLM** client routing model calls through **OpenRouter**. Our contribution is the operational world model and its retrieval, learning, and execution-support adaptation, not a new general-purpose agent loop.

The default model is **Nemotron 3 Super 120B A12B (free)**, configured as `openrouter/nvidia/nemotron-3-super-120b-a12b:free` in [configs/model.toml](configs/model.toml). The live smoke test passed on 2026-10-18. All model-based tests use free Nemotron. The previous Qwen settings are preserved in [configs/model-qwen.toml](configs/model-qwen.toml) for explicit selection; there is no automatic paid-model fallback. The `openrouter/` prefix is LiteLLM's provider-routing prefix, not part of OpenRouter's native model ID.

Model tests load `OPENROUTER_API_KEY` from the repository-root `.env`; an explicitly supplied environment variable takes precedence. Never print credentials, include them in model context, or save them in experiment metadata. `.env` is excluded by `.gitignore`. The configuration loader validates routing and budgets and keeps credentials separate from saved settings.

Before benchmark runs, make generation limits, reasoning settings, context-management policy, retries, and tool access explicit and matched across experimental arms. Do not silently substitute another model on failure. Benchmark adapters must expose only permitted tools and keep hidden scoring information outside agent context.

References: [Stirrup](https://github.com/ArtificialAnalysis/Stirrup), [LiteLLM OpenRouter configuration](https://docs.litellm.ai/docs/providers/openrouter), [selected OpenRouter model](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free).

### Running the initial implementation

Python 3.13+ and `uv` are required. From the repository root:

```sh
uv sync --locked
uv run worldmodel check-config
uv run pytest -q
uv run ruff check src tests
# Opt-in live API calls; smoke defaults to the free test profile:
uv run worldmodel smoke
```

Use `--root /absolute/path/to/repository` when launching the CLI elsewhere. Offline tests use a scripted client through the real Stirrup loop and never call a model. The live smoke run exposes only create/lookup tools and a finish tool: no shell, browser, filesystem, or grader access. Warnings about missing default tools are intentional.

The diagnostic fixture creates a ticket but returns a timeout on the first creation. Lookup can reveal its ID; a blind retry creates a duplicate and fails the separate grader. A fresh fixture resets all live state. This is an integration check, not a published benchmark or evidence of learned transfer.

Runs save agent-visible boundary observations and a summary under `runs/<run-id>/`. Summaries include public model settings, dependency versions, protocol limits, and returned token usage. Evidence is currently buffered and written on exit (including ordinary errors/cancellation), not crash-durable. Token usage covers responses returned by Stirrup, not billing for failed provider attempts; this is not yet the research cost ledger. Reasoning uses the provider default for this smoke test. LiteLLM retries are disabled, but Stirrup can make up to three attempts on transient errors. The run has six turns, six environment calls, and a 180-second wall timeout; a blocked call cannot mutate fixture state.

### AutomationBench integration

The optional dependency pins upstream commit `4a8e1061254004d9dac807054eed33fad7d1ff14`, not the local GraphTool fork. Install and run:

```sh
uv sync --locked --extra automationbench
uv run --extra automationbench pytest -q
# Baseline, no retained knowledge:
uv run --extra automationbench worldmodel automationbench --domain simple --memory none
# Development run: learn contracts only from permitted API discovery responses:
uv run --extra automationbench worldmodel automationbench --domain simple --memory graph
# Use the development run ID printed above; choose a different task:
uv run --extra automationbench worldmodel automationbench --domain simple --offset 1 \
  --phase frozen --memory graph --knowledge runs/DEVELOPMENT_RUN_ID/knowledge.sqlite
```

`--domain sales` selects the public sales tasks; `simple` is a tool-use diagnostic excluded from the benchmark's main score. `--task NAME`, `--limit`, and `--offset` select tasks. `--memory flat` uses the same claim representation without dependency/neighbor expansion. `--memory episodes` retrieves historical interactions instead of consolidated claims; its development mode retains raw episodes without invoking consolidation. Historical records remain explicitly separate from current task state. These are implemented access variants, **not yet a completed Q1/Q4 research evaluation**.

All memory modes budget the whole injected message, including metadata and any artifact menu. The renderer retains mandatory dependency bundles intact or omits them; full retrieval-gap diagnostics remain in run logs. Adaptive artifact menus are selected by current scope, prerequisites, dependency validity, and lexical subgoal relevance, rather than listing every active artifact.

Each task receives a newly initialized upstream world. Only task prompts and registered API tools reach Stirrup; world initialization and upstream assertion scoring stay in the controller. Frozen knowledge opens read-only, refuses tasks used for learning, and sends evaluation observations to a separate evidence database. Online mode allows learning between tasks and is explicitly a custom stream, not an official leaderboard protocol.

Learning now retains discovered endpoint contracts and structural behavioral observations. Output-field types start as hypotheses and need evidence from distinct tasks and tenants before they become supported empirical regularities. Applicable predictions are attached to the next call and compared only against covered response fields; missing fields and unknown outcomes do not count as contradictions. Error codes and changed-request recovery sequences are retained as hypotheses, not established causes or safe retry rules. Reusable patterns contain request names/types and endpoint templates rather than live record values. Complete effect, identity, input/output composition, diagnosis, and recovery models remain unfinished.

The composition learner now records `observed_value_flow` relationships from covered, successful, task-local pre-action evidence to a later successful call's structured body/query fields or documented URL parameters. Claims retain both endpoint dependencies, structural paths, request shapes, and episode provenance—not matching record values. Ambiguous matches, SQL substrings, unknown outcomes, short constants, and cross-task bindings are excluded. Distinct values across independent tasks/tenants can support an **observed reuse regularity**, not a causal identity rule or an automatically safe adapter. Changes to either endpoint trigger reconsideration. Transformation inference and executable binding validation remain unfinished.

`--information-policy task` adds `inspect_contract`: the agent names an uncertainty and alternative observations before using the existing API search operation. A target endpoint-claim ID tags a refresh; an untargeted search is a probe. Intent is retained in the actual episode, separately from observations; no automatic claim of uncertainty resolution is made. `--max-information-calls 4` caps this tool, and every call also consumes the shared environment budget. Ordinary API search/diagnosis remains available. `--information-policy future` additionally permits requests declared useful only for future tasks, in learning phases with persistent memory. These are initial information-gathering interfaces, **not a calibrated value-of-information policy or a completed Q3 comparison**. The default `off` preserves prior toolsets. Retrieval exposes bounded gap details rather than only a gap count, within the same whole-message context budget.

`--adaptive` exposes a checked `run_procedure` tool for active artifacts already in the knowledge store (with `--memory flat` or `graph`). Artifacts are append-only/versioned, require independent validation evidence and estimated benefit before activation, and use the same registered tools and budgets. Relevant dependency changes suspend required shortcuts; monitoring dependencies remain available. Ordinary task runs do not automatically initiate adaptation: use the separate development command below. This switch alone is not a completed adaptive-harness experiment.

Dependency checks preserve the original version pins while accepting evidence-only revisions and unchanged declared payload fields. Nested contract changes invalidate only overlapping field paths; missing required fields, changed conditions, disputed claims, and changed-then-reverted dependencies remain blocked. Execution rechecks stored suspension before each step, including after a tool call triggers invalidation. Retrieval and artifact execution share compatibility checks. These deterministic regression tests cover change handling, **not yet a completed notified/unannounced shift-and-repair benchmark (Q5)**.

### Artifact proposal and isolated validation

The development controller can ask the configured model for a typed artifact, check its bindings/tool access, execute it in distinct fresh AutomationBench worlds, and activate it only after the validation and estimated-benefit gates pass. No grader is used for this process: validation checks the artifact's declared observable subgoal, not benchmark task completion. The model cannot set its own adoption status, validation evidence, or cost estimates.

```sh
uv run --extra automationbench worldmodel adapt \
  --knowledge runs/DEVELOPMENT_RUN_ID/knowledge.sqlite \
  --adaptation-config configs/adaptation-soql.json
```

The example configuration declares the desired retrieval recipe, its input interface, two development-validation tasks with different inputs, and explicit reuse/savings assumptions. It is an integration example, **not automatic opportunity selection or a measured savings claim**. The proposal budget defaults to 8,192 output tokens; benchmark acting settings remain unchanged. This command intentionally updates the development store with candidate versions, validation evidence, and adopted artifacts. Both validation tasks become ineligible for frozen evaluation from that store. Failed candidates remain inspectable; corrected candidates are re-executed rather than activated through replay alone.

`configs/adaptation-message-read.json` demonstrates a multi-call recipe: search Gmail, bind the first returned ID into a message fetch, then verify both identity and body shape. Its subgoal explicitly permits the first match; it does not establish uniqueness or complete a business task. Observed value-flow hypotheses are supplied separately from supported contracts and cannot satisfy required-claim checks. URL components use a fixed `url_encode` transform, not generated code. The example uses a 16,384-token proposal allowance; acting settings remain unchanged.

For explicit repair, add `"repair": {"procedure_id": "<stored-id>", "suspended_version": 3, "reason": "<observed change>"}` to the adaptation configuration, using the actual suspended version. Refresh affected claims first. The proposal receives the suspended program and current dependency knowledge; the controller preserves its scope, input interface, and required dependency identities. Adoption requires fresh isolated executions and a new cost/benefit gate. Failed repairs remain suspended; a retry targets that newer suspended version. Historical validation is not reused. Controlled fixture tests change an output field: repinning alone fails, while correcting the binding and re-executing passes. This is repair-path integration coverage, not an AutomationBench shift result or an autonomous change detector. Interrupted validation recovery remains unfinished.

In run `eb8ace8b5f994e3cac07894b55367c40`, Qwen proposed a parameterized Salesforce query recipe, validation used two environment calls, and adoption succeeded. A previous candidate (`8207583b46d8419b9605fb3b59a472be`) failed both cases because its bindings used unsupported string interpolation; it was not activated. Static validation and the proposal prompt now make the reference/transform language explicit. A prior proposal attempt exceeded the smaller output limit; failed-provider token usage is not recovered by the current client.

Subsequent frozen adaptive run `dce3f9c0d3924d489ebdfd35a72a4798` passed `simple.email_sf_contact_account_update` with 17 environment calls, but its saved history contains **zero `run_procedure` calls**. The agent's success therefore does not demonstrate artifact reuse or benefit. Current runtime instrumentation also attaches artifact/step/execution identifiers to underlying evidence and reports artifact invocation counts for new runs.

### Paired four-arm pilot

```sh
uv run --extra automationbench worldmodel compare --domain simple --offset 5 --limit 1 \
  --phase frozen --knowledge runs/DEVELOPMENT_RUN_ID/knowledge.sqlite
```

This runs flat/fixed, graph/fixed, flat/adaptive, and graph/adaptive sequentially against the same immutable knowledge snapshot and task selection. A file hash checks frozen-store integrity between arms; paired reporting rejects task, model, runtime, protocol, and budget mismatches. Results include success, environment-call, and token contrasts and the graph-by-adaptation interaction. This is a descriptive pilot, not uncertainty estimation or a complete net-value calculation; order is currently fixed and bootstrap/maintenance costs still require complete accounting.

Pilot `eb93b6e8217e4acb8f7be0d436c0c75e` used the same simple lead-description task for all four arms:

| Arm | Strict success | Environment calls | Total returned tokens | Artifact invocations |
| --- | --- | --- | --- | --- |
| Flat / fixed | 1 | 5 | 33,324 | 0 |
| Graph / fixed | 1 | 5 | 27,543 | 0 |
| Flat / adaptive | 1 | 5 | 29,288 | 0 |
| Graph / adaptive | 1 | 5 | 34,566 | 0 |

Both adaptive arms received the applicable recipe in their menu but did not execute it. Token differences from this single stochastic stream do not establish an adaptation or graph benefit. The outcome specifically leaves useful multi-step artifact selection and execution reuse unproven.

Each run writes `summary.json`, an evaluation-only `evidence.sqlite`, per-task conversation histories, and (for new development/online stores) `knowledge.sqlite`. Grader results appear only after interaction and never enter learned memory. The default run has 20 turns, 40 environment calls, and a 300-second per-task timeout. `base64_encode` is counted as a local utility rather than an environment call. Token usage is reported separately; failed-provider billing and full adaptation costs are not yet accounted for.

### System-design implementation coverage

The full [system design](docs/system-design.md) remains the implementation target. Current coverage against its seven implementation steps (including opt-in paths, not a claim that the default stream runs all mechanisms):

| Step | Current evidence | Still required |
| --- | --- | --- |
| 1. Claims, scopes, evidence, live state | Typed records; append-only SQLite episodes/claim versions; task-local overlay; pre-evidence and predictions recorded at API boundary | Full binding and later-verification provenance |
| 2. Diagnostic fixture | Unknown timeout outcomes, lookup, duplicate detection, isolated tasks | Identifier transformations and controlled environment shifts |
| 3. Retrieval and freshness | Scoped lexical selection, dependency bundles, whole-message budgets, local targeted lookup, artifact applicability, costed task-local GET/HEAD refresh in operational mode | Richer entity/capability parsing, learned check values, discovery of new refresh routes, matched retrieval/refresh ablations |
| 4. Graph-informed execution and comparison | Operational endpoint/dependency gating; scoped semantic hypotheses and screening; pre-call output predictions, observed comparison, next-decision feedback | Supported semantic-rule induction, equivalent-hypothesis reconciliation, business/permission/identity prerequisites, evidence-backed task-completion checks, full functional models |
| 5. Adaptive execution support | Trace-driven opportunity selection, typed proposals, four named edit surfaces, applicability-sensitive selection, dependency suspension; separate isolated-development adoption and between-round experimental repair paths | Evidence gathering for deferred repairs, independently validated repair (not part of the requested in-sample curve), demonstrated repair/reuse value, full surface-specific adaptation |
| 6. Exploration and ablations | Shared call budget; contract inspection, hypothesis probes, live refresh; declared competing observations; model-call timing/returned usage and consolidation costs | Learned future-value policy, justified diagnostic/remedy selection, failed-provider billing, complete total-cost accounting, controlled Q1–Q5 comparisons |
| 7. Benchmark protocols | AutomationBench reset/scoring, development/frozen/online modes, repeated same-50 in-sample curve, frozen four-arm pilot, unannounced synthetic response-shift detection | ITSMBench adapter, notified/unannounced shift-and-repair evaluation, replicated controlled comparisons/statistical reporting, demonstrated net learning benefit |

The default stream and the ongoing 50-task curve still leave `operational` and
`semantic_learning` off; the original curve is not evidence for the newer path.
Semantic screening never promotes a hypothesis to a supported rule. Proposed
alternatives/remedies are not automatically executed; prediction contradictions
do not themselves revise claims or authorize retries. Refresh estimates are
model-supplied, not learned. The repeated-task protocol intentionally has no
held-out tasks, as requested; its experimental repairs are not represented as
independently validated or as a completed Q5 experiment.

Operational regression coverage now uses separate frozen knowledge and current
task-evidence stores, checks that rejected/pre-observation refresh attempts retain
the prior observation unchanged, and checks actual information-tool routing and
invocation audits. `information_invocations` counts typed calls to either
`inspect_contract` or `investigate_hypothesis`, including local refusals and
interruptions; `information_invocations_by_tool` and `information_audit` expose
that distinction. These invocation counts are not environment-call counts or
evidence that an uncertainty was resolved.

Passing integration tasks establishes plumbing, not the hypothesis that learned graphs or adaptation improve task-stream value. New runs write summaries under `runs/<run-id>/`. That directory is local and is not a tracked source artifact.

## The argument

The goal is to improve execution across tasks by retaining useful operational relationships, not to maximize the size of a graph or the number of harness changes.

1. **Problem:** agents repeatedly rediscover tool/data relationships, input mappings, failure explanations, and recovery conditions.
2. **Literature:** explicit rule learning, procedural memory, harness search, and world-model-guided adaptation already address parts of this problem. Their mathematics establishes the alternatives and overlaps.
3. **Common formulation:** separate environment dynamics, reusable knowledge, runtime state, harness, acting policy, and update mechanism; retain each method's evidence and protocol assumptions.
4. **Our hypothesis:** scoped operational claims, selective retrieval/refresh, and dependencies to reusable execution artifacts can make learning and adaptation more valuable.
5. **Design and tests:** the layers and functional models implement that hypothesis; matched-budget experiments determine which mechanisms earn their cost in ITSMBench and AutomationBench use cases.

Five questions connect the documents: **Q1 retained knowledge, Q2 selective use, Q3 useful learning, Q4 reusable execution support, and Q5 adaptation under change**. Their common endpoint is [task-stream value after all costs](docs/formulations.md#8-overall-objective). The [research questions](docs/research-questions.md) state what each one must establish and how a negative answer narrows the design.

## Documentation

| Document | Purpose |
| --- | --- |
| [Problem framing](docs/problem-framing.md) | The concrete problem, intended contribution, use cases, and research questions |
| [Research questions](docs/research-questions.md) | What Q1–Q5 must establish, the evidence that answers them, and how a negative result narrows the design |
| [Benchmarks](docs/benchmarks.md) | Candidate environments, Q1–Q5 fit, selection priorities, access caveats, and standard versus custom protocols |
| [Literature formulations](docs/literature-and-formulation.md) | Cumulative argument from existing work, comparison figures, and all 17 mathematical profiles |
| [Methods, prompts, and code](docs/methods-prompts-and-code.md) | All 17 methods, prompt contracts and source locations, pinned code inspections, and prioritized implementation candidates |
| [Mathematical formulation](docs/formulations.md) | Common framework, seven overlapping mathematical families, method instantiations, and our layered specialization |
| [System design](docs/system-design.md) | Mechanisms traced to Q1–Q5, worked examples, and controlled tests |
| [Paper guide](papers/README.md) | Local PDFs, additional research references, and benchmark sources |

Procedural Graphs motivates the distinction between environment knowledge and execution structure. WorldAbduct and continual neurosymbolic learning are close comparisons. Cognitive models motivate testable design choices, rather than anatomical claims. ITSMBench and AutomationBench are motivating use cases; continual task streams require an explicit evaluation protocol.

Read the [literature synthesis](docs/literature-and-formulation.md#from-existing-formulations-to-our-research-question) for the argument. Follow its mathematics in order: [individual papers](docs/literature-and-formulation.md#paper-by-paper-mathematical-formulations) → [standardized framework](docs/formulations.md#part-i-standardized-comparison-framework) → [our layered formulation](docs/formulations.md#part-ii-proposed-layered-specialization). Cognitive and neurosymbolic references remain in the literature review; they are not collapsed into harness-search methods.

The mathematical hierarchy is [common framework](docs/formulations.md#part-i-standardized-comparison-framework) → [family-level templates](docs/formulations.md#families-of-similar-formulations) → [individual method substitutions](docs/formulations.md#method-instantiations) → [our specialization](docs/formulations.md#deriving-our-formulation-from-the-common-interface). The [standardization criterion](docs/formulations.md#what-makes-the-formulation-standardized) explains what each mapping must preserve. All 17 local papers remain mapped, with additional neurosymbolic connections and explicit limits.

## Visual reading guide

- [Overall learning and execution loop](docs/problem-framing.md#1-the-problem)
- [Information layers and functional models](docs/system-design.md#41-connections-between-information-layers)
- [Change propagation](docs/system-design.md#43-change-propagation)
- [Task-time retrieval and refresh](docs/system-design.md#6-retrieval-and-refresh)
- [Formulation as a computation graph](docs/formulations.md)
- [Comparison with neighboring formulations](docs/literature-and-formulation.md#1-closest-conceptual-and-methodological-references)

Figures use editable Mermaid diagrams embedded in Markdown.

### Iterative evaluation

The same tasks run once per iteration. `reset` starts every task from an empty graph. `learn` keeps one graph across every domain, task, and iteration, so a later run can reuse a contract found earlier. Both arms can still learn within a task. The grader runs afterward and its score is not written into the graph.

```sh
WORLDMODEL_MODEL_CONFIG=configs/model-qwen.toml .venv/bin/python scripts/iterative_eval.py runs/iterative-public
.venv/bin/python scripts/analyze_iterations.py runs/iterative-public
```

Use a new output directory. The runner refuses to overwrite one. The default samples five tasks from each of the six public domains and repeats that list for five iterations: 30 tasks, two arms, 300 executions. Each task gets 50 turns and no added wall-clock or environment-call cap. Repeat `--task NAME` to fix the list, or pass `--domain simple --per-domain 5` for a smaller slice. `--semantic-learning` adds consolidation; it is off unless requested. The model must be the Qwen profile in [configs/model-qwen.toml](configs/model-qwen.toml).

After each task the script appends `results.jsonl` and rewrites `summary.json`. Read accuracy first. If the learn arm's strict accuracy is below the reset arm on the latest iteration, fewer discovery calls are not a win. Flat accuracy with fewer discovery calls is an efficiency win. The target pattern is strict accuracy rising across iterations while discovery calls fall and task outcomes stay at least as accurate as the reset arm. One stream does not establish that a particular learned claim caused the change.
