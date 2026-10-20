# Research questions

Status: questions this proposal must answer. They do not add a second objective. [Q1–Q5](problem-framing.md#8-contribution-to-investigate) and the [task-stream objective](formulations.md#8-overall-objective) remain canonical. Each section distinguishes a scientific hypothesis, its controlled comparisons, diagnostic measurements, and the limits of interpreting a negative result. Shared implementation requirements are stated separately.

The [system design](system-design.md#12-evaluation) specifies the controlled comparisons. The [formulation](formulations.md#deriving-our-formulation-from-the-common-interface) derives the representation those comparisons require. This document states what still has to be established.

## 1. The governing question

> When do qualified operational relationships, used selectively and linked to reusable execution artifacts, improve task-stream value by more than their learning, retrieval, and maintenance costs?

Task-stream value is verified task success minus environment, compute, and adaptation costs over a declared protocol. A larger graph, more layers, or more harness edits does not answer the question.

One case makes the subsidiary questions concrete. A create call times out. The agent must learn whether a later lookup can resolve the outcome, which of that relationship's facts must be fresh, whether spending a call to distinguish "created" from "not created" is worth it, when the relationship deserves a reusable verification procedure, and what to repair if the lookup contract changes. Q1–Q5 are that case stated so a task stream can decide it.

## 2. Questions that make an answer interpretable

Q1–Q5 are comparable only after the protocol and the ledger are fixed. These questions have to be answered in the experiment specification, before any arm is run.

| Question | What must be fixed |
| --- | --- |
| What persists across tasks? | For frozen transfer, online learning, and controlled shift: which of $G$, $H$, live state, and the evidence log carry, reset, or stay task-local. Logging for later analysis must not influence later frozen-transfer tasks. |
| What may each component see, and when? | Declare documentation/trajectory/probe/score/diagnostic/demonstration channels, released fields, availability time, role access and evaluator version. Scoring-only output is reporter-visible; feedback to a selector or learner is supervision even without actor access. |
| What is held equal? | Base model, tools, task split, context/call/learning budgets and observation channels, except the explicitly varied channel in a feedback experiment. A probe, refresh and task call share the environment ledger; feedback generation and processing are also costed. |
| Where is each cost charged? | $C^{\mathrm{env}}$ for task calls, probes, and refresh; $C^{\mathrm{compute}}$ for retrieval, prediction, reasoning, and graph learning; $C^{\mathrm{adapt}}$ for artifact construction, validation compute, and maintenance. A validation call already charged to $C^{\mathrm{env}}$ is not charged again. Bootstrap cost is inside the horizon or reported as an explicit amortization. |
| Which results are which protocol? | Standard ITSMBench and AutomationBench scores are reported separately from custom continual streams and controlled shifts. Versions, reset rules, shared scopes, and authorized information are fixed in advance. |
| Which contrast identifies which mechanism? | Representation, retrieval, exploration, harness adaptation, and dependency repair each have a comparison that changes that mechanism and holds the others at a matched budget. |

ITSMBench and AutomationBench do not by themselves define a persistent shared tenant. The continual and shift protocols have to state scope, carryover, and what a task is forbidden to inherit from another task's live state.

### Cross-cutting contrast: which evidence channels help?

This is an axis across Q1–Q5, not a sixth graph layer or an additional optimization objective.

| Configuration | Additional learner/selector access | Question isolated |
| --- | --- | --- |
| Trajectory-only | Documentation, actual interactions and permitted probes; grader stays reporting-only | What can operational learning achieve without correctness supervision? |
| Score-assisted | Scalar task/subtask score after the attempt, without hidden assertion text | Does success/failure feedback improve prioritization and selection? |
| Diagnostic-assisted | Explicitly released assertion outcomes and diagnostics, with coverage/evaluator provenance | Does localized feedback improve diagnosis, verification and artifact repair beyond scalar scores? |
| Demonstration-assisted | Authorized reference traces or answer-conditioned examples, tagged as such | What does stronger target-solution supervision add, at its construction cost? |

Hold task stream, model, reset, tools and budgets fixed except for source access. Declare release timing and which roles can consume each payload. Match feedback-query/count and teacher-compute allowances where meaningful; report unmatched source richness rather than pretending these are identical observations. Include a same-feedback direct-harness learner without graph organization to test whether graph connections add value beyond access to better labels.

Cross this source axis with Q1's episode/claim/graph comparison and Q4's fixed/adaptive harness where feasible; do not require a full factorial initially. Track pre-feedback score, later retry/round scores, feedback costs, diagnostic accuracy, artifact invocation and propagated mistakes. Grader disagreement, incompleteness and version shifts are useful robustness cases, not automatically world contradictions.

Our first experiment uses the same 50 tasks for learning and scoring without a validation split. If any feedback influences learning or artifact selection, subsequent influenced outcomes are **feedback-assisted in-sample results**, not independent evaluation. Task-success feedback alone cannot certify an operational rule or causal explanation. A true transfer/independent-evaluation study would need its own uncontaminated protocol.

### Shared implementation requirements

These are conditions for a valid experiment, not competing scientific hypotheses:

- A timeout is not proof of failure; an unreturned field is not proof of absence. Preserve observation coverage and distinguish observation from inference.
- Scope claims to the evidence; repeated observations of one event are not independent confirmations. Preserve identity ambiguities and distinguish epistemic status from confidence.
- Required permission and safety checks apply to every arm. When a prerequisite remains unresolved, use a permitted alternative, seek evidence, or report the limitation.
- Keep task-local facts separate from durable knowledge. Predicted or simulated probe outcomes do not become observed evidence.
- Use only permitted information for learning and validation. Shared claim sets in controlled comparisons come from permitted development evidence, not test solutions.

Correctness tests verify these requirements. Experiments then ask whether consolidation, representation, retrieval, or adaptation improves performance beyond that common minimum.

### Conditions that operationalize when

The governing question requires boundary conditions, not just one average comparison. Select a feasible set of regimes before evaluation:

| Variable | Controlled variation | Expected mechanism to examine |
| --- | --- | --- |
| Reuse frequency | Repeat operational relationships across different tasks versus mostly novel relations | Amortization of claim learning and artifacts |
| Environment volatility | Stable worlds versus declared rates and types of change | Refresh and maintenance costs |
| Task diversity | Related workflows versus different goals, tenants, or tool combinations | Transfer and scope generalization |
| Observation ambiguity | Immediately verifiable outcomes versus partial/delayed evidence | Value of diagnosis and information gathering |
| Learning horizon | Short versus long streams and differing time remaining | Break-even point for exploration/adaptation |
| Available resources | Declared context, call, and compute budgets | Success–cost tradeoffs and feasibility |

These manipulations apply to controlled fixtures or explicitly labeled benchmark extensions where supported; do not imply every official benchmark exposes them. Fix task difficulty and observation access as far as the contrast permits, and document unavoidable changes. Use selected interactions, especially reuse × volatility and horizon × exploration, rather than requiring an impractical full factorial across every variable.

### Measurement and uncertainty

Predeclare the primary contrasts, utility weights, practical effect threshold, and evaluation horizon. Report verified success and each cost component separately, as well as weighted task value and sensitivity to plausible cost weights. Include success–cost tradeoffs rather than choosing a winner solely through one convenient weighting.

For the repeated-task learning objective, prioritize **checkpoint reliability (pass^k)**:
all k attempts must succeed, alongside pass@1 and complementary pass@k (at least one
success). Compare fixed k across learning checkpoints. Reset environment and local
state for each replicate, and hold the checkpoint's persistent graph and harness fixed
across its replicates. Learning continues between checkpoints, not between sibling
measurement trials. The same tasks may still be used for learning and measurement;
this is explicitly in-sample reliability, not unseen-task evaluation. Never pool changing
online graph states as independent trials of one checkpoint. Neither metric is guaranteed
to improve with learning; report regressions and cost. See
[τ-bench's original definitions](https://arxiv.org/html/2406.12045v1).

Use matched task streams, environment initializations, and randomized run assignments where feasible. Repeat stochastic runs and task orders; use paired contrasts when pairing is valid. Treat correlated tasks within a continually learned stream as dependent: estimate uncertainty at the independent stream/world/run level, with the unit and resampling or estimation method stated. Do not present repeated decisions from one learned trajectory as independent replications.

For independent/final evaluation claims, keep development tuning separate. The requested repeated-task acceptance experiment instead reports in-sample learning, with no held-out split and explicit feedback labels. Distinguish evidence of improvement, evidence that effects are below a practical threshold, and an inconclusive interval that permits both. A null or negative result is conditional on the tested implementation, budgets, distribution, horizon and precision.


## 3. Q1 — Retained operational knowledge

**Do qualified relationships improve transfer beyond retrieving past episodes, and what does graph organization add beyond consolidation?**

**Hypotheses.** Consolidating experience into scoped claims can improve transfer; organizing those same claims as an operational graph may provide an additional benefit. These are separate claims.

**Controlled comparisons.** Use the same development experience and fixed executor:

| Arm | Retained information | What the contrast isolates |
| --- | --- | --- |
| Episode memory | Raw or retrievable episodes, with a competitive retrieval implementation | Reference for learning from past experience |
| Consolidated non-graph memory | Qualified claims in text or structured records, with scope, evidence, confidence/status, and validity metadata | Versus episodes: consolidation and reusable claim extraction |
| Operational graph | The same qualified claims and metadata, with explicit relational indexing and graph access | Versus non-graph claims: graph organization and access |

The non-graph control retains the same relationship meanings in its records; it is not deprived of input/output correspondences or evidence. Graph traversal is the intervention, not access to additional facts. Start with a shared claim set to isolate representation; then compare end-to-end learning pipelines, charging claim extraction, graph construction, and retrieval costs. State the retrieval interface because organization and access cannot be interpreted independently.

**Subsidiary tests.** Compare evidence-aware consolidation with immediate promotion; operational contracts with descriptive co-occurrence links; and explicit identity/status handling with matched variants. Measure which relation types help held-out binding, diagnosis, effect prediction, and recovery. These are diagnostics of later-task value, not independent success criteria. Scope generalization should be tested on declared held-out tasks, tenants, or versions, rather than recurrence within one template.

**Interpretation.** If claims beat episodes but graphs do not beat the matched claim store, the result supports consolidation, not graph necessity. If neither improves task value, neither is justified for that fixed-executor setting. This does not exclude a graph–adaptation interaction in Q4, or benefits under another reuse horizon or shift regime.

## 4. Q2 — Selective use

**Do dependency-aware retrieval and refresh improve decisions under a fixed context and interaction budget, and which mechanisms account for the gain?**

**Hypotheses.** Dependency expansion can recover decision prerequisites; layer-aware selection can allocate context more usefully; selective refresh can reduce errors from stale information. None follows merely from storing data in six named layers.

**Controlled comparisons.** Hold the learned records, metadata, executor, and permitted observations fixed. Use a competitive flat record retriever that can filter on scope, status, and freshness. Vary:

1. **Layer-aware selection:** policies that explicitly use the six information categories versus record-level selection using the same metadata, without layer quotas or routing.
2. **Dependency expansion:** follow typed prerequisites and consumer links versus retrieval without explicit expansion.
3. **Optional refresh:** decision-sensitive checks versus a stated baseline such as age-based or ordinary task-directed checking.

Use targeted factorial comparisons to separate these effects and selected interactions; a full-package comparison plus one freshness ablation cannot attribute all three. Report the tested combinations rather than assuming untested interactions are absent. Mandatory authorization and safety checks remain identical in every arm.

**Measurements.** Task value is primary. Diagnostics include required-dependency recall, cross-scope errors, stale-use errors, context occupancy, and check cost. Establish required dependencies using independently specified tasks or fixtures, not solely the agent's own graph. Count unique records once when expansion produces overlapping neighborhoods.

**Interpretation.** Layers help only if layer-aware policies outperform the metadata-rich control. Dependency expansion or refresh may help even when explicit layering does not. Retain the supported subset under the tested budget; a package-level negative result does not justify removing every component.

## 5. Q3 — Useful learning

**Does action selection that values future learning improve task-stream value beyond learning incidentally from task execution?**

**Hypothesis.** Paying to resolve uncertainty can be worthwhile when the information improves enough later decisions. Naming an uncertainty or calling an action a probe does not establish this effect.

**Controlled comparisons.** Keep the initial memory, learner, permitted actions, and total budget the same. Compare policies that differ in an explicit learning-value term:

$$
a_t^{(\beta)}\in\arg\max_{a\in\mathcal A_t^{\mathrm{allowed}}}
\left[
\widehat V_{\mathrm{current}}(a)
-\widehat C(a)
+\beta\,\widehat V_{\mathrm{future\ learning}}(a)
\right].
$$

The baseline uses $\beta=0$ and still permits diagnosis, verification, and information gathering needed to complete the current task. Both arms learn from every eligible observation. The treatment adds estimated benefit to later tasks; $\widehat V_{\mathrm{current}}$ must exclude that benefit to avoid double counting. This is an operational comparison rule, not a claim that the value estimates are exact. Select settings on development data.

**Subsidiary tests.** Vary reuse frequency and remaining horizon; compare reads, documentation inspection, and authorized experiments; measure whether acquired evidence changes subsequent decisions. Log a baseline-policy recommendation at shared decision states where feasible, without executing an extra action, to describe selection differences. Such logs are diagnostics, not counterfactual outcomes. After policies diverge, compare complete matched streams rather than treating their observations as identical.

**Measurements.** Include probe costs and opportunity costs from foregone task calls. Assess held-out prediction/diagnosis quality, but answer Q3 through total task value and the time needed to recover the investment. Noisy or nondiscriminating observations may leave uncertainty unresolved; they do not automatically establish a hypothesis.

**Interpretation.** An inconclusive or negative result limits support for the tested learning-value policy, estimator, and horizon. It does not establish that all exploration is useless or that necessary task-time diagnosis should be removed.

## 6. Q4 — Reusable execution support

**Does adapting the harness add value beyond fixed-harness knowledge use, and does graph organization improve that adaptation?**

**Hypotheses.** Adaptation can reduce repeated execution costs; graph organization may improve an adaptive system over an equally informed non-graph representation. The stronger claim that the graph increases the benefit of adaptation is an interaction claim, not merely a claim about the full system's performance.

**Controlled comparisons.** Cross representation with adaptation:

| | Fixed harness | Adaptive harness |
| --- | --- | --- |
| Matched non-graph claim memory | $V_{00}$: qualified claims used by a generic executor | $V_{01}$: direct harness adaptation with those claims and raw evidence |
| Operational graph | $V_{10}$: the same claims accessed through the graph | $V_{11}$: graph-dependent harness adaptation |

$V_{ra}$ is expected task-stream value under representation $r$ and adaptation setting $a$, including all applicable costs. Non-graph memory is the strong Q1 control, not an absence of memory. Adaptive arms share edit surfaces, proposer resources, validation access, and budget ceilings. Fixed arms incur no fictitious adaptation cost or forced unused expenditure.

The principal contrasts are:

$$
\Delta_{\mathrm{adapt}\mid G}=V_{11}-V_{10},\qquad
\Delta_{G\mid\mathrm{adapt}}=V_{11}-V_{01},
$$

$$
\Delta_{\mathrm{interaction}}
=(V_{11}-V_{10})-(V_{01}-V_{00}).
$$

A positive interaction means adaptation adds more value with the graph on this measurement scale. It does not alone establish that $V_{11}$ is best or cost-effective; report all four values and their uncertainty. An additive benefit can still make the combined system useful without supporting a synergy claim.

**Two stages.** First freeze the shared claim content and fit adaptive artifacts on permitted development tasks; evaluate fixed and adapted harnesses on held-out tasks, including build/validation costs over a declared reuse horizon. This isolates execution reuse without differences in learned knowledge. Then run the full online loop in which knowledge and harnesses may evolve and trajectories diverge. Equalize starting evidence and access, not the evidence each policy subsequently chooses to collect.

**Subsidiary tests.** Ablate adapters, retrieval recipes, conditional procedures, and recovery checks. Compare typed execution with graph-as-text using the same claim content; count checking compute as well as context. Compare applicability-sensitive reuse with replanning and reuse wherever legally admissible. The interpreter, external permission enforcement, and acting weights remain fixed. Comparisons must not disable safety enforcement to manufacture a benefit.

**Interpretation.** If $\Delta_{\mathrm{adapt}\mid G}$ has no practically meaningful benefit, adaptation is unsupported in that graph setting. If $\Delta_{G\mid\mathrm{adapt}}$ is nonpositive, direct adaptation is competitive under those conditions. Distinguish imprecise estimates from evidence of negligible benefit; neither result rules out effects under different reuse or volatility conditions.

## 7. Q5 — Adaptation under change

**Do revised claims enable useful repair of dependent procedures, separately from the ability to detect the change?**

**Hypotheses.** Scoped dependencies can reduce repair cost after a known change; a complete detector-and-repair loop may also improve value under unannounced change. These require different experiments.

**Controlled comparisons.**

1. **Repair given a known change.** Supply every arm with the same verified change notification at the same time. The notification identifies the changed contract/rule/fact, not the affected procedures or correct repair. Compare scoped dependency propagation, no proactive propagation with ordinary reactive fallback, and broad revalidation. Hold the initial artifacts, evidence, and repair budget fixed.
2. **End-to-end adaptation.** Remove the privileged notification. Require changes to be detected through permitted observations, charging detection and repair costs. Start with a common detector across repair variants, then vary detection separately if it is itself a proposed contribution.

The first stage is a labeled diagnostic experiment with oracle change notice; it is not a standard deployment result. Maintain identical external safety enforcement in both stages. “No propagation” means no proactive dependency maintenance, not permission to execute known-invalid or unauthorized actions.

**Subsidiary tests.** Vary changed fields, permissions, routing, effects, and recovery behavior. Test whether scope/condition checks improve on suspending every reachable consumer. Distinguish task-local rebinding from durable knowledge revision. Test shortcuts that rely on a hypothesis separately from diagnostic procedures designed to investigate it. Ground-truth affected artifacts should come from controlled fixtures or independent assessment, not the proposed dependency graph alone.

**Measurements.** With supplied notice, measure time/cost from notice to restored task performance, stale-artifact use, unnecessary suspension, and task value. End-to-end, separately report detection delay, false alarms, missed changes, post-detection repair cost, and total value. Detection gains must not be attributed to propagation.

**Interpretation.** A negative repair result supports a simpler repair strategy for the tested changes and dependency density. Good repair with poor end-to-end results can implicate detection cost or delay rather than invalidate dependency repair. Neither stage justifies a universal decision to discard maintenance.


## 8. What each use case must show

The [benchmark-selection document](benchmarks.md) compares additional settings against these same questions and defines the boundary between standard benchmark results and custom continual/shift experiments.

ITSMBench and AutomationBench instantiate the same five questions. They are not two research programs. Each still has to exhibit the relationships the questions refer to, or the measured effect has nothing to attach to.

| | ITSMBench | AutomationBench |
| --- | --- | --- |
| Q1 | Do identity correspondences, system dependencies, and diagnostic evidence paths transfer to later incidents? | Do endpoint contracts, transformations, prerequisites, and outcome checks transfer to later workflows? |
| Q2 | Which account, assignment, and device facts must be refreshed before a remediation decision? | Which contact, ownership, availability, and operation outcomes must be fresh before a dependent write? |
| Q3 | Which queries distinguish causes, and when does that distinction change the remediation? | Which calls reveal output composition, failure behavior, or the outcome of an uncertain write? |
| Q4 | When does a conditional diagnostic or remediation branch repay its validation cost? | When does an adapter or workflow fragment repay its validation cost? |
| Q5 | After a dependency or permission change, which branches are revalidated, and which keep working? | After a contract change, which adapters and verification steps are suspended or rebound? |

Standard benchmark scores stay separate from any stream in which the agent keeps $G$ or $H$ across tasks.

## 9. Order of answering

Fix protocol and cost accounting before interpreting any result. The following is an efficient investigation order, not a rule that every component must succeed alone before interactions can be tested:

1. Validate the shared requirements and define regimes, primary contrasts, and uncertainty estimates.
2. Run Q1's episode/claim/graph comparisons with a fixed executor to separate consolidation from organization.
3. Run Q2's retrieval, dependency, and refresh contrasts on common records.
4. Run Q3's policy comparison with the same learner and starting evidence; vary reuse and remaining horizon.
5. Run Q4's four-arm experiment on frozen claim content, followed by the coupled online loop.
6. Run Q5's notified-change repair experiment, then unannounced-change detection and repair.

A weak Q1 main effect does not preclude the Q4 interaction: graph organization could be useful primarily for constructing or repairing procedures. Test that possibility directly rather than attributing a full-system gain to every component. Report stable and shifted conditions and task-order sensitivity for online results.

## 10. What a complete answer contains

An answer to the governing question identifies:

- which approaches improve task value in which regimes, with effect sizes, uncertainty, and separate success/cost outcomes;
- whether gains come from consolidation, graph organization, selection, exploration, adaptation, repair, or measured interactions;
- the limits of transfer across tasks, tenants, versions, horizons, and change regimes;
- which simpler designs remain competitive, and which comparisons remain inconclusive.

A proposed mechanism should affect relevant later decisions in controlled cases; every individual claim need not change an action. Independently selected tasks and probes test use beyond paths the agent already favors. Component counts, accepted edits, and belief citations alone do not establish the mechanism.

Negative results narrow the supported scope of the design. They do not automatically delete a component from every future setting, and absence of statistical significance is not proof of equivalence.
