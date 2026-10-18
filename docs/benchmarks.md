# Benchmarks for operational world learning

Status: benchmark-selection proposal, not experimental results. Source review: 2026-10-15. No benchmark installation or end-to-end run is established by this document; pin releases and verify access before committing an experiment.

The goal is to test whether learned operational relationships improve task-stream value after learning, retrieval, and maintenance costs. Benchmarks supply environments and scoring; the [research questions](research-questions.md) supply the controlled comparisons, and the [formulation](formulations.md#8-overall-objective) supplies the objective. A high standard benchmark score alone does not establish cross-task learning or world-model-guided harness adaptation.

## 1. Selection recommendation

Keep **AutomationBench and ITSMBench** as the motivating use cases. Prioritize additional benchmarks by the evidence they add:

- **EnterpriseOps-Gym:** strongest additional candidate for tool–table relationships and cross-domain enterprise execution.
- **AppWorld:** strong candidate for controlled API composition, state verification, and reusable execution support.
- **ToolSandbox:** complementary mechanism-level setting for prerequisites, state tracking, and information gathering.
- **τ-bench telecom, CRMArena-Pro, WorkArena++, and TheAgentCompany:** secondary validation settings when conversational coordination, CRM specialization, browser interaction, or broader workplace behavior matters.

These priorities are our assessment, not claims made by the benchmark authors. Do not run every benchmark merely to enlarge the suite. Select a diagnostic environment and an end-to-end environment with complementary failure modes.

If only one new enterprise benchmark is added, investigate EnterpriseOps-Gym first. If the main need is clean ablations before scaling, investigate ToolSandbox or AppWorld first. Existing AutomationBench and ITSMBench experiments should not be displaced without checking which new evidence the replacement would provide.

## 2. Candidate comparison

“Fit” below means a proposed experimental use, not that the standard benchmark directly measures the corresponding research question.

| Benchmark and primary source | Native setting and evaluation | Proposed fit | Main qualification |
| --- | --- | --- | --- |
| [AutomationBench](https://github.com/zapier/AutomationBench) | Simulated business workflows across SaaS tools; assertion-based final-state evaluation | Q1 composition, Q2 prerequisite selection, Q4 adapters/workflows | Task initialization is not a continual shared tenant; public tasks differ from the private official leaderboard set |
| [ITSMBench](https://www.atomicwork.com/itsm-bench) | IT service-management investigation and remediation across systems | Q1 identity/dependency knowledge, Q2 evidence selection, Q3 diagnosis, Q4 remediation procedures | Confirm executable access, reset semantics, and evaluator integration before selecting it for controlled experiments |
| [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym) | Containerized, resettable enterprise simulation; MCP tools and SQL final-state verifiers | Q1 tool–table relations, Q2 retrieval among tools, Q4 cross-domain procedures | Tool-access mode and public/private split affect comparisons |
| [AppWorld](https://github.com/StonyBrookNLP/appworld) | Multi-app API environment with database-state tests, including collateral-change checks | Q1 output/input and identity relations, Q4 composition/reuse; candidate substrate for custom shifts | Each task has its own initial database; agent-accessible rollback would change the protocol |
| [ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox) | Stateful conversational tool use; milestone-DAG and state-constraint evaluation | Q2 prerequisites/working state, Q3 informative queries; mechanism tests for Q4 | Milestones are evaluator knowledge, not an agent's learned graph |
| [τ-bench family](https://github.com/sierra-research/tau2-bench) | Policy-constrained customer-service tasks with tools and simulated users | Q2 diagnostic state, Q3 information gathering, Q4 conditional support procedures | Simulator behavior and conversation quality can confound operational-learning effects; pin domain and release |
| [CRMArena-Pro](https://github.com/SalesforceAIResearch/CRMArena) | Salesforce CRM tasks with B2B/B2C settings and interactive scenarios | Q1 object relationships and scoped knowledge, Q2 relevant record selection | Hosted-platform access and reproducibility need checking; arbitrary platform changes are not assumed available |
| [WorkArena and WorkArena++](https://github.com/ServiceNow/WorkArena) | Browser-based enterprise tasks, including compositional WorkArena++ tasks | Q4 procedure transfer through a different interface | Browser navigation can dominate failure and hide the effect of operational knowledge |
| [TheAgentCompany](https://the-agent-company.com/) | Simulated workplace using web services, coding, and coworker interaction | Broader validation of organizational relationships and reusable execution support | Multiple services and conversational dependencies complicate attribution; some grading uses LLMs |

### EnterpriseOps-Gym

The repository describes 1,150 tasks across eight domains, 512 tools, and 164 database tables. Seed databases and domain MCP servers support resettable execution. The public release covers 60% of benchmark samples. Its runner distinguishes oracle tool access from settings with added tools. [Repository and setup](https://github.com/ServiceNow/EnterpriseOps-Gym/blob/main/README.md).

For our experiments, tool visibility must be held constant across arms. Oracle access can help isolate planning/composition, while non-oracle settings are more informative about retrieval; report them separately. Keep SQL verifier logic and seed internals out of the acting agent's context. A table relationship visible to the evaluator is not automatically an observed relation available to the learner.

### AppWorld

AppWorld exposes API interactions and state-based evaluation. Its tooling permits saving and restoring environment state, but explicitly warns that agent use of arbitrary rollback gives an unrealistic advantage over agents without it. Each task starts from its own database state. [Official repository](https://github.com/StonyBrookNLP/appworld).

Use reset/checkpoint capabilities for evaluator-controlled repeatability, not free agent exploration. Retain contracts and qualified relationship knowledge across a declared stream, while rebinding task-specific records. A custom shift study could modify selected contracts or behaviors only if the implementation can support consistent tasks and independent grading; native checkpoint support alone does not establish that capability.

### ToolSandbox

ToolSandbox evaluates trajectories against milestone dependencies and database constraints. That makes intermediate state requirements inspectable to the experimenter. [Official repository](https://github.com/apple-aiml-research/ToolSandbox).

Use it to construct independently specified prerequisite and observation-coverage tests. Do not hand the milestone DAG to the agent and then call its traversal learned world knowledge. A successful mechanism test here is not sufficient evidence of transfer to a large enterprise tool catalog.

### Secondary candidates and access considerations

- **τ-bench:** the telecom setting in τ²-bench includes agent and user actions in a shared environment. The current repository also includes τ³ extensions and task fixes. For this project, begin with a pinned text-mode domain rather than adding voice variability. Keep the user model/configuration fixed across arms and repeat runs. [Telecom paper](https://sierra.ai/-/cdn/document?src=https%3A%2F%2Fcdn.sanity.io%2Ffiles%2Fca4jck6w%2Fproduction%2F91c12fc38719eae97e812c6d59a37de5de288c75.pdf), [repository](https://github.com/sierra-research/tau2-bench).
- **CRMArena-Pro:** the repository provides evaluation code and Salesforce API setup. It states that GUI access is no longer publicly available without a request. Check API availability separately; source instructions do not establish that an account is currently usable. Do not copy shared credentials into project documentation. [Official setup](https://github.com/SalesforceAIResearch/CRMArena).
- **WorkArena++:** useful if interface-independent procedure reuse becomes part of the claim. Keep browser tooling fixed, and separate UI interaction failures from incorrect operational reasoning in diagnostics. [Official repository](https://github.com/ServiceNow/WorkArena).
- **TheAgentCompany:** the evaluation protocol initializes tasks by resetting dependent services and uses environment LLMs for some grading/coworker behavior. Record those model versions separately from the acting model. Its operational overhead makes it a later broad validation setting rather than the first mechanism experiment. [Evaluation protocol](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/docs/EVALUATION.md).

## 3. Existing world-model benchmarks in the literature

Do not lose the diagnostic references already present in the [paper guide](../papers/README.md#benchmark-sources):

| Reference | Role in our evaluation reasoning | Boundary |
| --- | --- | --- |
| [World of Workflows](../papers/2601.22130-world-of-workflows.pdf) | Partial observation and enterprise transition/effect prediction | Prediction results do not alone establish improved task execution or harness reuse |
| [EnterpriseWorldShift / continual discovery](../papers/2609.19551-continual-enterprise-wm.pdf) | Rule-model revision across defined world changes | Discovery/application phases are not automatically our task-interleaved online protocol |
| [Enterprise learned world models / CascadeBench](../papers/2605.12178-enterprise-learned-wm.pdf) | Prediction under configuration variation; learned behavior versus runtime grounding | Readable configuration and learned evidence must have explicitly matched access |

These are complementary diagnostics, not replacements for end-to-end execution. This document does not establish that their executable releases are available or compatible with our harness; verify that separately before adding them to the implementation plan.

## 4. Connecting benchmark choice to Q1–Q5

Use the same scientific comparisons across environments, with benchmark-specific tasks and observations.

| Question | Required experiment | Promising settings | Additional protocol needed |
| --- | --- | --- | --- |
| Q1 Retained knowledge | Episodes versus consolidated non-graph claims versus the same claims organized as a graph | AppWorld, EnterpriseOps-Gym, AutomationBench; CRM specialization in CRMArena-Pro | Shared development evidence, held-out transfer, scoped memory/reset rules |
| Q2 Selective use | Separate layer-aware selection, dependency expansion, and optional refresh against a metadata-rich control | ToolSandbox and EnterpriseOps-Gym; ITSM diagnosis | Matched tool visibility and budgets; independently specified prerequisites |
| Q3 Useful learning | Same learner, with versus without action selection's future-learning value | ToolSandbox, τ telecom, controlled AppWorld/ITSM task streams | Repeated opportunities to reuse knowledge; declared horizon and permitted probes |
| Q4 Execution support | Graph/non-graph memory crossed with fixed/adaptive harness | AppWorld, AutomationBench, EnterpriseOps-Gym | Development-fitted artifacts first; coupled online adaptation separately |
| Q5 Change and repair | Common notified change for repair, then unannounced change for detection plus repair | Controlled fixtures first; extensible environments after feasibility checks | Validated change generator, independent affected-artifact assessment, safety enforcement |

This table prioritizes experimental suitability; it is not a measured ranking. Before choosing a task subset, check that the relevant relationships actually occur and can be reused. A single-task benchmark can supply tasks for Q1–Q4, but its standard score does not by itself identify learning across tasks.

## 5. Standard evaluation and custom protocols

### Feedback availability, release and carryover

The following separates evaluator mechanisms described above from **proposed feedback adapters**. Except for AutomationBench's existing reporting path, these adapters are not implemented or confirmed end-to-end here. Availability of an evaluator does not imply an official protocol permits exposing it.

| Benchmark | Candidate feedback source / availability boundary | Proposed release and consumers | Permitted persistent result |
| --- | --- | --- | --- |
| AutomationBench | Current runner records final score and assertion details after execution; learner access is not wired | Reporting-only now; a declared assisted adapter could release scalar score or selected diagnostics to learner/selector after recording the attempt | Source-tagged feedback, learned scoped claims and procedure versions; never hidden initial state or current records by default |
| ITSMBench | Executable evaluator access and diagnostic payload still need verification | No assumed channel until an authorized API/export is established | Only actually released feedback and derived scoped artifacts |
| EnterpriseOps-Gym | Final-state SQL verifier results; exact export and permitted diagnostic fields require adapter validation | Post-attempt score/diagnostics to allowed roles; do not expose SQL or database internals implicitly | Evaluator-versioned outcomes and evidence-linked learning |
| AppWorld | Task-state tests, including collateral-change checks; confirm released test-result payload | Post-attempt authorized outcomes; test source, expected values and solutions are separate access choices | Test-result evidence and derived rules/artifacts, not restored task database values |
| ToolSandbox | Milestone/state-constraint evaluator; internal milestone DAG is not ordinary interaction evidence | Release a score or declared milestone outcomes after attempt; full DAG access would be stronger supervision | Provenanced assessment links, not an unlabeled copy of the oracle graph |
| τ-bench family | Task evaluator plus simulated-user interaction; pin domain/release and actual reward fields | User conversation is interaction evidence; evaluator result is a separate post-attempt channel | Scoped dialogue/operational evidence, released reward and artifact lineage |
| CRMArena-Pro | Task evaluation under its platform setup; permissions/export need confirmation | Use only feedback explicitly exported by an authorized evaluator | Released feedback and scoped learned relationships, not private Salesforce tenant state |
| WorkArena / WorkArena++ | Task validators and documented diagnostic tools; confirm payload and privileges | Ordinary browser observations versus declared validator/diagnostic channels | Task/version-tagged evaluations and reusable procedures |
| TheAgentCompany | Checkpoints/final evaluation, including some LLM-based assessments | Release authorized results with evaluator/model version and uncertainty; pin simulator settings | Assessment provenance and learned artifacts, not unqualified truth from an LLM verdict |

Default access in the current runtime is reporter-only for grading. Assisted experiments explicitly choose actor/learner/selector access and release time. Scores, diagnostics and answer-conditioned demonstrations are distinct channels, not interchangeable forms of “feedback.” Reference generation consumes a separately declared teacher/data budget; it is not assumed available from any benchmark's evaluator.

Across resets retain the declared graph/evidence/artifact views, including feedback lineage when enabled; reset current records and task-local bindings. Record pre-release score before feedback can affect a retry. Once feedback affects learning or selection, subsequent influenced scores are **feedback-assisted**, not independent evaluation. A failed requirement does not automatically establish its cause or a general environment rule.

The first requested learning curve repeats the same 50 tasks as learning and scoring tasks, without a validation split. This is an in-sample custom protocol. The transfer-oriented suggestions elsewhere in this document are separate studies, not additional requirements for that curve.

Keep three types of result distinct:

1. **Standard benchmark evaluation:** preserve the benchmark's task initialization, allowed observations/actions, split, grading, and submission rules. Report the pinned version and any deviations.
2. **Continual-learning extension:** explicitly allow knowledge/artifact carryover across a declared task stream, while resetting live state as specified. Report order sensitivity, learning costs, and what scopes can transfer.
3. **Controlled-shift extension:** introduce specified environmental changes and independently verify task solvability and scoring after each change. Report these as custom experiments, not unmodified benchmark results.

Possible shift interventions include changed output fields, identity mappings, prerequisites, permissions, or error/recovery behavior. These are proposed interventions, not claimed built-in capabilities of the listed benchmarks. Do not mutate a shared hosted service or external tenant without authorization; use isolated test instances where changes are supported.

For Q5, separate:

- **Repair with notice:** provide the same verified changed fact/contract to every arm, without providing the dependent-artifact list or repair solution.
- **Detection and repair:** remove that notice and require discovery through permitted observations. Charge detection calls and false alarms separately from repair.

A native environment reset is not an environment change experiment. Retaining another task's live identifiers is not learning. Readable evaluator state is not permissible agent knowledge merely because the benchmark code is public.

## 6. Proposed execution sequence

1. **Integration fixture:** implement a small controlled tool/data environment with ambiguous outcomes, identity bindings, and independent grading. Validate the graph/non-graph controls and cost ledger.
2. **Mechanism setting:** investigate ToolSandbox for Q2/Q3 and prerequisite diagnostics.
3. **Composition setting:** use AppWorld or the existing AutomationBench plan for Q1/Q4; choose based on actual reset control, integration effort, and claim reuse.
4. **Enterprise validation:** add EnterpriseOps-Gym or proceed with ITSMBench once executable access and evaluation control are confirmed.
5. **Shift study:** build validated changes in an isolated environment; start with notified repair before full detection.
6. **Broader transfer:** add τ telecom, CRMArena-Pro, WorkArena++, or TheAgentCompany only when it tests a new claimed capability.

This is a proposed order, not a commitment to install all candidates. Retain the same Q1–Q5 comparisons and cost accounting as the setting changes; do not select different favorable baselines for each benchmark.

## 7. Acceptance checklist before a benchmark run

Record the following in the experiment specification:

- Exact repository commit, task/data version, evaluator version, split, and permitted license/use.
- Executable access confirmed by a small smoke run; setup cost and hosted-service dependencies.
- Task reset behavior, environment/tenant identity, record-ID reuse, and allowed knowledge/artifact carryover.
- Agent-visible tools, documentation, configuration, observations, and interaction interface; oracle modes clearly labeled.
- Strict separation between agent inputs and solutions, verifier assertions, milestone graphs, audit-only state, and implementation internals.
- Repeatable initial states and independent task/outcome verification, including side effects where supported.
- Same base model, proposer resources, safety constraints, and budget ceilings for the relevant comparison; simulator/judge models recorded separately.
- A ledger for calls, ordinary compute, and adaptation costs, with no double counting or free bootstrap exploration.
- Streams/repeats sufficient to quantify uncertainty at an independent run/world level, plus the declared reuse, volatility, diversity, ambiguity, and horizon regimes.
- Standard success and component costs reported alongside the shared utility; public-split, official, and custom scores never conflated.

The detailed controls and interpretation rules remain in [research questions](research-questions.md#measurement-and-uncertainty). Benchmark selection is successful when it makes those questions answerable, not when it maximizes the number of leaderboard names.
