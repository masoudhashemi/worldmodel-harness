# Paper and benchmark guide

For implementation details, see the [methods, prompts, and code tables](../docs/methods-prompts-and-code.md): all 17 papers, supplemental cognitive/neurosymbolic references, pinned author-code inspections, missing-source disclosures, and candidate improvements mapped to our harness.

References support one research question: when does retaining operational relationships improve task execution and reusable harness adaptation enough to repay its costs? Environment papers establish what must be observed or learned; knowledge/procedure papers distinguish reusable representations; harness papers establish competing adaptation mechanisms; cognitive and neurosymbolic work suggests implementations to test. Benchmarks supply use cases, not a different objective. Local PDFs are linked directly; additional external references are not bundled.

Every one of the 17 local PDFs has a [mathematical profile](../docs/literature-and-formulation.md#paper-by-paper-mathematical-formulations), distinguishing source equations from reconstructed update rules. The [standardized framework and mapping](../docs/formulations.md#part-i-standardized-comparison-framework) compares their representations, mutable objects, evidence, schedules, and objectives before deriving our proposal. For the project itself, start with [problem framing](../docs/problem-framing.md).

The [method instantiations](../docs/formulations.md#method-instantiations) map native objects and updates back to the shared equations C1–C7. Each mathematical profile links to its corresponding instantiation, so the comparison can be read in either direction.

For mathematical grouping, use the [family templates](../docs/formulations.md#families-of-similar-formulations): environment/prediction, external knowledge learning, harness search, coupled reuse/adaptation, adaptation-mechanism learning, population specialization, and acting-policy learning. These overlap and are distinct from the reading-priority groups below.

## Primary comparisons

| Reference | Role | Local PDF |
| --- | --- | --- |
| Procedural Graphs | Conceptual separation of factual knowledge and procedural structure | [2609.09153](2609.09153-procedural-graphs.pdf) |
| E-Ledger and WorldAbduct | World knowledge, verification, and harness evolution | [2610.11552](2610.11552-eledger-worldabduct.pdf) |
| Continual Enterprise World Model Discovery | Persistent environment knowledge under change | [2609.19551](2609.19551-continual-enterprise-wm.pdf) |
| Harness2 ([arXiv](https://arxiv.org/pdf/2610.14224), [code](https://github.com/google-research/harness2)) | Two-level recursive harnessing: task-level Propose–Probe–Compose and cross-task improvement library without ground-truth verifiers | [2610.14224](2610.14224-harness2.pdf) |
| Meta-Harness | Direct optimization of the agent's surrounding program | [2603.28052](2603.28052-meta-harness.pdf) |
| Do Enterprise Systems Need Learned World Models? | Runtime configuration discovery and deployment shift | [2605.12178](2605.12178-enterprise-learned-wm.pdf) |
| World of Workflows | Partial observation and enterprise effects | [2601.22130](2601.22130-world-of-workflows.pdf) |

Procedural Graphs is a conceptual starting point, not a claim that the paper learns two coupled graphs. WorldAbduct is a close comparison, not a harness-only method. Harness2 is our closest open-world score-free harness-evolution baseline, contrasting multi-rollout task-level probing of 8 file components with our single-pass stateful execution over a typed operational world model.

## Additional local harness and execution papers

These remain a broader comparison set. Their inclusion does not commit the project to importing their algorithms or establish their equivalence.

| Paper | Local PDF |
| --- | --- |
| Agentic Harness Engineering | [2604.25850](2604.25850-ahe.pdf) |
| HarnessCompass | [2608.01918](2608.01918-harnesscompass.pdf) |
| HarnessEvolve | [2609.00829](2609.00829-harnessevolve.pdf) |
| StarHarness | [2608.24804](2608.24804-starharness.pdf) |
| Evo2 | [2610.13025](2610.13025-evo2.pdf) |
| RRSI | [2609.24972](2609.24972-rrsi.pdf) |
| Hyperagents | [2603.19461](2603.19461-hyperagents.pdf) |
| Mixture of Self-Improving Branches | [2609.37834](2609.37834-mixture-branches.pdf) |
| TreeFlex | [2610.20334](2610.20334-treeflex.pdf) |
| Harness-G | [2607.27652](2607.27652-harness-g.pdf) |

## Neurosymbolic research

| Source | Why it matters |
| --- | --- |
| [The Neuro-Symbolic Concept Learner, 2019](https://arxiv.org/abs/1904.12584) | Learned interpretation combined with symbolic program execution |
| [NeSyC, 2025](https://arxiv.org/abs/2503.00870) | Continual knowledge formulation, validation, and refinement with neural and symbolic components |

## Cognitive models

| Source | Functional connection |
| --- | --- |
| [What Learning Systems do Intelligent Agents Need?, 2016](https://stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf) | Episodes and consolidation in complementary learning systems |
| [Making Working Memory Work, 2006](https://ccnlab.org/papers/OReillyFrank06.pdf) | Task-dependent maintenance and updating |
| [The hippocampus as a predictive map, 2017](https://www.nature.com/articles/nn.4650) | Predictive relational representations |
| [Uncertainty-based competition for behavioral control, 2005](https://www.nature.com/articles/nn1560) | Arbitration between economical and flexible control |

These motivate testable mechanisms, not claims that the architecture reproduces brain anatomy.

## Benchmark sources

See the [benchmark-selection document](../docs/benchmarks.md) for the expanded candidate set, including EnterpriseOps-Gym, AppWorld, ToolSandbox, the τ-bench family, CRMArena-Pro, WorkArena++, and TheAgentCompany. It separates source capabilities from proposed Q1–Q5 experiments and custom continual/shift protocols. The references below retain the original use cases and world-model diagnostics.

| Benchmark | Sources | Intended role |
| --- | --- | --- |
| ITSMBench | [Official description](https://www.atomicwork.com/itsm-bench) | Investigation, dependencies, diagnosis, and remediation |
| AutomationBench | [Repository](https://github.com/zapier/AutomationBench), [paper](https://arxiv.org/abs/2604.18934) | Cross-application composition and outcome verification |
| World of Workflows | [Local paper](2601.22130-world-of-workflows.pdf) | Hidden effects and partial observation |
| EnterpriseWorldShift | [Local CDA paper](2609.19551-continual-enterprise-wm.pdf) | Discovery and revision under specified environment changes |

A custom task stream with persistent learning is a separate protocol from the standard benchmark setup. Freeze benchmark versions and define permitted evidence before experiments.
