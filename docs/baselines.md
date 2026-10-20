# Comparative Baselines and Experimental Contrasts

Status: reference specification and experimental guide. This document details the external literature baselines and internal ablation arms required to evaluate the operational world graph and adaptive harness architecture in `worldmodel-generation-harness`.

---

## 1. The Baseline Landscape

The core question investigated by this repository is:
> *Does maintaining an explicit, layered operational world graph outperform (a) direct code-level harness evolution, (b) procedural memory alone, (c) flat/episodic retrieval, or (d) pure runtime discovery?*

To isolate the sources of improvement, methods from the 17 literature papers ([docs/literature-and-formulation.md](literature-and-formulation.md) and [docs/methods-prompts-and-code.md](methods-prompts-and-code.md)) are categorized along the spectrum below:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AGENT LEARNING SPECTRUM                                         │
├──────────────────────┬────────────────────────┬───────────────────────────────┬────────────────────────┤
│  Procedural Memory   │ Coupled WM + Harness   │ Direct Harness / Code Search  │ Flat / Episodic Memory │
│  (How to act only)   │ (World + Procedures)   │ (No explicit world model)     │ (Raw instance storage) │
├──────────────────────┼────────────────────────┼───────────────────────────────┼────────────────────────┤
│ • Procedural Graphs  │ • WorldAbduct/E-Ledger │ • Harness2 (Recursive search) │ • Trajectory RAG       │
│   (P05, 2609.09153)  │   (P04, 2610.11552)    │ • Meta-Harness (P06, 2603.28) │   (Few-shot history)   │
│                      │ • CDA (Continual WM)   │ • StarHarness (P10, 2608.24)  │ • Reflexion / ReAct    │
│                      │   (P03, 2609.19551)    │ • AHE (P07, 2604.25850)       │   (Single-task memory) │
│                      │ • THIS REPO            │ • RRSI (P13, 2609.24972)      │                        │
└──────────────────────┴────────────────────────┴───────────────────────────────┴────────────────────────┘
```

---

## 2. Primary External Baselines (from the Literature)

### 2.1 The Closest Architectural Baseline: WorldAbduct / E-Ledger (P04)
* **Reference:** *E-Ledger and WorldAbduct: Diagnostic Learning of World Knowledge and Agent Harnesses* ([2610.11552](../papers/2610.11552-eledger-worldabduct.pdf)).
* **Method:**
  * Maintains an evolving world ledger ($K$), dynamic state tracking ($z$), a policy gate, and editable harness components ($H$).
  * Diagnoses failed trajectories to abduce hidden-rule hypotheses, tests them via targeted verification in sandboxes, and commits confirmed rules to the ledger while updating harness components.
* **Why it is an essential baseline:** It is the closest existing conceptual competitor that also couples world models with harness updates.
* **What our comparison tests:**
  * WorldAbduct treats rules and state monotonically and relies on LLM-judged rule verification without explicit lifetime boundaries.
  * *Hypothesis:* Partitioning knowledge into six explicit information lifetimes (`INTERFACE`, `CONFIGURATION`, `BEHAVIOR`, `PROCEDURES`, `LIVE`, `EVIDENCE`) with objective `GET`/`HEAD` verifications prevents context bloat and hallucinated constants better than monolithic abduction.

---

### 2.2 The Direct Code-Search / Harness-Evolution Baseline: Harness2 (P11)
* **Reference:** *Harness2: Two-Level Recursive Harnessing for Agentic Execution* ([2610.14224](../papers/2610.14224-harness2.pdf), Google Research).
* **Method:**
  * Uses a two-level optimization scheme: task-level Propose–Probe–Compose recursion (running $3k+1$ fresh-container task rollouts) over an 8-file component hierarchy (`SurfaceSpec`), combined with a cross-task `EXPERIENCE.md` library.
  * Operates without access to ground-truth grading rubrics or scores in the improvement context (`KEEP_BASE` attribution).
* **Why it is an essential baseline:** Harness2 represents the leading alternative to world modeling: *“Instead of modeling the environment, mutate the harness code directly using multi-rollout contrastive search.”*
* **What our comparison tests:**
  * Harness2 incurs massive sample and compute costs ($3k+1$ container runs per task) and requires a fresh container environment per rollout.
  * *Hypothesis:* Stateful single-pass execution over an operational world model achieves superior task value and consistency at a fraction of the compute and interaction budget required by brute-force container rollouts.

---

### 2.3 The Pure Procedural Memory Baseline: Procedural Graphs (P05)
* **Reference:** *Procedural Graphs for Agent Reasoning* ([2609.09153](../papers/2609.09153-procedural-graphs.pdf)).
* **Method:**
  * Maintains an attributed procedural graph representing execution workflows.
  * During inference, it provides localized neighborhood guidance to a base solver model. Between task batches, it refines the graph using success/failure traces and validation score gates.
* **Why it is an essential baseline:** It isolates procedural structure (how to act) from factual/operational knowledge (how the environment works).
* **What our comparison tests:**
  * Procedural Graphs has no underlying operational model of endpoints, input/output schemas, or error causes. If an API contract drifts or an endpoint schema changes, the procedural graph cannot perform root-cause attribution.
  * *Hypothesis:* Linking procedures explicitly to dependent environment claims (`required_claims` in [`Procedure`](file:///Users/masoud/Documents/Repos/worldmodel-generation-harness/src/worldmodel_harness/procedures.py#L47)) enables proactive invalidation and repair when environment contracts change, avoiding silent regressions.

---

### 2.4 The Continual Rule Discovery Baseline: CDA (P03)
* **Reference:** *Continual Enterprise World Model Discovery* ([2609.19551](../papers/2609.19551-continual-enterprise-wm.pdf)).
* **Method:**
  * Discovers structured condition/effect rules (constants, formulas, transformations, lookup tables) and carries them across changing enterprise deployments.
  * Operates in two distinct phases: a dedicated *Discovery* phase (exploratory queries) followed by an *Application* phase (predicting and executing without further experiments).
* **Why it is an essential baseline:** It is the primary reference for persistent enterprise rule discovery and rule carryover under domain shifts.
* **What our comparison tests:**
  * CDA separates discovery from execution into rigid phases and struggles to retire invalid rules when environments shift.
  * *Hypothesis:* Interleaved online learning during regular task execution, combined with dependency-aware invalidation, adapts faster and wastes fewer environment interactions than disjoint discovery phases.

---

### 2.5 Performance-Guided Program Search: Meta-Harness (P06) & StarHarness (P10)
* **References:**
  * *Meta-Harness: Direct Program Search for Agentic Harnesses* ([2603.28052](../papers/2603.28052-meta-harness.pdf)).
  * *StarHarness: Targeted Harness Evolution on AutomationBench* ([2608.24804](../papers/2608.24804-starharness.pdf)).
* **Method:**
  * Uses an LLM proposer to directly edit harness Python files, prompt templates, and tool wrappers based on execution logs and benchmark assertion feedback.
* **Why they matter:** StarHarness is directly integrated with AutomationBench and Stirrup.
* **What our comparison tests:**
  * These systems generate arbitrary Python scripts, risking unconstrained bugs, security vulnerabilities, and sandbox escapes.
  * *Hypothesis:* Bounded, data-only AST procedures ([`Step`](file:///Users/masoud/Documents/Repos/worldmodel-generation-harness/src/worldmodel_harness/procedures.py#L24)) executing through gated dispatch achieve the speedups of harness adaptation while preserving safety, deterministic auditability, and budget enforcement.

---

### 2.6 Dynamic Runtime Grounding: Enterprise Learned World Models (P02)
* **Reference:** *Do Enterprise Systems Need Learned World Models?* ([2605.12178](../papers/2605.12178-enterprise-learned-wm.pdf)).
* **Method:**
  * Compares static prompted models against dynamic runtime configuration discovery (using tools like `snow_query` to query live tables, schemas, and SLAs on the fly).
* **What our comparison tests:**
  * Proves that reading configuration dynamically is often superior to memorizing static transition rules.
  * *Hypothesis:* Storing explicit provenance, timestamps, and invalidation rules—and treating "reading configuration" as a consequential, costed action—balances fresh state retrieval against amortized reuse.

---

## 3. Comparison Matrix of Literature Baselines

| Method / Paper | Representation ($K$) | Harness Support ($H$) | Update Schedule | Feedback Permitted | Limits / Failure Modes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P01 World of Workflows** | Audit diffs & schemas | Fixed agent loop | Trajectory prediction | Grader audit diffs | Passive observer; no reusable harness evolution. |
| **P02 Enterprise Learned WM** | Runtime config queries | Fixed ReAct prompt | Per-task query | None (grounded reads) | High latency; up to 15 recursive calls per prediction. |
| **P03 Continual Discovery (CDA)** | Structured condition/effect rules | Fixed predictor | Disjoint Discovery vs. Application | Interactive env feedback | Rigid phases; difficulty retiring obsolete rules. |
| **P04 WorldAbduct / E-Ledger** | World ledger (monolithic) | Direct component edits | Per-trajectory diagnosis | Sandbox verification | Rule verification lacks objective record groundings. |
| **P05 Procedural Graphs** | None (no environment model) | Procedural transition graph | Batch trace refinement | Task pass/fail score | No environment contracts; breaks when APIs change. |
| **P06 Meta-Harness** | None | Executable Python code | Iterative propose/eval | Empirical task rewards | Unbounded code generation; risk of overfitting. |
| **P10 StarHarness** | None | Stirrup component edits | Search intervention loop | Assertion-level grading | Proposer sees grader feedback directly (leakage risk). |
| **P11 Harness2** | `EXPERIENCE.md` library | 8-file component tree | Task rollout recursion | Score-free (`KEEP_BASE`) | Expensive ($3k+1$ runs/task); requires fresh containers. |
| **This Repository** | Layered Operational Graph (6 layers) | Declarative AST Procedures | Online task-interleaved | Objective `GET`/`HEAD` (grader isolated) | Data-only procedures (no arbitrary Python generation). |

---

## 4. Controlled Internal Ablation Matrix (Q1–Q4)

To prove that improvements stem from our specific architectural commitments (rather than base LLM capability or prompting quirks), experiments must execute the **2×2 Factorial Design** defined in [docs/research-questions.md](research-questions.md#6-q4--reusable-execution-support):

```
                       ┌──────────────────────────────┬──────────────────────────────┐
                       │   Fixed Harness (No Tips)    │   Adaptive Harness (Tips)    │
┌──────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ Matched Non-Graph    │             V₀₀              │             V₀₁              │
│ Claim Memory         │  (Flat RAG / Lexical Search) │ (Direct Harness Adaptation)  │
├──────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ Operational          │             V₁₀              │             V₁₁              │
│ World Graph          │   (Graph-Guided Planning)    │ (Coupled WM + Procedures)    │
└──────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

### The Primary Mathematical Contrasts:

1. **Value of Procedure Adaptation on the Graph ($\Delta_{\text{adapt}\mid G}$):**
   $$\Delta_{\text{adapt}\mid G} = V_{11} - V_{10}$$
   *Does compiling repeated workflows into reusable procedures save tokens and calls compared to planning from scratch using the same operational world graph?*

2. **Value of the World Graph for Harness Adaptation ($\Delta_{G\mid\text{adapt}}$):**
   $$\Delta_{G\mid\text{adapt}} = V_{11} - V_{01}$$
   *Does adapting execution procedures with explicit world-model dependency links outperform adapting procedures blindly from raw traces?*

3. **Synergy / Interaction Effect ($\Delta_{\text{interaction}}$):**
   $$\Delta_{\text{interaction}} = (V_{11} - V_{10}) - (V_{01} - V_{00})$$
   *Is there a true interaction between world modeling and procedural compilation, or are their benefits purely additive?*

### Specific Targeted Ablation Arms:
* **Arm $V_{\text{raw}}$ (Raw Episode RAG, Q1):** The agent stores raw interaction traces and retrieves the $k$ most similar trajectories into the prompt. Isolates whether consolidating episodes into structured claims adds value over simple few-shot retrieval.
* **Arm $V_{\text{flat}}$ (Flat Consolidated Claims, Q1):** The agent retains the exact same claims as the operational graph, but stores them as unstructured text or flat JSON records without relational edges or dependency traversal.
* **Arm $\beta=0$ (Passive Observation, Q3):** The agent learns incidentally from task execution but sets exploration weight $\beta = 0$ (never spending calls on exploratory probing). Isolates whether targeted uncertainty resolution repays its cost.
* **Arm Blind-Revalidation (Q5):** When an API schema changes, the agent re-executes all historical validations rather than using dependency-graph invalidation.

---

## 5. Budget-Constrained Evaluation: The Minimal Viable Suite

If operating under constrained compute or API token budgets, running the full suite of literature baselines is cost-prohibitive. For example, **Harness2 (P11)** requires $3k+1$ fresh-container task rollouts per task, and **Meta-Harness (P06)** incurs massive token burn through unconstrained code-search loops.

To maintain scientific defensibility while minimizing expense, use the **Minimal Viable 4-Arm Suite**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE MINIMAL BUDGET COMPARISON SUITE                             │
├─────────────────────────┬───────────────────────────┬──────────────────────────────────┤
│ Baseline                │ How to Run / What it is   │ What it Proves / Isolates        │
├─────────────────────────┼───────────────────────────┼──────────────────────────────────┤
│ 1. Zero-Memory Base     │ Standard Stirrup          │ Anchor baseline; raw LLM variance│
│ 2. Few-Shot RAG         │ Raw Trajectory Retrieval  │ Value of graph vs. raw history   │
│ 3. World Model Only     │ `worldmodel --operational`│ Isolates world knowledge (V₁₀)   │
│ 4. Full System          │ `--operational            │ Synergy of world model +         │
│                         │   --semantic-learning`    │ procedural compilation (V₁₁)     │
└─────────────────────────┴───────────────────────────┴──────────────────────────────────┘
```

### 5.1 The Must-Have Comparisons

1. **Zero-Memory Base Agent (The Ground Floor):**
   * *What it is:* The vanilla agent executing tasks with an empty store across every round.
   * *Why it is mandatory:* Establishes baseline task difficulty and stochastic model variance without memory.
   * *Cost:* $1\times$ inference cost.

2. **Few-Shot Episodic RAG (The Standard Industry Baseline):**
   * *What it is:* Store past raw successful trajectories in a simple vector/text store and inject the 1–2 most relevant past traces into the prompt as few-shot demonstrations.
   * *Why it is mandatory:* Answers the essential reviewer critique: *"Why build complex operational world graphs when you could just retrieve past successful trajectories using standard RAG?"*
   * *Cost:* Negligible learning cost (zero synthesis passes; only adds prompt tokens at inference).

3. **Operational World Model ONLY ($V_{10}$ — Key Internal Ablation):**
   * *What it is:* Run with `--operational`, but with procedural compilation turned off (`--semantic-learning` disabled).
   * *Why it is mandatory:* Isolates the value of the **environment knowledge layer** (endpoint schemas, dependency resolution, objective `verify_goal` checks) from procedural execution.
   * *Cost:* Built directly into the harness CLI with no extra overhead.

4. **The Full System ($V_{11}$ — Proposed Coupled Method):**
   * *What it is:* Run with `--operational --semantic-learning` (Operational World Graph + Procedural Tips).
   * *Contrast with $V_{10}$:* Measures the exact marginal savings of compiling subroutines into reusable procedures ($\Delta_{\text{adapt}\mid G} = V_{11} - V_{10}$).

### 5.2 Single External Literature Competitor to Retain: WorldAbduct (P04)
If an external published baseline is required by reviewers:
* Compare against **WorldAbduct / Flat Rule Abduction**: Extract unstructured text rules from execution traces (e.g., *"Endpoint POST /tickets requires priority"* or *"Error 409 means customer exists"*), but **without** 6-layer lifetime separation or graph dependency invalidation.
* *Why:* It directly isolates the value of the multi-layer graph architecture over monolithic rule abduction.

---

## 6. Practical Execution Protocol & Cost Optimization

To minimize token usage and runtime during testing on AutomationBench:

### 6.1 Practical Budget Guidelines
1. **Reduce Task Count (10–15 Tasks instead of 50):**
   Select a focused subset of 10–15 representative tasks containing recurring subgoals (e.g. creating users, searching records, linking entities across tools).
2. **Run 2 Rounds instead of 3:**
   The delta between Round 1 (discovery) and Round 2 (amortized reuse) yields ~90% of the learning curve insight while cutting total execution cost by 33%.
3. **Use Free / High-Efficiency Model Profiles:**
   Use the repository's pre-configured free OpenRouter Nemotron endpoint (`configs/model-free-test.toml`) or an efficient high-speed model (e.g. `qwen-2.5-72b-instruct` or `gpt-4o-mini`) rather than expensive frontier models.

### 6.2 Execution Commands

```sh
# Override with the free testing profile:
export WORLDMODEL_MODEL_CONFIG=configs/model-free-test.toml

# 1. Base Agent (Zero Memory / Control)
uv run --extra automationbench worldmodel rounds \
  --task-count 10 --rounds 2 --seed 1729

# 2. Operational World Model Only (V10: Graph without procedures)
uv run --extra automationbench worldmodel rounds \
  --task-count 10 --rounds 2 --seed 1729 \
  --operational

# 3. Full Coupled System (V11: Operational Graph + Adaptive Procedures)
uv run --extra automationbench worldmodel rounds \
  --task-count 10 --rounds 2 --seed 1729 \
  --operational --semantic-learning
```

### 6.3 Reporting Requirements
* **Checkpoint Reliability ($\text{pass}^k$):** Measure whether all $k$ rounds succeed on each task using [`curve_report.py`](file:///Users/masoud/Documents/Repos/worldmodel-generation-harness/src/worldmodel_harness/curve_report.py):
  ```sh
  uv run --extra automationbench python -m worldmodel_harness.curve_report runs/<run_id>/learning_curve.json
  ```
* **Efficiency Metrics:** Report environment API calls per solved task, total token usage (including failed attempts), and adaptation compute overhead.
* **Leakage Integrity:** Ensure grader assertion rubrics are strictly isolated from the proposer model.

