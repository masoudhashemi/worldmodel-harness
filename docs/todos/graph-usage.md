# Graph Usage in the World Model Layer

This document details how graphs are modeled, where graph structures are used across the codebase, and which graph-theoretic properties and algorithms are applied.

---

## 1. Executive Summary

The harness does not treat memory as a flat bag of text chunks or raw conversation turns. Instead, it maintains an explicit **operational knowledge graph** stored in SQLite and projected into working memory during task execution.

Graphs are used in three primary layers:
1. **Operational Claim & Dependency Graph** (`knowledge.py`, `store.py`, `retrieval.py`)
2. **Data-Flow / Value-Flow Composition Graph** (`composition.py`)
3. **Control-Flow Execution Graph (CFG)** (`procedures.py`)

A 2×2 factorial evaluation framework (`comparison.py`) explicitly measures the empirical contribution of graph representation (`graph` vs. `flat`) crossed with execution adaptation (`fixed` vs. `adaptive`).

---

## 2. Where Graphs are Used

### A. The Operational Knowledge Graph (`knowledge.py`, `store.py`)

The core repository memory is modeled as an attributed property hypergraph:

* **Nodes / Hyperedges (`Claim`):**
  * **Entity Participants:** Each claim defines `participants: tuple[str, ...]`, which are the semantic entities involved (e.g., service names, endpoints, resource types, or config keys).
  * **Relations (Edge Types):** `relation: str` specifies the relationship type, including:
    * `endpoint_contract`: Route, HTTP method, schema, parameters.
    * `tested_response_rule`: Expected response structures and error codes.
    * `observed_value_flow`: Output-to-input parameter flow between two endpoints.
    * `invalidation_link`: Events that invalidate certain cached facts.
  * **Architectural Layering:** Vertically tiered into `interface`, `configuration`, and `behavior` layers.

* **Directed Dependency Edges (`Dependency`):**
  * Claims specify directed dependency edges to prerequisite claims:
    ```python
    dependencies: tuple[Dependency, ...]
    # Dependency(claim_id, version, fields, mode="requires"|"monitors")
    ```
  * Mode `requires`: Hard prerequisite. If the target claim is missing, stale, or disputed, the current claim cannot be utilized.
  * Mode `monitors`: Passive linkage for change tracking without blocking execution.

* **Bipartite Provenance Edges (`Episode` $\leftrightarrow$ `Claim`):**
  * Directed edges connect claims to concrete execution episodes:
    * `supporting_episodes`: Evidence proving the claim.
    * `contradicting_episodes`: Evidence violating or disputing the claim.

* **Storage (`store.py`):**
  * Stored in SQLite with an inverted index on participants (`participants(claim_id, version, entity)`).

---

### B. Directed Data-Flow Graph (`composition.py`)

In [`composition.py`](file:///Users/masoud/Documents/Repos/worldmodel-generation-harness/src/worldmodel_harness/composition.py), the `CompositionLearner` constructs an explicit **Data-Flow Graph** across distinct API endpoints:

* **Nodes:** Endpoint contracts (`source_endpoint` and `target_endpoint`).
* **Directed Edges (`observed_value_flow`):**
  * Connects an output property of the source endpoint to an input property of the target endpoint:
    $$\text{Endpoint}_A.\text{output\_path} \xrightarrow{\text{value\_flow}} \text{Endpoint}_B.\text{input\_path}$$
  * Retains abstract paths and structural value kinds (e.g., `id`, `email`, `timestamp`), never literal runtime values.
  * Used to guide parameter binding across multi-step API chains without repetitive exploratory searches.

---

### C. Control-Flow Execution Graph (`procedures.py`)

In [`procedures.py`](file:///Users/masoud/Documents/Repos/worldmodel-generation-harness/src/worldmodel_harness/procedures.py), reusable execution routines (`Procedure`) are structured as **Control Flow Graphs (CFG)**:

* **Nodes (`Step`):** Individual operational actions (`call`, `query`, `transform`, `branch`, `verify`, `stop`).
* **Edges:** Control-flow transitions:
  * Default transition: `next_step`
  * Conditional branching: `on_false`
  * Error recovery paths: `on_error`
* **Cross-Graph Linkage:** Each procedure links to the operational claim graph through `required_claims: tuple[Dependency, ...]`, ensuring a procedure only runs when its prerequisite environmental contracts are valid and supported.

---

### D. The 2×2 Factorial Evaluation (`comparison.py`, `system-design.md`)

The repository explicitly isolates the utility of graph structures through factorial experimentation:

$$\begin{array}{c|c|c}
& \textbf{Fixed Executor} & \textbf{Adaptive Executor} \\
\hline
\textbf{Flat Memory} & \text{flat\_fixed} & \text{flat\_adaptive} \\
\hline
\textbf{Graph Memory} & \text{graph\_fixed} & \text{graph\_adaptive} \\
\end{array}$$

* **Metric:** Evaluates `graph_adaptation_delta` and `interaction = graph - flat` to determine whether relational graph structures provide statistically significant gains over flat lexical retrieval at equal character and token budgets.

---

## 3. Graph-Theoretic Properties & Algorithms Applied

| Graph Property / Operation | Implementation Location | Mechanism |
| :--- | :--- | :--- |
| **Transitive Dependency Closure (DAG Traversal)** | `retrieval.py` (`retrieve`) | When a claim is selected, a stack-based traversal (`while pending: claim = pending.pop()`) traverses outgoing `claim.dependencies`. It bundles the entire dependency tree or flags explicit gap nodes (`dependency:<id>@<version>`) if disconnected. |
| **Relational Neighborhood Expansion (1-Hop)** | `retrieval.py`, `store.py` (`related`) | Distinguishes `memory="graph"` from `memory="flat"`. Queries the participant index for all claims sharing entity nodes with the lexical seed (`store.related(set(seed.participants))`), expanding the search frontier to the immediate relational neighborhood. |
| **Directed Branching & Cycle Prevention** | `procedures.py` (`execute`) | Step transitions obey an explicit step budget (`max_steps=32` to `256`), preventing infinite execution loops along cyclic branch edges. |
| **Versioned Subgraph Invalidation** | `store.py` (`dependency_compatible`) | Checks version contiguousness and signature consistency along dependency edges. If an upstream claim updates or becomes disputed, downstream claims automatically fail compatibility checks. |

---

## 4. Intentionally Excluded Graph Features

To prevent hallucinations, combinatorial explosion, and high latency, the harness deliberately avoids:
1. **Unbounded Multi-Hop Walks:** Graph traversal is strictly bounded to 1-hop participant expansion and explicit dependency DAG chains.
2. **Black-box Graph Neural Networks (GNNs):** All edges and nodes are inspectable, typed symbolic records rather than opaque latent embeddings.
3. **Global Topology Metrics:** PageRank, betweenness centrality, and graph diameter are not computed; ranking relies on task relevance, operational layer priority, and direct dependency requirements.

---

## 5. Brainstorming: Path Traversal, Node Frequencies & Confidence-Weighted Execution

### A. Motivation & The Bridge Between Graphs

Currently, operational knowledge tracking in the harness is coarse:
* **Nodes/Claims** store binary episode counts (`support_count`, `contradiction_count`).
* **Value-Flow Edges** require a fixed heuristic threshold ($N \ge 2$ independent task confirmations) to graduate from `"hypothesis"` to `"supported"`.
* **Procedures** are validated offline with discrete verdicts (`"helped"`, `"hurt"`).

The system does **not** yet count the number of times a node is visited or a path is traversed during execution.

Tracking node visit frequencies, edge traversals, and transition outcomes creates a principled, mathematical bridge between the **Environment Graph** (what endpoints, schemas, and bindings are possible) and the **Execution Graph** (what sequence of actions should be executed). Frequent, high-confidence paths through the environment graph can be directly mined and compiled into executable procedures.

---

### B. Non-Bandit Formulations: Empirical Counting & Graph Statistics

A deterministic, counting-based statistical approach that does not require reinforcement learning:

#### 1. Transition Statistics
For each directed edge $e = (u \xrightarrow{a} v)$ (where $u$ is the source endpoint/state, $a$ is the action/binding, and $v$ is the target endpoint/state):
* $N(u)$: Total visit count to node $u$.
* $T(e)$: Total traversal count along edge $e$.
* $S(e)$: Successful transitions (e.g., HTTP $200$, expected schema received, valid ID extracted).
* $F(e)$: Failed transitions (e.g., $4xx/5xx$, schema mismatch, null field).

The empirical transition reliability is:
$$P(\text{success} \mid u \xrightarrow{a} v) = \frac{S(e)}{T(e)}$$

#### 2. Path Confidence Scoring
For a candidate multi-step workflow path $\mathcal{P} = (e_1, e_2, \dots, e_k)$:
$$\text{Confidence}(\mathcal{P}) = \prod_{i=1}^k P(\text{success} \mid e_i) \quad \text{or} \quad \min_{i=1 \dots k} P(\text{success} \mid e_i)$$

#### 3. Automatic Procedure Synthesis (Frequent Path Mining)
Instead of relying on the LLM to invent procedural workflows from scratch:
* **Mining:** Mine frequent sequential paths through the environment graph that satisfy:
  $$T(\mathcal{P}) \ge K_{\text{threshold}} \quad \text{and} \quad \text{Confidence}(\mathcal{P}) \ge \theta_{\text{min}}$$
* **Compilation:** Automatically compile these paths into stored `Procedure` artifacts with typed input/output bindings.
* **Pruning:** Prune redundant exploratory searches from the path, keeping only the minimal generative subgraph.

#### 4. Confidence-Gated Execution
* **High Confidence ($\ge \theta_{\text{high}}$):** Fast-path execution. Execute the multi-step path as an atomic batch (`run_procedure`), skipping intermediate model prompts.
* **Low / Moderate Confidence ($< \theta_{\text{high}}$):** Step-by-step model supervision. Enforce verification gates (`check_effect`) after each call.
* **Degraded Edge:** If $P(\text{success} \mid e)$ drops below a threshold, trigger surgical edge invalidation without discarding the rest of the graph.

---

### C. Multi-Armed Bandit (MAB) & Reinforcement Learning Formulations

A principled exploration-vs-exploitation framework using Upper Confidence Bounds, Contextual Bandits, and Non-Stationary formulations:

#### 1. Edge-Level Action Selection (Standard UCB1)
When at node $u$ with multiple alternative outbound edges $e \in \text{Out}(u)$ (e.g., alternative search endpoints or candidate parameter bindings):
$$\text{UCB}(u, e) = \hat{\mu}_e + c \sqrt{\frac{\ln N(u)}{T(e)}}$$
* $\hat{\mu}_e = \frac{S(e)}{T(e)}$: Empirical reward / success rate.
* $c \sqrt{\frac{\ln N(u)}{T(e)}}$: Exploration bonus.
* **Behavior:**
  * If an edge is well-tested and reliable ($T(e) \gg 0$, $\hat{\mu}_e \approx 1$), the agent exploits it deterministically.
  * If an edge is untested ($T(e) \approx 0$), the exploration term dominates, encouraging the agent to probe the alternative.

#### 2. Contextual Bandits for Procedure Selection (LinUCB)
When matching a user task to an available macro-procedure vs. raw model planning, the choice depends heavily on task context (domain, user intent, schema versions).

Using **LinUCB with Disjoint Linear Models**:
* **Arms $a \in \mathcal{A}$:** Candidate Procedure 1, Candidate Procedure 2, ..., Raw LLM Execution.
* **Context Features $\mathbf{x}_{t} \in \mathbb{R}^d$:**
  * Task domain one-hot (`finance`, `hr`, `sales`, etc.)
  * Semantic embedding of task prompt / sub-goal
  * Tenant and role identifiers
  * Tool/endpoint schema versions
* **Model:** Expected payoff of arm $a$ is linear in its context:
  $$\mathbb{E}[r_{t,a} \mid \mathbf{x}_{t}] = \mathbf{x}_{t}^\top \boldsymbol{\theta}_a^*$$
* **Ridge Regression Updates:** For each arm $a$, maintain design matrix covariance $\mathbf{A}_a \in \mathbb{R}^{d \times d}$ and response vector $\mathbf{b}_a \in \mathbb{R}^d$:
  $$\mathbf{A}_a \leftarrow \mathbf{A}_a + \mathbf{x}_t \mathbf{x}_t^\top, \qquad \mathbf{b}_a \leftarrow \mathbf{b}_a + r_t \mathbf{x}_t$$
  $$\hat{\boldsymbol{\theta}}_a = \mathbf{A}_a^{-1} \mathbf{b}_a$$
* **Selection Rule:**
  $$a_t = \arg\max_{a \in \mathcal{A}} \left( \mathbf{x}_t^\top \hat{\boldsymbol{\theta}}_a + \alpha \sqrt{\mathbf{x}_t^\top \mathbf{A}_a^{-1} \mathbf{x}_t} \right)$$
* **Hybrid LinUCB Extension:** Shared parameters $\boldsymbol{\beta}$ for domain-general properties + arm-specific parameters $\boldsymbol{\theta}_a$ for procedure-specific quirks.

#### 3. Sequential Path Planning via UCT (Upper Confidence Bounds for Trees/Graphs / MCTS)
Extend edge-level UCB to multi-step planning through the environment graph:
* Treat the environment graph as a state-action search space.
* Use Monte Carlo rollouts guided by UCB to discover the optimal route from initial task input to the final desired state (e.g., `invoice_retrieved` $\to$ `tax_calculated` $\to$ `sheets_appended` $\to$ `email_sent`).
* Backpropagate task completion rewards to update $S(e)$ and $T(e)$ along the entire path.

#### 4. Handling SaaS Drift via Non-Stationary Bandits

SaaS environments are non-stationary: APIs update, endpoints get deprecated, quotas reset, and schemas change. Standard UCB assumes stationary reward distributions. Three formulations address this:

##### a. Discounted UCB (D-UCB)
Applies an exponential decay factor $\gamma \in (0, 1)$ to down-weight historical observations:
$$N_t(\gamma, e) = \sum_{s=1}^t \gamma^{t-s} \cdot \mathbb{I}(e_s = e), \qquad \hat{\mu}_t(\gamma, e) = \frac{\sum_{s=1}^t \gamma^{t-s} r_s \cdot \mathbb{I}(e_s = e)}{N_t(\gamma, e)}$$
$$\text{D-UCB}(e) = \hat{\mu}_t(\gamma, e) + 2 c \sqrt{\frac{\ln \left(\sum_{e'} N_t(\gamma, e')\right)}{N_t(\gamma, e)}}$$
* When an endpoint breaks, recent failures heavily degrade $\hat{\mu}_t(\gamma, e)$, while the exploration term swells, naturally forcing the agent to discover fallback routes.

##### b. Sliding-Window UCB (SW-UCB)
Restricts statistics to the most recent window of $\tau$ interactions:
$$N_t(\tau, e) = \sum_{s=\max(1, t-\tau+1)}^t \mathbb{I}(e_s = e), \qquad \hat{\mu}_t(\tau, e) = \frac{1}{N_t(\tau, e)} \sum_{s=\max(1, t-\tau+1)}^t r_s \mathbb{I}(e_s = e)$$
$$\text{SW-UCB}(e) = \hat{\mu}_t(\tau, e) + c \sqrt{\frac{\ln (\min(t, \tau))}{N_t(\tau, e)}}$$

##### c. Change-Detection Bandits (CUSUM / Page-Hinkley UCB)
* Runs an explicit statistical change-detection algorithm (CUSUM) on edge error rates.
* When a statistically significant change is detected (e.g. cumulative sum of residuals crosses a threshold), the harness resets the statistics for that specific edge, triggering an immediate active probe.

#### 5. Cost-Aware & Budget-Constrained Bandits (Budgeted MAB)
In SaaS agent execution, calls are not free—they consume environment call budget, token quota, rate limits, and latency:
* **Cost Metric:** For edge $e$:
  $$\text{Cost}(e) = w_1 \cdot \text{latency\_ms} + w_2 \cdot \text{token\_cost} + w_3 \cdot \text{env\_call\_quota}$$
* **Reward-to-Cost UCB Rule:**
  $$\text{Score}(u, e) = \frac{\hat{\mu}_e + c \sqrt{\frac{\ln N(u)}{T(e)}}}{\mathbb{E}[\text{Cost}(e)]}$$
* Prevents the agent from choosing an endpoint that succeeds but consumes 10x the token budget or takes 5 seconds, when an efficient alternative exists.

#### 6. Bayesian Thompson Sampling Formulation
As an alternative to UCB, Thompson Sampling handles small-sample uncertainty smoothly:
* For each edge $e$, assume Bernoulli successes with Beta conjugate prior:
  $$\theta_e \sim \text{Beta}(\alpha_e, \beta_e) \quad (\text{initially } \alpha=1, \beta=1)$$
* **Update:** On transition success: $\alpha_e \leftarrow \alpha_e + 1$; on failure: $\beta_e \leftarrow \beta_e + 1$.
* **Sampling:** Draw sample $\tilde{\theta}_e \sim \text{Beta}(\alpha_e, \beta_e)$ for each candidate edge and choose $e^* = \arg\max \tilde{\theta}_e$.

#### 7. Hierarchical Bandits (Two-Tier Decision Architecture)
* **Macro-Bandit (Tier 1):** Selects execution mode:
  1. *Run Stored Procedure* (fast, 0 model turns)
  2. *UCT Graph-Guided Execution* (guided search over environment graph)
  3. *Unconstrained LLM Reasoning* (fallback when graph coverage is absent)
* **Micro-Bandit (Tier 2):** When Tier 1 chooses Graph-Guided Execution, UCB/Thompson Sampling selects the individual step transitions.

---

### D. Proposed Implementation & Testing Roadmap

1. **Storage Layer (`store.py`):**
   * Add a `transitions` table in SQLite:
     ```sql
     CREATE TABLE IF NOT EXISTS transitions (
         source_claim_id TEXT NOT NULL,
         target_claim_id TEXT NOT NULL,
         edge_relation TEXT NOT NULL,
         traversals INTEGER NOT NULL DEFAULT 0,
         successes INTEGER NOT NULL DEFAULT 0,
         discounted_traversals REAL NOT NULL DEFAULT 0.0,
         discounted_successes REAL NOT NULL DEFAULT 0.0,
         mean_latency_ms REAL NOT NULL DEFAULT 0.0,
         mean_token_cost REAL NOT NULL DEFAULT 0.0,
         last_traversed_at TIMESTAMP NOT NULL,
         PRIMARY KEY (source_claim_id, target_claim_id, edge_relation)
     );
     ```
2. **Telemetry Hook (`runtime.py` / `decision_runtime.py`):**
   * After each `execute_decision` outcome, record the transition from previous endpoint to current endpoint, updating both raw and discounted counters.
3. **Evaluation Protocol:**
   * Test across 4 arms:
     * **Arm 0 (Baseline):** Current unweighted graph.
     * **Arm 1 (Frequency-Weighted Path Mining):** Formulation 1 without bandits.
     * **Arm 2 (UCB1 / UCT):** Edge-level UCB and graph planning.
     * **Arm 3 (LinUCB + Discounted UCB):** Contextual procedure selection with non-stationary drift adaptation.
   * **Target Metrics:**
     * Task completion accuracy.
     * Reduction in discovery calls ($api\_search$) across iterations.
     * Recovery turns required when an API route breaks (testing non-stationary drift recovery).
     * Total token consumption and wall-clock execution time.


