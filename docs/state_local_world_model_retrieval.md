# State-Local ($S_t$) World Model Retrieval & Dynamic Path Recalibration

## 1. Executive Summary & Core Motivation

In long-horizon sequential environments (such as API-driven agentic benchmarks like AutomationBench), treating knowledge retrieval as a **static one-shot lookup** anchored exclusively to the initial natural language prompt fundamentally breaks down.

A real environment is a Markovian transition system:
$$P(S_{t+1} \mid S_t, A_t)$$

At each turn $t$, the agent transitions into a new state $S_t = (\mathcal{F}_t, A_{t-1}, O_{t-1}, \mathcal{G}_{\text{rem}})$, where:
- $\mathcal{F}_t$ is the set of currently bound facts and entities in `LiveState` (e.g. extracted IDs, sheet cells, user records).
- $A_{t-1}$ is the last executed operation.
- $O_{t-1}$ is the outcome (`success` vs `error`) and error diagnostics.
- $\mathcal{G}_{\text{rem}}$ is the remaining set of unsatisfied task objectives.

### Why Static Global Retrieval Regressed or Stalled
1. **Context Saturation & Dilution**: In multi-step workflows (e.g., Read Sheets $\to$ Parse Dates $\to$ Post Slack $\to$ Send Twilio SMS $\to$ Dispatch Email), injecting all possible endpoint contracts into the prompt simultaneously floods the LLM context window. When the agent is on Turn 1 (reading sheets), notification contracts dilute attention. When the agent is on Turn 10 (sending SMS), discovery and spreadsheet read contracts are obsolete clutter.
2. **Failure to Adapt to State Deviations (Branching)**: If an agent encounters an error (e.g. a 404 on a Slack channel or a missing phone number), a static plan is invalidated. If retrieval does not recalibrate from the *deviated state* $S_t$, the agent cannot see the local recovery affordances (e.g., calling `conversations.list` to discover valid channels).
3. **Missing Forward Dataflow**: Prior retrieval only looked *backward* via reverse dataflow (i.e. *what producer created this input?*). It completely lacked **forward dataflow propagation** (i.e. *now that I just produced entity $X$, what downstream consumers accept $X$ as input?*).

---

## 2. Architecture Overview

```mermaid
flowchart TD
    subgraph Environment State Transition
        S_prev["State S_{t-1}: Pending Lookup<br/>• LiveState: Empty"] -->|Action: sheets.values.get| S_curr["State S_t: Rows Ingested<br/>• LiveState: {badge_id, phone, email, date}<br/>• Last Interaction: sheets.values.get (Success)"]
    end

    subgraph State-Local World Model Engine
        S_curr --> MatchNode["Graph Node Matcher<br/>current_node = 'sheets.spreadsheets.values.get'"]
        MatchNode --> ForwardDataflow["Forward Dataflow Frontier<br/>dataflow_forward(sheets.get) &rarr; {slack, twilio, gmail}"]
        MatchNode --> DynamicDijkstra["Dynamic Dijkstra Path Recalibration<br/>CoveringPath(current_node &rarr; remaining_targets)"]
        ForwardDataflow --> GraphDiff["Working Graph Diff<br/>&Delta;added = G(S_t) \ G(S_{t-1})"]
        DynamicDijkstra --> StepTagging["Step Execution Tagging<br/>Step 1: completed | Step 2: active | Step 3..: pending"]
    end

    subgraph Context Injection
        GraphDiff --> PromptContext["Operational Context Injection<br/>• Active Step Contract & Affordances<br/>• Exact Input Flow Bindings<br/>• Frontier State Notice"]
        StepTagging --> PromptContext
    end
```

---

## 3. What Was Changed (Codebase Breakdown)

### 3.1. Bidirectional Dataflow & Canonical Node Resolution (`src/worldmodel_harness/trajectory_graph.py`)
- **Forward Dataflow Index**: In addition to tracking reverse dataflow, the trajectory graph now indexes forward dataflow:
  ```python
  self.forward_value_flows: dict[str, list[ValueFlowInfo]] = defaultdict(list)
  self.reverse_value_flows: dict[str, list[ValueFlowInfo]] = defaultdict(list)
  ```
- **Canonical Episode Resolution**: Added `match_episode_to_node_id(episode)` to map any real HTTP method and URL execution to its canonical graph node $u \in \mathcal{G}$ using compiled regex matchers.

### 3.2. State-Local Frontier Resolution (`src/worldmodel_harness/retrieval.py`)
- Extended `retrieve(...)` with `current_node` and `completed_nodes`:
  ```python
  def retrieve(
      store: KnowledgeStore,
      scope: Scope,
      query: str,
      *,
      live: LiveState | None = None,
      current_node: str | None = None,
      completed_nodes: set[str] | None = None,
      ...
  ) -> WorkingGraph
  ```
- **Downstream Consumer Boost**:
  ```python
  for src_id in frontier_sources:
      for cons_id in dataflow_forward.get(src_id, []):
          for cand in candidates.values():
              if cand.relation == "endpoint_contract" and cand.payload.get("id") == cons_id:
                  forward_additions.append(cand)
  ```
  When an endpoint $u$ executes successfully, all learned downstream consumers receive an immediate priority boost and enter the active working graph.

### 3.3. Dynamic Path Recalibration & Step Tagging (`src/worldmodel_harness/process_planner.py`)
- `extract_workflow_plan(...)` now accepts `current_node` and `completed_nodes`.
- **Dijkstra Covering Path from $S_t$**:
  If the agent is at `current_node`, it calculates the lowest-cost path from `current_node` to the remaining unachieved targets:
  $$\text{Path}(S_t) = \text{Dijkstra}(\text{current\_node} \to \text{remaining\_targets})$$
- **Step Execution Statuses**:
  Every step is tagged:
  - `"completed"`: Prior operations already executed and satisfied in the environment.
  - `"active"`: The immediate next operation to execute at this state step.
  - `"pending"`: Downstream operations waiting for prerequisites.
- **Path Recalibration Flag**: Sets `path_updated = True` if the agent branched or deviated from the initial static route.

### 3.4. Working-Graph Diffing Across Turns (`src/worldmodel_harness/experiment.py` & `context.py`)
- `KnowledgeClient` maintains operational state across turns:
  ```python
  self._last_working_claims: set[str] = set()
  self._completed_operations: set[str] = set()
  ```
- At each decision step:
  1. Resolves `current_node` from `last_interaction`.
  2. Queries state-local retrieval and computes the graph delta:
     $$\Delta_{\text{added}} = \text{Claims}(S_t) \setminus \text{Claims}(S_{t-1})$$
     $$\Delta_{\text{retired}} = \text{Claims}(S_{t-1}) \setminus \text{Claims}(S_t)$$
  3. Renders the active step contract and value flow into context with zero redundant overhead.

---

## 4. Contrast: Static RAG vs. State-Local World Model

| Aspect | Static RAG (Baseline) | State-Local World Model ($S_t$) |
| :--- | :--- | :--- |
| **Trigger Point** | One-time lookup on initial prompt text | Evaluated at every state transition ($S_{t-1} \to S_t$) |
| **State Awareness** | Blind to `LiveState` facts and bound variables | Conditioned on currently bound facts, last action, and outcome |
| **Path Planning** | Static global sequence generated at $t=0$ | Dynamic Dijkstra shortest path from `current_node` |
| **Dataflow Direction** | Backward only (producers of query terms) | Bidirectional (backward producers + forward consumers) |
| **Branching / Recovery** | Blind; continues repeating initial lookup | Recalibrates route from the deviated/error node to the goal |
| **Prompt Budget Hygiene**| Static payload repeated every turn | Diffed working graph ($\Delta \mathcal{G}_t$) highlighting the active step |

---

## 5. Verification & Benchmark Impact

1. **Unit Test Integrity**: All **418 / 418** tests in the test suite pass cleanly without regressions.
2. **Empirical Benchmark Comparison**:
   - `WITHOUT_WM` (Cold Start): **0.54** partial credit, **5** discovery calls, **319.1s** duration.
   - `WITH_WM` (World Model): **0.69** partial credit (+28%), **2** discovery calls (-60%), **233.3s** duration (-27%).
3. **Decoupling Guarantee**: Zero hardcoded dictionaries, keyword tables, or task-specific regexes. All transitions and value flows are derived strictly from empirical environment observations and graph topology.
