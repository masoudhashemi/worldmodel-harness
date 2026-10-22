# Architecture & Topology of the Learned World Model (Environment Layer)

A learned **World Model** in this harness is not a generic free-text summary or raw token cache. It is a **grounded, multi-layer typed knowledge graph** that captures the real environment's interfaces, capabilities, output schemas, causal failure modes, and data-flow dependencies.

---

## 1. Topological Graph of the Real Learned Model

The plot below is rendered directly from the SQLite store (`runs/focused-env-audit/store.sqlite`) populated by our runs on `operations.contractor_badge_expiration` and `sales.zoom_meeting_followup`:

![Learned World Model Topology](/Users/masoud/.gemini/antigravity-cli/brain/0310ea49-122f-4371-b14b-e4c012e57910/world_model_graph.png)

### Key Visual Components:
- **Service Hubs (Blue)**: Partition boundaries (e.g. `SHEETS`, `TWILIO`, `SALESFORCE`, `ZOOM`). These enforce strict service scoping and eliminate cross-service lexical leakage.
- **Read Endpoints (Green)**: Verified inspect/read operations (`sheets.spreadsheets.get`, `salesforce.query`, etc.).
- **Write Endpoints (Orange)**: Verified state-mutating actions (`twilio.messages.create`, `salesforce.sobjects.opportunity.update`, etc.).
- **Observed Output Schemas (Gray Nodes, Dashed Green Edges)**: Concrete response structures confirmed against the live environment (`records:array`, `spreadsheetId:string`, `status:string`).
- **Value Flows / Composition Edges (Orange/Red Arrows)**: Causal dependencies where an output parameter from one call binds directly to the input parameter of another (e.g., `spreadsheetId -> path.spreadsheetId`).

---

## 2. The 4-Layer World Model Ontology

The environment layer is structured hierarchically across 4 complementary layers:

```mermaid
flowchart TD
    subgraph Layer4 ["Layer 4: TASK & GOALS"]
        T1["Active Subgoal / Unresolved Requirement"]
        T2["Identity Anchors & Preconditions"]
    end

    subgraph Layer3 ["Layer 3: SYSTEM INVARIANTS & RECOVERY"]
        S1["Possible Error Causes (e.g. URL query param vs params dict)"]
        S2["Environmental Blockers (e.g. api_search is doc-only, not instance data)"]
        S3["Contrastive Divergence Pairs"]
    end

    subgraph Layer2 ["Layer 2: BEHAVIOR & VALUE COMPOSITION"]
        B1["Observed Output Types (status:string, records:array)"]
        B2["Observed Value Flows (source[id] -> target[id])"]
        B3["State Invalidation Triggers (writes invalidate cached reads)"]
    end

    subgraph Layer1 ["Layer 1: INTERFACE & CONTRACTS"]
        I1["Endpoint Contract (Method + URL Pattern)"]
        I2["Documented Operation Semantics (Read vs Write Effect)"]
        I3["Active Service Partition (Sheets, Twilio, Salesforce)"]
    end

    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
```

---

## 3. Concrete Example: How a Learned Action Loop Operates

When the agent explores and acts in the environment, the World Model records, verifies, and connects claims in a closed feedback loop:

```mermaid
flowchart LR
    A["api_search"] -->|"Discovers Contract"| B["sheets.spreadsheets.get<br/>(GET https://sheets.../{id})"]
    B -->|"Dispatches Call"| C["Live WorldState Response"]
    C -->|"Extracts Output Schema"| D["Observed Output:<br/>spreadsheetId: str<br/>sheets: list<br/>properties: dict"]
    D -->|"Observes Flow"| E["sheets.spreadsheets.values.get<br/>(path.spreadsheetId = spreadsheetId)"]
    
    subgraph ErrorLearned ["Causal Invariant Learned"]
        F["Error: URL with query string<br/>interpreted as ID"]
        F -->|"Constraint"| G["Normalize URL query into params dict"]
    end
    
    E -.-> ErrorLearned
```

---

## 4. Graph Node & Edge Taxonomy

| Element | Graph Representation | Role in Decision Making | Ground-Truth Check |
|---|---|---|---|
| **Service Hub** | Large Blue Node | Restricts context retrieval strictly to the active task domain | `world.meta.allowed_services` |
| **Endpoint Contract** | Green/Orange Node | Precondition gate for invoking `api_fetch` | AutomationBench URL router handler |
| **Documented Semantics** | Node Effect Attribute | Distinguishes harmless inspection from state mutation | `effect in {"read", "write"}` |
| **Output Type** | Small Gray Node (Dashed) | Informs the planner which keys exist in the response payload | Checked against live response JSON keys |
| **Value Flow Edge** | Directed Orange/Red Arrow | Allows the agent to automatically chain multi-step pipelines | Validates parameter input/output types |
| **Error Cause / Invariant** | Graph Annotation / Gate | Blocks repeated harness errors before execution | Evaluated against precondition check |
