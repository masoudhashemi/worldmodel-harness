# Consolidated Implementation Plan: Injection Fixes & Contrastive Precondition Learning

**Date:** October 19, 2026  
**Status:** Unified Specification & Implementation Roadmap  
**Target File:** `docs/todos/consolidated-injection-and-contrastive.md`  
**Consolidates:**
- `docs/todos/error-analysis.md` (Todo 2: Injection, schema coercion, and retrieval fixes)
- `docs/todos/contrastive-semantic-learning.md` (Todo 3: Multi-modal contrastive analysis, silent failure detection, and precondition induction)
- Cross-references: `docs/todos/graph-usage.md` (Multi-arm bandits and graph algorithms)

---

## 1. Executive Summary & Rationale for Consolidation

Rather than applying piecemeal changes that require multiple refactorings and multiple benchmark restarts, this specification consolidates **Todo 2 (Prompt Injections & Schema Defect Fixes)** and **Todo 3 (Cross-Task Contrastive Semantic Learning)** into a single unified implementation pass.

### The Direct Synergy Between Todo 2 and Todo 3
1. **Todo 3 produces new causal knowledge:** It mines counterfactual pairs of episodes—comparing failed/unproductive actions (e.g. silent failures with $200\text{ OK}$ and `updated: 0`, missing authentication, unnested params, or verifier assertion failures) with successful counterpart actions across tasks—and induces **preconditions, silent failure signatures, and operational invariants**.
2. **Todo 2 defines how knowledge is retrieved and injected:** Without fixing Todo 2, any new contrastive claims would either be dumped as 12,000 characters of unranked noise or appended as a trailing `UserMessage` that destroys the LLM's tool feedback loop.
3. **Decision runtime enforces both:** When the agent emits an action (whether using shorthand `"a1"` or full JSON), the runtime gate validates both formal schema requirements AND contrastive preconditions before spending environment calls.

```mermaid
flowchart TD
    subgraph Execution & Episode Logging
        A["Agent Action (Coerced 'a1' -> ProposedAction)"] --> B["Tool Execution / Environment"]
        B --> C["Rich Multi-Modal Observation:<br/>Arguments + Raw Response + State Delta + Feedback"]
        C --> D["KnowledgeStore (Episodes)"]
    end

    subgraph Contrastive Causal Induction (Todo 3)
        D --> E["Contrastive Miner:<br/>Pairs Negative Trajectory (τ-) with Positive Trajectory (τ+)"]
        E --> F["LLM Causal Attribution Engine:<br/>Analyzes Raw Response, State Diff, and Verifier Feedback"]
        F --> G["KnowledgeStore:<br/>Injects 'contrastive_precondition' Claims into Graph"]
    end

    subgraph Consolidated Retrieval & System Injection (Todo 2)
        G --> H["retrieval.py:<br/>Relevance-Ranked & Bounded (remembered capped to 2)"]
        H --> I["context.py:<br/>Renders Contracts + Concise Causal Invariants"]
        I --> J["experiment.py:<br/>Injected into SystemMessage (Tool Frontier Preserved)"]
        J --> K["decisions.py (check_decision):<br/>Action Coercion + Contrastive Precondition Gate"]
        K --> A
    end
```

---

## 2. Unified Technical Architecture & File-by-File Changes

### A. Message Placement Fix (`src/worldmodel_harness/experiment.py`)
- **Problem:** Appending `UserMessage(content=content)` at the end of `selected_messages` pushes tool outputs/errors out of the conversation frontier, causing the model to miss error feedback and enter a 50-turn infinite deadlock loop.
- **Unified Fix:**
  1. Inject `content` (operational memory, contracts, AND contrastive invariants) into `selected_messages[0]` (`SystemMessage`). If no system message exists, insert one at index 0.
  2. Eliminate the trailing `UserMessage` append in both operational memory and episodic memory blocks.
  3. The latest `ToolMessage` remains the conversation frontier at all times.

```python
# In src/worldmodel_harness/experiment.py
if content:
    header = "### Operational Memory & Learned Invariants\n"
    if selected_messages and selected_messages[0].role == "system":
        selected_messages[0] = SystemMessage(
            content=selected_messages[0].content + "\n\n" + header + content
        )
    else:
        selected_messages.insert(0, SystemMessage(content=header + content))
return await self._generate(selected_messages, tools)
```

---

### B. Action Shorthand Coercion & Precondition Gating (`src/worldmodel_harness/decisions.py`)
- **Problem 1:** Prompt tells the model: *"Execute only the next action using its short ID or name (e.g. a1)"*. When the model emits `{"action": "a1"}`, `json.loads("a1")` crashes with `JSONDecodeError`, burning Turn 1 on every task.
- **Problem 2:** The gate in `check_decision()` only checks static schema dependencies; it does not enforce learned contrastive business preconditions (e.g., requiring subscription creation before activation).
- **Unified Fix:**
  1. In `CompactDecision.decode_serialized_action`, support bare string identifiers (`"a1"`, `"api_search"`), coercing them into `{"tool": value.strip(), "arguments": {}}`.
  2. In `check_decision()`, check whether `decision.action.tool` matches any active `contrastive_precondition` claims in `store`. If the action violates a required preceding action or state condition, return an explicit gap `contrastive_precondition:<rule>`.

```python
# In src/worldmodel_harness/decisions.py
@field_validator("action", mode="before")
@classmethod
def decode_serialized_action(cls, value):
    if isinstance(value, str):
        value_str = value.strip()
        try:
            parsed = json.loads(value_str)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                return {"tool": parsed.strip(), "arguments": {}}
        except (json.JSONDecodeError, ValueError):
            pass
        return {"tool": value_str, "arguments": {}}
    return value
```

---

### C. Bounded, Relevance-Ranked Retrieval (`src/worldmodel_harness/retrieval.py`)
- **Problem:** When `focus_entities` is empty at the start of a task, `remembered` gathers every supported contract in the database, flooding the prompt with 12,000 characters of unrelated SaaS endpoints (Calendly, Airtable, Jira).
- **Unified Fix:**
  1. Cap `remembered` unranked fallback to at most **2 contracts**, preventing context flooding across tasks while preserving the unit test requirement (`test_supported_endpoint_contract_survives_unrelated_wording`).
  2. Add `contrastive_precondition` to actionable relations in `rank(claim)` so that learned causal invariants receive high retrieval priority when their associated tool or entities are active.

```python
# In src/worldmodel_harness/retrieval.py
actionable = (
    2
    if claim.relation in ("tested_response_rule", "endpoint_contract", "contrastive_precondition")
    or claim.payload.get("response_rule") is not None
    else (1 if claim.payload.get("workspace_fact") else 0)
)
...
remembered = (
    []
    if focus_entities
    else [
        c
        for c in candidates.values()
        if c.claim_id not in queued
        and c.relation in ("endpoint_contract", "contrastive_precondition")
        and c.status == "supported"
        and not c.needs_review
        and (c.valid_until is None or c.valid_until > now)
    ][:2]  # Strict cap to prevent cross-task context flooding
)
```

---

### D. New Module: Contrastive Causal Induction (`src/worldmodel_harness/contrastive.py`)
- **Purpose:** Extracts causal invariants from multi-task episodes using holistic evidence (arguments, response bodies, silent failure indicators like `updated: 0` or empty `[]`, and verifier assertions).
- **Key Functions:**
  1. `mine_contrastive_pairs(store, scope)`: Scans historical episodes to find counterfactual pairs:
     - Negative episode ($\tau^-$): An action on tool/endpoint $E$ resulting in silent failure (`updated: 0`), explicit error, or verifier failure.
     - Positive episode ($\tau^+$): An action on tool/endpoint $E$ resulting in verified success.
  2. `induce_preconditions(client, store, scope, pairs)`: Formulates an LLM prompt containing the full counterfactual evidence ($R_{\text{raw}}$, $\Delta S$, $F$) to diagnose the missing precondition or parameter formatting rule.
  3. Ingestion: Stores verified invariants in `KnowledgeStore` with `relation="contrastive_precondition"`, linking supporting and contradicting episodes.

---

### E. Compact Invariant Rendering (`src/worldmodel_harness/context.py`)
- **Purpose:** In `render_context()`, format contrastive claims into a concise, easily digestible text block rather than dumping thousands of characters of raw JSON:
  ```
  ### Learned Causal Invariants & Avoided Failures:
  - [crm.update_status]: Setting status='Active' silently fails (updated: 0) unless billing.create_subscription is executed first.
  - [sheets.values.get]: Querying Sheet1 fails when custom tabs exist; invoke get_metadata first to resolve tab names.
  ```

---

## 3. Implementation Checklist & Verification Plan

- [ ] **Step 1: Implement Injection & Schema Fixes**
  - Update `experiment.py` (system prompt injection).
  - Update `decisions.py` (action shorthand coercion).
  - Update `retrieval.py` (cap `remembered` to 2; rank `contrastive_precondition`).
- [ ] **Step 2: Implement Contrastive Learning Engine**
  - Create `src/worldmodel_harness/contrastive.py` (pair mining and causal induction).
  - Update `src/worldmodel_harness/context.py` to render contrastive invariants cleanly.
  - Connect contrastive precondition checks to `check_decision()` in `decisions.py`.
- [ ] **Step 3: Unit & Regression Testing**
  - Run `uv run pytest` to ensure all 348 existing tests pass.
  - Add unit tests for action shorthand coercion (`"a1"`, `"api_search"`).
  - Add unit tests for message frontier order in `experiment.py`.
  - Add unit tests for contrastive pair mining and claim generation.
- [ ] **Step 4: Clean Corrupted Data & Re-launch**
  - Remove zero-call `learn` rows from `runs/iterative-gemma/results.jsonl`.
  - Reset `runs/iterative-gemma/stores/learn.sqlite`.
  - Launch iterative evaluation and verify that Task 2+ makes $\ge 20$ environment calls and achieves positive scores.
