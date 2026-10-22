# The World Model as a State Transition System ($S_t \xrightarrow{a_t} S_{t+1}$)

In contrast to a static documentation catalog, a **World Model** in an agentic environment is fundamentally a **Markov Decision Process (MDP) / State Transition System**:

$$\text{State } S_t \xrightarrow{\text{Action / Tool Call } a_t} \text{State } S_{t+1}$$

Where:
- **State ($S_t$)**: The combined state of the environment (`WorldState`: CRM records, Google Sheets rows, sent messages) and the agent's grounded knowledge (discovered contracts, bound identifiers, invariants).
- **Edge ($a_t$)**: The executable action dispatched to the environment (tool name, HTTP method, URL, and arguments).
- **Transition ($\mathcal{T}$)**: The deterministic or observed mutation of world state and schema observations produced by that call.

---

## 1. Grounded Trajectory Plot

The diagram below maps the exact execution trajectory of `operations.contractor_badge_expiration` from [`runs/focused-env-audit/store.sqlite`](file:///Users/masoud/Documents/Repos/worldmodel-generation-harness/runs/focused-env-audit/store.sqlite):

![World Model State Transitions](/Users/masoud/.gemini/antigravity-cli/brain/0310ea49-122f-4371-b14b-e4c012e57910/state_transition_world_model.png)

---

## 2. State-Machine Diagram of the Execution Loop

```mermaid
stateDiagram-v2
    [*] --> S0_Initial: Task Prompt Received
    
    state S0_Initial {
        Spreadsheet: unread
        BadgeExpiry: unknown
        Notification: pending
    }

    S0_Initial --> S1_CapabilityDiscovered: api_search("spreadsheet ss_badges")
    
    state S1_CapabilityDiscovered {
        Contract: GET /v4/spreadsheets/{id}
        TargetID: ss_badges
    }

    S1_CapabilityDiscovered --> S_err_FormatFailure: api_fetch(GET ...?includeGridData) [400 Error]
    
    state S_err_FormatFailure {
        CausalInvariant: "Query params must not be appended to path"
        RecoveryAction: Strip query & pass clean path
    }

    S_err_FormatFailure --> S2_MetadataGrounded: api_fetch(GET .../ss_badges) [Fixed]
    S1_CapabilityDiscovered --> S2_MetadataGrounded: api_fetch(GET .../ss_badges)

    state S2_MetadataGrounded {
        SheetTitle: "Badge Tracker"
        SheetRange: "A1:Z100"
        DataflowReady: True
    }

    S2_MetadataGrounded --> S3_DataGrounded: api_fetch(GET .../values/Badge Tracker!A1:Z100)

    state S3_DataGrounded {
        ExpiringContractors: 3 found
        PhoneNumbers: Grounded
        NotificationNeeded: True
    }

    S3_DataGrounded --> S4_NotifierDiscovered: api_search("send text message sms")

    state S4_NotifierDiscovered {
        Contract: POST /Accounts/{Sid}/Messages.json
        RequiredParams: [To, Body]
    }

    S4_NotifierDiscovered --> S5_WorldMutated: api_fetch(POST .../Messages.json, To="+15551234567")

    state S5_WorldMutated {
        TwilioStatus: "queued / sent"
        MessageSID: SM8z0337aa0ga...
        ContractorNotified: True
    }

    S5_WorldMutated --> S_Final_Complete: execute_decision -> finish()

    state S_Final_Complete {
        StrictVerification: 100% Passed
        AssertionsSatisfied: All
    }

    S_Final_Complete --> [*]
```

---

## 3. Direct Connection to `graph-grounded-automationbench`

Your work in [graph-grounded-automationbench](https://github.com/masoudhashemi/graph-grounded-automationbench) establishes the precise formal foundation for this state-transition view:

### A. The 5 Typed Graph Edges as State Operators

| Edge Kind in `graph-grounded` | Role in State Transition | Trajectory Example |
|---|---|---|
| **`PRODUCES`** | $S_t \to S_{t+1}$ generates a typed value into state | `sheets.spreadsheets.get` **`PRODUCES`** `record_identifier` (`spreadsheetId`) |
| **`ACCEPTS_VALUE`** | State precondition requiring a grounded value | `sheets.spreadsheets.values.get` **`ACCEPTS_VALUE`** (`spreadsheetId`) |
| **`DATAFLOW_TO`** | Connects producer state to consumer action | `sheets.get[spreadsheetId]` $\xrightarrow{\text{DATAFLOW\_TO}}$ `values.get[spreadsheetId]` |
| **`MUTATES`** | Action that changes external environment reality | `twilio.messages.create` **`MUTATES`** `WorldState.twilio.messages` |
| **`PRECEDES`** | Directs macro workflow stage ordering | `Lookup Stage` $\xrightarrow{\text{PRECEDES}}$ `Mutation Stage` $\xrightarrow{\text{PRECEDES}}$ `Notification Stage` |

### B. Workflow Motifs as State Paths
In `graph-grounded-automationbench`, complex tasks match reusable motifs:
$$\text{Motif: } \text{cross\_service\_lookup\_then\_mutate\_then\_notify}$$

1. **Lookup State ($S_0 \to S_3$)**:
   - Discovers `sheets.spreadsheets.get` & `values.get`.
   - Reads rows, identifies expiring dates, binds phone numbers.
2. **Mutation / Notification State ($S_3 \to S_5$)**:
   - Authorizes `twilio.messages.create`.
   - Binds `To: +15551234567` from row data and dispatches SMS.
3. **Verification State ($S_5 \to S_{\text{final}}$)**:
   - Evaluates environment goal assertions before invoking `finish`.

---

## 4. How the World Model Harness Implements This

When the agent uses this harness:
1. **Preconditions Gate (`check_decision`)**:
   Checks whether the current state $S_t$ satisfies all mandatory fact bindings (e.g. valid `spreadsheetId` or `opportunity_id`) before the action edge is allowed to fire.
2. **Causal Invariants (`possible_error_cause`)**:
   When an action leads to an error state $S_{\text{err}}$, the reflection records the causal difference so future transitions bypass the failure.
3. **State Invalidation (`resource_invalidation`)**:
   When a mutating action fires along edge $a_t$ (`MUTATES`), the world model automatically invalidates stale cached reads of that entity, updating the state belief to reflect the mutation.
