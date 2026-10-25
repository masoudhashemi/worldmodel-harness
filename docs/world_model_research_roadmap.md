# World Model Research & Execution Roadmap

This document tracks the phased progression of the World Model Harness across algorithmic implementation, unit verification, and empirical benchmark evaluations.

---

## Phase A: Algorithmic Implementation & Unit Verification

- [x] **Task 1: Commit & Snapshot Validated Graph & Journal Milestone**
  - Committed ground-truth comparison engine (`src/worldmodel_harness/graph_comparison.py`), test suite (`tests/test_worldmodel_ground_truth_comparison.py`), and journal entry `E007`.
  - Commit: `cf38216`.

- [x] **Task 2: Graph-Conditioned Pre-Execution Planning (Implementation)**
  - Implemented ontological stage rank ordering (`lookup -> mutate -> notify`) and multi-objective target-coverage ranking in `src/worldmodel_harness/process_planner.py`.
  - Added unit test `test_multi_objective_target_coverage_ranking` in `tests/test_process_planner.py`.
  - Commit: `aa16de4`.

- [x] **Task 3: Continual Learning & Consolidation (Implementation)**
  - Implemented and verified multi-iteration value flow consolidation from `hypothesis` to `supported` across distinct task/tenant contexts ($k=1 \to k=2$).
  - Verified parameter slot generalization and non-catastrophic retention in `tests/test_continual_learning_consolidation.py`.
  - Commit: `e0fd257`.

- [x] **Task 4: Active Frontier Discovery (Implementation)**
  - Implemented `discover_and_index_frontier()` in `src/worldmodel_harness/frontier_discovery.py` with grounded discovery evidence episodes.
  - Verified 100.0% recall across all 47 services (758 endpoints) with 0.0% hallucinations in `tests/test_frontier_discovery.py`.
  - Commit: `098f3a7`.

---

## Phase B: Empirical Benchmark Evaluations (Live AutomationBench Tasks)

- [x] **Experiment E008: Empirical Evaluation of Task 2 (Graph-Conditioned Pre-Planning)** *(COMPLETED)*
  - **Task**: `operations.sheets_monday_maintenance_queue`
  - **Model**: `openai/gemma-4-31B-it` via Toolkit cluster
  - **Result**: **0.33 Partial Credit** (up from 0.00 baseline).
  - **Verification**: Escalated stagnant query relaxation successfully guided the model to omit `board_id` from `items:find`, resolving `itm_700` and `brd_ops`. Model mutated Monday status to `Scheduled` (graded PASS) and dispatched Slack notification to `#ops-updates`.
  - **Artifact**: `runs/eval-task2-preplanning-monday-queue-v2/summary.json` (Run ID: `4a804e9ea2544d27a3534dfd2282c24f`).
  - **Journal Entry**: `E008` in `docs/experiment-journal.md`.

- [x] **Experiment E009: Empirical Evaluation of Task 3 (Continual Learning & Multi-Iteration Consolidation)** *(COMPLETED)*
  - **Benchmark**: Sequential continual learning across isomorphic perturbation tasks (`operations.contractor_badge_expiration_isomorphic`, `operations.twilio_afterhours_incident_isomorphic`) using carried store seeded from 5-task gated run.
  - **Results**: **0.75 Macro Partial Credit** (0.70 on contractor badges, 0.80 on twilio incidents); **0.0% Anchor Hazard Rate** (0 poison literal hits across runs).
  - **Consolidation**: `google_drive.files.list -> sheets.spreadsheets.get` promoted from `hypothesis` to `status="supported"` in SQLite; 6 total value-flow claims now supported.
  - **Artifact**: `runs/eval-task3-continual-consolidation/summary.json`.
  - **Journal Entry**: `E009` in `docs/experiment-journal.md`.

- [x] **Experiment E010: Empirical Evaluation of Task 4 (Frontier Discovery on Cold-Start Tasks)** *(COMPLETED)*
  - **Benchmark**: Cold-start evaluation on `operations.pipefy_slack_purchase_request` comparing:
    - Arm A: Cold-start baseline (empty store)
    - Arm B: Frontier-discovered World Model (pre-indexed full catalog via `discover_and_index_frontier`)
  - **Results**: **0.50 Partial Credit** (vs 0.00 baseline); **63.6% Faster Duration** (120.4s vs 330.8s); **48.1% Fewer Calls** (14 vs 27 environment calls).
  - **Fidelity**: Pre-indexed `INTERFACE` layer delivered exact parameter names and HTTP verbs, correctly evaluating policy rules and dispatching the approval alert to Slack.
  - **Artifact**: `runs/eval-task4-frontier-benchmark/summary.json`.
  - **Journal Entry**: `E010` in `docs/experiment-journal.md`.

---

## Phase C: Extended Cross-Domain Evaluation (Salesforce, Sheets, Gmail)

- [ ] **Experiment E011: Extended Cross-Domain Iterative Benchmark (Sales Domain)** *(RUNNING)*
  - **Tasks**: `sales.multi_hop_lookup`, `sales.negative_selection`, `sales.recency_selection`
  - **Model**: `openai/gemma-4-31B-it` via Toolkit cluster
  - **Seed Store**: `runs/eval-task4-frontier-benchmark/stores/full_frontier.sqlite` (758 pre-indexed endpoints)
  - **Iterations**: 2 iterations with persistent memory accumulation (`worldmodel_carried.sqlite`)
  - **Goal**: Evaluate iterative cross-domain knowledge accumulation ($k=1 \to k=2$) across Salesforce, Sheets, and Gmail, measuring execution speedup, call reduction, and learning curve.
  - **Output**: `runs/eval-extended-sales-benchmark/summary.json`.

---
*Updated: 2026-10-25*
