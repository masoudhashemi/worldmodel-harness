# v3.2 verification reliability experiment

Written before live calls. Repeat the twelve-trial [checkpoint protocol](checkpoint-reliability-plan.md)
with v3.2 outcome contracts and separate path/freshness diagnostics. Do not change
runtime, prompts, model, budgets, or source checkpoints during measurement.

## Controls

- Baseline: `runs/checkpoint-reliability-8c2197dd48c04115b1ce84ceed0be33c/manifest.json`.
  Use its exact immutable empty/learned sources; verify hashes and identical seeded
  schedule. The runner creates disposable SQLite backups per trial; their file
  headers need not be byte-identical to the input.
- Same two simple tasks, two arms, three reset replicates, sequential calls.
- Free Nemotron, 8,192 output tokens, 20 turns, 24 environment calls, 240 seconds,
  12,000 retrieval characters. No paid fallback. No semantic learning, procedures,
  information probes or grader feedback to the actor/learner.
- Disposable per-trial knowledge copy; ordinary within-task updates remain enabled.
  No learning passes between replicates. Source graph construction cost excluded,
  not zero. This is in-sample checkpoint reliability, not a learning curve.
- Stop after two consecutive zero-interaction trials; preserve failures and missing
  usage. Do not replace unsuccessful samples or pool v3.1/v3.2 outcomes.

## Measurements

Report per-task raw outcomes and per-arm grader, clean-finish and checked-finish
pass^1/^2/^3 and pass@1/@2/@3 using the existing combinatorial estimators. Checked
status has stricter semantics than v3.1; compare labels with this caveat.

Audit final checks for value-specific equality, target resource and source grounding;
separate unresolved finishes, turn exhaustion, timeouts and provider errors. Record
inner/outer rejections, repeated unchanged calls where traces permit, discovery and
environment calls, returned tokens, missing usage and elapsed model time. Inspect
available histories for path repair, stale-evidence reads and weak-check rejection.
If a mechanism is not exercised or histories are unavailable, report that explicitly.

The immediate acceptance question is whether v3.2 prevents weak checked finishes
without recurrent unrepairable verification loops. A grader pass alone is insufficient.
Report regressions, even if stronger verification explains them. No significance or
generalization claim from two tasks and three replicates. Model-defined requirement
coverage and semantic adequacy remain limits of the contract implementation.

## Execution and next decision

Run `scripts/checkpoint_reliability.py` with this plan and the baseline manifest,
using `WORLDMODEL_MODEL_CONFIG=configs/model-free-test.toml`. Save results separately
in `docs/verification-v32-experiment-results.md` and link the generated manifest.
Offline baseline: 263 tests pass. The 50-task run remains stopped.

Do not automatically expand the benchmark. If this loop succeeds, propose a
predeclared roughly ten-task, diverse cross-task learning experiment; if it fails,
identify the smallest trace-grounded repair first.
