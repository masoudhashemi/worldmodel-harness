# Repaired operational interface: checkpoint reliability pilot

Written before live execution. Implement actionable gate errors, typed array evidence
paths, historical/current goal-evidence modes, conservative resource invalidation,
and executable semantic-rule validation. No model, prompt tuning or runtime changes
after measurement begins. No new procedures or grader-assisted learning.

## Fixed protocol

- Model: `configs/model-free-test.toml`, Nemotron free, 8,192 output tokens.
- Tasks: `simple.email_sf_contact_phone_update` and
  `simple.email_sf_contact_email_update`, identical to the earlier pilot.
- Arms: empty checkpoint versus
  `runs/small-sample-bd095ff2395c485bbe451ecfe3be3881/E-learned-repeat-after.sqlite`.
- Three reset replicates per task per arm: twelve executions, sequential, seeded
  randomized arm order within task/replicate pairs. Record the schedule before calls.
- Identical operational graph interface, no adaptive procedures, no semantic
  consolidation/hints and no information-policy probes. This isolates access to
  retained supported graph knowledge, not the semantic-learning mechanism.
- Each task: 20 acting turns, 24 environment calls, 240-second acting timeout,
  12,000-character retrieval allowance. Provider policy unchanged; no paid fallback.
- Each replicate starts from its immutable checkpoint in a disposable SQLite copy,
  a fresh environment and fresh conversation/live state. Fresh API discovery and
  ordinary within-task knowledge updates are allowed in that copy in both arms;
  they never carry to another replicate or mutate the checkpoint. This allows the
  empty arm to discover contracts without giving it prior-task learning. No
  between-trial selection or adaptation. Save/hash checkpoints and verify unchanged.
- Explicit **in-sample checkpoint** mode permits previously learned task names;
  independent frozen-transfer overlap checks remain unchanged. This is not unseen
  task evaluation. Previously retained procedures stay inaccessible.

## Repair acceptance tests

1. Rejected arguments report all field paths/error types and checked schema, without
   echoing invalid input values or executing the tool. Missing/ambiguous contracts
   identify method, URL and matching candidates and suggest discovery.
2. Dictionary keys and typed array indexes work, including legacy numeric strings
   on arrays. Invalid/negative indexes and uncovered/missing fields remain unobserved.
3. Current-state checks require live evidence; explicitly historical read evidence
   remains usable for a past observation after writes, never as a live prerequisite.
4. A known write invalidates reads across its service origin (including queries and
   collections) plus known cross-service value-flow dependents, not only its URL.
   Unknown write identity/origin falls back to broad invalidation. This is a declared
   conservative service-boundary policy, not learned causal effect precision.
5. Rule conditions normalize JSON-string params/body consistently without changing
   raw evidence. Retained executable rules must have applicable cited observations
   with supported paths; this checks testability, not truth. Unobserved novel ideas
   can remain narrative hypotheses without a response rule. Promotion still requires
   subsequent evidence, and no causal authority comes from fitting a source episode.

## Outcomes

Preflight amendment: the first measurement attempt
`checkpoint-reliability-741a102ba4844383934a78122622e7e8` was stopped after two
completed executions and one interrupted execution. It exposed generic schema errors
at Stirrup's outer tool boundary, before the repaired inner gate. Those results are
an incomplete diagnostic, excluded from the restarted twelve-trial measurement.
The repair now enriches both boundaries and logs outer schema failures. The restart
uses the same checkpoints, task schedule, budgets and free model; revision v3.1.

Primary: strict grader pass^1, pass^2, pass^3 per arm, computed per task as
`comb(successes,k)/comb(3,k)` then averaged. Also report pass@k as
`1-comb(3-successes,k)/comb(3,k)`, clean acting completion, evidence-checked completion,
registered requirement coverage, discovery calls, blocked/schema calls, returned
tokens and latency. Include failures in denominators, missing usage as unknown.
No claim of reliability improvement from two tasks without adequate uncertainty;
report raw per-task outcomes. Check coverage is model-defined, not independent proof.

This pilot compares an empty and one learned checkpoint after identical repairs.
It does not establish a monotonic learning curve, individual repair effects, total
amortized learning benefit or generalization. Include prior graph-construction costs
as excluded from measurement, not zero. If endpoint unavailability prevents trials,
preserve failures and stop after two consecutive zero-interaction trials; do not mix
models inside the comparison. The earlier 50-task run remains stopped.
