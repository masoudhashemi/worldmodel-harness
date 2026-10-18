# v3.2 outcome-verification pilot

Status: completed, twelve executions. **Acceptance criterion not met.**
[Predeclared plan](verification-v32-experiment-plan.md).
[Measurement manifest](../runs/checkpoint-reliability-3fb928a703c34f10bab86be7819e8a89/manifest.json).
[Historical v3.1 results](checkpoint-reliability-results.md) remain separate.

Preflight: 263 tests and lint pass. A runner-only preflight stopped before model
calls because SQLite backup headers were not byte-identical. The runner now uses
the original immutable baseline files as inputs and creates disposable per-trial
backups as before. No runtime change or unsuccessful model trial was discarded.

The experiment uses the free Nemotron model, the same two tasks, initial graph
sources, twelve-trial schedule and budgets as v3.1. It tests stricter outcome
verification, not semantic/procedure learning or independent task generalization.
No between-replicate learning is allowed. Sources remain unchanged and hashed.

Outcome contracts are model-selected. Even matched value/resource checks do not
prove complete task interpretation or correct record identity. Report grader
success, clean finish and verification separately; do not pool revisions.

## Results

All twelve v3.2 summaries match their scheduled v3.1 counterparts' task and model
settings, budgets and trajectory-only protocol. The exact baseline source hashes
and seeded schedule were verified unchanged. No runtime edits occurred during
measurement. No paid fallback or larger benchmark run was started.

| Outcome / total over six trials | v3.1 empty | v3.2 empty | v3.1 learned | v3.2 learned |
| --- | ---: | ---: | ---: | ---: |
| Strict grader passes | 6 | 4 | 6 | 6 |
| Clean finishes | 4 | 1 | 6 | 0 |
| Checked finishes | 4 | 0 | 6 | 0 |
| Timeouts | 1 | 5 | 0 | 5 |
| Environment calls | 55 | 59 | 49 | 50 |
| Discovery calls | 26 | 35 | 19 | 21 |
| Model calls | 100 | 96 | 91 | 86 |
| Returned tokens | 1,225,185 | 1,221,179 | 1,178,737 | 1,086,208 |
| Calls without usage | 1 | 5 | 0 | 5 |
| Model-call elapsed seconds | 1,076.9 | 1,402.4 | 988.1 | 1,420.9 |
| Inner gate rejections | 5 | 5 | 1 | 1 |
| Outer schema rejections | 2 | 9 | 4 | 16 |

The single v3.2 clean finish was **unresolved and grader-failing**. There were ten
timeouts and one turn-exhaustion failure. No run established checked completion.
The v3.1 checked label had weaker semantics and included one presence-only learned
finish. Nevertheless, zero v3.2 checked finishes is not a verification-quality win:
the agent did not demonstrate that it could use the stronger contract successfully.

Token totals are lower bounds, with ten calls missing usage. Lower observed tokens
are not an efficiency improvement when calls are interrupted and completion falls.
Graph-construction costs are excluded. Provider latency and sample variation can
contribute to cross-revision differences; this is not a concurrent randomized
ablation isolating the verifier's causal effect.

## Reliability

Values are per-task combinatorial estimates averaged across the two tasks.

| v3.2 outcome | Empty pass^1 / pass^2 / pass^3 | Learned pass^1 / pass^2 / pass^3 |
| --- | --- | --- |
| Strict grader success | 66.7% / 50% / 50% | 100% / 100% / 100% |
| Clean completion, including unresolved | 16.7% / 0% / 0% | 0% / 0% / 0% |
| Checked completion | 0% / 0% / 0% | 0% / 0% / 0% |

Grader pass@1/@2/@3 is 66.7%/83.3%/100% for empty and 100% throughout for learned.
Clean-finish pass@1/@2/@3 is 16.7%/33.3%/50% for empty and zero throughout for learned.
Checked pass@k is zero for both. In v3.1, both arms' grader pass^k was 100%; clean
pass^1/^2/^3 was 66.7%/33.3%/0% empty and 100% throughout learned. These small empirical
samples do not establish true probabilities or unseen-task reliability.

Raw v3.2 outcomes in replicate order:

| Task / checkpoint | Grader | Acting outcome |
| --- | --- | --- |
| Email / empty | pass, pass, pass | timeout, timeout, timeout |
| Email / learned | pass, pass, pass | timeout, timeout, timeout |
| Phone / empty | pass, fail, fail | timeout, unresolved finish, timeout |
| Phone / learned | pass, pass, pass | turn exhaustion, timeout, timeout |

## Trace findings

1. **The interface repeatedly failed before verification.** There were 25 outer
   schema rejections: 22 malformed nested `action` JSON errors and three calls
   putting `goal` and `rationale` inside `action`. Learned/phone replicate 1
   (`296c701e1c0e4da8a45eb9ef5e8773e9`) had ten outer rejections. Its complete history
   contains one exact action-argument string six times and another twice. Generic
   full-schema repair feedback did not break that loop.
2. **Grounding source observations is awkward under an all-equality contract.**
   Empty/email replicate 1 repeated an ungrounded `find_email` update. Empty/phone
   replicate 1 progressed from ungrounded source-email content to circular grounding,
   then repeated the circular-grounding failure. A requirement to find/read a source
   document differs from a requirement to make a target field equal its extracted
   value. v3.2 does not express that distinction adequately. Re-reading a source
   solely to obtain a different episode ID would not solve the semantic problem.
3. **Weak checks were rejected, but replacements did not finish.** Examples include
   learned/email replicate 2 and empty/phone replicate 3. Learned/phone replicate 2
   ended with a `present` source-body predicate marked `insufficient_outcome_predicate`.
   Several invalid updates were rejected atomically, leaving empty ledgers. Seven
   trials ended with no registered checks; none ended with matched checks.
4. **Path diagnosis worked in a live failure.** Learned/phone replicate 2
   (`25fdd5e341bc4d79a0fdf54c812034ac`) tried `Phone` on an empty response object.
   The report returned `invalid_evidence_path`, `failed_at: 0`, and no available keys,
   rather than stale evidence. It did not repair before timeout. No stale-evidence
   refresh or successful wrong-path repair was demonstrated by this pilot.
5. **Schema/evidence assembly remained costly.** Empty/email replicate 2 put
   `evidence_ids` inside checks, where `episode_id` was required. It repaired those
   fields but ended with `missing_outcome_contract` for both requirements. Across
   all trials there were 24 inner-audited goal-update attempts, including one exact
   duplicate beyond its first occurrence; malformed outer calls are not in that
   count. Semantically similar but differently worded updates are not counted as
   exact duplicates.

The [saved console log](../runs/checkpoint-reliability-3fb928a703c34f10bab86be7819e8a89/console.log)
contains execution progress and visible rejection feedback. The SDK abbreviates
long tool outputs in this display; it is not a full model-input transcript. Ten
timeout runs lack saved `history.json`; their summaries retain decision inputs,
validation audits, cost records and final checks, and their evidence databases
retain environment calls. Full histories exist for the two non-timeout runs.
Do not interpret absent console text as proof that a mechanism never occurred.

## Decision

**Do not expand to ten or fifty tasks yet.** The safeguards prevented weak checked
finishes but the usable verification loop regressed. Offline test success did not
predict live usability.

The smallest next experiment should simplify the generic interface: expose a
typed verification operation without a nested action JSON payload or duplicated
target/observation structures; distinguish task-grounded source observation from
desired-effect equality; preserve source-to-target value and identity bindings;
and return a specific missing-field/path repair. Source-observation checks should
establish task relevance and provenance, not require advance knowledge of a whole
message body. Effect checks should still require grounded desired values and
current target reads. Do not weaken these into unrestricted presence checks or
hard-code benchmark solutions.

Replay the captured failure cases offline first, then use a small predeclared live
probe before another full reliability comparison. This recommendation is not yet
implemented. The experiment itself is complete and the runtime remains v3.2.
