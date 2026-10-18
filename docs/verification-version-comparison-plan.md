# Interleaved verification-version comparison

Status: preflight only; **live execution blocked on baseline reproducibility**.
No model calls have been made for this experiment. The user approved a twelve-run
v3.2 versus v3.3.1 comparison; this document records the intended protocol before
measurement, not a completed comparison.

Current preflight: 275 tests pass; the learned checkpoint hash is unchanged.
Fingerprint of the sorted per-file SHA256 map for `src/worldmodel_harness/*.py`,
`uv.lock`, and `configs/model-free-test.toml` (JSON with sorted keys):
`2e2c03cd29f8479977a89cb220a413346d11b47b5842b2c889c5f655cc627e50`.
This identifies the current preflight files; it is not an archive of either version.

## Reproducibility gate

The workspace is not a Git repository and no archived v3.2 source snapshot was
found. Historical summaries identify `operational-learning-v3.2`; saved schemas,
audits and some histories constrain its behavior but do not constitute a complete
executable source snapshot. Running current code under an old revision label or
reusing historical scores would not implement the approved comparison.

Two paths are possible, requiring a user choice before live calls:

1. Supply/recover the original v3.2 source snapshot, then freeze both versions.
2. Explicitly authorize a **reconstructed v3.2 baseline**, validate its schemas,
   prompts and known behavior against historical artifacts, and label it as a
   reconstruction throughout. This would test that reconstructed implementation,
   not establish exact historical-version equivalence.

Before execution, archive/hash both executable source trees, dependency lockfile,
model config, runner and scoring audit. Keep the working runtime unchanged during
measurement and run each version in its own process with the same dependency set.
Record any differences beyond the intended verification implementation.

## Fixed live protocol

- Two tasks: `simple.email_sf_contact_email_update` and
  `simple.email_sf_contact_phone_update`.
- Two versions, three reset trials per task/version: twelve executions total.
- Generate and save a seed-1729 schedule before calls. Shuffle task order per
  replicate and version order within each task/replicate pair; execute sequentially.
- Both versions use the same learned input checkpoint referenced by
  `runs/checkpoint-reliability-3fb928a703c34f10bab86be7819e8a89/manifest.json`.
  Hash-check it before and after every trial. Fresh environment, conversation,
  live state and disposable knowledge copy per trial. No cross-trial learning.
- Free Nemotron via `configs/model-free-test.toml`; identical model settings,
  8,192 output tokens, 20 turns, 24 environment calls, 240-second acting timeout,
  12,000 retrieval characters. No paid fallback, semantic learning, procedures,
  information probes, grader-fed learning or mid-run prompt/code changes.
- Preserve all failed trials and missing usage. Stop after two consecutive trials
  with no environment interaction; report an incomplete experiment rather than
  replacing samples. Do not restart the 50-task stream or expand the task sample.

## Primary outcome and audit

Joint success requires all three:

1. Correct final state according to the benchmark grader.
2. Clean termination within the declared budgets, not an unresolved finish.
3. Adequate source and effect verification under a common post-run audit, not merely
   the version-specific `checked` label.

The post-run audit must identify task-relevant source evidence, the requested value
grounded in that evidence or task, a successful current target read, matching value
and target identity, and coverage of both source and effect requirements. Bare
response presence, write acknowledgments and unrelated matching records fail the
audit. Apply the same rubric to both versions' actual evidence; do not demand a
v3.3-specific field name or grant older checks a weaker success definition. If
adequacy cannot be established from saved artifacts, mark it unestablished and do
not count joint success. Save reasons and evidence IDs. Never pass audit/grader
outcomes to the actor or learner during the experiment.

Report joint-success pass^1/^2/^3 and pass@1/@2/@3 using per-task combinatorial
estimators, then average across the two tasks. Report raw per-task trials alongside
grader, clean, unresolved and nominal checked rates. Three trials on two tasks
cannot establish general reliability or unseen-task performance.

## Secondary diagnostics

- Malformed outer calls, inner gate failures and verification rejections separately.
- Exact unchanged retries and successful repairs after rejection, with denominators.
- Incorrectly accepted predicates under the common audit.
- Environment/discovery/model calls, returned tokens, missing usage and elapsed time
  for every attempt, including failures. Lower cost from timing out is not a win.
- Lost histories or missing diagnostics explicitly reported. Do not infer absence
  of an event from abbreviated console output.

Any improvement claim requires higher joint success without a verification-safety
regression under matched budgets. With this sample size, describe results as pilot
evidence rather than statistical proof. The test compares verification versions;
it does not measure learning between tasks. Shared-learning/checkpoint evaluation
is a separate subsequent experiment.
