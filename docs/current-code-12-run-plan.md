# Current-code twelve-run test

Written before execution, 2026-10-16. Test current code only: two tasks (email and
phone), empty versus learned checkpoint, three reset replicates per arm/task.
Use the existing seeded sequential schedule in `scripts/checkpoint_reliability.py`.

Input/settings reference:
`runs/checkpoint-reliability-3fb928a703c34f10bab86be7819e8a89/manifest.json`.
Its historical results are not current-code observations. Do not pool prior probes.

Controls: free Nemotron, 8,192 output tokens, 20 turns, 24 environment calls,
240 seconds per attempt, existing 12,000-character retrieval setting. No paid
fallback, automatic retries, semantic learning, procedures or grader feedback.
Each attempt resets its environment and receives a disposable checkpoint copy;
ordinary within-task learning is enabled, but no learning passes between trials.
Verify immutable input hashes. Archive source/config/lockfile without credentials
before starting. Do not change runtime during measurement. Stop after two
consecutive attempts with no environment interaction, preserving all failures.

Report grader success, clean finish and checked finish separately, per arm and
task, including pass^k and pass@k for k=1,2,3. Audit verification failures, schema
rejections, calls, tokens and unavailable usage. Verify whether checked evidence
establishes the requested value and identity. Grader success alone is insufficient.
This is a small in-sample checkpoint comparison, not a learning curve or evidence
of generalization. Prior graph construction cost is excluded, not zero.

Save the manifest, console log, source snapshot and a results document. Do not
expand the task set or implement repairs as part of this test.
