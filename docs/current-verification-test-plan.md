# Current-code verification test

User requested testing current code only. No reconstructed or historical-version
arm. Written before live calls.

1. Run lint, all offline tests and the captured-failure replay.
2. Run two learned-checkpoint trials using the existing runner's `--probe` mode:
   email then phone, one attempt each, current v3.3.1 code.
3. Use the same immutable source checkpoint and free Nemotron configuration as
   `checkpoint-reliability-3fb928a703c34f10bab86be7819e8a89`. That manifest supplies
   inputs/settings only; its outcomes are not a comparison arm.
4. Keep 20 turns, 24 environment calls, 240 seconds, 8,192 output tokens and 12,000
   retrieval characters. Reset environment/conversation and disposable working
   graph per task. No cross-trial learning, semantic/procedure updates, paid fallback,
   larger benchmark, or runtime changes during measurement.
5. Report grader success separately from clean/checked finish. Inspect source and
   effect checks, identity/value grounding, failed repairs, schema errors and costs.
   Preserve failures and missing usage. Two attempts test usability, not reliability
   or learning improvement. No automatic retries.

Archive the current runtime source, dependency lockfile and model configuration
before live calls, excluding `.env` and all credentials. Record the manifest and
results in `docs/current-verification-test-results.md`.
