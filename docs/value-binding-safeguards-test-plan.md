# Value abstraction and binding safeguards: v3.4

Implementation test plan, written before live execution. Preserve the user's
value-kind learning and route deduplication; add conservative boundaries:

- Unknown/constraint rule leaves stay literal. Example/parameter abstraction must
  name explicit leaf paths. A generalized kind predicate is not a copy/equality rule.
- Rule identities include roles, scope and exact constraints. Hypotheses remain
  hypotheses until the existing future-task evidence tests promote them. Replayed
  generating events are not independent evidence. Original observations stay in
  the append-only episode log.
- Automatic source binding requires one matched source with the exact scalar
  value and a matching task identity on the target record. Arbitrary message prose
  is not automatically interpreted as ownership. Keep explicit source links for it.
  Record original request and inferred binding provenance. Outcome verification
  rejects kind markers; literal equality and current-read checks still apply.
- Same-route aliases merge only within the same scope and with matching documented
  payload (excluding endpoint id). Conflicting aliases retain their own claims;
  old checkpoint conflicts also block selection and report schemas/scopes. No
  arbitrary preference for one schema. Same-ID authoritative updates retain the
  existing versioned-documentation semantics; this is not full schema reconciliation.

## Offline

Run full pytest, Ruff, and the captured-failure replay. Cover wrong same-kind value,
wrong entity, ambiguous source, prose-only binding, literal cutoff/URL preservation,
explicit abstraction, invalid abstraction paths, duplicate event evidence, schema
conflicts and tenant isolation. Do not weaken checks to obtain completion.

## Live smoke comparison

Four attempts: email and phone, empty/learned arms, one reset replicate each. Use
`scripts/checkpoint_reliability.py --replicates 1` with the prior full current-code
manifest as the settings/checkpoint reference. Default runner remains three
replicates; this bounded test intentionally does not estimate pass^2 or pass^3.

Free Nemotron, 8,192 output tokens, 20 turns, 24 environment calls, 240 seconds.
Disposable working graph per attempt; unchanged source checkpoints, no cross-trial
learning, semantic/procedure learning or grader feedback. Archive current code and
lockfile before calls; no runtime edits during measurement. No replacement trials
or paid fallback. Stop on the runner's two-consecutive-zero-interaction condition.

Pre-completion correction: initial smoke run
`checkpoint-reliability-6b7ad98664854c3490f252058762df37` was interrupted after a
code inspection found nested-dictionary aliasing in rule normalization. Preserve
its partial observations but exclude them from the final smoke comparison. Fix
normalization to copy before rewriting and test that the input rule is unchanged;
restart from the same immutable checkpoints with a new source archive and manifest.

Report grader/clean/checked separately, verification and binding audit, gate gaps,
tokens and unavailable usage. A four-attempt smoke comparison can reveal failures,
not establish reliability improvement. Rule abstraction is exercised by offline
tests, not this semantic-learning-disabled live arm. Do not pool historical trials.
