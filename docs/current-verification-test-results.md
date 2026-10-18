# Current-code verification test

Status: completed. **Current code passes offline checks but does not complete
live verification within the test budget.** Current v3.3.1 only, not a version comparison.
[Plan](current-verification-test-plan.md).
[Manifest](../runs/checkpoint-reliability-e2d9199088b74363856110e24043af62/manifest.json).

Preflight: 275 tests passed, lint passed, captured-failure replay passed without
model/environment calls. Two free-Nemotron trials use unchanged learned input
knowledge, fresh task environments and disposable per-task graph copies.

The tested code, tests, runner, dependency lockfile and model configuration were
[archived before calls](../runs/current-verification-source.RhjgRd/source.tar.gz).
Archive SHA256: `caf02793e9a8f8c50b509f0c0d1be8cd49fd67299c8ae7973eb9138a833abf77`.
No `.env` or credentials are included. No runtime changes during measurement.

## Live results

| Outcome | Email | Phone |
| --- | ---: | ---: |
| Grader success | yes | yes |
| Clean / checked finish | no / no | no / no |
| Error | TimeoutError | TimeoutError |
| Environment / discovery calls | 8 / 3 | 10 / 4 |
| Model calls | 14 | 16 |
| Direct verification calls | 4 | 4 |
| Successful verification calls | 1 (source) | 0 |
| Inner / outer schema rejections | 0 / 0 | 0 / 0 |
| Returned tokens | 146,575 | 177,579 |
| Calls without usage | 1 | 1 |

Each task had a 240-second acting timeout. Returned tokens are lower bounds; one
interrupted call per task has no reported usage. Source checkpoint hashes remained
unchanged. All archived source/config/test/runner files match the files after the
run (macOS AppleDouble metadata entries excluded from the file-content comparison).
Both summaries identify `operational-learning-v3.3.1`.

## What actually worked and failed

**Email:** the first source check omitted `anchor_path`. A subsequent request used
`Maria Santos` from the public task and `from` in the source message; it matched.
However, the actor created a new requirement ID, leaving its first pending check
unresolved. Its effect attempt omitted identity information, then selected
`records / 0 / Id` as the identity path. That record ID does not match the inherited
name anchor, so the check correctly returned `target_identity_mismatch`. Another
pending effect requirement remained as well. Final ledger: one matched source,
two missing-evidence requirements and one mismatched effect identity.

**Phone:** effect requests omitted the verification resource, then the target
identity. Source requests used the requested phone number and later a discovered
email address as task anchors, neither literally present in the task. The richer
`missing_task_anchor` feedback supplied the full public task and repair instructions,
but no valid replacement was recorded before timeout. Final ledger: one pending
effect and one source with an invalid task anchor.

The direct interface produced no schema failures in these attempts. The source
check worked once, and inadequate effect identity was not falsely accepted.
Nevertheless neither session achieved joint success (correct state, clean finish,
adequate verification). This test does **not** establish improved reliability,
learning benefit or a usable end-to-end verification loop.

Artifacts: [console log](../runs/checkpoint-reliability-e2d9199088b74363856110e24043af62/console.log),
[email summary](../runs/64f349768cfb400d91b5e8c7884807cf/summary.json),
[phone summary](../runs/feba4eaf4d5b4d338030cb2681d3fd1e/summary.json).
The summaries retain full typed verification requests/results despite timeout;
console tool-output displays may be abbreviated by the SDK. No failed sample was
replaced, no model was changed, and no further run or runtime fix was started.
