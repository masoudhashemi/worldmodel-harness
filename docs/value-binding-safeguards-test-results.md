# Value/binding safeguards: implementation and test results

Implemented as `operational-learning-v3.4`. **292 offline tests and lint pass;
the live smoke comparison still has zero verified finishes.**

[Plan](value-binding-safeguards-test-plan.md) ·
[Manifest](../runs/checkpoint-reliability-5ab23b4631b24967bd76ab0993aa2118/manifest.json) ·
[Console](../runs/value-safeguards-source.Noh7dZ/console.log) ·
[Code archive](../runs/value-safeguards-source.Noh7dZ/source.tar.gz).

## Implemented boundaries

- Slot kinds cannot establish literal task outcomes. Effect grounding rejects
  `$kind` markers, including nested markers. Exact desired values, target identity,
  current reads and resource checks remain separate obligations.
- Automatic source selection requires one matched source with the exact recognized
  scalar value and a matching task identity on the target's same record. Same-kind
  alternatives, wrong entities, ambiguous sources and prose-only containment do
  not attach automatically. Audit records retain original and resolved requests
  plus source/target episode IDs and paths for inferred bindings.
- Response-rule literals stay exact unless an explicit leaf role requests example
  or parameter abstraction. Constraints/unknown roles remain literal. Structured
  rule identities include roles and retained constants. Kind abstraction is a type
  regularity, not a relational input/output equality. Normalization deep-copies
  before rewriting; it does not change the original proposal or raw observations.
- Existing prospective promotion still needs later evidence across two tasks;
  generating/relogged events cannot count as independent support. Counterevidence
  disputes the prediction, not merely its narrative explanation.
- Same-route aliases merge only for identical documented payloads (except endpoint
  ID) in exactly the same scope. New documentation evidence is retained. Conflicting
  alias schemas remain separate and block selection, including on old checkpoints.
  Diagnostics include schemas/scopes, rather than suggesting an arbitrary choice.

Full schema reconciliation, semantic identity resolution, prose ownership inference,
and reconstruction of already-generalized legacy literals are not implemented.
Same-ID authoritative documentation retains existing versioned-update semantics.

## Offline verification

292 tests pass (279 before this change), Ruff passes. Regression coverage includes
wrong same-kind values, wrong people, duplicate candidate sources, prose-only
inference, kind-only outcome rejection, distinct literal dates/URLs/emails/phones,
explicit abstraction roles, malformed role paths, input-rule immutability,
duplicate event support, incompatible schemas, and tenant isolation. Existing
prospective-promotion and contradiction tests continue to pass.

The saved-failure replay succeeds without model/environment calls: source matches,
write acknowledgment fails with `invalid_evidence_path`, correct current readback
matches. This is a fixture check, not evidence of autonomous live completion.

## Live smoke comparison

Free Nemotron; email and phone tasks; one reset replicate per empty/learned arm.
Same immutable source checkpoints as the previous twelve-run experiment; 8,192
output tokens, 20 turns, 24 environment calls, 240 seconds. No cross-trial updates,
semantic learning, procedures, grader feedback or paid fallback. These are four
new observations, not pooled with historical trials. No pass^2/pass^3 estimate is
possible from one attempt per task/arm.

| Metric | Empty | Learned |
| --- | ---: | ---: |
| Grader passes | 2/2 | 2/2 |
| Clean / checked finishes | 0/2 / 0/2 | 0/2 / 0/2 |
| Timeouts | 2 | 2 |
| Environment / discovery calls | 17 / 7 | 16 / 6 |
| Model calls | 29 | 26 |
| Returned tokens | 294,283 | 282,869 |
| Calls without usage | 2 | 2 |
| Non-invoked decision calls | 1 | 0 |
| Outer schema rejections | 0 | 0 |
| Automatic source bindings | 0 | 0 |

Returned-token totals exclude unavailable usage. Prior checkpoint construction cost
is excluded, not zero. Small differences in calls/tokens do not establish improvement.

| Task/arm | Saved summary | Verification failures |
| --- | --- | --- |
| Email / empty | [267e635](../runs/267e635eb33448c4be7dcc6f0f643dae/summary.json) | Missing target identity; ungrounded outcome value |
| Email / learned | [5a0691d](../runs/5a0691d7e3d6481ab7463eec0b52113a/summary.json) | Missing source fields; invalid task anchor |
| Phone / empty | [864db47](../runs/864db478ce91429798bccd9d86586586/summary.json) | Missing target identity; ungrounded outcome value; one unresolved gate requirement |
| Phone / learned | [484cbeb](../runs/484cbeb722dd43e4803d9c97161ba227/summary.json) | Missing effect fields/target identity; unverified source; missing source fields |

No source or effect check matched. The final ledgers contain missing evidence or
an invalid task anchor. No automatic source binding fired: its positive case and
rejection behavior were tested offline, not demonstrated by these live attempts.
Likewise semantic rule abstraction was not enabled live. There were no learned-arm
gate rejections in this smoke run, but one attempt per task cannot establish that
contract-selection reliability has improved. There are no checked finishes to
audit for false acceptance.

## Interrupted preliminary attempt and integrity

An initial attempt in
`runs/checkpoint-reliability-6b7ad98664854c3490f252058762df37` was interrupted when
code inspection found aliasing of nested rule dictionaries. It had no completed
trial result. Its [console](../runs/value-safeguards-source.jW0817/console.log) and
[earlier source archive](../runs/value-safeguards-source.jW0817/source.tar.gz) remain
intact; its partial usage is not included in the four completed trial totals.
This was a code-correction restart, not a replacement of a failed scored sample.

After fixing aliasing and rerunning tests, all four trials used one unchanged
archive: SHA256 `2df564d2b7541827f0bb7ca207df015b2fa66e3cc78b1898810ce4bc93d62522`.
All 86 archived files match post-run files (macOS metadata excluded); source
checkpoint hashes also match the manifest. No credentials are archived.

## Conclusion

The requested safety boundaries are implemented and covered by regression tests.
The agent still cannot reliably assemble verification evidence and finish within
budget. Do not describe this as a performance improvement. The next usability work
must make missing fields and explicit source/effect links easier to repair while
preserving these boundaries, rather than accepting kind matches as task success.
