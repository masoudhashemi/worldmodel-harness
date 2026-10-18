# v3.3 direct verification probe

Status: completed. **Direct calls worked, but verification usability still failed.**
[Plan](verification-v33-probe-plan.md).
[Manifest](../runs/checkpoint-reliability-645f0c7fe1b244b5a23cf7fcb6c68c0a/manifest.json).

Offline: 273 tests pass, including a real Stirrup session using the direct typed
tool for source/effect verification and checked finish. Lint passes. The saved
v3.2 phone failure was replayed with explicit diagnostic bindings: source matched,
write acknowledgment failed by path, existing readback matched value and identity.
No model/environment calls were made by the replay, and its bindings were not
provided to the live agent. This is not autonomous model repair evidence.

Live: two learned-checkpoint tasks, one attempt each, same free model and budgets
as v3.2, disposable working graphs and unchanged source checkpoint. No semantic
learning, procedure generation, grader feedback or paid fallback. No runtime
changes after calls began. This is a usability probe, not a reliability estimate.

## Measured v3.3 results

| Outcome | Email | Phone |
| --- | ---: | ---: |
| Grader pass | yes | yes |
| Clean / checked finish | no / no | no / no |
| Acting error | timeout | timeout |
| Direct verification calls | 6 | 3 |
| Successful direct verification calls | 0 | 0 |
| Inner / outer schema rejections | 0 / 0 | 0 / 0 |
| Environment / discovery calls | 7 / 2 | 9 / 4 |
| Model calls | 17 | 14 |
| Returned tokens | 185,853 | 152,283 |
| Calls without usage | 1 | 1 |

The source checkpoints remained unchanged; both summaries report
`operational-learning-v3.3`. The [console log](../runs/checkpoint-reliability-645f0c7fe1b244b5a23cf7fcb6c68c0a/console.log)
and each summary's `goal_verification_audit` preserve the outcome. Both task histories
timed out, but the new audit retained every dispatched verification request and result.
Observed token totals are lower bounds. There is no paid fallback or extra live rerun.

The actor used the direct tool successfully at the schema level. In contrast to
the v3.2 nested-action failures, none of these nine verification calls failed JSON
validation. With only two attempts this is evidence of interface use, not a proven
reduction in error probability.

Source checks failed because the actor used a newly discovered value/address as
the anchor rather than an identity named in the task. Email repeated the same
anchor under several IDs; phone also produced duplicate source requirements.
Effect attempts omitted resource or identity fields. No source or effect matched,
so the implementation did not demonstrate verified completion despite grader success.

## Post-probe correction: v3.3.1, offline only

Inspection found that an unverified anchor could become locked before the model
could repair it. The correction allows repair until the source is matched; a
matched source's identity remains fixed. Anchor membership now uses the complete
public task rather than only a shorter requirement quote. `missing_task_anchor`
returns the field, full public task, and explicit instructions to reuse the
requirement ID and select a task-named identity rather than the discovered answer.
The tool description also warns against duplicate requirement IDs during repair.

Regression tests cover correcting an invalid anchor on the same requirement,
using an identity elsewhere in the public task, preserving the anchor after a
match, and rejecting a task-literal effect on the wrong identity. These corrections
were made **after both measured runs ended**. Current runtime v3.3.1 has not been
tested with a live model; do not attribute the v3.3 measurements to it.

The next action is a small fresh usability probe of the correction, not a larger
benchmark or a claim of learning improvement. Complete task interpretation and
unique entity identity remain open limitations. No further live calls were started.

Final offline verification: **275 tests pass**, lint passes, and the saved-failure
replay still passes with zero model/environment calls.
