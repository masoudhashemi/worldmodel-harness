# Current v3.3.1: twelve-run results

Completed on 2026-10-16. **The learned graph reduced calls and returned tokens,
but did not improve task success or establish reliable verified completion.**
All twelve attempts used `operational-learning-v3.3.1`; these are not the older
v3.2 results and do not include the preceding two-attempt probe.

[Predeclared plan](current-code-12-run-plan.md) ·
[Manifest and metrics](../runs/checkpoint-reliability-3226dbcfffca430a9551c6dd120cac7f/manifest.json) ·
[Console log](../runs/current-12-source.CKmUaj/console.log) ·
[Source archive](../runs/current-12-source.CKmUaj/source.tar.gz).

## Controls and integrity

Two tasks, empty/learned graphs, three reset attempts per task/arm. Free Nemotron,
8,192 output tokens, 20 turns, 24 environment calls, 240 seconds per attempt.
Same seeded schedule and immutable source checkpoints as the historical reference;
its results are not pooled. Disposable working graph per attempt; within-task
learning remains enabled, but no learning passes between attempts. No semantic
learning, procedures, grader feedback, paid fallback, replacement trials or runtime
edits. This is an in-sample fixed-checkpoint comparison, not a learning curve.

Preflight: **275 tests passed; lint passed**. All 86 archived source/test/runner/
config/plan files matched after execution (macOS metadata entries ignored).
Archive SHA256: `a61c65781c4256292e7866cf349bb3d3549890663dd935862f370d1f6ed51094`.
No credentials are archived. Both input checkpoint hashes remained unchanged:

- Empty: `47f6e91a5103efd4098d3d669e89f40f9c00b96b341e18f779e135e5690fe0f0`.
- Learned: `39953dca5e94fc00cc1d39b6f2688a5b8f4f5e554eccc44351bbbaa5a1d64f53`.

## Outcomes and costs

| Metric | Empty graph | Learned graph |
| --- | ---: | ---: |
| Grader passes | 6/6 | 5/6 |
| Clean finishes | 0/6 | 1/6 (unresolved) |
| Checked finishes | 0/6 | 0/6 |
| Timeouts | 4 | 5 |
| Turn-exhausted attempts | 2 | 0 |
| Environment calls | 64 | 48 |
| Discovery calls | 34 | 19 |
| Model calls | 109 | 86 |
| Returned tokens | 1,389,237 | 967,338 |
| Calls without usage | 4 | 5 |
| Summed model seconds | 1,279.8 | 1,420.0 |
| Non-invoked decision calls | 7 | 6 |
| Outer schema rejections | 1 | 0 |

Learned-arm totals were 25% lower for environment calls, 44.1% lower for discovery,
and 30.4% lower for returned tokens. These are descriptive totals, not equal-success
efficiency estimates: the learned arm also failed one task and used more model time.
Returned tokens exclude unavailable usage from nine calls. Prior graph-construction
cost is excluded, not zero. Provider timing and the tiny task set limit inference.

## Consistency

For each task, compute the probability of all k successes (`pass^k`) or at least
one success (`pass@k`) when selecting k of its three observations without
replacement, then average over the two tasks. These are small-sample combinatorial
estimates at a fixed learning state, not evidence that performance improves over rounds.

| Arm / outcome | pass^1 | pass^2 | pass^3 | pass@1 | pass@2 | pass@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empty / grader | 100% | 100% | 100% | 100% | 100% | 100% |
| Learned / grader | 83.3% | 66.7% | 50% | 83.3% | 100% | 100% |
| Empty / clean | 0% | 0% | 0% | 0% | 0% | 0% |
| Learned / clean | 16.7% | 0% | 0% | 16.7% | 33.3% | 50% |
| Either arm / checked | 0% | 0% | 0% | 0% | 0% | 0% |

Grader pass@3 hides the learned arm's inconsistency: it reaches 100%, while
grader pass^3 is only 50%. Verified completion is zero under either measure.

## Individual attempts

All attempts below have `checked=false`. The clean finish is explicitly unresolved.

| Arm | Task | Replicate | Grader | Termination | Summary |
| --- | --- | ---: | --- | --- | --- |
| Empty | Email | 1 | pass | turn limit | [04e83b5](../runs/04e83b5d9302407a8bf1082da71a8faa/summary.json) |
| Learned | Email | 1 | pass | timeout | [c9f2e4d](../runs/c9f2e4dc840d41a09c1834775554030b/summary.json) |
| Empty | Phone | 1 | pass | timeout | [5fc795c](../runs/5fc795ce376046be84d428ea13396e26/summary.json) |
| Learned | Phone | 1 | fail | timeout | [5d17c86](../runs/5d17c8610fd94bcea9b9f32ec333c886/summary.json) |
| Learned | Email | 2 | pass | timeout | [2be3740](../runs/2be3740db5d04ba29fa810a52777b489/summary.json) |
| Empty | Email | 2 | pass | timeout | [d192db8](../runs/d192db80b49e457486221c8ebad8e95e/summary.json) |
| Learned | Phone | 2 | pass | timeout | [8825a1c](../runs/8825a1c1d706457ab3dcd301c1b830b5/summary.json) |
| Empty | Phone | 2 | pass | timeout | [e92ebb2](../runs/e92ebb26424c4be381e56159360d20b3/summary.json) |
| Empty | Phone | 3 | pass | turn limit | [efd39ed](../runs/efd39edaa69c4bd0a64dc92931808ae1/summary.json) |
| Learned | Phone | 3 | pass | clean, unresolved | [c38fcf5](../runs/c38fcf52a44a48a9824cea6e4695e9e3/summary.json) |
| Empty | Email | 3 | pass | timeout | [d63447b](../runs/d63447bf608f454a8181e3fb4a43e22e/summary.json) |
| Learned | Email | 3 | pass | timeout | [b6868fa](../runs/b6868fac61e44dd9b05b4eedf12b1bb5/summary.json) |

## What failed and what worked

**Verification remained unusable end to end.** The direct-tool audits contain:

| Verification result | Empty | Learned |
| --- | ---: | ---: |
| Missing target identity | 11 | 4 |
| Missing effect fields | 3 | 1 |
| Missing source fields | 1 | 4 |
| Ungrounded outcome value | 1 | 4 |
| Inexact task quote | 2 | 1 |
| Missing task anchor | 0 | 1 |
| Matched source check | 0 | 2 |

No effect check matched. Final ledgers contain seven missing-evidence requirements
in the empty arm; five missing-evidence requirements and two matched sources in the
learned arm. These missing-evidence labels often mean an attempted update was
rejected before evidence was attached, not that no environment read happened.
There are no final invalid-path or stale-evidence diagnoses here; this run therefore
does not establish the live effectiveness of that distinction.

**Anchor repair worked once, but did not complete the chain.** In learned email
replicate 3, an invalid source anchor was repaired to `Maria Santos` at `from` and
the source check matched. The subsequent effect request omitted
`source_requirement_id`, so its value remained ungrounded despite that matched
source. Learned email replicate 1 also matched a source without finishing the
effect. Neither constitutes verified outcome success. There are no checked finishes
to audit for false acceptance; the absence of false positives here does not prove
verification soundness or requirement coverage.

**The learned graph also introduced a concrete action blocker.** In the failed
first learned-phone attempt, four PATCH decisions were rejected with
`ambiguous_contract`: two stored contracts matched the same Contact update route.
Rediscovery did not resolve it before timeout; no PATCH was invoked. This is an
observed memory/gate interaction, not just a verification timeout or a missing
provider response. The other two learned-arm blocked decisions tried to dispatch
`verify_goal` through `execute_decision` instead of calling it directly. Empty-arm
blocked decisions had seven `requirements:unresolved` gaps. These counts are not
schema-error counts. One separate malformed nested-action JSON rejection occurred
in the empty-phone first replicate; no direct verification schema rejection occurred.

## Conclusion and next decision

The full current-code test fails the practical success criterion: no verified
completion in either arm, and no observed reliability gain from the learned graph.
Reduced discovery is useful, but insufficient. Do not claim improvement from the
275 passing offline tests or from the prior v3.2 results.

The next repair candidates supported by these traces are (1) resolving competing
applicable contracts without permanently blocking a legitimate action, and (2)
making source-to-effect verification bindings and missing-field repairs usable by
the actor without supplying benchmark answers or weakening outcome checks. Test
those mechanisms on captured traces before another live comparison. No such fix
or additional run was implemented as part of this experiment.
