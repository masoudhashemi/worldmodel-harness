# Graph-learning framework

Branch `graph-learning`, 2026-11-03. This replaces the mechanisms of `world-model-v2`
(E013–E034) with one framework: an experience graph learned from runs, which advises the
agent through its tool results. The rule behind it is in `CLAUDE.md`. Everything that changes
what the agent sees comes from the graph, and the core knows nothing about any one benchmark.

## Layout

| Part | Where | Knows about the benchmark? |
|---|---|---|
| Traces: `Step`, `Episode`, the `Domain` interface, generic argument leaves (JSON and base64 opened) | `core/trace.py` | no |
| Lifting: query templates, value origins, argument sources (`Query`, `Resp(field\|anchor) <- action`), filling them from a run's state | `core/lifting.py` | no |
| The experience graph | `core/graph.py` | no |
| The advisor: candidate messages from the graph, and a policy that picks which to show | `core/advisor.py` | no |
| `HttpDomain`: actions are method plus URL template; slots are path ids, query parameters, and params and body leaves | `automationbench/domain.py` | yes |
| Tools on the upstream world, with one hook: after every call, the advisor may add text | `automationbench/env.py` | yes |
| Reading Stirrup trajectories into episodes | `automationbench/traces.py` | yes (harness format) |

To move to another harness, write a `Domain` (or use the generic `ToolDomain`: the action is the
tool plus its argument names, and the slots are the argument leaves), a trace loader, and a
hook that passes each tool call to `Advisor.after`.

## What the graph learns

It learns from tool calls and results only, never from grades. Counts are per episode, and
every rule is gated by a Wilson lower bound (`THETA`).

| Content | From | Used for |
|---|---|---|
| Next successful action from a state (template plus the last two actions) | every run | `plan` |
| Where each argument value came from | every successful call | filling calls from this run's state; chaining |
| Which read held each written value | every write | `sources`; branch content |
| Actions most runs made | every run | `unmade` |
| At the first visit of a state, the next action and whether later written values were found or made up | every run | `branch` |
| After a failed call, the next successful call of the same tool, with its argument shape | every failure | `recovery` |

## How the agent is advised

After each tool call, `Advisor.after` builds candidates of the kinds above for the run's
state. A `Policy` decides which candidates to show, and every candidate is logged with
whether it was shown and the step it followed.

`ShowNew` shows a candidate whenever its text has changed. It is the untuned default.
`LearnedPolicy` (`core/policy.py`) learns which kinds to show (E035):
- **Unit.** Within a run, each kind is shown or held back for the whole run, decided at its
  first offer.
- **Outcome.** The calls the run still made after that first offer, minus the calls the same
  task's control trial (no advisor) still made after the same situation: the same last
  successful action, the same call and result, and the same repeat flag, at the same
  occurrence. This is from tool calls alone; in E035 it tracked graded outcomes. Raw calls left
  across tasks mostly measured task length and made the E036 decisions flip. Offers with no
  matched point are not learned from.
- **Comparison.** Shown runs against held-back runs, per (query template, kind), falling back
  to the kind.
- **Decision.** Thompson sampling: show with the probability that showing leaves fewer calls,
  from a Student t posterior of the difference of the means. It is random (1/2) while either
  side has fewer than 2 runs. Draws are seeded by task and kind.

## What was removed, and why

All of these were decided by reading failures, or were specific to AutomationBench. They are
still on `world-model-v2`.

| Removed | Kind |
|---|---|
| Fixed pointer for word-for-word repeated searches (E024) | hand-made rule |
| "Repeating will fail" after 3 identical failures | hand-made rule |
| "No account is connected" note (E021) | hand-made rule, benchmark error text |
| Fix-note wording tuned on Gemma probes, and the "base64_encode produces it" hint | hand-made wording, benchmark tool |
| Stall gate (2 repeated searches) and show-once (E029) | hand-tuned policy |
| Dropping "calls not made yet" and ignoring values under 4 characters | hand decisions from graded tallies |
| Near-copy email write check (E034) | hand-made rule |
| Gmail `raw` header decoding | benchmark-specific; replaced by generic opening of base64 and JSON strings |
| API-version regex for URLs | hand-made; versions are now learned as constants across workflows |
| Advisory note, generic checklist, entity-ledger review, v1 claim store (E007–E017) | earlier designs, not graph-based |

**Kept as plumbing, the same in every arm:**
- Gemma text tool-call parsing, and schema-error feedback (`repairing_agent.py`);
- the output cap with one retry after a cut-off reply (`runtime.py`);
- surface variants (`variants.py`);
- step metrics (`scripts/step_metrics.py`).

These change nothing between the baseline and the graph arm.
