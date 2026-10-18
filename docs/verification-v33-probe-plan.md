# v3.3 direct verification usability probe

Written before live calls. This is two executions, not a reliability comparison.

## Changes under test

- Direct flat `verify_goal` tool, one requirement per call; no nested action JSON
  or duplicated outcome/observation predicate needed on this interface.
- Source checks require a successful historical read, nonempty covered content,
  and a task-quoted identity anchor in the same record. Case/punctuation-normalized
  token matching permits a name in a sender address; it is not semantic entity resolution.
- Effect checks require a current target GET/HEAD, fixed desired value/resource,
  grounding in a verified source's content or literal task text, and same-record
  identity. Source-derived effects inherit the source's task anchor. Task-literal
  effects must supply a task-quoted anchor. Value and identity must share a parent.
- Failed evidence updates retain pending requirements. Paths remain repairable;
  values, kind, resource and grounding cannot change to fit observations.
- Verification requests/results are saved in a dedicated audit, including on timeout.
  No grader input, benchmark-specific runtime branches or automatic write retries.

## Offline acceptance

Run the full tests including the actual Stirrup direct-tool session. Replay saved
v3.2 run `25fdd5e341bc4d79a0fdf54c812034ac` with explicitly supplied fixture bindings:
source matches, empty write acknowledgment fails by path, actual subsequent read
matches identity and value. The replay is a diagnostic, not autonomous model repair;
its bindings are not sent to the live actor or installed as a procedure.

## Live protocol

Use `scripts/checkpoint_reliability.py --probe` with baseline
`runs/checkpoint-reliability-3fb928a703c34f10bab86be7819e8a89/manifest.json` and this plan.
Select only the learned-arm first-replicate email and phone tasks, in that order,
from the previous schedule. Same immutable learned checkpoint and model settings:
free Nemotron, 8,192 output tokens, 20 turns, 24 environment calls, 240 seconds,
12,000 retrieval characters. Fresh task environment and disposable working graph
per trial. No semantic learning, procedures, probes or paid fallback. No runtime
changes after measurement starts.

Report grader, clean/checked/unresolved completion, direct-tool use and rejections,
inner/outer failures, source/effect evidence, cost and missing usage. A checked
finish must include a source and a value/identity-grounded effect on these tasks.
Inspect full audit, not just the completion label. One sample per task cannot
establish pass^k consistency, causality or learning benefit. Do not expand or
rerun failures automatically. Preserve both outcomes and decide next steps afterward.

## Limits

Requirement selection and anchor interpretation remain model-proposed. Lexical
identity matching cannot distinguish duplicate names or prove unique identity;
same-parent paths are a conservative structural check, not a semantic join model.
Only literal/source-contained values are supported, not arbitrary transformations
or aggregates. The legacy goal-update backend remains for compatibility but its
schema is no longer advertised in the acting action catalog. No claim of full
natural-language entailment or complete task coverage is made.
