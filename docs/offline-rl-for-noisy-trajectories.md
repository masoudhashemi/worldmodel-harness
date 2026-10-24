# Offline RL ideas for learning an enterprise world model from noisy trajectories

## Motivation

The strongest idea is to **learn environment behavior separately from whether the agent made a good decision**. A failed trajectory can still teach us how an API works. Conversely, a successful API call does not establish correct record selection or task completion.

We currently group several different problems under “noise”:

- Poor actions or incorrect record selection.
- Incomplete observations and unresolved references.
- Incorrect success labels, including unsupported final summaries.
- Changing policies or environmental conditions.

These problems need different treatments. The applications below are proposed adaptations to this harness, not claims that the cited papers already solve enterprise SaaS tasks. Some are related model-based or off-policy RL ideas rather than strictly offline algorithms.

## Related ideas and their application

| Idea / reference | What it learns or addresses | How it could help our harness | Important limitation |
|---|---|---|---|
| **Probabilistic environment models and ensembles** — [PETS](https://arxiv.org/abs/1805.12114) | Predicts action outcomes while representing uncertainty. | Learn `(observed state, operation, arguments) → response / observed change` from successful **and failed** trajectories. A wrong-recipient SMS still teaches sending mechanics, without teaching that the recipient was appropriate. | Agreement is not proof. Models trained on the same biased data can share mistakes; uncertainty needs calibration against held-out observations. |
| **Belief states for partial observability** — [PlaNet](https://arxiv.org/abs/1811.04551) | Maintains an inferred state from observation history rather than treating the latest observation as complete state. | Represent “sheet says P2; monitoring reference unresolved” or “policy worksheet exists but has not been read.” Distinguish **unknown eligibility** from eligible/ineligible. | Missing policy contents cannot be recovered reliably from nothing. The model must retain uncertainty until relevant evidence arrives. |
| **Pessimistic model-based planning** — [MOPO](https://arxiv.org/abs/2005.13239), [MOReL](https://arxiv.org/abs/2005.05951) | Discourages plans that exploit unreliable regions of a learned environment model. | Reduce confidence in a procedure when its prerequisite, recipient binding, or identifier mapping lacks evidence. Prefer a supported path or further investigation over an unsupported shortcut. | These methods address uncertainty in learned dynamics; they do not automatically recognize a semantic mistake. Their guarantees do not transfer directly to our symbolic graph. |
| **Hindsight learning from achieved subgoals** — [HER](https://arxiv.org/abs/1707.01495) | Reuses failed attempts by learning from goals they actually achieved. | A failed end-to-end job may still teach project lookup, task creation, or email encoding. Store those demonstrated capabilities separately from the failed job. | Relabel only demonstrably achieved subgoals. “Encoded email” must not become “sent email,” and an unauthorized action must not become an approved procedure. |
| **Conservative value estimation** — [CQL](https://arxiv.org/abs/2006.04779) | Reduces overestimation of actions insufficiently represented in offline data. | Rank action or procedure candidates conservatively when evidence for their success in the current state is sparse. This could restrain speculative graph paths. | Learns action value, not environment semantics. It requires meaningful rewards and state representations; bad success labels remain a problem. |
| **Learning better policies from mixed-quality behavior** — [IQL](https://arxiv.org/abs/2110.06169) | Learns from offline data without evaluating unseen actions during its policy-improvement training step. | Learn which observed next steps work better in comparable states—potentially favoring policy/reference investigation over immediate mutation when reliable outcomes support that choice. | Needs trustworthy reward signals and sufficient coverage. Repeated mistakes do not become useful positive examples merely because they are frequent. |
| **Reward learning from ranked imperfect demonstrations** — [T-REX](https://proceedings.mlr.press/v97/brown19a.html) | Infers reward from approximately ranked, potentially poor trajectories rather than assuming demonstrations are optimal. | Compare attempts at the **same task** and learn what distinguishes better outcomes. We need not wait for perfect trajectories before learning. | Rankings must contain information. Ranking by speed, API success, or an unreliable auditor could reward confidently wrong behavior. |
| **Preference learning over short trajectory segments** — [Learning from human preferences](https://arxiv.org/abs/1706.03741) | Learns a reward model from comparisons between behavior segments. | Ask for focused judgments: “Which attempt better establishes eligibility?” or “Which summary is supported?” This can supply targeted supervision where our auditor repeatedly fails. | Human or independent feedback is needed initially. Comparisons generated solely by the same mistaken model can reinforce its errors. |
| **Delayed reward redistribution / credit assignment** — [RUDDER](https://arxiv.org/abs/1806.07857) | Attributes delayed returns to earlier events that help predict them. | Help identify whether reading a policy or following a reference explains later success, rather than concentrating learning on the final `finish` rejection. | Predictive contribution is not causal proof. It also cannot repair an incorrect terminal reward. |
| **Off-policy evaluation before deployment** — [Doubly robust evaluation](https://proceedings.mlr.press/v48/jiang16.html) | Estimates a new policy’s performance using historical interactions. | Evaluate a proposed routing or procedure policy before enabling it online, with uncertainty rather than a single optimistic score. | Requires assumptions about coverage and behavior probabilities. Arbitrary LLM tool actions make these difficult; replay and controlled evaluations remain necessary. |

## Recommended order of work

For our current stage, prioritize **environment prediction, explicit unknown state, and targeted preference supervision**, rather than immediately training a full offline RL policy.

1. **Learn local environment models from observations.** Predict response shapes, parameter relationships, observed effects, and failure conditions. Keep actor summaries and auditor judgments as separate, potentially noisy annotations.
2. **Evaluate those models independently.** Hold out trajectories and test prediction accuracy, uncertainty calibration, and generalization across changed identities and policies. Claim count is not a useful substitute.
3. **Learn task-quality judgments from evidence-backed comparisons.** Use a small reviewed set to test and improve eligibility, completion, and summary assessment. Keep hidden benchmark assertions out of this training channel.
4. **Only then learn procedure selection.** Use conservative offline policy ideas once state and reward signals are sufficiently reliable.

## Why ordinary imitation or inverse RL is insufficient

**Observed behavior does not uniquely reveal the intended objective.** If the agent repeatedly notifies the wrong person, learning a reward that makes those demonstrations look optimal can reproduce the mistake. Ranked imperfect demonstrations are more appropriate than blindly treating our trajectories as expert behavior. [T-REX](https://proceedings.mlr.press/v97/brown19a.html)

This distinction also preserves the design goal: learning reusable environmental knowledge from interactions, rather than hardcoding benchmark-specific identities, exclusions, or business rules.

## Initial offline implementation

`src/worldmodel_harness/experience.py` implements the evidence and structural
retrieval foundation. `scripts/replay_experience.py` evaluates it using separate
training and test evidence databases. Neither is connected to actor prompts,
execution gates, or the online knowledge store.

- Failed and successful environment calls both supply transitions. Harness
  rejections do not become observations about environment dynamics.
- Literal scalar instances become variables; repeated values share variables,
  while distinct badges and array positions remain distinct. Schema field names
  remain visible. This is not a privacy filter for arbitrary dynamic object keys.
- State is a bounded history of observations, not a claim that old observations
  are still current. Omitted history and truncated structures are explicit.
- REST capability links use endpoint templates from preceding discovery
  responses. Ambiguous or undiscovered routes abstain. Native tool operations
  remain separate from REST endpoints.
- Retrieval separates matching observed structure from mere capability matches.
  Neither establishes semantic equivalence. Opposite status values can have the
  same structure; this version does not infer eligibility or encoding rules.
- Duplicate events do not add support. Conflicting outcomes remain visible even
  when the returned example count is limited. Counts are observations, not
  calibrated probabilities or independent causal experiments.
- Replay builds each query before observing its response, excludes overlapping
  train/test task instances and events, and never trains on test transitions.
  It reads only episode records, with input hashes checked afterward.

Example:

```bash
.venv/bin/python scripts/replay_experience.py \
  --train runs/<training-run>/evidence.sqlite \
  --test runs/<held-out-run>/evidence.sqlite \
  --output runs/<new-output>/replay.json
```

The first three-task replay used baseline and iteration 1 episodes as training
and iteration 3 as held-out episodes from `iterative-eval-post-shadow`. It stored
90 transitions; 51 of 55 test calls could be represented and 50 retrieved
historical examples. Only 7 calls had matching observed structure with support
from at least two training task instances (4 searches and 3 fetches). Their
API-outcome Brier score was zero, which does **not** demonstrate semantic state
prediction, task improvement, or general calibration. The artifact is
`runs/offline-experience-three-task/replay.json`.

Next: extract evidence-linked semantic predicates, retain unknown and conflicting
conditions, and measure whether those predicates distinguish outcome-relevant
states on held-out trajectories. Preconditions inferred from recovery remain
hypotheses until evidence distinguishes competing explanations. Policy content
reuse additionally needs scope and freshness checks. CQL/IQL-style policy
optimization remains deferred because this implementation has no trustworthy
task-reward signal.

## Evidence from the current harness

In the staged completion-auditor retry recorded in `runs/shadow-staged-completion-retry`, **0/3 audits completed validation**, and semantic misses persisted. Those outputs should remain noisy candidate labels, not the reward authority for the next learning loop.

The local [retry findings](../runs/shadow-staged-completion-retry/review.json) record that:

- Incident requirement analysis approved the duplicate incident without unresolved conditions; action assessment failed validation.
- Asana requirement analysis confused record identifiers with evidence references, omitted requested effects, and used an excessively large catalog context. Tag-mapping accuracy was not evaluated.
- Badge analysis retained separate records but omitted notification requirements and unresolved policy conditions; action assessment failed validation.

These are findings from that specific retry, not results for the RL methods above. Run artifacts may be unavailable in a fresh checkout.
