# Project rules

## The harness learns; we do not hand-fix it

This project tests whether a graph learned from runs makes an agent better. Every
mechanism that changes what the agent sees or does must come from that graph.

- **No hand-made improvements.** Do not read failures (trajectories, grades) and then write a
  rule, threshold, note, filter or wording that targets them. Examples rejected so far:
  - a fixed pointer for repeated searches;
  - "repeating will fail" after 3 failures;
  - a "service not connected" note;
  - the stall gate and show-once;
  - dropping a hint after tallying its wins and losses;
  - ignoring values under 4 characters;
  - an edit-distance check on email addresses.
- **What is allowed.** Generic learning machinery (how traces become graph nodes, edges and
  counts; confidence bounds) and the learned content itself. When and whether to intervene
  must also be learned from outcomes in tool calls and results, not tuned by us.
- **Grades are for diagnosis only.** They never feed learning or the choice of mechanism.
- **The framework must transfer to other harnesses and benchmarks.** Keep code specific to a
  benchmark or service out of the learning core; it belongs in a thin adapter. Examples: Gmail
  `raw` decoding, AutomationBench's `api_search` and `api_fetch` tools, HTTP method and URL
  parsing, error-text regexes.
- **Before building anything, check:** what in this comes from the graph, and what did I
  decide by reading failures? If the second part changes the agent's behaviour, don't build it;
  make it something the graph learns.

## Evaluation

The success criterion is no regression against A/A noise, plus gains on familiar steps.
Task accuracy is a guardrail. See `docs/evaluation-steps.md`.

## Git

Commit or push only when asked. Never push.
