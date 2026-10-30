# Advisory Operational Memory

An agent working in an API environment rediscovers the same endpoints on every task.
This harness lets it remember **routes that actually worked** and tells it about them in a
short note, without changing how it acts. The question it tests is whether that memory
makes the agent cheaper (time, tokens, environment calls) at equal or better accuracy.

It runs [AutomationBench](https://github.com/zapier/AutomationBench) tasks in API mode on the
[Stirrup](https://pypi.org/project/stirrup/) agent loop. Grading uses the upstream rubric after
the task ends; scores are never written to memory.

## How it works

There are two arms.

- **Baseline:** the plain agent loop. The model sees the task and the tools `api_search`,
  `api_fetch`, `base64_encode` and `finish`, can call several tools per turn, and stops when
  it calls `finish`. Nothing is stored.
- **Advisory:** the same loop and tools, plus one short note added to the system message.

The note is built once per task from a store carried across tasks
(`src/worldmodel_harness/advisory.py`):

1. Find the services the task names, from words ("email", "spreadsheet") and literals
   (`ss_…` IDs, email addresses, `#channels`).
2. List up to 3 endpoints per service that **succeeded in at least 2 earlier tasks**, with
   the documented request format of write endpoints.
3. Add value flows observed between them, e.g. a Drive file ID reused as a spreadsheet ID.
4. Name what is missing: services with no confirmed route, and resources the task names
   that the listed endpoints do not cover ("gmail draft").
5. Say the hints are incomplete and that anything not listed should be discovered with
   `api_search`.

The note stays the same for the whole task (about 0.8–2k characters). If the agent makes the
same tool call 3 times within its last 6 calls, the note is withdrawn for the rest of the task.

Every call made in the advisory arm is also learned from (`automationbench.py`):

| Claim | Learned by | From |
| --- | --- | --- |
| `endpoint_contract` | `learning.Learner.document` | `api_search` results |
| `observed_output_type` | `behavior.BehaviorLearner` | successful `api_fetch` responses |
| `observed_value_flow` | `composition.CompositionLearner` | a value from one response reused in a later request |

Claims keep endpoint IDs and paths, never record values. A claim counts as supported only
with evidence from independent tasks.

## Running

```bash
uv sync --extra automationbench
# model profile: configs/model.toml (override with WORLDMODEL_MODEL_CONFIG); key in .env

# 3 baseline repeats, then 3 advisory passes with a carried store
uv run python scripts/run_eval.py --task-file configs/eval_53_tasks.json \
  --baseline-repeats 3 --iterations 3 \
  --seed-store runs/focused-env-audit/store.sqlite --output-dir runs/my-eval

# reuse an earlier run's baseline trials instead of rerunning them
uv run python scripts/run_eval.py --task-file configs/eval_53_tasks.json \
  --baseline-from runs/e014-controlled/summary.json --iterations 3 \
  --seed-store runs/focused-env-audit/store.sqlite --output-dir runs/my-eval
```

`summary.json` holds every trial plus aggregates per arm and iteration: strict pass rate,
pass^k, partial credit, environment calls, model turns, input tokens, wall time, and how
often the note was withdrawn. Add `--resume` to continue an interrupted run.

## Layout

```
src/worldmodel_harness/
  advisory.py        note building and loop detection
  experiment.py      one trial: baseline or advisory
  automationbench.py upstream tools, grading, and learning from each call
  learning.py, behavior.py, composition.py   the three claim learners
  knowledge.py, store.py                     claim/episode records and SQLite store
  runtime.py, config.py, repairing_agent.py  model client, settings, Stirrup agent
scripts/run_eval.py  the evaluation driver
configs/             model profiles and the 53-task evaluation set
docs/                research framing and the experiment journal
```

## History

Earlier versions wrapped the agent in an operational layer: a decision tool, execution
guards, Thompson-sampled workflow plans, LLM-written hypotheses, and a large memory block
injected every turn. In E014 that design was slower (about 4× wall time) and less accurate
than the baseline. Its trajectories showed one action per turn, narrowed exploration, and
output slips caused by the large context. This branch keeps only the parts the advisory arm
uses. The full earlier code is in the history (commit `371b184` and before);
`docs/experiment-journal.md` records the experiments.
