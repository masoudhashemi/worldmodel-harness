# Graph-Learning Harness

Does an agent get better at tasks it has seen before when a graph learned from its earlier
runs advises it? This harness tests that on [AutomationBench](https://github.com/zapier/AutomationBench)
tasks in API mode, on the [Stirrup](https://pypi.org/project/stirrup/) agent loop, with
Gemma-4-31B-it served by vLLM at temperature 0.

The graph is learned from tool calls and results only. Grades are used for evaluation, never
for learning. When and whether the agent is shown what the graph knows is also learned, from
the same tool calls (`core/policy.py`). The project rules are in `CLAUDE.md`.

## How it works

There are two arms, run back to back on each task:

- **Baseline:** the plain agent loop with the tools `api_search`, `api_fetch`,
  `base64_encode` and `finish`.
- **Graph:** the same loop. After every tool call, the advisor may add a short
  `[Experience]` note to the tool result. The note can hold the likeliest next calls, where
  earlier runs found the values they wrote, calls most runs made, and how earlier runs
  recovered from the same error. A learned policy picks which kinds of note to show.

The graph starts from every earlier run under `runs/` and learns from each graph-arm trial as
the run goes on. `docs/graph-learning-framework.md` describes what it learns and how the core
stays independent of the benchmark.

## Running

```bash
uv sync --extra automationbench
# model profile: configs/model.toml (override with WORLDMODEL_MODEL_CONFIG); key in .env

# baseline vs graph, 2 rounds; resumes from <out>/summary.json if interrupted
uv run python scripts/run_arm_eval.py runs/my-run runs/my-run/tasks.json 2 baseline,graph

# step-level evaluation: guardrails, regression against A/A, steps by familiarity
uv run python scripts/step_metrics.py runs/my-run baseline graph not-shown
```

`tasks.json` is `{"tasks": ["domain.task~v1", ...]}`. Surface variants (`~vN`) rename every
literal of a task (`src/worldmodel_harness/variants.py`), so experience has to transfer
across names and ids.

## Layout

```
src/worldmodel_harness/
  core/              benchmark-agnostic: traces, lifting, the graph, the advisor, the policy
  automationbench/   adapter: HTTP domain, upstream tools, trajectory loading
  experiment.py      one trial
  runtime.py, repairing_agent.py, config.py   model client, Gemma tool-call parsing, settings
  variants.py        surface variants of tasks
scripts/             run_arm_eval.py, step_metrics.py, offline analyses; archive/ runs on world-model-v2
docs/                graph-learning-framework.md, evaluation-steps.md, experiment-journal.md
```

## History

E001–E021 tried an operational layer, an advisory note built from a claim store, checklists
and fix notes. E022–E034 learned an experience graph, but tied its use to rules written by
hand after reading failures. Both designs are on the `world-model-v2` branch. This branch
keeps the graph and learns its use as well. `docs/experiment-journal.md` records every
experiment.
