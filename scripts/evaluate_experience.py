"""Paired offline predictive evaluation; no actor, grader, reward, or LLM access."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from replay_experience import load, replay, task_keys

from worldmodel_harness import experience
from worldmodel_harness.experience import ExperienceMemory, PrefixExperience

LABELS = ("success", "error", "blocked", "unknown")
ARMS = ("no_experience", "global_history", "capability_history", "state_history")


def fields(snapshot):
    """Response schema with wildcard array positions, excluding instance literals."""
    result = set()

    def walk(value, path):
        if "object" in value:
            result.add((path, "object"))
            for key, child in value["object"].items():
                walk(child, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif "array" in value:
            result.add((path, "array"))
            for child in value["array"]:
                walk(child, path + "/*")
        elif "kind" in value:
            result.add((path, value["kind"]))

    walk(snapshot.structure, "")
    return tuple(sorted(result))


def forecast(candidates):
    counts = Counter(t.outcome for t in candidates)
    probabilities = {
        label: counts[label] / len(candidates) if candidates else 0.25 for label in LABELS
    }
    schemas = Counter(
        fields(t.observed_response) for t in candidates if not t.observed_response.truncated
    )
    schema = schemas.most_common(1)[0][0] if schemas else None
    # Bounded examples; aggregate counts use the entire eligible evidence pool.
    examples = [
        {
            "episode_id": t.episode_id,
            "outcome": t.outcome,
            "response": t.observed_response.model_dump(),
        }
        for t in candidates[:3]
    ]
    return {
        "probabilities": probabilities,
        "response_fields": schema,
        "evidence_count": len(candidates),
        "independent_tasks": len({t.query.task_id for t in candidates}),
        "example_context_chars": len(json.dumps(examples, separators=(",", ":")))
        if examples
        else 0,
        "evidence_episodes": [t.episode_id for t in candidates],
    }


def predict(query, memory):
    scoped = [
        t
        for t in memory.transitions
        if t.query.task_id != query.task_id
        and t.query.scope.model_dump(exclude={"task_id"})
        == query.scope.model_dump(exclude={"task_id"})
    ]
    capability = [t for t in scoped if t.query.capability.key == query.capability.key]
    exact = [t for t in capability if t.query.state_signature() == query.state_signature()]
    return {
        "no_experience": forecast([]),
        "global_history": forecast(scoped),
        "capability_history": forecast(capability),
        "state_history": forecast(exact or capability),
    }, bool(exact)


def score(prediction, outcome, response):
    probabilities = prediction["probabilities"]
    result = {
        "outcome_brier": sum(
            (probabilities[label] - int(label == outcome)) ** 2 for label in LABELS
        ),
        # Fixed label order breaks ties; uniform baseline is explicitly labeled.
        "outcome_correct": int(max(LABELS, key=lambda label: probabilities[label]) == outcome),
    }
    if response.truncated:
        return result
    actual = set(fields(response))
    predicted = set(prediction["response_fields"] or ())
    result["response_fields_f1"] = (
        2 * len(actual & predicted) / (len(actual) + len(predicted)) if actual or predicted else 1.0
    )
    result["response_structure_correct"] = int(
        prediction["response_fields"] is not None and actual == predicted
    )
    return result


def summarize(rows):
    summaries = {}
    for arm in ARMS:
        scores = [r["arms"][arm]["scores"] for r in rows if "arms" in r]
        summary = {"paired_calls": len(scores)}
        for key in (
            "outcome_brier",
            "outcome_correct",
            "response_fields_f1",
            "response_structure_correct",
        ):
            values = [s[key] for s in scores if key in s]
            summary[key] = sum(values) / len(values) if values else None
            summary[key + "_calls"] = len(values)
        forecasts = [r["arms"][arm] for r in rows if "arms" in r]
        summary["mean_example_context_chars"] = (
            sum(p["example_context_chars"] for p in forecasts) / len(forecasts) if forecasts else 0
        )
        summary["calls_with_evidence"] = sum(bool(p["evidence_count"]) for p in forecasts)
        summaries[arm] = summary
    return summaries


def evaluate(episodes, memory):
    prefixes, seen, rows = {}, set(), []
    for episode in episodes:
        if episode.origin != "environment" or episode.environment_called is False:
            continue
        key = (episode.scope.model_dump_json(), episode.task_id)
        if (key, episode.event_id) in seen:
            continue
        seen.add((key, episode.event_id))
        prefix = prefixes.setdefault(key, PrefixExperience(episode.scope, episode.task_id))
        query = prefix.query(episode.operation, episode.arguments)
        predictions, exact = predict(query, memory) if query else ({}, False)
        transition = prefix.observe(episode, query)
        row = {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "operation": episode.operation,
            "query_available": query is not None,
            "exact_structure_match": exact,
            "observed_outcome": episode.outcome,
        }
        if transition:
            row["arms"] = {
                name: p | {"scores": score(p, episode.outcome, transition.observed_response)}
                for name, p in predictions.items()
            }
            row["context_truncated"] = bool(
                query.history_omitted
                or query.action.truncated
                or any(o.response.truncated for o in query.before)
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = args.train + args.test
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(p.resolve()): digest(p) for p in paths}
    train, test = ([e for p in paths for e in load(p)] for paths in (args.train, args.test))
    if task_keys(train) & task_keys(test) or {e.event_id for e in train} & {
        e.event_id for e in test
    }:
        parser.error("Training and test tasks/events must be disjoint")
    memory = ExperienceMemory()
    replay(train, memory, training=True)
    rows = evaluate(test, memory)
    exact = [r for r in rows if r["exact_structure_match"]]
    different = [
        r
        for r in rows
        if "arms" in r
        and any(
            r["arms"]["capability_history"][key] != r["arms"]["state_history"][key]
            for key in ("probabilities", "response_fields")
        )
    ]
    report = {
        "experiment": "E001",
        "mode": "deterministic_offline_predictive_comparison",
        "input_sha256": hashes,
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path(experience.__file__),
                Path(__file__).with_name("replay_experience.py"),
            )
        },
        "training_transitions": len(memory.transitions),
        "test_calls": len(rows),
        "unresolved_calls": sum(not r["query_available"] for r in rows),
        "observed_outcomes": dict(Counter(r["observed_outcome"] for r in rows)),
        "exact_structure_calls": len(exact),
        "state_changed_prediction_calls": len(different),
        "overall": summarize(rows),
        "exact_structure_subset": summarize(exact),
        "per_task": {
            task: summarize([r for r in rows if r["task_id"] == task])
            for task in sorted({r["task_id"] for r in rows})
        },
        "limitations": [
            "Repeated task instances; not unseen task families",
            "Response schemas and API outcomes do not establish semantic effects",
            "Uniform no-experience baseline is not an LLM baseline",
            "No calibration or statistical significance claim from this small sample",
            "State-history falls back to capability-history when unmatched",
        ],
        "results": rows,
    }
    if any(digest(p) != hashes[str(p.resolve())] for p in paths):
        raise RuntimeError("Evidence changed during evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("results", "per_task", "input_sha256", "source_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
