"""Bounded model consolidation during a task, driven by observed repetition.

The trigger measures repetition; it does not diagnose causes or prescribe tools.
All explanations are model proposals, remain hypotheses, and retain provenance.
"""

import asyncio
import hashlib
import json
from collections import Counter
from time import perf_counter

from .semantic_learning import learn_semantics
from .semantic_review import review_semantics


def interaction_signature(episode):
    response = episode.response
    if isinstance(response, dict):
        response = response.get("result", response.get("content", response))
    arguments = {k: v for k, v in episode.arguments.items() if k not in ("goal", "rationale")}
    return hashlib.sha256(json.dumps(
        [episode.operation, arguments, response, episode.outcome], sort_keys=True
    ).encode()).hexdigest()


class OnlineLearner:
    def __init__(self, adapter, client, *, max_reflections=2, timeout_seconds=180,
                 evidence_chars=36000):
        self.adapter = adapter
        self.client = client
        self.max_reflections = max_reflections
        self.timeout_seconds = timeout_seconds
        self.evidence_chars = evidence_chars
        self.audit = []
        self.last_boundary = 0
        self.latest_hypotheses = []
        self.pending_experiment = None

    async def before_decision(self):
        if self.client is None or self.adapter.knowledge.frozen:
            return
        episodes = self.adapter.knowledge.episodes(self.adapter.scope)
        calls = [e for e in episodes if e.origin == "harness" and e.operation.startswith("tool_call:")]
        if self.pending_experiment is not None and len(calls) > self.last_boundary:
            from .recovery_experiment import assess_experiment

            self.audit[-1]["experiment_outcome"] = assess_experiment(
                self.adapter, self.pending_experiment, calls[self.last_boundary])
            self.pending_experiment = None
        if len(self.audit) >= self.max_reflections or len(calls) - self.last_boundary < 3:
            return
        recent = calls[max(self.last_boundary, len(calls) - 8):]
        counts = Counter(interaction_signature(e) for e in recent)
        repeated = counts[interaction_signature(recent[-1])] >= 3
        failures = all(e.outcome in ("error", "blocked", "unknown") for e in recent[-3:])
        if not (repeated or failures):
            return
        self.last_boundary = len(calls)
        related = {eid for e in recent for eid in e.related_episode_ids}
        recovery_evidence = [e for e in episodes if (
            e in recent or e.origin == "environment" and e.episode_id in related
        )]
        names = {e.operation.removeprefix("tool_call:") for e in recent}
        preferred = [e.episode_id for e in episodes if (
            e in recent or e.episode_id in related
            or e.operation == "tool_schema" and e.arguments.get("name") in names
        )]
        record = {
            "trigger": "repeated_observation" if repeated else "repeated_failure",
            "after_interaction_id": recent[-1].interaction_id,
            "evidence_ids": preferred,
            "learning": {}, "review": {},
        }
        self.audit.append(record)
        started = perf_counter()
        call_start = len(self.client.calls)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                ids = await learn_semantics(
                    self.client, self.adapter.knowledge, self.adapter.scope,
                    accounting=record["learning"], max_chars=self.evidence_chars,
                    preferred_episode_ids=preferred,
                    task_context=" ".join(m["content"] for m in self.adapter.task.prompt
                                          if m.get("role") != "system"),
                )
                record["hypothesis_ids"] = ids
                self.latest_hypotheses = ids
                record["reviewed_ids"] = await review_semantics(
                    self.client, self.adapter.knowledge, self.adapter.scope,
                    accounting=record["review"], max_chars=self.evidence_chars,
                )
                if getattr(self.adapter, "public_tools", None):
                    from .recovery_experiment import RecoveryPause, propose_experiment

                    record["experiment_accounting"] = {}
                    decision = await propose_experiment(
                        self.adapter, self.client, recovery_evidence,
                        [{"statement": self.adapter.knowledge.claim(i).statement,
                          "blocker": self.adapter.knowledge.claim(i).payload.get("blocker")} for i in ids],
                        record["experiment_accounting"])
                    if isinstance(decision, RecoveryPause):
                        record["pause"] = decision.model_dump(mode="json")
                    else:
                        self.pending_experiment = decision
                        record["experiment"] = decision.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 -- learner failure must not invent a remedy
            record["error_type"] = type(exc).__name__
        finally:
            elapsed = perf_counter() - started
            record["elapsed_seconds"] = elapsed
            record["model_call_costs"] = self.client.calls[call_start:]
            self.adapter.ledger.adaptation_seconds += elapsed

    def context(self, max_chars=2400):
        prefix = "New model hypotheses from this task, not verified facts or permissions:\n"
        experiment = ""
        if self.audit and self.audit[-1].get("experiment"):
            experiment = "\nModel-proposed distinguishing action and observed follow-through: " + json.dumps({
                "proposal": self.audit[-1]["experiment"],
                "outcome": self.audit[-1].get("experiment_outcome", "pending actor decision"),
            })
            if len(prefix + experiment) > max_chars:
                experiment = ""
        if self.audit and self.audit[-1].get("pause"):
            experiment = "\nModel-proposed scoped pause, advisory only: " + json.dumps(self.audit[-1]["pause"])
            if len(prefix + experiment) > max_chars:
                experiment = ""
        rows = []
        for identity in self.latest_hypotheses:
            claim = self.adapter.knowledge.claim(identity)
            review = claim.payload.get("semantic_review", {})
            if claim.status != "hypothesis" or review.get("verdict") in {
                "counterevidence", "internally_inconsistent",
            }:
                continue
            row = {
                "claim_id": identity, "statement": claim.statement,
                "alternatives": claim.payload.get("alternatives", []),
                "distinguishing_observation": claim.payload.get("distinguishing_observation"),
                "evidence_ids": list(claim.supporting_episodes),
                "review": review.get("verdict", "unreviewed"),
            }
            if claim.payload.get("blocker"):
                row["blocker"] = claim.payload["blocker"]
            if len(prefix + experiment + json.dumps([*rows, row])) <= max_chars:
                rows.append(row)
        return prefix + json.dumps(rows) + experiment if rows or experiment else ""
