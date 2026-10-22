"""Costed refresh of task-local records using observed, permitted read calls."""

from pydantic import Field

from .decisions import Decision, ProposedAction, check_decision
from .knowledge import Record, response_coverage, utc_now
from .operation_semantics import is_observation_read
from .operational_requirements import api_requirements


class RefreshEstimate(Record):
    # Same units required. These are explicit estimates, not calibrated values.
    expected_stale_decision_loss: float = Field(ge=0, allow_inf_nan=False)
    check_cost: float = Field(ge=0, allow_inf_nan=False)
    mandatory: bool = False


class RefreshResult(Record):
    attempted: bool
    refreshed: bool
    reason: str
    source_episode_id: str | None = None
    observation_episode_id: str | None = None


async def refresh_live_fact(adapter, key: str, estimate: RefreshEstimate) -> RefreshResult:
    """Refresh is not permission to retry a write or to resolve an unknown effect.

    Replays only a previously successful known non-mutating read in this exact task/scope.
    Missing read provenance remains unresolved. Mandatory checks bypass the
    optional value heuristic, never the shared call budget or dependency gate.
    """
    from automationbench.tools.api import api_fetch

    if not estimate.mandatory and estimate.expected_stale_decision_loss <= estimate.check_cost:
        return RefreshResult(
            attempted=False, refreshed=False, reason="estimated_check_cost_exceeds_value"
        )
    candidates = []
    for episode in adapter.evidence.episodes(adapter.scope):
        if (
            episode.scope != adapter.scope
            or episode.task_id != adapter.live.task_id
            or episode.operation != "api_fetch"
            or episode.outcome != "success"
            or not is_observation_read(adapter, episode)
            or episode.timestamp > utc_now()
        ):
            continue
        namespace = "api_fetch:" + str(episode.arguments.get("url", "")) + ":"
        if isinstance(episode.response, (dict, list)) and any(
            key == namespace + field and field in response_coverage(episode.response)
            for field in episode.coverage
        ):
            candidates.append(episode)
    if not candidates:
        return RefreshResult(
            attempted=False, refreshed=False, reason="no_task_local_read_provenance"
        )
    source = max(candidates, key=lambda e: e.timestamp)
    action = ProposedAction(tool="api_fetch", arguments=source.arguments)
    try:
        requirements = api_requirements(adapter.knowledge, adapter.scope, action)
    except ValueError:
        return RefreshResult(
            attempted=False,
            refreshed=False,
            reason="contract_unresolved",
            source_episode_id=source.episode_id,
        )
    gate = check_decision(
        Decision(
            goal="Refresh live fact " + key,
            rationale="Recheck a consequential observation",
            action=action,
        ),
        mandatory=requirements,
        store=adapter.knowledge,
        scope=adapter.scope,
        live=adapter.live,
        allowed_tools={"api_fetch"},
        evidence_store=adapter.evidence,
    )
    if not gate.allowed:
        return RefreshResult(
            attempted=False,
            refreshed=False,
            reason="unresolved_read_prerequisites",
            source_episode_id=source.episode_id,
        )
    before = {e.episode_id for e in adapter.evidence.episodes(adapter.scope)}

    def observations():
        return {
            e.episode_id: e
            for e in adapter.evidence.episodes(adapter.scope)
            if e.episode_id not in before
            and e.scope == adapter.scope
            and e.operation == "api_fetch"
            and e.arguments == source.arguments
            and e.purpose == "refresh"
        }

    try:
        result = await adapter.execute(api_fetch, source.arguments, purpose="refresh")
    except BaseException:
        # A rejected budget charge (or other pre-observation failure) is not a
        # new observation and must not erase the old fact. Preserve its original
        # timestamp/invalidation state; never mark it freshly verified.
        if observations():
            adapter.live.invalidate({key}, "refresh-interrupted")
        raise
    observed = observations()
    fact = next((f for f in adapter.live.facts() if f.key == key), None)
    refreshed = bool(
        result.success
        and fact is not None
        and fact.episode_id in observed
        and observed[fact.episode_id].outcome == "success"
    )
    if observed and not refreshed:
        adapter.live.invalidate({key}, "refresh-unresolved")
    return RefreshResult(
        attempted=bool(observed),
        refreshed=refreshed,
        reason="observed"
        if refreshed
        else "refresh_unresolved"
        if observed
        else "no_new_observation",
        source_episode_id=source.episode_id,
        observation_episode_id=fact.episode_id
        if fact is not None and fact.episode_id in observed
        else None,
    )
