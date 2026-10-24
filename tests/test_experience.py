import pytest

from worldmodel_harness.experience import ExperienceMemory, PrefixExperience, Symbolizer
from worldmodel_harness.knowledge import Episode, Scope


def episode(task="one", **changes):
    return Episode(
        **(
            {
                "task_id": task,
                "scope": Scope(environment="test", version="1", task_id=task),
                "operation": "send",
                "arguments": {"recipient": "alice@example.com"},
                "response": {"receipt": "private-id"},
                "outcome": "success",
            }
            | changes
        )
    )


def transition(e):
    prefix = PrefixExperience(e.scope, e.task_id)
    query = prefix.query(e.operation, e.arguments)
    return prefix.observe(e, query)


def test_symbolic_identity_preserves_two_badges_one_owner_and_renaming():
    def rows(person, first, second):
        return [{"owner": person, "badge": first}, {"owner": person, "badge": second}]

    left = Symbolizer().snapshot(rows("Alice", "BDG-1", "BDG-2"))
    right = Symbolizer().snapshot(rows("Bob", "OTHER-1", "OTHER-2"))
    assert left == right
    values = left.structure["array"]
    assert values[0]["object"]["owner"] == values[1]["object"]["owner"]
    assert values[0]["object"]["badge"] != values[1]["object"]["badge"]
    assert "Alice" not in left.model_dump_json()
    assert "BDG-1" not in left.model_dump_json()


def test_unknown_is_not_false_and_partial_observation_is_explicit():
    assert Symbolizer().snapshot({}) != Symbolizer().snapshot({"eligible": False})
    snapshot = Symbolizer(max_nodes=4).snapshot(list(range(100)))
    assert snapshot.truncated
    assert snapshot.structure["length"] == 100
    assert len(snapshot.structure["array"]) == 3


def test_failed_tasks_supply_local_transitions_without_rewards():
    memory = ExperienceMemory()
    bad = transition(episode(outcome="error", response={"error": "encoding rejected"}))
    good = transition(episode("two"))
    memory.append(bad)
    memory.append(bad)
    memory.append(good)
    query = transition(episode("heldout")).query
    report = memory.retrieve(query)
    assert len(memory.transitions) == 2
    assert report.outcome_counts == {"error": 1, "success": 1}
    assert report.independent_tasks == 2
    assert any("disagree" in u for u in report.uncertainty)
    assert all(m.match_kind == "same_observed_structure" for m in report.matches)


def test_scope_and_same_task_exclusion():
    memory = ExperienceMemory()
    stored = transition(episode())
    memory.append(stored)
    assert not memory.retrieve(stored.query).matches
    heldout = transition(episode("test")).query
    for field, value in (("version", "2"), ("tenant", "other"), ("role", "admin")):
        query = heldout.model_copy(
            update={"scope": heldout.scope.model_copy(update={field: value})}
        )
        assert not memory.retrieve(query).matches


def test_discovery_only_resolves_subsequent_calls_and_keeps_interface_distinction():
    e = episode()
    prefix = PrefixExperience(e.scope, e.task_id)
    args = {"method": "POST", "url": "https://example.test/messages/secret/send"}
    assert prefix.query("api_fetch", args) is None
    discovery = episode(
        operation="api_search",
        arguments={},
        response={
            "results": [{"method": "POST", "url": "https://example.test/messages/{id}/send"}]
        },
    )
    prefix.observe(discovery, prefix.query(discovery.operation, discovery.arguments))
    query = prefix.query("api_fetch", args)
    assert query.capability.route == "https://example.test/messages/{id}/send"
    assert "secret" not in query.model_dump_json()
    assert query.capability.key != prefix.query("send", {}).capability.key


def test_harness_rejections_do_not_teach_environment_dynamics():
    e = episode(origin="harness", outcome="blocked")
    prefix = PrefixExperience(e.scope, e.task_id)
    assert prefix.observe(e, prefix.query(e.operation, e.arguments)) is None
    assert not prefix.history


def test_prefix_query_does_not_include_current_response_or_future_evidence():
    e = episode()
    prefix = PrefixExperience(e.scope, e.task_id)
    query = prefix.query(e.operation, e.arguments)
    assert query.before == ()
    first = prefix.observe(e, query)
    assert first.query.before == ()
    assert prefix.query("send", e.arguments).before[0].episode_id == e.episode_id
    assert query.before == ()
    with pytest.raises(ValueError, match="mix task"):
        prefix.observe(episode("another"), query)


def test_structural_match_is_never_presented_as_semantic_equivalence():
    memory = ExperienceMemory()
    memory.append(transition(episode(arguments={"status": "eligible"})))
    query = transition(episode("heldout", arguments={"status": "ineligible"})).query
    report = memory.retrieve(query)
    assert report.matches[0].match_kind == "same_observed_structure"
    assert any("does not establish semantic" in u for u in report.uncertainty)


def test_alias_numbering_is_canonicalized_across_irrelevant_older_values():
    first = PrefixExperience(episode().scope, "one")
    second = PrefixExperience(episode("two").scope, "two")
    second.symbolizer.snapshot("irrelevant older instance")
    assert first.query("send", {"to": "a@example.com"}).state_signature() == (
        second.query("send", {"to": "b@example.com"}).state_signature()
    )


def test_evidence_counts_and_disagreement_survive_example_limit():
    memory = ExperienceMemory()
    for i in range(6):
        memory.append(transition(episode(str(i))))
    memory.append(transition(episode("failed", outcome="error")))
    report = memory.retrieve(transition(episode("heldout")).query, limit=1)
    assert len(report.matches) == 1
    assert report.outcome_counts == {"success": 6, "error": 1}
    assert report.independent_tasks == 7
    assert any("disagree" in u for u in report.uncertainty)


def test_different_action_cannot_be_attached_to_a_query():
    e = episode()
    prefix = PrefixExperience(e.scope, e.task_id)
    query = prefix.query("lookup", {})
    with pytest.raises(ValueError, match="different action"):
        prefix.observe(e, query)


def test_replay_retrieval_is_invariant_to_heldout_response_and_does_not_train():
    import runpy

    replay = runpy.run_path("scripts/replay_experience.py")["replay"]
    memory = ExperienceMemory()
    replay([episode("train")], memory, training=True)
    test = episode("heldout")
    first = replay([test], memory, training=False)
    changed = test.model_copy(update={"response": {"totally": "different"}, "outcome": "error"})
    second = replay([changed], memory, training=False)
    assert first[0]["retrieval"] == second[0]["retrieval"]
    assert len(memory.transitions) == 1


def test_predictive_eval_does_not_use_test_response(monkeypatch):
    import runpy

    monkeypatch.syspath_prepend("scripts")
    evaluate = runpy.run_path("scripts/evaluate_experience.py")["evaluate"]
    memory = ExperienceMemory()
    memory.append(transition(episode("train")))
    heldout = episode("heldout")
    first = evaluate([heldout], memory)[0]
    second = evaluate([heldout.model_copy(update={"outcome": "error", "response": []})], memory)[0]
    for arm in first["arms"]:
        before = {k: v for k, v in first["arms"][arm].items() if k != "scores"}
        after = {k: v for k, v in second["arms"][arm].items() if k != "scores"}
        assert before == after
    assert len(memory.transitions) == 1


def test_predictive_eval_reports_fallback_and_uses_paired_denominator(monkeypatch):
    import runpy

    monkeypatch.syspath_prepend("scripts")
    functions = runpy.run_path("scripts/evaluate_experience.py")
    memory = ExperienceMemory()
    memory.append(transition(episode("train")))
    rows = functions["evaluate"](
        [
            episode("heldout", arguments={"different": [1, 2]}),
            episode("heldout", operation="api_fetch", arguments={"url": "https://unknown.test"}),
        ],
        memory,
    )
    assert not rows[0]["exact_structure_match"]
    assert rows[0]["arms"]["state_history"] == rows[0]["arms"]["capability_history"]
    assert not rows[1]["query_available"]
    assert all(s["paired_calls"] == 1 for s in functions["summarize"](rows).values())


def test_response_schema_scoring_ignores_literals_and_reports_abstention(monkeypatch):
    import runpy

    monkeypatch.syspath_prepend("scripts")
    functions = runpy.run_path("scripts/evaluate_experience.py")
    first = Symbolizer().snapshot({"items": [{"id": "one"}]})
    second = Symbolizer().snapshot({"items": [{"id": "two"}, {"id": "three"}]})
    assert functions["fields"](first) == functions["fields"](second)
    scores = functions["score"](functions["forecast"]([]), "success", first)
    assert scores["response_structure_correct"] == 0
    assert scores["outcome_brier"] == 0.75


def test_llm_payload_does_not_include_future_response(monkeypatch):
    import runpy

    monkeypatch.syspath_prepend("scripts")
    functions = runpy.run_path("scripts/evaluate_experience_llm.py")
    e = episode("heldout")
    query = PrefixExperience(e.scope, e.task_id).query(e.operation, e.arguments)
    case = {"episode": e, "query": query, "capability": [], "state": []}
    first = functions["payload"](case, "no_experience")
    second = functions["payload"](
        case | {"episode": e.model_copy(update={"response": {"future": "secret"}})},
        "no_experience",
    )
    assert first == second
    assert "private-id" not in str(first)


def test_llm_metrics_do_not_hide_invalid_arms(monkeypatch):
    import runpy

    monkeypatch.syspath_prepend("scripts")
    functions = runpy.run_path("scripts/evaluate_experience_llm.py")
    result = functions["metrics"](
        [
            {"episode_id": "one", "arm": "no_experience", "scores": {"outcome_brier": 0}},
            {"episode_id": "one", "arm": "capability_history", "error_type": "ValueError"},
        ]
    )
    assert result["no_experience"]["valid"] == 1
    assert result["no_experience"]["paired_cases"] == 0
    assert result["capability_history"]["valid"] == 0


def test_llm_response_paths_require_json_pointer_syntax(monkeypatch):
    import runpy

    from pydantic import ValidationError

    monkeypatch.syspath_prepend("scripts")
    response_field = runpy.run_path("scripts/evaluate_experience_llm.py")["ResponseField"]
    with pytest.raises(ValidationError):
        response_field(path="items/*/id", kind="string")
    assert response_field(path="/items/*/id", kind="string").path == "/items/*/id"
    assert response_field(path="", kind="object").path == ""
