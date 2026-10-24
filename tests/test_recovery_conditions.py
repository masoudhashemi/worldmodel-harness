import base64
import json

from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.recovery_conditions import RecoveryMemory, request_conditions


def events(task="one", field="payload", wire="bad", success="good"):
    scope = Scope(environment="test", version="1", task_id=task)

    def e(**kw):
        return Episode(scope=scope, task_id=task, **kw)

    return [
        e(
            operation="api_search",
            arguments={},
            outcome="success",
            response={"results": [{"method": "POST", "url": "https://example.test/items/{id}"}]},
        ),
        e(
            operation="api_fetch",
            arguments={"method": "POST", "url": "https://example.test/items/42", "body": wire},
            outcome="error",
            response={"error": "invalid"},
        ),
        e(
            operation="api_fetch",
            arguments={"method": "POST", "url": "https://example.test/items/42", "body": success},
            outcome="success",
            response={"ok": True},
        ),
    ]


def test_features_preserve_representation_but_not_real_values():
    raw = "To: secret@example.com\nSubject: Private Name\n\nPrivate body"
    encoded = base64.urlsafe_b64encode(raw.encode()).decode()
    direct = request_conditions({"body": json.dumps({"raw": raw})})
    wrapped = request_conditions({"body": json.dumps({"raw": encoded})})
    assert not direct["/body/json/raw/base64_utf8"]
    assert wrapped["/body/json/raw/base64_utf8"]
    assert wrapped["/body/json/raw/decoded/header_names"] == ("subject", "to")
    serialized = json.dumps(wrapped)
    assert "secret@example.com" not in serialized
    assert encoded not in serialized
    assert "Private" not in serialized
    assert not any("form_keys" in k for k in wrapped)


def test_json_and_form_differ_without_endpoint_rules():
    form = request_conditions({"body": "target=alice&content=hello"})
    obj = request_conditions({"body": '{"target":"alice","content":"hello"}'})
    assert form["/body/json_parseable"] is False
    assert obj["/body/json_parseable"] is True
    assert form["/body/form_keys"] == ("content", "target")


def test_learn_contrast_without_successful_task_or_invented_counts():
    es = events(wire="x=one&y=two", success='{"x":"one","y":"two"}')
    memory = RecoveryMemory()
    memory.learn(es + es)
    assert len(memory.observations) == 2
    assert len(memory.contrasts) == 1
    c = memory.contrasts[0]
    assert c.failed_episode == es[1].episode_id
    assert c.accepted_episode == es[2].episode_id
    assert c.status == "hypothesis"
    assert c.confounded  # Many measured conditions changed; no single cause asserted.
    assert memory.retrieve(c.capability, c.scope) == []  # no same-task leakage
    scope = c.scope.model_copy(update={"task_id": "new"})
    assert len(memory.retrieve(c.capability, scope)) == 1
    result = memory.assess(c.capability, scope, es[1].arguments)
    assert result["observed_outcomes"] == {"error": 1}
    assert memory.retrieve(c.capability, scope.model_copy(update={"version": "2"})) == []


def test_errors_from_harness_and_different_resources_do_not_form_pairs():
    es = events(wire="a=b&c=d", success='{"a":"b","c":"d"}')
    memory = RecoveryMemory()
    memory.learn([es[0], es[1].model_copy(update={"origin": "harness"}), es[2]])
    assert not memory.contrasts
    memory = RecoveryMemory()
    changed = es[2].model_copy(
        update={"arguments": es[2].arguments | {"url": "https://example.test/items/99"}}
    )
    memory.learn([es[0], es[1], changed])
    assert not memory.contrasts


def test_contradictory_conditions_are_retained_as_outcome_distribution():
    es = events(wire="bad", success="bad")
    memory = RecoveryMemory()
    memory.learn(es)
    c = memory.contrasts[0]
    assert c.changed_paths == ()
    scope = c.scope.model_copy(update={"task_id": "heldout"})
    assert memory.retrieve(c.capability, scope) == []
    result = memory.assess(c.capability, scope, es[1].arguments)
    assert result["observed_outcomes"] == {"error": 1, "success": 1}


def test_choice_prompt_never_exposes_heldout_outcomes(monkeypatch):
    import runpy

    monkeypatch.syspath_prepend("scripts")
    api = runpy.run_path("scripts/evaluate_recovery_choices.py")
    train, heldout = RecoveryMemory(), RecoveryMemory()
    train.learn(events("train", wire="x=a&y=b", success='{"x":"a","y":"b"}'))
    heldout.learn(events("heldout", wire="x=c&y=d", success='{"x":"c","y":"d"}'))
    cases, excluded = api["cases_from"](train, heldout)
    assert not excluded
    case = cases[0]
    prompt = api["make_prompt"](case, train, "learned_conditions")
    changed = case | {
        "episodes": [
            e.model_copy(update={"outcome": "unknown", "response": "FUTURE"})
            for e in case["episodes"]
        ]
    }
    assert prompt == api["make_prompt"](changed, train, "learned_conditions")
    assert case["contrast"].failed_episode not in json.dumps(prompt)
    assert "FUTURE" not in json.dumps(prompt)


def test_duplicate_action_pairs_cannot_count_as_heldout_improvement(monkeypatch):
    import runpy

    monkeypatch.syspath_prepend("scripts")
    api = runpy.run_path("scripts/evaluate_recovery_choices.py")
    train, test = RecoveryMemory(), RecoveryMemory()
    for memory, task in ((train, "train"), (test, "test")):
        memory.learn(events(task, wire="x=a&y=b", success='{"x":"a","y":"b"}'))
    cases, excluded = api["cases_from"](train, test)
    assert not cases
    assert excluded[0]["reason"] == "duplicate_action_pair"


def test_frozen_training_contracts_resolve_missing_discovery_without_learning_labels():
    train, test = RecoveryMemory(), RecoveryMemory()
    train.learn(events("train", wire="x=a&y=b", success='{"x":"a","y":"b"}'))
    heldout = events("heldout", wire="x=c&y=d", success='{"x":"c","y":"d"}')[1:]
    test.learn(heldout)
    assert not test.observations
    test.learn(heldout, seed_evidence=train.observations)
    assert len(test.contrasts) == 1
    assert len(train.contrasts) == 1
    assert (
        test.observations[0].capability.contract_episode
        == train.observations[0].capability.contract_episode
    )
