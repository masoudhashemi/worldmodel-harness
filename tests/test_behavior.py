from worldmodel_harness.behavior import BehaviorLearner, endpoint_contract
from worldmodel_harness.knowledge import Episode, Scope
from worldmodel_harness.learning import Learner
from worldmodel_harness.store import KnowledgeStore


def setup(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    learner = Learner(store)
    scope = Scope(environment="test", version="1", tenant="a", task_id="first")
    endpoint = {"id": "records.get", "method": "GET", "url": "https://test/records/{id}"}
    episode = Episode(
        task_id="first",
        scope=scope,
        operation="api_search",
        arguments={},
        response={"results": [endpoint]},
        outcome="success",
    )
    store.append_episode(episode)
    learner.document(
        episode,
        relation="endpoint_contract",
        participants=("records.get",),
        statement="Get a record",
        payload=endpoint,
        scope=Scope(environment="test", version="1"),
    )
    return store, learner, scope


def observation(scope, value="secret-live-value", **changes):
    base = {
        "task_id": scope.task_id,
        "scope": scope,
        "operation": "api_fetch",
        "arguments": {"method": "GET", "url": "https://test/records/private-id"},
        "response": {"id": value},
        "outcome": "success",
        "coverage": ("id",),
    }
    return Episode(**(base | changes))


def test_one_instance_is_not_a_general_rule_or_reusable_identifier(tmp_path):
    store, learner, scope = setup(tmp_path)
    episode = observation(scope)
    store.append_episode(episode)
    claim = BehaviorLearner(learner).observe(episode)[0]
    assert claim.status == "hypothesis"
    assert "secret-live-value" not in claim.model_dump_json()
    assert "private-id" not in claim.model_dump_json()
    store.close()


def test_multiple_reads_in_one_task_do_not_promote(tmp_path):
    store, learner, scope = setup(tmp_path)
    behavior = BehaviorLearner(learner)
    for _ in range(3):
        episode = observation(scope)
        store.append_episode(episode)
        claim = behavior.observe(episode)[0]
    assert claim.status == "hypothesis"
    store.close()


def test_independent_tasks_promote_an_output_type(tmp_path):
    store, learner, scope = setup(tmp_path)
    behavior = BehaviorLearner(learner)
    for target in (scope, scope.model_copy(update={"tenant": "b", "task_id": "second"})):
        episode = observation(target)
        store.append_episode(episode)
        claim = behavior.observe(episode)[0]
    assert claim.status == "supported"
    assert claim.payload == {
        "endpoint": "records.get", "request_shape": (), "field": "id", "type": "string"
    }
    store.close()


def test_errors_do_not_train_behavior(tmp_path):
    store, learner, scope = setup(tmp_path)
    failed = observation(
        scope,
        outcome="error",
        response={"error": {"code": 400, "message": "private-value"}},
        coverage=("error",),
    )
    store.append_episode(failed)
    assert BehaviorLearner(learner).observe(failed) == []
    store.close()


def test_unknown_outcome_cannot_train_behavior(tmp_path):
    store, learner, scope = setup(tmp_path)
    episode = observation(scope, outcome="unknown")
    store.append_episode(episode)
    assert BehaviorLearner(learner).observe(episode) == []
    store.close()


def test_phone_numbers_share_one_kind_and_another_kind_disputes_the_slot(tmp_path):
    store, learner, scope = setup(tmp_path)
    behavior = BehaviorLearner(learner)
    claim = None
    for number, tenant in (("+1-415-555-0100", "a"), ("(415) 555-0199", "b")):
        episode = observation(
            scope.model_copy(update={"tenant": tenant, "task_id": tenant}), value=number
        )
        store.append_episode(episode)
        claim = behavior.observe(episode)[0]
    assert claim is not None
    assert claim.payload["type"] == "phone"
    assert claim.status == "supported"
    assert "+1-415-555-0100" not in claim.model_dump_json()
    assert "(415) 555-0199" not in claim.model_dump_json()
    email = observation(scope, value="avery@example.com")
    store.append_episode(email)
    disputed = behavior.observe(email)[0]
    assert disputed.claim_id == claim.claim_id
    assert disputed.status == "disputed"
    assert disputed.payload["conflicting_type"] == "email"
    assert "avery@example.com" not in disputed.model_dump_json()
    named = observation(scope, response={"Phone": "hello"}, coverage=("Phone",))
    store.append_episode(named)
    assert behavior.observe(named)[0].payload["type"] == "string"
    store.close()


def test_same_route_with_another_endpoint_id_stays_one_contract(tmp_path):
    store, learner, scope = setup(tmp_path)
    search = next(episode for episode in store.episodes(scope) if episode.operation == "api_search")
    alias = {"id": "records.get.alias", "method": "GET", "url": "https://test/records/{id}"}
    learner.document(
        search,
        relation="endpoint_contract",
        participants=(alias["id"], alias["url"], alias["method"]),
        statement="Alias of the same route",
        payload=alias,
        scope=Scope(environment="test", version="1"),
    )
    contracts = [claim for claim in store.claims() if claim.relation == "endpoint_contract"]
    assert len(contracts) == 1
    assert endpoint_contract(
        store, scope, {"method": "GET", "url": "https://test/records/another-id"}
    )
    concrete = {
        "id": "records.special",
        "method": "GET",
        "url": "https://test/records/special",
    }
    learner.document(
        search,
        relation="endpoint_contract",
        participants=(concrete["id"], concrete["url"], concrete["method"]),
        statement="A different template",
        payload=concrete,
        scope=Scope(environment="test", version="1"),
    )
    assert (
        endpoint_contract(store, scope, {"method": "GET", "url": "https://test/records/special"})
        is None
    )
    store.close()


def test_same_route_conflicting_schema_keeps_both_and_grounds_neither(tmp_path):
    store, learner, scope = setup(tmp_path)
    search = next(e for e in store.episodes(scope) if e.operation == "api_search")
    payload = {"id": "other", "method": "GET", "url": "https://test/records/{id}",
               "parameters": {"permission": {"required": True}}}
    learner.document(search, relation="endpoint_contract", participants=("other",),
                     statement="Different schema", payload=payload,
                     scope=Scope(environment="test", version="1"))
    assert len([c for c in store.claims() if c.relation == "endpoint_contract"]) == 2
    args = {"method": "GET", "url": "https://test/records/1"}
    assert endpoint_contract(store, scope, args) is None
    store.close()


def test_same_route_tenant_schema_does_not_overwrite_environment(tmp_path):
    store, learner, scope = setup(tmp_path)
    search = next(e for e in store.episodes(scope) if e.operation == "api_search")
    scoped = scope.model_copy(update={"task_id": None})
    payload = {"id": "tenant-route", "method": "GET", "url": "https://test/records/{id}",
               "parameters": {"tenantToken": {"required": True}}}
    learner.document(search, relation="endpoint_contract", participants=("tenant-route",),
                     statement="Tenant schema", payload=payload, scope=scoped)
    contracts = [c for c in store.claims() if c.relation == "endpoint_contract"]
    assert {c.scope.tenant for c in contracts} == {None, "a"}
    args = {"method": "GET", "url": "https://test/records/1"}
    assert endpoint_contract(store, scope, args) is None
    assert endpoint_contract(store, scope.model_copy(update={"tenant": "b"}), args)
    store.close()


def test_ambiguous_or_unknown_contract_cannot_ground_generalization(tmp_path):
    store, learner, scope = setup(tmp_path)
    episode = observation(scope, arguments={"method": "GET", "url": "https://unknown/records/x"})
    assert endpoint_contract(store, scope, episode.arguments) is None
    assert BehaviorLearner(learner).observe(episode) == []
    store.close()


def test_specific_parameterized_route_beats_generic_placeholder_route(tmp_path):
    store, learner, scope = setup(tmp_path)
    search = next(e for e in store.episodes(scope) if e.operation == "api_search")
    generic = {
        "id": "records.update_generic",
        "method": "PATCH",
        "url": "https://test/sobjects/{sObjectType}/{id}",
    }
    specific = {
        "id": "records.update_contact",
        "method": "PATCH",
        "url": "https://test/sobjects/Contact/{id}",
    }
    for payload in (generic, specific):
        learner.document(
            search,
            relation="endpoint_contract",
            participants=(payload["id"], payload["url"], payload["method"]),
            statement="Route contract",
            payload=payload,
            scope=Scope(environment="test", version="1"),
        )
    chosen = endpoint_contract(
        store, scope, {"method": "PATCH", "url": "https://test/sobjects/Contact/003001"}
    )
    assert chosen is not None
    assert chosen.payload["url"] == "https://test/sobjects/Contact/{id}"
    store.close()

