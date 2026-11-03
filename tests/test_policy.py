from worldmodel_harness.core.policy import EXPLORE, MIN_RUNS, LearnedPolicy


def log(kind, step, shown):
    return [{"kind": kind, "step": step, "shown": shown}, {"kind": kind, "step": step + 3, "shown": False}]


def test_explores_until_both_sides_have_runs_then_shows_what_left_fewer_calls():
    policy = LearnedPolicy()
    first = policy.for_run("t.a~v1")
    first.show("plan", "T", "s", "x")
    assert first.decisions["plan"]["reason"] == "explore"
    for n in range(MIN_RUNS):
        policy.learn("T", log("plan", 5, True), 15)  # 10 calls left after a shown plan
        policy.learn("T", log("plan", 5, False), 35)  # 30 left when held back
    assert policy.estimate("T", "plan") == (10, 30, MIN_RUNS, MIN_RUNS)
    decisions = [policy.for_run(f"t.a~v{n}") for n in range(200)]
    shown = [r.show("plan", "T", "s", "x") for r in decisions]
    assert all(r.decisions["plan"]["reason"] in ("learned", "explore") for r in decisions)
    assert 0.8 < sum(shown) / len(shown) < 1 - EXPLORE / 3


def test_falls_back_to_the_kind_across_templates_and_is_seeded_by_task():
    policy = LearnedPolicy()
    for n in range(MIN_RUNS):
        policy.learn(f"T{n}", log("sources", 2, True), 40)
        policy.learn(f"U{n}", log("sources", 2, False), 12)
    assert policy.estimate("new template", "sources") == (38, 10, MIN_RUNS, MIN_RUNS)
    a, b = policy.for_run("t.b~v1"), policy.for_run("t.b~v1")
    assert a.show("sources", "new template", "s", "x") == b.show("sources", "new template", "s", "x")


def test_a_shown_kind_is_shown_when_its_text_changes_and_decided_once_per_run():
    policy = LearnedPolicy()
    for n in range(MIN_RUNS):
        policy.learn("T", log("plan", 1, True), 5)
        policy.learn("T", log("plan", 1, False), 50)
    run = next(r for r in (policy.for_run(f"t~v{n}") for n in range(50)) if r.show("plan", "T", "s", "a"))
    assert not run.show("plan", "T", "s", "a")
    assert run.show("plan", "T", "s", "b")
