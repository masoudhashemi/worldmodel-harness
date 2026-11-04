from worldmodel_harness.core.policy import MIN_RUNS, LearnedPolicy, matched, situation
from worldmodel_harness.core.trace import Step, ToolDomain

D = ToolDomain()


def calls(*names):
    return [Step(n, {"q": n}, "{}", True) for n in names]


def log(kind, step, shown):
    return [{"kind": kind, "step": step, "shown": shown}, {"kind": kind, "step": step + 3, "shown": False}]


def learn(policy, template, kind, shown, left, control_left):
    """A run whose kind was first offered after its 2nd call, against a control run in the
    same situation at its 2nd call."""
    run = calls("a", "b", *["c"] * left)
    control = calls("a", "b", *["d"] * control_left)
    policy.learn(template, log(kind, 2, shown), run, control)


def test_matches_the_control_in_the_same_situation_at_the_same_occurrence():
    run = calls("a", "b", "a", "b")
    control = calls("x", "a", "b", "a", "b", "y")
    here = situation(D, run, 3)
    assert here == ("a(q)", "b(q)", True, True)
    assert matched(D, control, here, 0) == 4
    assert matched(D, control, situation(D, run, 1), 0) == 2
    assert matched(D, calls("z"), here, 0) is None


def test_learns_calls_left_against_the_control_not_raw_calls_left():
    policy = LearnedPolicy(D)
    first = policy.for_run("t.a~v1")
    first.show("plan", "T", "s", "x")
    assert first.decisions["plan"]["reason"] == "explore"
    for left, control in ((40, 50), (45, 54), (30, 41)):
        learn(policy, "T", "plan", True, left, control)  # long tasks, fewer calls than control
    for left, control in ((12, 10), (8, 7), (15, 12)):
        learn(policy, "T", "plan", False, left, control)  # short tasks, more calls
    shown_mean, held_mean, se, *_ = policy.estimate("T", "plan")
    assert (shown_mean, held_mean) == (-10, 2) and se > 0
    decisions = [policy.for_run(f"t.a~v{n}") for n in range(200)]
    shown = [r.show("plan", "T", "s", "x") for r in decisions]
    assert decisions[0].decisions["plan"]["reason"] == "sampled"
    assert sum(shown) / len(shown) > 0.95


def test_keeps_exploring_while_the_sides_overlap():
    policy = LearnedPolicy(D)
    for left, control in ((10, 9), (2, 8), (20, 15)):
        learn(policy, "T", "plan", True, left, control)
        learn(policy, "T", "plan", False, left + 1, control)
    shown = [policy.for_run(f"t.a~v{n}").show("plan", "T", "s", "x") for n in range(200)]
    assert 0.25 < sum(shown) / len(shown) < 0.75


def test_offers_without_a_matched_control_point_are_not_learned():
    policy = LearnedPolicy(D)
    policy.learn("T", log("plan", 2, True), calls("a", "b", "c"), calls("z"))
    assert not policy.records


def test_falls_back_to_the_kind_across_templates_and_is_seeded_by_task():
    policy = LearnedPolicy(D)
    for n in range(MIN_RUNS):
        learn(policy, f"T{n}", "sources", True, 5, 3)
        learn(policy, f"U{n}", "sources", False, 3, 5)
    assert policy.estimate("new template", "sources")[:2] == (2, -2)
    a, b = policy.for_run("t.b~v1"), policy.for_run("t.b~v1")
    assert a.show("sources", "new template", "s", "x") == b.show("sources", "new template", "s", "x")


def test_a_shown_kind_is_shown_when_its_text_changes_and_decided_once_per_run():
    policy = LearnedPolicy(D)
    for n in range(MIN_RUNS):
        learn(policy, "T", "plan", True, 1, 9)
        learn(policy, "T", "plan", False, 9, 1)
    run = next(r for r in (policy.for_run(f"t~v{n}") for n in range(50)) if r.show("plan", "T", "s", "a"))
    assert not run.show("plan", "T", "s", "a")
    assert run.show("plan", "T", "s", "b")
