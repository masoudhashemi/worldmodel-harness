from worldmodel_harness.core.policy import MIN_RUNS, LearnedPolicy, matched, situation
from worldmodel_harness.core.trace import Step, ToolDomain

D = ToolDomain()


def calls(*names):
    return [Step(n, {"q": n}, "{}", True) for n in names]


def log(kind, step, shown):
    return [{"kind": kind, "step": step, "shown": shown}, {"kind": kind, "step": step + 3, "shown": False}]


class Writes(ToolDomain):
    def writes(self, step):
        return step.tool.startswith("w")


def learn(policy, template, kind, shown, left, control_left, writes=0, control_writes=0):
    """A run whose kind was first offered after its 2nd call, against a control run in the
    same situation at its 2nd call."""
    run = calls("a", "b", *[f"w{n}" for n in range(writes)], *["c"] * (left - writes))
    control = calls("a", "b", *[f"w{n}" for n in range(control_writes)], *["d"] * (control_left - control_writes))
    policy.learn(template, log(kind, 2, shown), run, control)


def test_matches_the_control_in_the_same_situation_at_the_same_occurrence():
    run = calls("a", "b", "a", "b")
    control = calls("x", "a", "b", "a", "b", "y")
    here = situation(D, run, 3)
    assert here == ("a(q)", "b(q)", True, True)
    assert matched(D, control, here, 0) == 4
    assert matched(D, control, situation(D, run, 1), 0) == 2
    assert matched(D, calls("z"), here, 0) is None
    assert matched(D, control, here, 1) is None  # no fallback to another occurrence


def test_learns_calls_left_against_the_control_not_raw_calls_left():
    policy = LearnedPolicy(D)
    first = policy.for_run("t.a~v1")
    first.show("plan", "T", "s", "x")
    assert first.decisions["plan"]["reason"] == "explore"
    for left, control in ((40, 50), (45, 54), (30, 41)):
        learn(policy, "T", "plan", True, left, control)  # long tasks, fewer calls than control
    for left, control in ((12, 10), (8, 7), (15, 12)):
        learn(policy, "T", "plan", False, left, control)  # short tasks, more calls
    e = policy.estimate("T", "plan")
    assert (e.shown, e.held) == (-10, 2) and e.se > 0
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
    e = policy.estimate("new template", "sources")
    assert (e.shown, e.held) == (2, -2)
    a, b = policy.for_run("t.b~v1"), policy.for_run("t.b~v1")
    assert a.show("sources", "new template", "s", "x") == b.show("sources", "new template", "s", "x")


def test_a_template_with_runs_on_one_side_only_is_explored_not_averaged_away():
    policy = LearnedPolicy(D)
    for n in range(MIN_RUNS):
        learn(policy, f"U{n}", "unmade", True, 3, 5)
        learn(policy, f"V{n}", "unmade", False, 5, 3)
    for _ in range(MIN_RUNS):
        learn(policy, "T", "unmade", True, 200, 10)  # this template only ever shown, and badly
    assert policy.estimate("T", "unmade") is None
    assert policy.estimate("new template", "unmade") is not None
    learn(policy, "T", "unmade", False, 10, 10)
    learn(policy, "T", "unmade", False, 11, 10)
    e = policy.estimate("T", "unmade")
    assert (e.n_shown, e.n_held) == (MIN_RUNS, 2) and e.shown > e.held


def test_a_shown_kind_is_shown_when_its_text_changes_and_decided_once_per_run():
    policy = LearnedPolicy(D)
    for n in range(MIN_RUNS):
        learn(policy, "T", "plan", True, 1, 9)
        learn(policy, "T", "plan", False, 9, 1)
    run = next(r for r in (policy.for_run(f"t~v{n}") for n in range(50)) if r.show("plan", "T", "s", "a"))
    assert not run.show("plan", "T", "s", "a")
    assert run.show("plan", "T", "s", "b")


def test_fewer_calls_do_not_count_when_the_work_is_not_done():
    policy = LearnedPolicy(Writes())
    for left, control in ((5, 30), (8, 35), (6, 28)):
        learn(policy, "T", "plan", True, left, control, writes=0, control_writes=3)  # stopped early
    for left, control in ((30, 30), (33, 31), (29, 30)):
        learn(policy, "T", "plan", False, left, control, writes=3, control_writes=3)
    e = policy.estimate("T", "plan")
    assert e.shown < e.held and e.writes == -3
    runs = [policy.for_run("t.a~v1", f"r{n}") for n in range(20)]
    assert not any(r.show("plan", "T", "s", "x") for r in runs)
    assert runs[0].decisions["plan"]["reason"] == "fewer writes"


def test_repeated_trials_of_a_task_draw_afresh():
    policy = LearnedPolicy(D)
    shown = {policy.for_run("t.a~v1", f"r{n}").show("plan", "T", "s", "x") for n in range(20)}
    assert shown == {True, False}
    assert policy.for_run("t.a~v1", "r1").show("plan", "T", "s", "x") == policy.for_run("t.a~v1", "r1").show("plan", "T", "s", "x")
