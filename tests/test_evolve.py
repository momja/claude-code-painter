"""The conductor on a toy problem: numbers that score against a partner, with niches."""

import pytest

from conveyor.claude import BudgetExhausted
from conveyor.evolve import Conductor
from conveyor.evolve import Evaluation
from conveyor.evolve import Mutator
from conveyor.evolve import Node
from conveyor.evolve import Organism
from conveyor.harness import SessionCutOff
from conveyor.harness import SessionFailed
from conveyor.store import Store

NICHES = ["low", "high"]


class Nudge(Mutator):
    name = "nudge"

    def __init__(self, step: float) -> None:
        self.step = step
        self.contexts = []

    def propose(self, ctx):
        self.contexts.append(ctx)
        x = ctx.parent.genome["x"] + self.step
        return [Organism(node=ctx.node, genome={"x": x}, summary=f"+{self.step}", niche="high" if x > 5 else "low")]


class Broken(Mutator):
    name = "broken"

    def propose(self, ctx):
        raise ValueError("no idea")


class CutOffOnce(Nudge):
    """A nudge whose first session a usage limit ends partway."""
    name = "cut-off-once"

    def __init__(self, step: float) -> None:
        super().__init__(step)
        self.cut = False

    def propose(self, ctx):
        if not self.cut:
            self.cut = True
            raise SessionCutOff("the claude usage limit ended a mutate session partway")
        return super().propose(ctx)


def cut_off_once(evaluate):
    """`evaluate`, but its first call is ended partway by a usage limit."""
    calls = []

    def wrapped(org, partner, reason, sample=0):
        calls.append(org.id)
        if len(calls) == 1:
            raise SessionCutOff("the claude usage limit ended a paint session partway")
        return evaluate(org, partner, reason, sample)
    return wrapped


def score_against(partner_node, luck=None):
    """x plus a tenth of the partner's x. `luck(org, sample)` adds noise to one evaluation."""
    def evaluate(org, partner, reason, sample=0):
        px = partner.genome["x"] if partner else 0
        noise = luck(org, sample) if luck else 0.0
        return Evaluation(organism_id=org.id, score=org.genome["x"] + 0.1 * px + noise,
                          partner_id=partner.id if partner else None, reason=reason)
    return evaluate


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "e.db")
    yield s
    s.close()


def nodes(a_mutators, b_mutators, luck=None, confirm=3):
    return [
        Node(name="a", seeds=[Organism(node="a", genome={"x": 1}, niche="low")], evaluate=score_against("b", luck),
             mutators=a_mutators, partner="b", archive=True, all_niches=NICHES, parents=2, confirm=confirm),
        Node(name="b", seeds=[Organism(node="b", genome={"x": 0})], evaluate=score_against("a"), mutators=b_mutators,
             partner="a", parents=1),
    ]


def test_champions_improve_and_partners_get_rescored(store):
    nudge = Nudge(3)
    conductor = Conductor(nodes([nudge], [Nudge(1)]), store, lanes=2)
    conductor.run(3)
    a, b = conductor.champions["a"], conductor.champions["b"]
    assert a.genome["x"] > 1 and b.genome["x"] > 0
    # b's champion was measured against a's current champion, not the seed it started with.
    assert conductor.pops["b"].evals[b.id].partner_id == a.id
    assert set(conductor.pops["a"].niches()) == {"low", "high"}
    # invent-style aiming: with one niche filled at the start, the first mutations were pointed at the empty one
    assert nudge.contexts[0].wanted_niche == "high" and nudge.contexts[0].empty_niches == ["high"]
    assert any(c.lineage for c in nudge.contexts)


def test_a_mutator_that_raises_is_recorded_not_fatal(store):
    conductor = Conductor(nodes([Broken()], [Nudge(1)]), store, lanes=1)
    conductor.run(2)
    assert conductor.champions["a"].genome["x"] == 1
    store.flush()
    import sqlite3

    errors = sqlite3.connect(store.path).execute(
        "SELECT count(*) FROM events WHERE kind='mutation' AND json_extract(data, '$.error') IS NOT NULL").fetchone()[0]
    assert errors == 4


def test_budget_exhaustion_stops_the_run(store):
    class Spendthrift(Mutator):
        name = "spend"

        def propose(self, ctx):
            raise BudgetExhausted("spent it all")

    conductor = Conductor(nodes([Spendthrift()], [Nudge(1)]), store, lanes=1)
    conductor.run(5)
    assert conductor.stopped and "spent it all" in conductor.stop_reason


def test_fixed_and_mutatorless_nodes_never_take_a_turn(store):
    conductor = Conductor(nodes([Nudge(1)], []), store, schedule=[("b", 1), ("a", 1)])
    assert conductor.schedule == [("a", 1)]


class Worse(Mutator):
    """Children a little worse than their parent, whose first evaluation is lucky enough to look better."""

    name = "worse"

    def propose(self, ctx):
        return [Organism(node=ctx.node, genome={"x": ctx.parent.genome["x"] - 1, "lucky": True}, niche="low")]


def lucky_first(org, sample):
    return 1.5 if org.genome.get("lucky") and sample == 0 else 0.0


def test_a_lucky_first_score_has_to_hold_up(store):
    conductor = Conductor(nodes([Worse()], [], luck=lucky_first), store, lanes=2)
    conductor.run(2)
    pop = conductor.pops["a"]
    # Every child scored 0.5 above the seed on its first painting and 1 below on its repeats; none took over.
    assert conductor.champions["a"].genome["x"] == 1
    challengers = [o for o in pop.organisms.values() if o.genome.get("lucky") and pop.count(o.id) == 3]
    assert challengers and all(abs(pop.evals[o.id].score - (o.genome["x"] + 0.5)) < 1e-9 for o in challengers)
    assert pop.evals[conductor.champions["a"].id].samples == 3  # the seed was painted again to defend its title


def test_with_confirm_off_a_lucky_score_wins(store):
    conductor = Conductor(nodes([Worse()], [], luck=lucky_first, confirm=1), store, lanes=2)
    conductor.run(1)
    assert conductor.champions["a"].genome.get("lucky")


def test_a_rescore_reuses_the_partners_repeats(store):
    calls = []

    def counting(partner_node):
        inner = score_against(partner_node)

        def evaluate(org, partner, reason, sample=0):
            calls.append((org.id, partner.id if partner else None, sample))
            return inner(org, partner, reason, sample)
        return evaluate

    ns = nodes([Nudge(3)], [Nudge(1)])
    ns[0].evaluate, ns[1].evaluate = counting("b"), counting("a")
    conductor = Conductor(ns, store, lanes=2)
    conductor.run(2)
    # Every evaluation asked for a sample number no earlier call used for the same organism and partner.
    assert len(calls) == len(set(calls))
    b = conductor.champions["b"]
    assert conductor.pops["b"].evals[b.id].samples == 3


def test_a_run_that_waits_out_limits_runs_cut_off_jobs_again(store):
    ns = nodes([CutOffOnce(3)], [Nudge(1)])
    ns[0].evaluate = cut_off_once(ns[0].evaluate)
    conductor = Conductor(ns, store, lanes=1, wait_out_limits=True)
    conductor.run(1)
    assert conductor.stop_reason is None
    store.flush()
    import sqlite3
    seed = ns[0].seeds[0]
    seed_evals = sqlite3.connect(store.path).execute(
        "SELECT score FROM evaluations WHERE organism_id=? AND reason='seed'", (seed.id,)).fetchall()
    assert seed_evals == [(1.0,)]  # the cut-off left no evaluation behind; the second try is the only one
    assert len(conductor.pops["a"].organisms) == 3  # the seed and both children, the cut-off one included


def test_without_waiting_a_cut_off_stops_the_run_and_scores_nothing(store):
    ns = nodes([Nudge(3)], [Nudge(1)])
    ns[0].evaluate = cut_off_once(ns[0].evaluate)
    conductor = Conductor(ns, store, lanes=1)
    conductor.run(2)
    assert conductor.stop_reason == "the claude usage limit ended a paint session partway"
    assert not conductor.pops["a"].evals


def fails(evaluate, when):
    """`evaluate`, but a session dies (a stall, a dropped stream) whenever `when(org, reason)`; counts every call."""
    calls = []

    def wrapped(org, partner, reason, sample=0):
        calls.append((org.id, reason))
        if when(org, reason):
            raise SessionFailed("the painting session ended after 0 of 10 actions: stalled")
        return evaluate(org, partner, reason, sample)
    wrapped.calls = calls
    return wrapped


def test_a_session_that_dies_is_run_again(store):
    ns = nodes([Nudge(3)], [Nudge(1)])
    seen = []
    ns[0].evaluate = fails(ns[0].evaluate, lambda org, reason: not seen.append(org.id) and len(seen) == 1)
    conductor = Conductor(ns, store, lanes=1)
    conductor.run(1)
    assert conductor.stop_reason is None
    store.flush()
    import sqlite3
    seed_evals = sqlite3.connect(store.path).execute(
        "SELECT score, viable FROM evaluations WHERE organism_id=? AND reason='seed'", (ns[0].seeds[0].id,)).fetchall()
    assert seed_evals == [(1.0, 1)]  # the failed try left nothing behind


def test_a_child_whose_sessions_keep_dying_is_not_marked_non_viable(store):
    ns = nodes([Nudge(3)], [Nudge(1)])
    ns[0].evaluate = fails(ns[0].evaluate, lambda org, reason: org.genome["x"] == 4)
    conductor = Conductor(ns, store, lanes=1)
    conductor.run(1)
    store.flush()
    import sqlite3
    db = sqlite3.connect(store.path)
    children = [o for o in conductor.pops["a"].organisms.values() if o.genome["x"] == 4]
    assert children
    for child in children:
        ev = conductor.pops["a"].evals[child.id]
        assert ev.inconclusive and not ev.viable
        [(viable,)] = db.execute("SELECT viable FROM organisms WHERE id=?", (child.id,)).fetchall()
        assert viable is None  # never evaluated, which is not the same as no good
    assert sum(1 for org_id, _ in ns[0].evaluate.calls if org_id == children[0].id) == 3  # once, then twice more
    [(n_bad, n_none)] = db.execute(
        "SELECT sum(json_extract(data, '$.n_nonviable')), sum(json_extract(data, '$.n_inconclusive')) "
        "FROM events WHERE kind='iteration'").fetchall()
    assert (n_bad, n_none) == (0, len(children))
    assert conductor.champions["a"].genome["x"] == 1  # the seed stands; nothing was lost to the failures


def test_a_rescore_whose_session_dies_leaves_the_champion_standing(store):
    ns = nodes([Nudge(3)], [Nudge(1)])
    ns[0].evaluate = fails(ns[0].evaluate, lambda org, reason: reason.startswith("rescore"))
    conductor = Conductor(ns, store, lanes=1)
    conductor.run(3)
    store.flush()
    import sqlite3
    assert any(reason.startswith("rescore") for _, reason in ns[0].evaluate.calls)  # b's champion did change
    assert conductor.champions["a"] is not None and conductor.pops["a"].champion() is not None
    bad = sqlite3.connect(store.path).execute("SELECT count(*) FROM organisms WHERE node='a' AND viable=0").fetchone()
    assert bad == (0,)


def test_an_inconclusive_evaluation_changes_no_standing():
    from conveyor.evolve import Population

    org = Organism(node="a", genome={"x": 1})
    pop = Population()
    pop.add(org, Evaluation(organism_id=org.id, score=2.0))
    pop.repeat(org.id, Evaluation(organism_id=org.id, score=4.0))
    unknown = Evaluation(organism_id=org.id, score=0.0, viable=False, inconclusive=True)
    pop.rescore(org.id, unknown)
    pop.repeat(org.id, unknown)
    assert pop.evals[org.id].score == 3.0 and pop.evals[org.id].viable and pop.count(org.id) == 2
    assert pop.champion()[0] is org
