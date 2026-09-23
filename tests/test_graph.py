"""Conductor behavior on a tiny two-node problem where each node's score depends on the other's champion."""

import json
import random

from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism

from conveyor.events import EventSink
from conveyor.events import connect
from conveyor.graph import Board
from conveyor.graph import Conductor
from conveyor.graph import Edge
from conveyor.graph import Node


class Num(Organism):
    value: float


Num.model_rebuild(_types_namespace={"EvaluationFailureCase": EvaluationFailureCase})


class CoupledEvaluator(Evaluator):
    """Score rewards a high value and agreement with the partner's champion."""

    def __init__(self, board: Board, partner: str) -> None:
        self.board = board
        self.partner = partner

    def evaluate(self, organism: Num) -> EvaluationResult:
        partner = self.board.champion(self.partner).value
        score = min(organism.value, 1.0) - 0.5 * abs(organism.value - partner)
        return EvaluationResult(
            score=score,
            trainable_failure_cases=[EvaluationFailureCase(data_point_id="x")],
            is_viable=organism.value < 5,
        )

    def evaluate_holdout(self, organism: Num) -> dict:
        return {"score": min(organism.value, 1.0) * 0.9}


class StepUp(Mutator):
    def mutate(self, organism, failure_cases, learning_log_entries):
        return [Num(value=organism.value + random.uniform(0.0, 0.3), from_change_summary="[up] step")]


class Broken(Mutator):
    def mutate(self, organism, failure_cases, learning_log_entries):
        raise ValueError("LLM returned garbage")


def _build(tmp_path):
    board = Board()
    nodes = [
        Node("a", initial_organism=Num(value=0.0), evaluator=CoupledEvaluator(board, "b"),
             mutators=[StepUp(), Broken()], partners=["b"], num_parents=2, rescore_top_k=2,
             mutator_concurrency=2, evaluator_concurrency=2),
        Node("b", initial_organism=Num(value=0.0), evaluator=CoupledEvaluator(board, "a"),
             mutators=[StepUp()], partners=["a"], num_parents=2, rescore_top_k=2,
             mutator_concurrency=2, evaluator_concurrency=2),
    ]
    sink = EventSink(tmp_path / "g.db")
    conductor = Conductor(nodes, [Edge("a", "b")], sink, board)
    return conductor, sink


def test_conductor_records_champions_rescores_and_mutator_errors(tmp_path):
    random.seed(0)
    conductor, sink = _build(tmp_path)
    conductor.run(cycles=4)
    sink.close()

    conn = connect(tmp_path / "g.db", readonly=True)
    kinds = {r[0]: r[1] for r in conn.execute("SELECT kind, count(*) FROM events GROUP BY kind")}
    assert kinds["graph"] == 1
    assert kinds["iteration"] == 8
    assert kinds["champion_changed"] > 2
    assert kinds["rescored"] > 0

    errors = [json.loads(r[0]) for r in conn.execute("SELECT data FROM events WHERE kind='mutate_call'")]
    assert any(e.get("error", "").startswith("ValueError") for e in errors)

    # Every evaluation records which partner versions it ran against.
    for (data,) in conn.execute("SELECT data FROM events WHERE kind='evaluated'"):
        partners = json.loads(data)["partners"]
        assert set(partners) in ({"a"}, {"b"})

    # A rescore replaced the stored score in the population.
    rescored = [(r[0], json.loads(r[1])) for r in conn.execute("SELECT organism_id, data FROM events WHERE kind='rescored'")]
    oid, data = rescored[-1]
    node = conn.execute("SELECT node FROM events WHERE kind='rescored' AND organism_id=?", (oid,)).fetchone()[0]
    stored = conductor.populations[node].result_for(next(o.id for o, _ in conductor.populations[node].organisms if str(o.id) == oid))
    latest_eval = json.loads(conn.execute(
        "SELECT data FROM events WHERE kind='evaluated' AND organism_id=? ORDER BY seq DESC LIMIT 1", (oid,)
    ).fetchone()[0])
    assert stored.score == latest_eval["score"]


def test_holdout_runs_on_the_first_champion_then_every_nth(tmp_path):
    random.seed(2)
    conductor, sink = _build(tmp_path)
    conductor.nodes["a"].holdout_every = 3  # a: the first champion, then every third
    conductor.run(cycles=4)
    sink.close()
    conn = connect(tmp_path / "g.db", readonly=True)

    def ids(node, kind):
        return [r[0] for r in conn.execute(
            "SELECT organism_id FROM events WHERE node=? AND kind=? ORDER BY seq", (node, kind))]

    assert ids("a", "holdout") == ids("a", "champion_changed")[::3]
    assert ids("b", "holdout") == ids("b", "champion_changed")  # every champion by default


def test_champion_on_board_matches_population_best(tmp_path):
    random.seed(1)
    conductor, sink = _build(tmp_path)
    conductor.run(cycles=3)
    sink.close()
    for name, pop in conductor.populations.items():
        assert conductor.board.champion(name).id == pop.top(1)[0][0].id
