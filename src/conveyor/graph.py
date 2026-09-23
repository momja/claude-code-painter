"""
A graph of evolver nodes.

Each evolving node owns a darwinian_evolver Evolver and its own population. Nodes read each other's
champions through a shared Board. The Conductor steps nodes on a schedule, notices when a champion
changes, and rescores the nodes that depend on it, because their stored scores were measured against
the old champion.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from darwinian_evolver.evolver import Evolver
from darwinian_evolver.learning_log_view import AncestorLearningLogView
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import MutatorContext
from darwinian_evolver.problem import Organism

from conveyor.events import EventSink
from conveyor.observe import ObservedEvaluator
from conveyor.observe import ObservedMutator
from conveyor.observe import emit_organism
from conveyor.population import RescorablePopulation

PERCENTILES = [0.0, 25.0, 50.0, 75.0, 90.0, 100.0]


@dataclass
class Node:
    name: str
    description: str = ""
    initial_organism: Organism | None = None
    evaluator: Evaluator | None = None
    mutators: list[Mutator] = field(default_factory=list)
    # Nodes whose champions this node's evaluator runs against. A champion change there triggers a rescore here.
    partners: list[str] = field(default_factory=list)
    # How many elites of each partner the evaluator uses. Must match what the evaluator actually reads.
    partner_k: int = 1
    # Fixed nodes (a critic, a simulator) sit in the pipeline and show up in the graph but do not evolve.
    fixed: bool = False
    # Run the held-out check on every Nth new champion. Champions change often early, and each check costs a
    # full evaluation, so 1 is right for cheap evaluators and 3 or more for an LLM painter.
    holdout_every: int = 1
    version: str = "v1"
    # For fixed nodes: show the thumbnail artifact from this node's champion evaluation.
    mirror: str | None = None
    thumbnail_artifact: str = "thumb"

    num_parents: int = 3
    rescore_top_k: int = 4
    elite_k: int = 3
    batch_size: int = 3
    verify_mutations: bool = False
    sharpness: float = 10.0
    midpoint_score_percentile: float = 75.0
    novelty_weight: float = 1.0
    learning_log: tuple[type, dict[str, Any]] = (AncestorLearningLogView, {"max_depth": 4})
    mutator_concurrency: int = 6
    evaluator_concurrency: int = 6


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "artifact"  # "artifact" flows forward through the pipeline, "feedback" carries failures back
    label: str = ""


class Board:
    """Current elites per node. Evaluators read partners from here."""

    def __init__(self) -> None:
        self._elites: dict[str, list[Organism]] = {}
        self._lock = threading.Lock()

    def set(self, name: str, elites: list[Organism]) -> None:
        with self._lock:
            self._elites[name] = list(elites)

    def champion(self, name: str) -> Organism:
        return self.elites(name, 1)[0]

    def elites(self, name: str, k: int) -> list[Organism]:
        with self._lock:
            elites = self._elites.get(name)
        if not elites:
            raise KeyError(f"No champion for node {name!r} yet")
        return elites[:k]

    def versions(self, names: list[str], k: int) -> dict[str, list[str]]:
        return {name: [str(o.id) for o in self.elites(name, k)] for name in names}


class Conductor:
    def __init__(
        self,
        nodes: list[Node],
        edges: list[Edge],
        sink: EventSink,
        board: Board,
        schedule: list[tuple[str, int]] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.log = log or (lambda message: None)
        self.nodes = {n.name: n for n in nodes}
        self.edges = edges
        self.sink = sink
        self.board = board
        self.schedule = schedule or [(n.name, 1) for n in nodes if not n.fixed]
        self.populations: dict[str, RescorablePopulation] = {}
        self.evolvers: dict[str, Evolver] = {}
        self.evaluators: dict[str, ObservedEvaluator] = {}
        self.iterations: dict[str, int] = defaultdict(int)
        self.champion_ids: dict[str, Any] = {}
        self._stop = threading.Event()
        self._ready = False
        self._holdout_done: set[tuple[str, Any]] = set()
        self._champions_seen: dict[str, int] = defaultdict(int)

    @property
    def evolving(self) -> list[Node]:
        return [n for n in self.nodes.values() if not n.fixed]

    def stop(self) -> None:
        self._stop.set()

    def setup(self) -> None:
        self.sink.emit(
            None,
            "graph",
            nodes=[
                dict(
                    name=n.name,
                    description=n.description,
                    fixed=n.fixed,
                    version=n.version,
                    partners=n.partners,
                    partner_k=n.partner_k,
                    mirror=n.mirror,
                    thumbnail_artifact=n.thumbnail_artifact,
                    mutators=[type(m).__name__ for m in n.mutators],
                    num_parents=n.num_parents,
                    verify_mutations=n.verify_mutations,
                )
                for n in self.nodes.values()
            ],
            edges=[e.__dict__ for e in self.edges],
            schedule=self.schedule,
        )

        # Seed the board with the unevaluated initial organisms so evaluators can find partners.
        for node in self.evolving:
            assert node.initial_organism is not None and node.evaluator is not None, node.name
            self.board.set(node.name, [node.initial_organism])

        for node in self.evolving:
            self.evaluators[node.name] = ObservedEvaluator(
                node.evaluator,
                node.name,
                self.sink,
                partners=lambda n=node: self.board.versions(n.partners, n.partner_k),
            )

        for node in self.evolving:
            self.sink.set_iteration(node.name, 0)
            self.sink.emit(node.name, "node_status", status="evaluating initial organism")
            emit_organism(self.sink, node.name, node.initial_organism, mutator=None)
            result = self.evaluators[node.name].evaluate_observed(node.initial_organism, reason="initial")
            if not result.is_viable:
                raise ValueError(f"Initial organism of node {node.name!r} is not viable")
            population = RescorablePopulation(
                node.initial_organism,
                result,
                sharpness=node.sharpness,
                midpoint_score_percentile=node.midpoint_score_percentile,
                novelty_weight=node.novelty_weight,
            )
            mutators = [ObservedMutator(m, node.name, self.sink) for m in node.mutators]
            for m in mutators:
                m.set_context(MutatorContext(population=population))
            self.populations[node.name] = population
            self.evolvers[node.name] = Evolver(
                population=population,
                mutators=mutators,
                evaluator=self.evaluators[node.name],
                learning_log_view_type=node.learning_log,
                mutator_concurrency=node.mutator_concurrency,
                evaluator_concurrency=node.evaluator_concurrency,
                batch_size=node.batch_size,
                should_verify_mutations=node.verify_mutations,
            )
            self.board.set(node.name, [o for o, _ in population.top(node.elite_k)])
            self.champion_ids[node.name] = node.initial_organism.id
            self.sink.emit(
                node.name,
                "champion_changed",
                node.initial_organism.id,
                old_id=None,
                old_score=None,
                new_score=result.score,
                reason="initial",
            )
            self._holdout_check(node.name, node.initial_organism, status_after="idle")
            self.sink.emit(node.name, "node_status", status="idle")
        self._ready = True

    def run(self, cycles: int) -> None:
        if not self._ready:
            self.setup()
        self.sink.emit(None, "run_started", cycles=cycles)
        for cycle in range(cycles):
            if self._stop.is_set():
                break
            self.sink.emit(None, "cycle", cycle=cycle, of=cycles)
            self.log(f"cycle {cycle + 1}/{cycles}")
            for name, n in self.schedule:
                if self._stop.is_set():
                    break
                self._step(name, n)
        self.sink.emit(None, "run_finished", stopped_early=self._stop.is_set())
        self.sink.flush()

    def _step(self, name: str, n: int) -> None:
        node = self.nodes[name]
        population = self.populations[name]
        self.sink.emit(name, "node_status", status="evolving")
        for _ in range(n):
            iteration = self.iterations[name] + 1
            self.sink.set_iteration(name, iteration)
            before = len(population.organisms)
            try:
                stats = self.evolvers[name].evolve_iteration(node.num_parents, iteration=iteration)
            except RuntimeError as e:
                self.sink.emit(name, "iteration_error", error=str(e))
                break
            self.iterations[name] = iteration
            new = population.organisms[before:]
            best_org, best_result = population.top(1)[0]
            self.sink.emit(
                name,
                "iteration",
                stats=stats.model_dump(),
                percentiles={str(int(k)): v for k, v in population.get_score_percentiles(PERCENTILES).items()},
                population_size=len(population.organisms),
                n_new=len(new),
                n_nonviable=sum(1 for _, r in new if not r.is_viable),
                best_id=str(best_org.id),
                best_score=best_result.score,
            )
            self._check_champion(name, cascade=True, reason="evolution")
        self.sink.emit(name, "node_status", status="idle")

    def _check_champion(self, name: str, cascade: bool, reason: str) -> None:
        node = self.nodes[name]
        population = self.populations[name]
        top = population.top(node.elite_k)
        if not top:
            return
        self.board.set(name, [o for o, _ in top])
        best_org, best_result = top[0]
        old_id = self.champion_ids.get(name)
        if best_org.id == old_id:
            return
        old_result = population.result_for(old_id) if old_id else None
        self.champion_ids[name] = best_org.id
        self.sink.emit(
            name,
            "champion_changed",
            best_org.id,
            old_id=str(old_id) if old_id else None,
            old_score=old_result.score if old_result else None,
            new_score=best_result.score,
            reason=reason,
        )
        old_text = f"{old_result.score:.3f}" if old_result else "none"
        self.log(f"  {name}: new champion {str(best_org.id)[:8]} {old_text} -> {best_result.score:.3f} ({reason})")
        self._holdout_check(name, best_org, status_after="evolving" if cascade else "idle")
        if cascade:
            for dependent in self.evolving:
                if name in dependent.partners:
                    self._rescore(dependent.name, because=name)

    def _holdout_check(self, name: str, organism: Organism, status_after: str) -> None:
        """
        Paint the held-out targets for a new champion, once per organism. Only champions feed the overfitting
        alarm, so doing it on every evaluation would double the cost of evolution for nothing.
        """
        key = (name, organism.id)
        evaluator = self.evaluators[name]
        if key in self._holdout_done or getattr(evaluator.inner, "evaluate_holdout", None) is None:
            return
        seen = self._champions_seen[name]
        self._champions_seen[name] += 1
        if seen % max(1, self.nodes[name].holdout_every):  # the first champion, then every Nth
            return
        self._holdout_done.add(key)
        self.sink.emit(name, "node_status", status="checking the held-out painting")
        data = evaluator.evaluate_holdout(organism)
        self.sink.emit(name, "node_status", status=status_after)
        if data and data.get("score") is not None:
            self.log(f"  {name}: champion {str(organism.id)[:8]} scored {data['score']:.3f} on the held-out painting")

    def _rescore(self, name: str, because: str) -> None:
        """Re-evaluate the top of `name`'s population against `because`'s new champion."""
        node = self.nodes[name]
        population = self.populations[name]
        evaluator = self.evaluators[name]
        self.sink.set_iteration(name, self.iterations[name])
        self.sink.emit(name, "node_status", status="rescoring", because=because)
        partner_id = str(self.champion_ids[because])
        fresh: set = set()

        def rescore_batch(pairs: list[tuple[Organism, EvaluationResult]]) -> None:
            with ThreadPoolExecutor(max_workers=node.evaluator_concurrency) as pool:
                futures = [
                    (org, old, pool.submit(evaluator.evaluate_observed, org, f"rescore after {because} changed"))
                    for org, old in pairs
                ]
                for org, old, future in futures:
                    new = future.result()
                    population.rescore(org.id, new)
                    fresh.add(org.id)
                    self.sink.emit(
                        name,
                        "rescored",
                        org.id,
                        old_score=old.score,
                        new_score=new.score,
                        because=because,
                        partner_id=partner_id,
                    )

        rescore_batch(population.top(node.rescore_top_k))
        # Everything outside the rescored set still carries a stale score. If one of those now sits on top,
        # rescore it too, so the champion we publish was measured against the current partners.
        for _ in range(node.rescore_top_k):
            best = population.top(1)
            if not best or best[0][0].id in fresh:
                break
            rescore_batch(best)

        self.sink.emit(name, "node_status", status="idle")
        self._check_champion(name, cascade=False, reason=f"rescore after {because} changed")
