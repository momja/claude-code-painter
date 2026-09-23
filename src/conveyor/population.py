from __future__ import annotations

from uuid import UUID

from darwinian_evolver.population import WeightedSamplingPopulation
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Organism


class RescorablePopulation(WeightedSamplingPopulation):
    """
    WeightedSamplingPopulation whose stored scores can be replaced.

    The base Population keeps each organism's first EvaluationResult forever. In a graph of nodes that is
    wrong as soon as a partner node's champion changes, because every stored score was measured against the
    old partner. The Conductor calls `rescore` on the top organisms after such a change.
    """

    def rescore(self, organism_id: UUID, result: EvaluationResult) -> EvaluationResult:
        organism, old = self._organisms_by_id[organism_id]
        for i, (o, _) in enumerate(self._organisms):
            if o.id == organism_id:
                self._organisms[i] = (organism, result)
                break
        self._organisms_by_id[organism_id] = (organism, result)
        return old

    def top(self, k: int) -> list[tuple[Organism, EvaluationResult]]:
        viable = [(o, r) for o, r in self._organisms if r.is_viable]
        return sorted(viable, key=lambda pair: pair[1].score, reverse=True)[:k]

    def result_for(self, organism_id: UUID) -> EvaluationResult | None:
        pair = self._organisms_by_id.get(organism_id)
        return pair[1] if pair else None
