"""
Wrappers that record what darwinian_evolver does without forking it.

Evolver only reaches problem code through `mutator.mutate()` and `evaluator.evaluate()` /
`evaluator.verify_mutation()`, so wrapping those three calls is enough to see every organism,
every score, and every tool call made during an evaluation.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any
from typing import Callable

from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import MutatorContext
from darwinian_evolver.problem import Organism

from conveyor.events import EventSink
from conveyor.events import observing

BASE_FIELDS = {
    "id",
    "parent",
    "additional_parents",
    "from_failure_cases",
    "from_learning_log_entries",
    "from_change_summary",
    "visualizer_props",
}
MAX_LOGGED_FAILURES = 16


def genome(organism: Organism) -> dict:
    """The problem-specific fields of an organism."""
    return organism.model_dump(exclude=BASE_FIELDS, mode="json")


def render_text(organism: Organism) -> str:
    """Human-readable form used for diffs. Organisms can define `render_text()` to control it."""
    fn = getattr(organism, "render_text", None)
    if callable(fn):
        return fn()
    return json.dumps(genome(organism), indent=2, sort_keys=True)


def generation(organism: Organism) -> int:
    n = 0
    current = organism.parent
    while current is not None:
        n += 1
        current = current.parent
    return n


def _failure_dump(failure: EvaluationFailureCase) -> dict:
    return failure.model_dump(mode="json")


def emit_organism(
    sink: EventSink,
    node: str,
    organism: Organism,
    mutator: str | None,
    parent_score: float | None = None,
    llm_call_ids: list[str] | None = None,
) -> None:
    failures = organism.from_failure_cases or []
    sink.emit(
        node,
        "organism",
        organism.id,
        parent_id=str(organism.parent.id) if organism.parent else None,
        additional_parent_ids=[str(p.id) for p in organism.additional_parents],
        generation=generation(organism),
        mutator=mutator,
        failure_type=failures[0].failure_type if failures else None,
        failure_ids=[f.data_point_id for f in failures],
        change_summary=organism.from_change_summary,
        parent_score=parent_score,
        llm_call_ids=llm_call_ids or [],
        genome=genome(organism),
        text=render_text(organism),
    )


class ObservedMutator(Mutator):
    """
    Records each mutate() call and every child it produces.

    One behavior change from calling the inner mutator directly: an exception is recorded and turned
    into "no children" instead of aborting the whole Evolver iteration. Unreliable mutators are normal
    in this setup and one bad LLM response should not kill a run.
    """

    def __init__(self, inner: Mutator, node: str, sink: EventSink) -> None:
        super().__init__()
        self.inner = inner
        self.node = node
        self.sink = sink
        self.name = type(inner).__name__

    def set_context(self, context: MutatorContext) -> None:
        super().set_context(context)
        self.inner.set_context(context)

    @property
    def supports_batch_mutation(self) -> bool:
        return self.inner.supports_batch_mutation

    def mutate(
        self,
        organism: Organism,
        failure_cases: list[EvaluationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[Organism]:
        parent_score = self._parent_score(organism)
        failure_type = failure_cases[0].failure_type if failure_cases else None
        started = time.perf_counter()
        llm_calls: list[str] = []  # the LLM client appends the id of every model call made inside this mutate()
        try:
            with observing(node=self.node, mutator=self.name, organism_id=str(organism.id), llm_calls=llm_calls):
                children = self.inner.mutate(organism, failure_cases, learning_log_entries)
        except Exception as e:  # noqa: BLE001
            self.sink.emit(
                self.node,
                "mutate_call",
                organism.id,
                mutator=self.name,
                failure_type=failure_type,
                n_failures=len(failure_cases),
                n_learning_log=len(learning_log_entries),
                n_children=0,
                duration=time.perf_counter() - started,
                llm_call_ids=llm_calls,
                error=f"{type(e).__name__}: {e}",
            )
            return []

        duration = time.perf_counter() - started
        for child in children:
            # Evolver fills these in after mutate() returns. Fill them now so the recorded event is complete.
            if child.parent is None:
                child.parent = organism
            if child.from_failure_cases is None:
                child.from_failure_cases = list(failure_cases)
            if child.from_learning_log_entries is None:
                child.from_learning_log_entries = learning_log_entries
            emit_organism(self.sink, self.node, child, self.name, parent_score, llm_call_ids=llm_calls)

        self.sink.emit(
            self.node,
            "mutate_call",
            organism.id,
            mutator=self.name,
            failure_type=failure_type,
            n_failures=len(failure_cases),
            n_learning_log=len(learning_log_entries),
            n_children=len(children),
            child_ids=[str(c.id) for c in children],
            llm_call_ids=llm_calls,
            duration=duration,
        )
        return children

    def _parent_score(self, organism: Organism) -> float | None:
        if self._context is None:
            return None
        population = self._context.population
        result_for = getattr(population, "result_for", None)
        if result_for is not None:
            result = result_for(organism.id)
            return result.score if result else None
        for o, r in population.organisms:
            if o.id == organism.id:
                return r.score
        return None


class ObservedEvaluator(Evaluator):
    """
    Records every evaluation: score, sub-scores, failure cases, artifacts, the partner versions it was
    measured against, and a trace of tool calls made inside it.
    """

    def __init__(
        self,
        inner: Evaluator,
        node: str,
        sink: EventSink,
        partners: Callable[[], dict[str, list[str]]] | None = None,
    ) -> None:
        self.inner = inner
        self.node = node
        self.sink = sink
        self.partners = partners or (lambda: {})

    def evaluate(self, organism: Organism) -> EvaluationResult:
        return self.evaluate_observed(organism, reason="new")

    def evaluate_observed(self, organism: Organism, reason: str) -> EvaluationResult:
        partners = self.partners()
        started = time.perf_counter()
        error: str | None = None
        with self.sink.trace() as trace, observing(
            node=self.node, mutator=None, organism_id=str(organism.id), llm_calls=None
        ):
            try:
                result = self.inner.evaluate(organism)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                result = EvaluationResult(score=0.0, trainable_failure_cases=[], is_viable=False)

        counts = Counter(f.failure_type for f in result.trainable_failure_cases)
        data: dict[str, Any] = dict(
            reason=reason,
            score=result.score,
            viable=result.is_viable,
            sub_scores=dict(result.visualizer_props),
            failure_counts=dict(counts),
            n_trainable=len(result.trainable_failure_cases),
            n_holdout=len(result.holdout_failure_cases),
            failures=[_failure_dump(f) for f in result.trainable_failure_cases[:MAX_LOGGED_FAILURES]],
            holdout_failures=[_failure_dump(f) for f in result.holdout_failure_cases[:MAX_LOGGED_FAILURES]],
            artifacts=dict(getattr(result, "artifacts", {}) or {}),
            partners=partners,
            trace_id=trace.id,
            n_spans=trace.num_spans,
            duration=time.perf_counter() - started,
        )
        if error:
            data["error"] = error
        notes = getattr(result, "notes", None)
        if notes:
            data["notes"] = list(notes)
        details = getattr(result, "details", None)
        if details:
            data["details"] = details
        self.sink.emit(self.node, "evaluated", organism.id, **data)
        return result

    def evaluate_holdout(self, organism: Organism) -> dict | None:
        """
        The held-out check, for evaluators that define `evaluate_holdout`. Recorded as a `holdout` event with its
        own trace; the organism's stored score is left alone, since held-out targets never count toward it.
        """
        fn = getattr(self.inner, "evaluate_holdout", None)
        if fn is None:
            return None
        partners = self.partners()
        started = time.perf_counter()
        error: str | None = None
        with self.sink.trace() as trace, observing(
            node=self.node, mutator=None, organism_id=str(organism.id), llm_calls=None
        ):
            try:
                data = dict(fn(organism) or {})
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                data = {"score": None}
        data.update(partners=partners, trace_id=trace.id, n_spans=trace.num_spans,
                    duration=time.perf_counter() - started)
        if error:
            data["error"] = error
        self.sink.emit(self.node, "holdout", organism.id, **data)
        return data

    def verify_mutation(self, organism: Organism) -> bool:
        started = time.perf_counter()
        with observing(node=self.node, mutator=None, organism_id=str(organism.id), llm_calls=None):
            passed = self.inner.verify_mutation(organism)
        self.sink.emit(
            self.node,
            "verified",
            organism.id,
            passed=bool(passed),
            parent_id=str(organism.parent.id) if organism.parent else None,
            duration=time.perf_counter() - started,
        )
        return passed

    def set_output_dir(self, output_dir: str) -> None:
        super().set_output_dir(output_dir)
        self.inner.set_output_dir(output_dir)
