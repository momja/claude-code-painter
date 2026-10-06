"""
Co-evolution of nodes that score each other.

Each node owns a population. An organism is scored against the current champion of its partner node, so a
score is only as fresh as the partner it was measured against. When a node's champion changes, the conductor
re-evaluates the top of every node that depends on it.

One evaluation can be noisy (a painting is one run of a model), and the best of many noisy scores is mostly a
lucky one. So an organism's score is the mean of its evaluations against the current partner, and a champion is
only replaced after the challenger and the champion have each been evaluated `confirm` times. A challenger that
loses on its first evaluation costs nothing extra; the repeats go only to organisms that look like winners.

Parent selection is where variance is won or lost. A node with `archive=True` keeps the best organism in each
niche (a behaviour descriptor the problem computes) and samples parents across niches rather than by score, so
an organism that is different but not yet better still gets children. Mutators are told which niches are empty,
and an `invent` mutator is aimed at one of them. A plain node samples parents by rank among its best.

A run that waits out usage limits (`wait_out_limits`) runs a mutation or an evaluation again when the limit ended
its session partway. The harness holds the new session until the limit resets.
"""

from __future__ import annotations

import random
import threading
import time
import traceback
from collections.abc import Callable
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any

from conveyor.claude import BudgetExhausted
from conveyor.claude import RateLimited
from conveyor.harness import SessionCutOff
from conveyor.harness import SessionFailed
from conveyor.store import Store
from conveyor.store import new_id

CUT_OFF_RETRIES = 3  # times one job runs again after a usage limit ends its session partway
SESSION_RETRIES = 2  # times one job runs again after its session is cut off by a stall, a crash or a timeout


@dataclass
class Organism:
    node: str
    genome: dict
    id: str = field(default_factory=new_id)
    parent_id: str | None = None
    parent2_id: str | None = None
    mutator: str | None = None
    summary: str = ""
    niche: str | None = None
    traits: dict = field(default_factory=dict)
    sheet: str | None = None  # artifact name of a picture of the organism (an instrument's demo sheet)
    note: str = ""  # why it isn't viable, when it isn't
    viable: bool = True
    session_id: str | None = None  # the Claude session that wrote it
    created: float = field(default_factory=time.time)


@dataclass
class Evaluation:
    organism_id: str
    score: float
    viable: bool = True
    partner_id: str | None = None
    reason: str = "new"
    details: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    session_id: str | None = None
    error: str | None = None
    feedback: dict = field(default_factory=dict)  # what a mutator gets to see: images, notes, worst regions
    # The sessions behind it kept failing, so nothing was learned about the organism. Unlike a non-viable result it
    # leaves the organism's viability and standing alone.
    inconclusive: bool = False
    id: str = field(default_factory=new_id)
    started: float = field(default_factory=time.time)
    ended: float | None = None
    samples: int = 1  # how many evaluations `score` is the mean of


@dataclass
class Context:
    """What a mutator is given for one mutation."""

    node: str
    parent: Organism
    parent_eval: Evaluation | None
    partner: Organism | None
    lineage: list[dict]  # the parent's ancestry, newest first: what each change was and how it scored
    niches: dict[str, tuple[Organism, Evaluation]]  # the archive: best organism per niche
    empty_niches: list[str]
    wanted_niche: str | None = None
    other: Organism | None = None  # second parent, for recombination
    other_eval: Evaluation | None = None


class Mutator:
    name = "mutator"
    weight = 1.0
    needs_other = False  # recombination needs a second parent from a different niche

    def propose(self, ctx: Context) -> list[Organism]:
        raise NotImplementedError


@dataclass
class Node:
    name: str
    seeds: list[Organism]
    # (organism, partner champion, reason, sample): sample n is the organism's nth evaluation against this partner.
    # An evaluator may hand back one it already has for that sample, which is how two nodes share a measurement.
    evaluate: Callable[[Organism, Organism | None, str, int], Evaluation]
    mutators: list[Mutator] = field(default_factory=list)
    partner: str | None = None
    description: str = ""
    archive: bool = False  # niche-based parent selection
    all_niches: list[str] = field(default_factory=list)
    niche_distance: Callable[[str | None, str], float] | None = None  # weights which niche an invention aims at
    fixed: bool = False
    parents: int = 2  # mutations per iteration
    rescore_k: int = 1
    champion_share: float = 0.3  # archive nodes: share of parents drawn from the champion rather than the niches
    confirm: int = 3  # evaluations a challenger and the champion each stand on before the champion changes; 1 is off


class Population:
    def __init__(self) -> None:
        self.organisms: dict[str, Organism] = {}
        # Each organism's standing: the mean of its viable evaluations against the current partner, carrying the
        # details and feedback of the latest one. A new partner starts a new tally.
        self.evals: dict[str, Evaluation] = {}
        self.samples: dict[str, list[Evaluation]] = {}
        self.tries: dict[str, int] = {}  # sample numbers used against the current partner, viable or not
        self._lock = threading.Lock()

    def add(self, org: Organism, ev: Evaluation) -> None:
        with self._lock:
            self.organisms[org.id] = org
            self._tally(org.id, [ev], tries=1)

    def rescore(self, org_id: str, ev: Evaluation) -> None:
        """A new standing against a new partner. An inconclusive evaluation leaves the old standing as it was."""
        if ev.inconclusive:
            return
        with self._lock:
            self._tally(org_id, [ev], tries=1)

    def repeat(self, org_id: str, ev: Evaluation) -> None:
        """One more evaluation against the same partner. A failed one leaves the standing as it was."""
        with self._lock:
            self._tally(org_id, self.samples.get(org_id, []) + [ev], tries=self.tries.get(org_id, 0))

    def claim(self, org_id: str, n: int) -> list[int]:
        """The next `n` sample numbers for this organism, so parallel repeats never ask for the same one."""
        with self._lock:
            start = self.tries.get(org_id, 0)
            self.tries[org_id] = start + n
            return list(range(start, start + n))

    def count(self, org_id: str) -> int:
        return len(self.samples.get(org_id, []))

    def _tally(self, org_id: str, evs: list[Evaluation], tries: int) -> None:
        viable = [e for e in evs if e.viable]
        self.tries[org_id] = tries
        self.samples[org_id] = viable
        if viable:
            self.evals[org_id] = replace(viable[-1], score=sum(e.score for e in viable) / len(viable),
                                         samples=len(viable))
        else:
            self.evals[org_id] = evs[-1]

    def ranked(self) -> list[tuple[Organism, Evaluation]]:
        with self._lock:
            pairs = [(o, self.evals[o.id]) for o in self.organisms.values() if self.evals[o.id].viable]
        return sorted(pairs, key=lambda p: p[1].score, reverse=True)

    def champion(self) -> tuple[Organism, Evaluation] | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def niches(self) -> dict[str, tuple[Organism, Evaluation]]:
        best: dict[str, tuple[Organism, Evaluation]] = {}
        for o, e in self.ranked():
            if o.niche and o.niche not in best:
                best[o.niche] = (o, e)
        return best

    def lineage(self, org: Organism, depth: int = 5) -> list[dict]:
        """The organism's ancestry, newest first, as a learning log: each change and what it did to the score."""
        out = []
        current: Organism | None = org
        while current is not None and len(out) < depth:
            parent = self.organisms.get(current.parent_id) if current.parent_id else None
            ev, pev = self.evals.get(current.id), self.evals.get(parent.id) if parent else None
            if current.summary:
                out.append({"id": current.id, "mutator": current.mutator, "summary": current.summary,
                            "niche": current.niche, "parent_niche": parent.niche if parent else None,
                            "score": ev.score if ev and ev.viable else None, "viable": bool(ev and ev.viable),
                            "note": current.note, "parent_score": pev.score if pev else None})
            current = parent
        return out


class Conductor:
    def __init__(self, nodes: list[Node], store: Store, *, lanes: int = 2, schedule: list[tuple[str, int]] | None = None,
                 log: Callable[[str], None] | None = None, on_stop: Callable[[], None] | None = None,
                 should_stop: Callable[[], bool | str | None] | None = None, wait_out_limits: bool = False) -> None:
        """`should_stop` is asked between steps; a string it returns is the reason for stopping."""
        self.nodes = {n.name: n for n in nodes}
        self.store = store
        self.lanes = max(1, lanes)
        # A fixed node, or one with nothing to mutate it, sits in the graph but never takes a turn.
        evolving = {n.name for n in nodes if not n.fixed and n.mutators}
        self.schedule = [(name, k) for name, k in (schedule or [(n, 1) for n in evolving]) if name in evolving]
        self.log = log or (lambda m: None)
        self.on_stop = on_stop
        self.should_stop = should_stop or (lambda: False)
        self.wait_out_limits = wait_out_limits
        self.pops: dict[str, Population] = {n.name: Population() for n in nodes}
        self.champions: dict[str, Organism] = {}
        self.iterations: dict[str, int] = {n.name: 0 for n in nodes}
        self._stop = threading.Event()
        self.stop_reason: str | None = None
        self._rng = random.Random()

    # ---- control ------------------------------------------------------------------------------------------

    def stop(self, reason: str = "stopped") -> None:
        if not self._stop.is_set():
            self.stop_reason = reason
            self._stop.set()
            self.store.emit("stopping", reason=reason)
            self.log(f"Stopping: {reason}")
            if self.on_stop:
                self.on_stop()

    @property
    def stopped(self) -> bool:
        if not self._stop.is_set():
            reason = self.should_stop()
            if reason:
                self.stop(reason if isinstance(reason, str) else "budget reached")
        return self._stop.is_set()

    @contextmanager
    def _pool(self) -> Iterator[ThreadPoolExecutor]:
        """Threads for one batch of sessions. Ctrl+C stops the run before the pool joins its threads, so one that
        is waiting out a usage limit lets go instead of holding the exit until the limit resets."""
        with ThreadPoolExecutor(max_workers=self.lanes) as pool:
            try:
                yield pool
            except KeyboardInterrupt:
                self.stop("interrupted")
                raise

    def _patiently(self, what: str, fn: Callable[[], Any]) -> Any:
        """`fn()`, run again when its session was cut off: by a usage limit (if this run waits out limits), or by
        a stall, a crash or a timeout (twice at most; after that the failure goes to the caller)."""
        cut_offs = failures = 0
        while True:
            try:
                return fn()
            except SessionCutOff as e:
                if not self.wait_out_limits or cut_offs == CUT_OFF_RETRIES or self.stopped:
                    raise
                cut_offs += 1
                self.log(f"  {what}: {e}; it runs again once the limit resets")
            except SessionFailed as e:
                if failures == SESSION_RETRIES or self.stopped:
                    raise
                failures += 1
                self.log(f"  {what}: {e}; trying again ({failures}/{SESSION_RETRIES})")

    def partner_of(self, node: Node) -> Organism | None:
        return self.champions.get(node.partner) if node.partner else None

    # ---- recording ----------------------------------------------------------------------------------------

    def _record_organism(self, org: Organism) -> None:
        self.store.organism({
            "id": org.id, "node": org.node, "parent_id": org.parent_id, "parent2_id": org.parent2_id,
            "mutator": org.mutator, "created": org.created, "summary": org.summary, "genome": org.genome,
            "text": org.genome.get("text") or org.genome.get("source") or org.genome.get("prompt"),
            "niche": org.niche, "traits": org.traits, "sheet": org.sheet, "session_id": org.session_id,
            "viable": None, "note": org.note,
        })

    def _record_eval(self, node: str, ev: Evaluation) -> None:
        self.store.evaluation({
            "id": ev.id, "node": node, "organism_id": ev.organism_id, "partner_id": ev.partner_id, "reason": ev.reason,
            "score": ev.score, "viable": int(ev.viable), "started": ev.started, "ended": ev.ended or time.time(),
            "details": ev.details, "artifacts": ev.artifacts, "session_id": ev.session_id, "error": ev.error,
        })
        if not ev.inconclusive:
            self.store.update_organism(ev.organism_id, viable=int(ev.viable))

    def _status(self, node: str, status: str, **extra: Any) -> None:
        self.store.emit("node_status", node=node, status=status, **extra)

    # ---- evaluating ---------------------------------------------------------------------------------------

    def _evaluate(self, node: Node, org: Organism, reason: str, sample: int = 0) -> Evaluation:
        partner = self.partner_of(node)
        if not org.viable:
            ev = Evaluation(organism_id=org.id, score=0.0, viable=False, reason=reason,
                            partner_id=partner.id if partner else None, error=org.note)
        else:
            try:
                ev = self._patiently(f"{node.name} {org.id[:6]} {reason}",
                                     lambda: node.evaluate(org, partner, reason, sample))
            except (BudgetExhausted, RateLimited):
                raise
            except SessionFailed as e:
                self.log(f"  {node.name} {org.id[:6]} ({reason}): no result, {e}")
                ev = Evaluation(organism_id=org.id, score=0.0, viable=False, inconclusive=True, reason=reason,
                                partner_id=partner.id if partner else None, error=str(e),
                                details={"inconclusive": True})
            except Exception as e:  # noqa: BLE001 - an evaluator crash is recorded, not fatal
                traceback.print_exc()
                ev = Evaluation(organism_id=org.id, score=0.0, viable=False, reason=reason,
                                partner_id=partner.id if partner else None, error=f"{type(e).__name__}: {e}")
        ev.ended = ev.ended or time.time()
        self._record_eval(node.name, ev)
        return ev

    # ---- setup --------------------------------------------------------------------------------------------

    def setup(self) -> None:
        self.store.emit("graph", nodes=[{
            "name": n.name, "description": n.description, "partner": n.partner, "fixed": n.fixed,
            "archive": n.archive, "all_niches": n.all_niches, "mutators": [m.name for m in n.mutators],
            "parents": n.parents,
        } for n in self.nodes.values()], schedule=self.schedule, lanes=self.lanes)
        # Every node's first seed is its provisional champion, so the other nodes have a partner to be scored
        # against before anything has been evaluated.
        for n in self.nodes.values():
            self.champions[n.name] = n.seeds[0]
        for n in self.nodes.values():
            self._status(n.name, "evaluating seeds")
            for seed in n.seeds:
                self._record_organism(seed)
                ev = self._evaluate(n, seed, "seed")
                self.pops[n.name].add(seed, ev)
                self.log(f"{n.name}: seed {seed.id[:6]} scored {ev.score:.3f}" + ("" if ev.viable else f" (not viable: {ev.error})"))
            champ = self.pops[n.name].champion()
            if champ is None:
                raise RuntimeError(f"no seed of node {n.name!r} is viable")
            self.champions[n.name] = champ[0]
            self.store.emit("champion_changed", node=n.name, organism_id=champ[0].id, old_id=None,
                            score=champ[1].score, reason="seed")
            self._status(n.name, "idle")

    # ---- running ------------------------------------------------------------------------------------------

    def run(self, cycles: int) -> None:
        try:
            if not self.champions:
                self.setup()
            self.store.emit("run_started", cycles=cycles)
            for cycle in range(cycles):
                if self.stopped:
                    break
                self.store.emit("cycle", cycle=cycle, of=cycles)
                self.log(f"cycle {cycle + 1}/{cycles}")
                for name, times in self.schedule:
                    for _ in range(times):
                        if self.stopped:
                            break
                        self.iterate(self.nodes[name])
        except (BudgetExhausted, RateLimited) as e:
            self.stop(str(e))
        finally:
            self.store.emit("run_finished", stopped_early=self._stop.is_set(), reason=self.stop_reason)
            self.store.flush()

    def _pick_parent(self, node: Node) -> Organism:
        pop = self.pops[node.name]
        if node.archive:
            niches = pop.niches()
            if niches and self._rng.random() >= node.champion_share:
                return self._rng.choice(list(niches.values()))[0]
            return self.champions[node.name]
        # A plain node gets its variety from the mutator, not from its parents, and often makes one child a cycle,
        # so it leans on its best: with four organisms the champion is picked about 70% of the time. Weights of
        # 1/rank once spent a whole painting on a seed that scored 0.10 under the champion.
        ranked = pop.ranked()[:8]
        weights = [1.0 / (i + 1) ** 2 for i in range(len(ranked))]
        return self._rng.choices([o for o, _ in ranked], weights=weights)[0]

    def _context(self, node: Node, parent: Organism, mutator: Mutator) -> Context:
        pop = self.pops[node.name]
        niches = pop.niches()
        empty = [n for n in node.all_niches if n not in niches]
        ctx = Context(node=node.name, parent=parent, parent_eval=pop.evals.get(parent.id),
                      partner=self.partner_of(node), lineage=pop.lineage(parent), niches=niches, empty_niches=empty)
        if node.archive and node.all_niches:
            # Aim at an empty niche when there is one, and otherwise at any niche but the parent's own, favouring
            # the niches most unlike the parent's.
            pool = empty or [n for n in node.all_niches if n != parent.niche] or node.all_niches
            weights = [node.niche_distance(parent.niche, n) if node.niche_distance else 1.0 for n in pool]
            ctx.wanted_niche = self._rng.choices(pool, weights=[max(w, 1e-6) for w in weights])[0]
        if mutator.needs_other:
            candidates = [o for n, (o, _) in niches.items() if o.id != parent.id and n != parent.niche]
            candidates = candidates or [o for o, _ in pop.ranked() if o.id != parent.id]
            if candidates:
                ctx.other = self._rng.choice(candidates)
                ctx.other_eval = pop.evals.get(ctx.other.id)
        return ctx

    def _pick_mutator(self, node: Node) -> Mutator:
        usable = [m for m in node.mutators if not m.needs_other or len(self.pops[node.name].ranked()) > 1]
        return self._rng.choices(usable, weights=[m.weight for m in usable])[0]

    def _mutate_and_evaluate(self, node: Node, parent: Organism, mutator: Mutator) -> list[tuple[Organism, Evaluation]]:
        if self.stopped:
            return []
        ctx = self._context(node, parent, mutator)
        started = time.time()
        try:
            children = self._patiently(f"{node.name} {mutator.name}", lambda: mutator.propose(ctx))
            error = None
        except (BudgetExhausted, RateLimited):
            raise
        except Exception as e:  # noqa: BLE001 - one failed mutation is data, not a crash
            traceback.print_exc()
            children, error = [], f"{type(e).__name__}: {e}"
        self.store.emit("mutation", node=node.name, organism_id=parent.id, mutator=mutator.name,
                        n_children=len(children), child_ids=[c.id for c in children], error=error,
                        wanted_niche=ctx.wanted_niche, other_id=ctx.other.id if ctx.other else None,
                        seconds=round(time.time() - started, 1))
        out = []
        for child in children:
            child.node = node.name
            child.parent_id = child.parent_id or parent.id
            child.mutator = child.mutator or mutator.name
            self._record_organism(child)
            if self.stopped:
                break
            ev = self._evaluate(node, child, "new")
            self.pops[node.name].add(child, ev)
            out.append((child, ev))
            parent_ev = self.pops[node.name].evals.get(parent.id)
            verdict = ("no result (its sessions kept failing)" if ev.inconclusive else
                       "not viable" if not ev.viable else
                       f"{ev.score:.3f} vs parent {parent_ev.score:.3f}" if parent_ev else f"{ev.score:.3f}")
            self.log(f"  {node.name}: {mutator.name} child {child.id[:6]} ({child.niche or '-'}) {verdict}")
        return out

    def iterate(self, node: Node) -> None:
        self.iterations[node.name] += 1
        self._status(node.name, "evolving", iteration=self.iterations[node.name])
        pairs = [(self._pick_parent(node), self._pick_mutator(node)) for _ in range(node.parents)]
        with self._pool() as pool:
            futures = [pool.submit(self._mutate_and_evaluate, node, p, m) for p, m in pairs]
            results = []
            for f in futures:
                try:
                    results.extend(f.result())
                except (BudgetExhausted, RateLimited) as e:
                    self.stop(str(e))
        pop = self.pops[node.name]
        champ = pop.champion()
        self.store.emit("iteration", node=node.name, iteration=self.iterations[node.name], n_new=len(results),
                        n_nonviable=sum(1 for _, e in results if not e.viable and not e.inconclusive),
                        n_inconclusive=sum(1 for _, e in results if e.inconclusive),
                        best_id=champ[0].id if champ else None, best_score=champ[1].score if champ else None,
                        population=len(pop.organisms), niches=sorted(pop.niches()))
        self._check_champion(node, cascade=True, reason="evolution")
        self._status(node.name, "idle")

    def _check_champion(self, node: Node, cascade: bool, reason: str) -> None:
        pop = self.pops[node.name]
        old = self.champions.get(node.name)
        # Whatever tops the ranking on fewer than `confirm` evaluations is evaluated again, and so is the champion
        # it would replace, until the means are settled. A repeat can drop the challenger below another organism
        # that has only one evaluation, so this goes round until the top is confirmed.
        for _ in range(node.parents + 4):
            champ = pop.champion()
            if champ is None or old is None or champ[0].id == old.id or self.stopped:
                break
            short = [o for o in (champ[0], old) if pop.count(o.id) < node.confirm and pop.evals[o.id].viable]
            if not short:
                break
            self._confirm(node, short)
        champ = pop.champion()
        if champ is None:
            return
        best, ev = champ
        if old is not None and best.id == old.id:
            return
        self.champions[node.name] = best
        old_ev = self.pops[node.name].evals.get(old.id) if old else None
        self.store.emit("champion_changed", node=node.name, organism_id=best.id, old_id=old.id if old else None,
                        score=ev.score, old_score=old_ev.score if old_ev else None, reason=reason,
                        samples=ev.samples, old_samples=old_ev.samples if old_ev else None)
        self.log(f"  {node.name}: new champion {best.id[:6]} {old_ev.score if old_ev else float('nan'):.3f} -> "
                 f"{ev.score:.3f} ({reason})")
        if cascade:
            for other in self.nodes.values():
                if other.partner == node.name and not self.stopped:
                    self._rescore(other, because=node.name)

    def _confirm(self, node: Node, orgs: list[Organism]) -> None:
        """Evaluate each organism again against the current partner until its score stands on `node.confirm`."""
        pop = self.pops[node.name]
        jobs = []
        for org in orgs:
            missing = node.confirm - pop.count(org.id)
            if missing > 0 and pop.evals[org.id].viable:
                jobs += [(org, k) for k in pop.claim(org.id, missing)]
        if not jobs or self.stopped:
            return
        self._status(node.name, "confirming", organisms=[o.id for o in orgs])
        self.log(f"  {node.name}: confirming " + ", ".join(f"{o.id[:6]} ({pop.evals[o.id].score:.3f} on "
                                                             f"{pop.count(o.id)})" for o in orgs))

        def one(job: tuple[Organism, int]) -> None:
            org, k = job
            if not self.stopped:
                pop.repeat(org.id, self._evaluate(node, org, "confirm", sample=k))

        with self._pool() as pool:
            list(pool.map(one, jobs))
        for org in orgs:
            ev = pop.evals[org.id]
            self.store.emit("confirmed", node=node.name, organism_id=org.id, score=ev.score, samples=ev.samples,
                            scores=[round(e.score, 4) for e in pop.samples.get(org.id, [])])
            self.log(f"  {node.name}: {org.id[:6]} stands at {ev.score:.3f} over {ev.samples}")

    def _rescore(self, node: Node, because: str) -> None:
        """Re-evaluate the top of `node` against `because`'s new champion, so its champion stays current."""
        pop = self.pops[node.name]
        self._status(node.name, "rescoring", because=because)
        done: set[str] = set()
        for _ in range(node.rescore_k * 2):  # the top k, then anything stale that climbs above them
            top = [o for o, _ in pop.ranked()[: node.rescore_k] if o.id not in done]
            if not top or self.stopped:
                break
            for org in top:
                old = pop.evals[org.id]
                ev = self._evaluate(node, org, f"rescore after {because} changed")
                pop.rescore(org.id, ev)
                done.add(org.id)
                if ev.inconclusive:
                    continue  # its standing is unchanged; the next rescore tries again
                self.store.emit("rescored", node=node.name, organism_id=org.id, old_score=old.score,
                                new_score=ev.score, because=because)
        # The champion's new standing gets its full count of evaluations too. That is usually free: the partner's
        # new champion was confirmed against this very organism, so the evaluator already has them.
        champion = self.champions.get(node.name)
        if champion is not None and champion.id in done and not self.stopped:
            self._confirm(node, [champion])
        self._check_champion(node, cascade=False, reason=f"rescore after {because} changed")
        self._status(node.name, "idle")
