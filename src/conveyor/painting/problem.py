"""Evaluators for the toolkit and agent nodes, and the graph that wires them to the critic."""

from __future__ import annotations

import zlib
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal

import numpy as np
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import EvaluationResult
from darwinian_evolver.problem import Evaluator
from pydantic import Field
from pydantic import computed_field

from conveyor.events import artifact
from conveyor.graph import Board
from conveyor.graph import Edge
from conveyor.graph import Node
from conveyor.painting.agent import RandomStrategyMutator
from conveyor.painting.agent import Strategy
from conveyor.painting.agent import TargetedStrategyMutator
from conveyor.painting.agent import paint
from conveyor.painting.canvas import Target
from conveyor.painting.canvas import default_targets
from conveyor.painting.canvas import heatmap
from conveyor.painting.canvas import physics_violations
from conveyor.painting.canvas import set_canvas_scale
from conveyor.painting.canvas import swatch_sheet
from conveyor.painting.canvas import to_png
from conveyor.painting.canvas import triptych
from conveyor.painting.critic import Critic
from conveyor.painting.toolkit import CrossoverToolkitMutator
from conveyor.painting.toolkit import OracleFitter
from conveyor.painting.toolkit import RandomToolkitMutator
from conveyor.painting.toolkit import TargetedToolkitMutator
from conveyor.painting.toolkit import Toolkit
from conveyor.painting.toolkit import initial_toolkit

if TYPE_CHECKING:
    from conveyor.llm import LLMClient

# Patch RMSE above this after the oracle's best effort: the tools cannot express the patch.
TOOL_THRESHOLD = 0.08
# Agent RMSE this far above the oracle's: the tools could do it and the agent did not.
GAP_THRESHOLD = 0.02
MAX_TRIPTYCHS = 8


class PatchFailure(EvaluationFailureCase):
    target: str
    row: int
    col: int
    blame: Literal["toolkit", "agent"]
    oracle_err: float
    agent_err: float
    triptych: str | None = None


class PaintResult(EvaluationResult):
    pixel: float = 0.0
    style: float = 0.0
    expressivity: float = 0.0
    blame_toolkit: int = 0
    blame_agent: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    # {"critic": [one Critic.report() per target]}: the score with its working shown, for the dashboard.
    details: dict = Field(default_factory=dict)

    @computed_field
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        return {
            "pixel": round(self.pixel, 4),
            "style": round(self.style, 4),
            "expressivity": round(self.expressivity, 4),
            "blame_toolkit": self.blame_toolkit,
            "blame_agent": self.blame_agent,
        }

    def format_observed_outcome(self, parent_result: EvaluationResult | None, ndigits: int = 3) -> str:
        if not self.is_viable:
            return "Not viable: " + "; ".join(self.notes)
        text = f"Score {self.score:.3f} (pixel {self.pixel:.3f}, style {self.style:.3f})"
        if parent_result is not None:
            ps = round(parent_result.score, ndigits)
            s = round(self.score, ndigits)
            if s > ps:
                text += f", better than the parent's {ps:.3f}."
            elif s < ps:
                text += f", worse than the parent's {ps:.3f}."
            else:
                text += f", same as the parent's {ps:.3f}."
        return text


def _seed(target: Target) -> int:
    return zlib.crc32(target.name.encode())


class PaintingEvaluator(Evaluator):
    """
    Paints the training targets with each (toolkit, strategy) combination and scores with the critic. The
    held-out targets are painted by `evaluate_holdout`, which the graph runner calls for new champions only.

    role="toolkit": the organism is a Toolkit, painted by the agent node's top `partner_k` strategies.
    role="agent":   the organism is a Strategy, painting with the toolkit node's top `partner_k` toolkits.
    Blame comes from the first combination, which is the champion pairing.
    """

    def __init__(
        self,
        role: Literal["toolkit", "agent"],
        board: Board,
        critic: Critic,
        oracle: OracleFitter,
        train: list[Target],
        holdout: list[Target],
        partner_k: int,
        painter: Callable[..., np.ndarray] = paint,
    ) -> None:
        self.painter = painter
        self.role = role
        self.board = board
        self.critic = critic
        self.oracle = oracle
        self.train = train
        self.holdout = holdout
        self.partner_k = partner_k

    def _pairings(self, organism) -> tuple[list[Toolkit], list[Strategy]]:
        if self.role == "toolkit":
            return [organism], self.board.elites("agent", self.partner_k)
        return self.board.elites("toolkit", self.partner_k), [organism]

    def evaluate(self, organism) -> PaintResult:
        toolkits, strategies = self._pairings(organism)

        artifacts: dict[str, str | None] = {}
        if self.role == "toolkit":
            artifacts["thumb"] = artifact(swatch_sheet(organism.brushes))
            violations = physics_violations(organism.brushes)
            if violations:
                return PaintResult(
                    score=0.0,
                    is_viable=False,
                    trainable_failure_cases=[],
                    notes=violations,
                    artifacts={k: v for k, v in artifacts.items() if v},
                )

        train_totals, pixels, styles, oracle_means = [], [], [], []
        train_failures: list[PatchFailure] = []
        blame_counts = {"toolkit": 0, "agent": 0}
        n_triptychs = 0
        reports: list[dict] = []

        for target in self.train:
            primary = None
            for tk in toolkits:
                oracle_errs, oracle_canvas = self.oracle.fit(tk, target)
                for st in strategies:
                    canvas = self.painter(st, tk, target, seed=_seed(target), record=primary is None)
                    scores = self.critic.score(canvas, target)
                    train_totals.append(scores["total"])
                    pixels.append(scores["pixel"])
                    styles.append(scores["style"])
                    if primary is None:
                        primary = (canvas, oracle_errs, oracle_canvas)

            canvas, oracle_errs, oracle_canvas = primary
            agent_errs = self.critic.patch_errors(canvas, target)
            oracle_means.append(float(oracle_errs.mean()))
            artifacts[f"canvas_{target.name}"] = artifact(to_png(canvas))
            artifacts[f"oracle_{target.name}"] = artifact(to_png(oracle_canvas))
            artifacts[f"heat_{target.name}"] = artifact(heatmap(self.critic.pixel_error(canvas, target)))
            reports.append({**self.critic.report(canvas, target, artifact), "train": True})

            rows, cols = target.grid
            for r in range(rows):
                for c in range(cols):
                    o, a = float(oracle_errs[r, c]), float(agent_errs[r, c])
                    if o > TOOL_THRESHOLD:
                        blame = "toolkit"
                    elif a - o > GAP_THRESHOLD:
                        blame = "agent"
                    else:
                        continue
                    blame_counts[blame] += 1
                    if blame != self.role:
                        continue
                    trip = None
                    if n_triptychs < MAX_TRIPTYCHS:
                        ys, xs = target.patch_slice(r, c)
                        trip = artifact(triptych(target.image[ys, xs], oracle_canvas[ys, xs], canvas[ys, xs], scale=1))
                        n_triptychs += 1
                    failure = PatchFailure(
                        data_point_id=f"{target.name}/r{r}c{c}",
                        failure_type=target.classes[(r, c)],
                        target=target.name,
                        row=r,
                        col=c,
                        blame=blame,
                        oracle_err=round(o, 4),
                        agent_err=round(a, 4),
                        triptych=trip,
                    )
                    train_failures.append(failure)

        first = self.train[0].name
        if self.role == "agent":
            artifacts["thumb"] = artifacts.get(f"canvas_{first}")
        artifacts["heat"] = artifacts.get(f"heat_{first}")

        # Worst patches first, so the mutator sees the clearest examples.
        train_failures.sort(key=lambda f: -(f.oracle_err if self.role == "toolkit" else f.agent_err - f.oracle_err))
        return PaintResult(
            score=float(np.mean(train_totals)),
            trainable_failure_cases=train_failures,
            pixel=float(np.mean(pixels)),
            style=float(np.mean(styles)),
            expressivity=max(0.0, 1.0 - float(np.mean(oracle_means)) / 0.25),
            blame_toolkit=blame_counts["toolkit"],
            blame_agent=blame_counts["agent"],
            artifacts={k: v for k, v in artifacts.items() if v},
            details={"critic": reports},
        )

    def evaluate_holdout(self, organism) -> dict:
        """
        Paint the held-out targets with the champion pairing and score them. The graph runner calls this once
        per new champion: only champions feed the overfitting alarm, so painting the held-out targets on every
        evaluation doubled the cost of evolution for a number nobody read.
        """
        toolkits, strategies = self._pairings(organism)
        tk, st = toolkits[0], strategies[0]
        reports, totals, artifacts = [], [], {}
        for target in self.holdout:
            canvas = self.painter(st, tk, target, seed=_seed(target), record=True)
            report = self.critic.report(canvas, target, artifact)
            reports.append({**report, "train": False})
            totals.append(report["total"])
            artifacts[f"canvas_{target.name}"] = artifact(to_png(canvas))
        return dict(
            score=float(np.mean(totals)) if totals else None,
            details={"critic": reports},
            artifacts={k: v for k, v in artifacts.items() if v},
        )

    def verify_mutation(self, organism) -> bool:
        """Toolkit only: did the change lower the oracle's error on the patches it was meant to fix?"""
        if self.role != "toolkit":
            raise NotImplementedError("Only the toolkit node verifies mutations")
        parent = organism.parent
        if parent is None or physics_violations(organism.brushes):
            return True  # let evaluation record it as non-viable, so it shows up in the lineage
        cells = defaultdict(list)
        for f in organism.from_failure_cases or []:
            cells[f.target].append((f.row, f.col))
        targets = {t.name: t for t in self.train + self.holdout}
        new_err = old_err = 0.0
        for name, rc in cells.items():
            new, _ = self.oracle.fit(organism, targets[name])
            old, _ = self.oracle.fit(parent, targets[name])
            new_err += sum(float(new[r, c]) for r, c in rc)
            old_err += sum(float(old[r, c]) for r, c in rc)
        return new_err < old_err - 1e-4


def build_painting_graph(
    width: int = 80, llm: LLMClient | None = None, harness: str = "pi", strokes: int = 500,
    parents: int | None = None, concurrency: int | None = None, paint_model: str | None = None
) -> tuple[Board, list[Node], list[Edge], list[tuple[str, int]]]:
    """
    The painting graph. With `llm`, the painter and some mutators run on that model; otherwise all scripted.
    `harness` picks how the painter talks to the model: "pi" (one Pi conversation with look/finish tools) or
    "stateless" (a fresh request every turn).
    """
    train, holdout = default_targets(width=width)
    # Brush sizes and their limits are in canvas pixels, so they scale with the canvas before anything is built.
    scale = set_canvas_scale(width)
    board = Board()
    critic = Critic()
    # `strokes` is the LLM painter's cap; the scripted painter evolves its own count and the oracle's floor covers it.
    oracle = OracleFitter(stroke_budget=strokes if llm is not None else None)
    edges = [
        Edge("toolkit", "agent", "artifact", "brush functions"),
        Edge("agent", "critic", "artifact", "painting"),
        Edge("critic", "toolkit", "feedback", "toolkit failures"),
        Edge("critic", "agent", "feedback", "agent failures"),
    ]
    critic_node = Node(
        name="critic",
        description="Pixel match at three scales plus a style proxy. Splits blame using the oracle fitter.",
        fixed=True,
        version=critic.version,
        mirror="agent",
        thumbnail_artifact="heat",
    )

    if llm is not None:
        from conveyor.painting.llm_agent import INITIAL_PROMPT
        from conveyor.painting.llm_agent import PromptStrategy
        from conveyor.painting.llm_agent import llm_paint
        from conveyor.painting.llm_mutators import LLMPromptMutator
        from conveyor.painting.llm_mutators import LLMToolkitMutator

        if harness == "pi":
            from conveyor.painting.pi_harness import PiHarness

            painter = PiHarness(llm, model=paint_model)
        else:
            painter = partial(llm_paint, client=llm)
        # Every evaluation is several model calls per painting, so one partner and small rescores. Parents and
        # lanes decide how many paintings overlap: a live run spent 93 minutes painting in 74 minutes of wall
        # clock, which is barely parallel at all.
        # Each painting runs its own node sidecar, so lanes cost memory as well as rate limit. Eight let five
        # paintings run at once and the machine ran out of memory; four is the safer default.
        lanes = concurrency or 4
        small = dict(num_parents=parents or 4, rescore_top_k=2, mutator_concurrency=lanes,
                     evaluator_concurrency=lanes, holdout_every=3)
        nodes = [
            Node(
                name="toolkit",
                description=f"Component A. Brush functions. LLM ({llm.model}) and rule-based mutators compete.",
                initial_organism=initial_toolkit(fine=True, scale=scale),
                evaluator=PaintingEvaluator("toolkit", board, critic, oracle, train, holdout, 1, painter),
                mutators=[LLMToolkitMutator(llm), TargetedToolkitMutator(), CrossoverToolkitMutator()],
                partners=["agent"],
                partner_k=1,
                verify_mutations=True,
                **small,
            ),
            Node(
                name="agent",
                description=f"Component B. An LLM painter ({paint_model or llm.model}, {harness} harness) "
                            "whose prompt evolves.",
                initial_organism=PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=strokes),
                evaluator=PaintingEvaluator("agent", board, critic, oracle, train, holdout, 1, painter),
                mutators=[LLMPromptMutator(llm)],
                partners=["toolkit"],
                partner_k=1,
                **small,
            ),
            critic_node,
        ]
        # The agent node is the cheaper one (one mutator, no verification), so it gets two iterations per cycle.
        return board, nodes, edges, [("agent", 2), ("toolkit", 1)]

    nodes = [
        Node(
            name="toolkit",
            description="Component A. Brush functions and canvas physics.",
            initial_organism=initial_toolkit(scale=scale),
            evaluator=PaintingEvaluator("toolkit", board, critic, oracle, train, holdout, partner_k=2),
            mutators=[TargetedToolkitMutator(), RandomToolkitMutator(), CrossoverToolkitMutator()],
            partners=["agent"],
            partner_k=2,
            verify_mutations=True,
            num_parents=3,
        ),
        Node(
            name="agent",
            description="Component B. The painter's strategy, standing in for the LLM agent's prompt.",
            initial_organism=Strategy(),
            evaluator=PaintingEvaluator("agent", board, critic, oracle, train, holdout, partner_k=1),
            mutators=[TargetedStrategyMutator(), RandomStrategyMutator()],
            partners=["toolkit"],
            partner_k=1,
            num_parents=3,
        ),
        critic_node,
    ]
    return board, nodes, edges, [("agent", 2), ("toolkit", 1)]
