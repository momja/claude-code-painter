"""
Component B: the painter.

`Strategy` plays the role of the LLM agent's prompt. It never names specific brushes, only sizes, so a
toolkit change cannot break it. `paint()` is the tool-use loop: each stroke is one tool call, recorded as a
span on the active trace.
"""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism
from pydantic import Field

from conveyor.events import current_trace
from conveyor.painting.canvas import Target
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import composite
from conveyor.painting.canvas import stroke_alpha
from conveyor.painting.canvas import to_png
from conveyor.painting.toolkit import Toolkit
from conveyor.painting.toolkit import tags_that_failed


class Strategy(Organism):
    n_strokes: int = 140
    coarse_fraction: float = 0.35
    candidates: int = 3
    focus: float = 1.0
    detail_bias: float = 0.4
    angle_mode: Literal["random", "gradient"] = "random"
    color_mode: Literal["center", "local_mean"] = "center"
    notes: list[str] = Field(default_factory=list)

    def render_text(self) -> str:
        angle = (
            "Lay each stroke along the nearest edge in the target."
            if self.angle_mode == "gradient"
            else "Pick stroke direction at random."
        )
        color = (
            "Mix the color from the average of the target under the brush."
            if self.color_mode == "local_mean"
            else "Take the color from the target pixel at the stroke's start."
        )
        lines = [
            "You are copying the target painting onto a simulated watercolor canvas.",
            f"You have {self.n_strokes} strokes.",
            f"Spend the first {self.coarse_fraction:.0%} of them on the largest brushes to block in washes.",
            f"Before each stroke, try {self.candidates} placements and keep the one that lowers error most.",
            f"Choose where to paint by sampling the error map with focus {self.focus:.2f}.",
            f"After the block-in, favor small brushes with weight {self.detail_bias:.2f}.",
            angle,
            color,
        ]
        if self.notes:
            lines.append("")
            lines.append("Notes from earlier attempts:")
            lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


Strategy.model_rebuild(_types_namespace={"EvaluationFailureCase": EvaluationFailureCase})


def _hex(color: np.ndarray) -> str:
    r, g, b = (np.clip(color, 0, 1) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def paint(strategy: Strategy, toolkit: Toolkit, target: Target, seed: int, record: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tgt_img = target.image
    height, width = tgt_img.shape[:2]
    canvas = blank(target)
    err = ((canvas - tgt_img) ** 2).sum(axis=-1)

    brushes = sorted(toolkit.brushes, key=lambda b: b.radius, reverse=True)
    radii = np.array([b.radius for b in brushes], dtype=np.float64)
    n_big = max(1, math.ceil(len(brushes) / 2))
    coarse_p = np.zeros(len(brushes))
    coarse_p[:n_big] = 1.0 / n_big
    fine_p = (1.0 / radii) ** (strategy.detail_bias * 3.0)
    fine_p /= fine_p.sum()

    trace = current_trace() if record else None
    # Stored at 1x; the dashboard scales them up. Snapshots are most of a run's disk use.
    thumb_every = max(1, strategy.n_strokes // 8)

    for i in range(strategy.n_strokes):
        probs = coarse_p if i / strategy.n_strokes < strategy.coarse_fraction else fine_p
        cdf = np.cumsum(((err + 1e-4) ** strategy.focus).ravel())
        best = None
        for _ in range(strategy.candidates):
            b = brushes[int(rng.choice(len(brushes), p=probs))]
            idx = min(int(np.searchsorted(cdf, rng.random() * cdf[-1])), cdf.size - 1)
            py, px = divmod(idx, width)
            x, y = px + rng.random(), py + rng.random()
            if strategy.angle_mode == "gradient":
                angle = float(target.edge_angle[py, px]) + rng.normal(0, 0.2)
            else:
                angle = rng.uniform(0, 2 * math.pi)
            hit = stroke_alpha(b, x, y, angle, height, width, rng)
            if hit is None:
                continue
            ys, xs, a = hit
            tgt = tgt_img[ys, xs]
            if strategy.color_mode == "local_mean":
                color = (tgt * a).sum(axis=(0, 1)) / (a.sum() + 1e-6)
            else:
                color = tgt_img[py, px]
            new = composite(canvas[ys, xs], a, color)
            new_err = ((new - tgt) ** 2).sum(axis=-1)
            delta = float(new_err.sum() - err[ys, xs].sum())
            if best is None or delta < best["delta"]:
                best = dict(delta=delta, brush=b, ys=ys, xs=xs, new=new, new_err=new_err, x=x, y=y, angle=angle, color=color)

        kept = best is not None and best["delta"] < 0
        if kept:
            canvas[best["ys"], best["xs"]] = best["new"]
            err[best["ys"], best["xs"]] = best["new_err"]
        if trace is not None:
            last = i == strategy.n_strokes - 1
            image = to_png(canvas) if (i % thumb_every == 0 or last) else None
            if best is None:
                trace.span("skip", result={"kept": False}, image=image)
            else:
                trace.span(
                    best["brush"].name,
                    args={
                        "x": round(best["x"], 1),
                        "y": round(best["y"], 1),
                        "angle": round(best["angle"] % (2 * math.pi), 2),
                        "color": _hex(best["color"]),
                    },
                    result={"delta_error": round(best["delta"], 4), "kept": kept},
                    image=image,
                )
    return canvas


class TargetedStrategyMutator(Mutator):
    """Adjusts the strategy for the failure type it is shown. Stands in for an LLM rewriting the prompt."""

    OPTIONS = {
        "fine_detail": ["more-strokes", "detail-bias", "more-candidates"],
        "soft_wash": ["longer-blockin", "local-color", "less-focus"],
        "hard_edge": ["follow-edges", "more-focus", "center-color"],
    }

    @property
    def supports_batch_mutation(self) -> bool:
        return True

    @staticmethod
    def _applicable(s: Strategy, action: str) -> bool:
        return {
            "more-strokes": s.n_strokes < 400,
            "detail-bias": s.detail_bias < 2.0,
            "more-candidates": s.candidates < 8,
            "longer-blockin": s.coarse_fraction < 0.8,
            "local-color": s.color_mode != "local_mean",
            "less-focus": s.focus > 0.3,
            "follow-edges": s.angle_mode != "gradient",
            "more-focus": s.focus < 4.0,
            "center-color": s.color_mode != "center",
        }[action]

    def mutate(self, organism, failure_cases, learning_log_entries):
        rng = random.Random()
        s: Strategy = organism
        ftype = failure_cases[0].failure_type
        avoid = tags_that_failed(learning_log_entries)
        options = [o for o in self.OPTIONS.get(ftype, []) if self._applicable(s, o)]
        if not options:
            return []
        action = rng.choice([o for o in options if o not in avoid] or options)
        where = ", ".join(f.data_point_id for f in failure_cases[:2])

        update: dict = {}
        if action == "more-strokes":
            update["n_strokes"] = min(400, round(s.n_strokes * 1.25))
            note = f"Detail at {where} stayed blurry. Use {update['n_strokes']} strokes."
        elif action == "detail-bias":
            update["detail_bias"] = round(min(2.0, s.detail_bias + 0.25), 2)
            note = f"Detail at {where} stayed blurry. Lean harder on small brushes."
        elif action == "more-candidates":
            update["candidates"] = s.candidates + 1
            note = f"Strokes at {where} missed. Try {update['candidates']} placements per stroke."
        elif action == "longer-blockin":
            update["coarse_fraction"] = round(min(0.8, s.coarse_fraction + 0.1), 2)
            note = f"Washes at {where} were patchy. Block in longer."
        elif action == "local-color":
            update["color_mode"] = "local_mean"
            note = f"Washes at {where} were blotchy. Average the color under the brush."
        elif action == "less-focus":
            update["focus"] = round(max(0.25, s.focus - 0.3), 2)
            note = f"Washes at {where} were neglected. Spread strokes more evenly."
        elif action == "follow-edges":
            update["angle_mode"] = "gradient"
            note = f"Edges at {where} were ragged. Stroke along the edges."
        elif action == "more-focus":
            update["focus"] = round(min(4.0, s.focus + 0.4), 2)
            note = f"Edges at {where} were missed. Concentrate on high-error areas."
        else:  # center-color
            update["color_mode"] = "center"
            note = f"Edges at {where} were muddy. Take color from the stroke's start pixel."

        update["notes"] = (s.notes + [note])[-6:]
        child = Strategy(**{**s.model_dump(include=set(Strategy.model_fields) - set(Organism.model_fields)), **update})
        child.from_change_summary = f"[{action}] {note}"
        return [child]


class RandomStrategyMutator(Mutator):
    """Nudges one strategy parameter at random."""

    def mutate(self, organism, failure_cases, learning_log_entries):
        rng = random.Random()
        s: Strategy = organism
        param = rng.choice(["n_strokes", "coarse_fraction", "candidates", "focus", "detail_bias", "angle_mode"])
        fields = s.model_dump(include=set(Strategy.model_fields) - set(Organism.model_fields))
        old = fields[param]
        if param == "n_strokes":
            new = int(min(400, max(40, old + rng.randint(-30, 30))))
        elif param == "candidates":
            new = int(min(8, max(1, old + rng.choice([-1, 1]))))
        elif param == "angle_mode":
            new = "gradient" if old == "random" else "random"
        elif param == "coarse_fraction":
            new = round(min(0.9, max(0.0, old + rng.uniform(-0.1, 0.1))), 2)
        else:
            new = round(max(0.0, old * rng.uniform(0.7, 1.4)), 2)
        fields[param] = new
        child = Strategy(**fields)
        child.from_change_summary = f"[nudge-{param}] Changed {param} from {old} to {new}."
        return [child]
