"""Component A: the brush toolkit, its mutators, and the oracle fitter that measures what it can express."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import zlib

import numpy as np
from darwinian_evolver.learning_log import LearningLogEntry
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import Mutator
from darwinian_evolver.problem import Organism

from conveyor.painting.brushcode import DEFAULTS
from conveyor.painting.brushcode import DOCS
from conveyor.painting.brushcode import jitter_source
from conveyor.painting.brushcode import render_source
from conveyor.painting.canvas import MAX_BRUSHES
from conveyor.painting.canvas import Brush
from conveyor.painting.canvas import Target
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import composite
from conveyor.painting.canvas import stroke_alpha


class Toolkit(Organism):
    brushes: list[Brush]

    def key(self) -> str:
        payload = json.dumps([b.model_dump() for b in self.brushes], sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def render_text(self) -> str:
        """The toolkit as the code it actually is. Each brush's own program, under a header giving its reach."""
        lines = ['"""Brushes the painter can call. Each call lays one stroke, drawn by the brush\'s own code."""', ""]
        for b in self.brushes:
            lines += [f"# {b.name}: reach {b.radius:.2f}px, path {b.length:.1f}px" + (f". {b.doc}" if b.doc else "")]
            lines += [b.source.rstrip().replace(f"def alpha(", f"def {b.name}(", 1), ""]
        return "\n".join(lines).rstrip() + "\n"


# Organism annotates `from_failure_cases` with a forward reference that pydantic resolves here.
Toolkit.model_rebuild(_types_namespace={"EvaluationFailureCase": EvaluationFailureCase})


def templated(name: str, template: str, radius: float, length: float = 0.0, doc: str = "", **params) -> Brush:
    """A brush written from one of the stock templates, keeping the knobs that produced it for later edits."""
    filled = {**DEFAULTS[template], **params}
    text = doc or DOCS[template]
    return Brush(
        name=name,
        radius=round(radius, 2),
        length=round(length, 2),
        doc=text,
        source=render_source(template, filled, text),
        template=template,
        params=filled,
    )


def retemplate(brush: Brush, **changes: float) -> Brush | None:
    """
    Re-render a templated brush with some knobs changed. None if this brush did not come from a template,
    which is the case for anything an LLM wrote as free code: that gets edited as code instead.
    """
    if not brush.template or brush.template not in DEFAULTS:
        return None
    params = {**brush.params, **changes}
    return brush.model_copy(update={"params": params, "source": render_source(brush.template, params, brush.doc)})


def initial_toolkit(fine: bool = False, scale: float = 1.0) -> Toolkit:
    """
    Thin by default, with nothing small enough for the swirl strokes in a Van Gogh background, so the scripted
    demo shows the toolkit evolving liners. `fine=True` adds an edge brush and a liner for the LLM painter: at
    64 px the eyes and the dark lapel lines are 2 to 3 px wide, and with only 7 and 14 px brushes a live
    500-stroke run learned that every dark stroke raised the error, and washed the painting out instead.

    Every seed brush is a plain capsule, so a run starts from the mark the old parameter-only toolkit could
    make and has to discover the rest. `scale` sizes the brushes to the canvas, so a wash covers the same
    fraction of a 128 px picture as it does of a 64 px one and the stroke budget still covers the canvas.
    """

    def brush(name: str, radius: float, length: float, **rest: float) -> Brush:
        return templated(name, "capsule", radius * scale, length * scale, **rest)

    brushes = [
        brush("flat_wash", 7.0, 8.0, softness=0.7, opacity=0.45, bleed=0.2),
        brush("round_mid", 3.5, 3.0, softness=0.5, opacity=0.6),
    ]
    if fine:
        brushes += [
            brush("edge", 1.5, 6.0, softness=0.15, opacity=0.75),
            brush("liner", 0.8, 3.0, softness=0.3, opacity=0.8),
        ]
    return Toolkit(brushes=brushes)


class OracleFitter:
    """
    Best reproduction of each patch the toolkit can manage, found by greedy search instead of the agent.

    High oracle error on a patch means the tools cannot express it (toolkit's fault). Low oracle error but
    high agent error means the agent did not find a stroke sequence the tools allow (agent's fault).
    Results depend only on the toolkit, so they are cached by toolkit hash.

    The oracle must get at least the agent's stroke budget per patch. Otherwise a well-funded agent beats
    it, the oracle's error reads as "the tools can't do this", and the toolkit gets blamed for patches the
    agent had already painted better than the oracle.

    Each stroke tries every brush, `TRIES_PER_BRUSH` placements each, and keeps the single best. A larger
    toolkit therefore costs more to fit, on purpose: the result has to say what the brushes can do, not how
    thinly a fixed search got spread across them.
    """

    STROKES = 18  # per patch, the floor
    # Placements tried per brush, per stroke, rather than a fixed number of tries spread over the toolkit.
    # It is per brush on purpose. When the oracle drew one brush at random and tried six placements whatever
    # the toolkit held, each extra brush meant fewer tries each, so a larger toolkit got a worse fit and was
    # blamed more for owning more tools: adding a rake, dots and a taper to the four seed brushes moved the
    # toolkit's blamed patches the wrong way, 19 up to 21. Every brush now gets the same number of tries, so
    # the fit measures the brushes instead of how many of them there are.
    TRIES_PER_BRUSH = 3
    # The LLM painter can press lightly, so the oracle can too. Otherwise it underrates the brushes.
    PRESSURES = (0.35, 0.7, 1.0)

    def __init__(self, stroke_budget: int | None = None) -> None:
        """`stroke_budget`: the agent's strokes for a whole painting, spread evenly over the patches."""
        self.stroke_budget = stroke_budget
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    def strokes_per_patch(self, target: Target) -> int:
        if not self.stroke_budget:
            return self.STROKES
        rows, cols = target.grid
        return max(self.STROKES, math.ceil(self.stroke_budget / (rows * cols)))

    def fit(self, toolkit: Toolkit, target: Target) -> tuple[np.ndarray, np.ndarray]:
        key = (toolkit.key(), target.name)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._fit(toolkit, target)
        with self._lock:
            self._cache[key] = result
        return result

    def _fit(self, toolkit: Toolkit, target: Target) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = target.grid
        p = target.patch
        per_patch = self.strokes_per_patch(target)
        errs = np.zeros((rows, cols), dtype=np.float32)
        canvas = blank(target)
        rng = np.random.default_rng(int(toolkit.key()[:8], 16) ^ zlib.crc32(target.name.encode()))
        brushes = toolkit.brushes
        for r in range(rows):
            for c in range(cols):
                ys, xs = target.patch_slice(r, c)
                tgt_patch = target.image[ys, xs]
                patch = canvas[ys, xs]
                for _ in range(per_patch):
                    best = None
                    for b in brushes:
                        for _ in range(self.TRIES_PER_BRUSH):
                            hit = stroke_alpha(b, rng.uniform(0, p), rng.uniform(0, p), rng.uniform(0, math.pi), p, p, rng)
                            if hit is None:
                                continue
                            sy, sx, a = hit
                            a = a * self.PRESSURES[int(rng.integers(len(self.PRESSURES)))]
                            region, tgt = patch[sy, sx], tgt_patch[sy, sx]
                            color = (tgt * a).sum(axis=(0, 1)) / (a.sum() + 1e-6)
                            new = composite(region, a, color)
                            delta = float(((new - tgt) ** 2).sum() - ((region - tgt) ** 2).sum())
                            if best is None or delta < best[0]:
                                best = (delta, sy, sx, new)
                    if best is not None and best[0] < 0:
                        patch[best[1], best[2]] = best[3]
                errs[r, c] = math.sqrt(float(((patch - tgt_patch) ** 2).mean()))
        return errs, canvas


def tags_that_failed(entries: list[LearningLogEntry], depth: int = 6) -> set[str]:
    """Action tags of recent ancestor changes that made things worse. Mutators avoid repeating them."""
    tags = set()
    for entry in entries[:depth]:
        outcome = entry.observed_outcome.lower()
        if "worse" in outcome or "not viable" in outcome:
            m = re.match(r"\[([\w-]+)\]", entry.attempted_change)
            if m:
                tags.add(m.group(1))
    return tags


def _unique_name(stem: str, brushes: list[Brush]) -> str:
    names = {b.name for b in brushes}
    i = 1
    while f"{stem}_{i}" in names:
        i += 1
    return f"{stem}_{i}"


def _add_brush(brushes: list[Brush], new: Brush) -> tuple[list[Brush], str]:
    if len(brushes) < MAX_BRUSHES:
        return brushes + [new], ""
    victim = min(brushes, key=lambda b: abs(b.radius - new.radius))
    return [new if b is victim else b for b in brushes], f", replacing {victim.name}"


class TargetedToolkitMutator(Mutator):
    """
    Reads the failure type and makes the change a person would try first. Stands in for an LLM that is
    shown the failing patches and asked to revise the brush code.

    Half the actions add a brush written from a template, so the scripted demo reaches the marks the old
    parameter-only toolkit could not make at all: a broken line of dots, a rake of parallel bristles, a
    stroke that thins as it is lifted. The other half edit a brush already in the set, by name where it came
    from a template and by its code where it did not.
    """

    OPTIONS = {
        "fine_detail": ["add-liner", "sharpen-small", "add-dotted"],
        "soft_wash": ["add-wash", "soften-large", "add-taper"],
        "hard_edge": ["add-edge", "lengthen-mid", "add-rake"],
    }

    @property
    def supports_batch_mutation(self) -> bool:
        return True

    def mutate(self, organism, failure_cases, learning_log_entries):
        rng = random.Random()
        ftype = failure_cases[0].failure_type
        options = self.OPTIONS.get(ftype, ["add-liner"])
        avoid = tags_that_failed(learning_log_entries)
        action = rng.choice([o for o in options if o not in avoid] or options)

        brushes = [b.model_copy() for b in organism.brushes]
        by_radius = sorted(brushes, key=lambda b: b.radius)
        extra = ""

        def replace(old: Brush, new: Brush) -> None:
            brushes[brushes.index(old)] = new

        def edit(target: Brush, what: str, **changes: float) -> str:
            """Change knobs by name on a templated brush, or nudge a number in the code of one that isn't."""
            fresh = retemplate(target, **changes)
            if fresh is not None:
                replace(target, fresh)
                said = ", ".join(f"{k} {v:.2f}" for k, v in changes.items())
                return f"{what} {target.name} to {said}"
            hit = jitter_source(target.source, rng)
            if hit is None:
                return f"Left {target.name} alone: its code has no number to turn"
            source, note = hit
            replace(target, target.model_copy(update={"source": source}))
            return f"{what} {target.name} by editing its code ({note})"

        if action == "add-liner":
            s = by_radius[0]
            new = templated(
                _unique_name("liner", brushes), "capsule",
                radius=max(0.6, s.radius * rng.uniform(0.4, 0.7)),
                length=rng.uniform(0.0, 4.0),
                softness=round(rng.uniform(0.15, 0.4), 2),
                opacity=round(rng.uniform(0.65, 0.9), 2),
            )
            brushes, extra = _add_brush(brushes, new)
            desc = f"Added brush {new.name} (reach {new.radius}, softness {new.params['softness']})"
        elif action == "sharpen-small":
            s = by_radius[0]
            desc = edit(s, "Sharpened", softness=round(s.params.get("softness", 0.5) * 0.6, 2),
                        opacity=round(min(1.0, s.params.get("opacity", 0.6) + 0.1), 2))
        elif action == "add-dotted":
            # Pigment laid as separate dots. Unreachable with one continuous capsule, whatever its numbers.
            s = by_radius[0]
            new = templated(
                _unique_name("stipple", brushes), "dotted",
                radius=max(0.6, s.radius * rng.uniform(0.6, 1.1)),
                length=rng.uniform(4.0, 12.0),
                spacing=round(rng.uniform(1.8, 3.2), 2),
                dot_scale=round(rng.uniform(0.4, 0.7), 2),
                opacity=round(rng.uniform(0.6, 0.9), 2),
            )
            brushes, extra = _add_brush(brushes, new)
            desc = f"Added brush {new.name}, a dotted line (dot every {new.params['spacing']} radii)"
        elif action == "add-wash":
            big = by_radius[-1]
            # Can overshoot the radius limit. The evaluator marks that toolkit non-viable.
            new = templated(
                _unique_name("wash", brushes), "capsule",
                radius=big.radius * rng.uniform(1.1, 1.7),
                length=rng.uniform(6.0, 14.0),
                softness=round(rng.uniform(0.8, 1.0), 2),
                opacity=round(rng.uniform(0.2, 0.35), 2),
                bleed=round(rng.uniform(0.3, 0.6), 2),
                granulation=round(rng.uniform(0.1, 0.3), 2),
            )
            brushes, extra = _add_brush(brushes, new)
            desc = f"Added brush {new.name} (reach {new.radius}, bleed {new.params['bleed']})"
        elif action == "soften-large":
            big = by_radius[-1]
            desc = edit(big, "Softened",
                        softness=round(min(1.0, big.params.get("softness", 0.5) + 0.15), 2),
                        bleed=round(min(1.0, big.params.get("bleed", 0.0) + 0.15), 2),
                        opacity=round(max(0.05, big.params.get("opacity", 0.6) - 0.05), 2))
        elif action == "add-taper":
            big = by_radius[-1]
            new = templated(
                _unique_name("taper", brushes), "taper",
                radius=max(0.6, big.radius * rng.uniform(0.5, 0.9)),
                length=rng.uniform(6.0, 14.0),
                taper=round(rng.uniform(0.5, 0.9), 2),
                fade=round(rng.uniform(0.2, 0.6), 2),
                opacity=round(rng.uniform(0.5, 0.8), 2),
            )
            brushes, extra = _add_brush(brushes, new)
            desc = f"Added brush {new.name}, thinning to {1 - new.params['taper']:.0%} by the end of the stroke"
        elif action == "add-edge":
            new = templated(
                _unique_name("edge", brushes), "capsule",
                radius=rng.uniform(1.5, 3.5),
                length=rng.uniform(6.0, 12.0),
                softness=round(rng.uniform(0.1, 0.25), 2),
                opacity=round(rng.uniform(0.7, 0.85), 2),
            )
            brushes, extra = _add_brush(brushes, new)
            desc = f"Added brush {new.name} (reach {new.radius}, path {new.length})"
        elif action == "add-rake":
            # Parallel bristles, the mark that makes up most of a van Gogh surface.
            mid = by_radius[len(by_radius) // 2]
            reach = max(1.5, mid.radius * rng.uniform(0.8, 1.4))
            new = templated(
                _unique_name("rake", brushes), "rake",
                radius=reach,
                length=rng.uniform(4.0, 12.0),
                # Spacing is in pixels and sized to the brush, so a rake always fits a few bristles in its width.
                gap=round(max(2.0, reach * rng.uniform(0.5, 0.9)), 2),
                width=round(rng.uniform(0.8, 1.4), 2),
                granulation=round(rng.uniform(0.0, 0.25), 2),
                opacity=round(rng.uniform(0.55, 0.85), 2),
            )
            brushes, extra = _add_brush(brushes, new)
            desc = f"Added brush {new.name}, parallel bristles {new.params['gap']}px apart"
        else:  # lengthen-mid
            mid = by_radius[len(by_radius) // 2]
            longer = mid.model_copy(update={"length": round(mid.length + 4.0, 1)})
            replace(mid, longer)
            desc = edit(longer, "Lengthened", softness=round(max(0.0, longer.params.get("softness", 0.5) - 0.1), 2))
            desc = f"{desc}, path now {longer.length}px"

        where = ", ".join(f.data_point_id for f in failure_cases[:3])
        summary = f"[{action}] {desc}{extra}. Targets {len(failure_cases)} {ftype} failures ({where})."
        return [Toolkit(brushes=brushes, from_change_summary=summary)]


class RandomToolkitMutator(Mutator):
    """
    Turns one number at random, with no clamping. The baseline every targeted mutator should beat.

    Most of a brush lives in its code now, so most of the time that number is a literal in the code. That
    keeps this mutator working on brushes an LLM wrote as free-form source, which have no named knobs at all.
    """

    FIELDS = ["radius", "length"]

    def mutate(self, organism, failure_cases, learning_log_entries):
        rng = random.Random()
        brushes = [b.model_copy() for b in organism.brushes]
        i = rng.randrange(len(brushes))
        b = brushes[i]
        if rng.random() < 0.7:
            hit = jitter_source(b.source, rng)
            if hit is not None:
                source, note = hit
                # The code is the brush now, so the params that used to describe it no longer do.
                brushes[i] = b.model_copy(update={"source": source, "params": {}, "template": None})
                return [Toolkit(brushes=brushes, from_change_summary=f"[jitter] Edited {b.name} code: {note}.")]
        field = rng.choice(self.FIELDS)
        old = getattr(b, field)
        new = round(old * rng.uniform(0.6, 1.5) + (0.05 if old == 0 else 0.0), 2)
        brushes[i] = b.model_copy(update={field: new})
        return [Toolkit(brushes=brushes, from_change_summary=f"[jitter] Scaled {b.name}.{field} from {old:.2f} to {new:.2f}.")]


class CrossoverToolkitMutator(Mutator):
    """Borrows a brush from another high-scoring toolkit in the population."""

    def mutate(self, organism, failure_cases, learning_log_entries):
        if self._context is None:
            return []
        rng = random.Random()
        pairs = [(o, r) for o, r in self._context.population.organisms if r.is_viable and o.id != organism.id]
        pairs.sort(key=lambda pair: pair[1].score, reverse=True)
        others = [o for o, _ in pairs[:6]]
        rng.shuffle(others)
        names = {b.name for b in organism.brushes}
        for other in others:
            candidates = [b for b in other.brushes if b.name not in names]
            if not candidates:
                continue
            pick = rng.choice(candidates).model_copy()
            brushes, extra = _add_brush([b.model_copy() for b in organism.brushes], pick)
            summary = f"[crossover] Borrowed {pick.name} (radius {pick.radius}) from toolkit {str(other.id)[:6]}{extra}."
            return [Toolkit(brushes=brushes, additional_parents=[other], from_change_summary=summary)]
        return []
