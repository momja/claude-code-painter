"""Brushes are programs. What they may do, what happens when they misbehave, and what that buys a painter."""

import itertools
import random

import numpy as np


from conveyor.painting.brushcode import BrushCodeError
from conveyor.painting.brushcode import compile_brush
from conveyor.painting.brushcode import jitter_source
from conveyor.painting.brushcode import render_source
from conveyor.painting.canvas import Brush
from conveyor.painting.canvas import physics_violations
from conveyor.painting.canvas import stroke_alpha
from conveyor.painting.toolkit import Toolkit
from conveyor.painting.toolkit import initial_toolkit
from conveyor.painting.toolkit import templated

DOTTED = """def alpha(u, v, rng, radius, length):
    phase = np.abs(np.mod(u, 6.0) - 3.0)
    return np.clip((radius - np.hypot(phase, v)) / 1.0 + 0.5, 0.0, 1.0)
"""


def _accepted(sources: dict[str, str]) -> list[str]:
    """Which of these the sandbox let through. Returned by name so a failure says what got in."""
    out = []
    for label, source in sources.items():
        try:
            compile_brush(source)
            out.append(label)
        except BrushCodeError:
            pass
    return out


def mask(brush: Brush, size: int = 40) -> np.ndarray:
    """The brush laid down the middle of a small canvas, as a plain 2D array."""
    hit = stroke_alpha(brush, 4.0, size / 2, 0.0, size, size, np.random.default_rng(0))
    assert hit is not None
    ys, xs, a = hit
    out = np.zeros((size, size), dtype=np.float32)
    out[ys, xs] = a[..., 0]
    return out


def test_a_brush_can_draw_a_dotted_line():
    """The mark the old parameter-only brush could not make at any setting: pigment with gaps in it."""
    brush = Brush(name="stipple", radius=1.4, length=24.0, source=DOTTED)
    along = mask(brush)[20]  # the row the stroke runs down
    inked = along > 0.5
    assert inked.any()
    # Runs of ink separated by runs of bare paper, rather than one continuous bar.
    runs = [(bool(v), len(list(g))) for v, g in itertools.groupby(inked)]
    gaps = [n for v, n in runs if not v and n > 1]
    dots = [n for v, n in runs if v]
    assert len(dots) >= 3, f"expected several dots, got runs {runs}"
    assert gaps, "a dotted line needs bare paper between the dots"

    # The stock capsule cannot do it, whatever its numbers: its ink is one unbroken run.
    solid = templated("solid", "capsule", radius=1.4, length=24.0, opacity=1.0, softness=0.0)
    solid_runs = [(bool(v), len(list(g))) for v, g in itertools.groupby(mask(solid)[20] > 0.5)]
    assert len([n for v, n in solid_runs if v]) == 1


def test_a_rake_lays_parallel_bristles():
    """Modulating across the path instead of along it, which is most of a van Gogh surface."""
    rake = templated("rake", "rake", radius=6.0, length=6.0, gap=3.5, width=1.0, granulation=0.0, opacity=1.0)
    across = mask(rake)[:, 6]  # a slice cut across the stroke
    bands = [v for v, _ in itertools.groupby(across > 0.3)]
    assert sum(1 for v in bands if v) >= 3, f"a rake should leave several separate ridges, got {across.round(2)}"


def test_the_sandbox_refuses_the_ways_out():
    attempts = {
        "import": "import os\ndef alpha(u, v, rng, radius, length):\n    return u * 0 + 1",
        "dunder": "def alpha(u, v, rng, radius, length):\n    return u.__class__",
        "numpy's ctypes door": "def alpha(u, v, rng, radius, length):\n    return np.ctypeslib.ctypes.CDLL('x')",
        "aliasing past it": "def alpha(u, v, rng, radius, length):\n    x = np\n    return x.ctypeslib",
        "builtins": "def alpha(u, v, rng, radius, length):\n    return open('/etc/passwd')",
        "unbounded loop": "def alpha(u, v, rng, radius, length):\n    while True:\n        pass",
    }
    assert _accepted(attempts) == []


def test_the_sandbox_refuses_code_that_is_not_a_brush():
    bad = {
        "no entry point": "def other(u, v, rng, radius, length):\n    return u",
        "wrong signature": "def alpha(u, v):\n    return u",
        "wrong shape": "def alpha(u, v, rng, radius, length):\n    return 1.0",
        "outside 0 to 1": "def alpha(u, v, rng, radius, length):\n    return u * 0 + 4.0",
        "lays nothing": "def alpha(u, v, rng, radius, length):\n    return u * 0",
        "raises": "def alpha(u, v, rng, radius, length):\n    return u[999999]",
        "syntax": "def alpha(u, v, rng, radius, length)\n    return u",
    }
    assert _accepted(bad) == []


def test_broken_brush_code_makes_the_toolkit_non_viable_instead_of_crashing():
    broken = Brush(name="bad", radius=2.0, source="def alpha(u, v, rng, radius, length):\n    return u * 0\n")
    problems = physics_violations([broken])
    assert len(problems) == 1 and problems[0].startswith("bad: ")
    assert physics_violations(initial_toolkit(fine=True).brushes) == []


def test_a_brush_that_breaks_only_at_paint_time_costs_a_stroke_not_the_run():
    """The probe runs on one grid shape. A brush that fails on another lays nothing and scores badly."""
    sometimes = Brush(
        name="flaky",
        radius=3.0,
        # Fine on the square probe grid, wrong shape on anything else.
        source="def alpha(u, v, rng, radius, length):\n    return np.clip(u * 0 + 0.5, 0, 1)[:9, :9]\n",
    )
    assert physics_violations([sometimes]) == []
    assert stroke_alpha(sometimes, 20.0, 20.0, 0.0, 64, 64, np.random.default_rng(0)) is None


def test_size_limits_still_bite_on_code_brushes():
    """Whatever the program does, it only ever draws inside the box `radius` and `length` buy it."""
    greedy = Brush(name="stamp", radius=3.0, length=0.0,
                   source="def alpha(u, v, rng, radius, length):\n    return u * 0 + 1.0\n")
    painted = mask(greedy, size=40) > 0
    ys, xs = np.nonzero(painted)
    assert ys.max() - ys.min() <= 2 * (3.0 + 1.5) + 1
    assert xs.max() - xs.min() <= 2 * (3.0 + 1.5) + 1


def test_the_capsule_template_still_draws_the_old_mark():
    """Seed toolkits start where the parameter-only ones did: a soft round head dragged along the path."""
    b = templated("round", "capsule", radius=4.0, length=6.0, softness=0.5, opacity=1.0)
    img = mask(b)                    # the path runs from x=4 to x=10 down row 20
    assert img[20, 6] > img[20, 18]  # ink along the path, bare paper past its end
    assert img[20, 6] > img[15, 6]   # solid on the centre line, fading across it
    assert img.max() <= 1.0
    assert (img[20, 4:11] > 0.9).all()  # one unbroken mark, which is the shape the old toolkit was stuck with


def test_jitter_turns_a_number_in_whatever_code_it_is_given():
    """How the random mutator keeps a grip on brushes an LLM wrote, which have no named knobs."""
    source = render_source("dotted", {})
    out = jitter_source(source, random.Random(3))
    assert out is not None
    edited, note = out
    assert edited != source and note.startswith("line ")
    compile_brush(edited)  # still a working brush

    # Nothing to turn: no literal in the body that isn't zero.
    bare = "def alpha(u, v, rng, radius, length):\n    return np.abs(np.clip(v, -radius, radius)) / radius\n"
    assert jitter_source(bare, random.Random(0)) is None


def test_toolkit_text_is_the_code_that_runs():
    text = Toolkit(brushes=[templated("stipple", "dotted", radius=2.0, length=8.0)]).render_text()
    assert "def stipple(u, v, rng, radius, length):" in text
    assert "np.mod(u" in text  # the actual body, not a summary of it
