"""The oracle keeps up with the agent's stroke budget, so it never loses to the agent it's judging."""

import math
from collections import Counter

from conveyor.painting import toolkit as toolkit_module
from conveyor.painting.canvas import TARGETS_DIR
from conveyor.painting.canvas import load_target
from conveyor.painting.toolkit import OracleFitter
from conveyor.painting.toolkit import Toolkit
from conveyor.painting.toolkit import initial_toolkit
from conveyor.painting.toolkit import templated


def test_oracle_budget_scales_with_the_agent():
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=64)
    rows, cols = target.grid
    assert OracleFitter().strokes_per_patch(target) == OracleFitter.STROKES
    assert OracleFitter(stroke_budget=100).strokes_per_patch(target) == OracleFitter.STROKES  # the floor wins
    per_patch = OracleFitter(stroke_budget=500).strokes_per_patch(target)
    assert per_patch == math.ceil(500 / (rows * cols))
    assert per_patch * rows * cols >= 500


def test_a_bigger_toolkit_never_fits_worse():
    """
    The oracle decides what the toolkit gets blamed for, so its search must not punish a toolkit for having
    more brushes. It used to: one brush drawn at random and a fixed six placements meant every extra brush
    got fewer tries, so adding brushes raised the toolkit's own blame. Every brush now gets the same tries.
    """
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=64)
    seed = initial_toolkit(fine=True).brushes
    richer = seed + [
        templated("rake", "rake", radius=2.0, length=5.0, gap=3.0, width=1.0, opacity=0.75),
        templated("dots", "dotted", radius=1.0, length=4.0, spacing=2.4, opacity=0.8),
        templated("taper", "taper", radius=2.5, length=6.0, taper=0.8, fade=0.4, opacity=0.75),
    ]
    thin, _ = OracleFitter().fit(Toolkit(brushes=seed), target)
    fat, _ = OracleFitter().fit(Toolkit(brushes=richer), target)
    assert fat.mean() <= thin.mean(), (
        f"a superset of the same brushes fit worse ({fat.mean():.4f} vs {thin.mean():.4f}): "
        "the oracle's search is being diluted by toolkit size again"
    )


def test_every_brush_gets_tried_the_same_number_of_times():
    """What the blame split rests on: the search is exhaustive over brushes, not a lottery between them."""
    calls = []
    real = toolkit_module.stroke_alpha

    def spy(brush, *args, **kwargs):
        calls.append(brush.name)
        return real(brush, *args, **kwargs)

    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=32)
    tk = initial_toolkit(fine=True)
    toolkit_module.stroke_alpha = spy
    try:
        OracleFitter().fit(tk, target)
    finally:
        toolkit_module.stroke_alpha = real
    counts = Counter(calls)
    assert set(counts) == {b.name for b in tk.brushes}
    assert len(set(counts.values())) == 1, f"brushes got uneven tries: {counts}"
