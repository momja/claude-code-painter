"""Error regions are labelled with pixel ranges, so the model and the harness can't count rows differently."""

from conveyor.painting.canvas import LIMITS
from conveyor.painting.canvas import TARGETS_DIR
from conveyor.painting.canvas import set_canvas_scale
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import load_target
from conveyor.painting.llm_agent import error_grid_text
from conveyor.painting.llm_agent import worst_regions_text
from conveyor.painting.toolkit import initial_toolkit


def test_grid_and_worst_regions_use_pixel_ranges():
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=64)
    canvas = blank(target)
    lines = error_grid_text(canvas, target).splitlines()
    assert "16 × 16 px" in lines[0]
    assert lines[1].split()[:4] == ["x", "0-16", "x", "16-32"]
    assert lines[2].startswith("y 0-16") and lines[-1].startswith("y 48-64")
    worst = worst_regions_text(canvas, target, k=2)
    assert worst.startswith("Worst regions: x ") and worst.count("; ") == 1


def test_grid_columns_stay_apart_on_a_bigger_canvas():
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=128)
    rows, cols = target.grid
    lines = error_grid_text(blank(target), target).splitlines()
    assert lines[1].split()[:2] == ["x", "0-16"]
    for line in lines[2:]:
        parts = line.split()  # "y", "96-112", then one number per column, never run together
        assert parts[0] == "y" and len(parts) == 2 + cols
        assert all(0.0 <= float(v) <= 1.0 for v in parts[2:])
    assert len(lines) == 2 + rows


def test_fine_toolkit_adds_brushes_thin_enough_for_ears():
    assert [b.name for b in initial_toolkit().brushes] == ["flat_wash", "round_mid"]
    fine = initial_toolkit(fine=True)
    assert min(2 * b.radius for b in fine.brushes) < 4  # the eyes are about 3 px wide at 64 px


def test_brush_sizes_and_limits_scale_with_the_canvas():
    base = initial_toolkit(fine=True)
    try:
        assert set_canvas_scale(128) == 2.0
        assert LIMITS["radius"][1] == 24.0 and LIMITS["length"][1] == 48.0
        big = initial_toolkit(fine=True, scale=2.0)
        assert [b.radius for b in big.brushes] == [2 * b.radius for b in base.brushes]
        assert [b.length for b in big.brushes] == [2 * b.length for b in base.brushes]
        assert set_canvas_scale(48) == 1.0  # small canvases keep the base limits
        assert LIMITS["radius"][1] == 12.0
    finally:
        set_canvas_scale(64)  # module-level limits: leave them as we found them
