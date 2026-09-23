"""The coordinate grid on the model's images, and stroke pressure."""

import io

import numpy as np
from PIL import Image

from conveyor.painting.canvas import TARGETS_DIR
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import load_target
from conveyor.painting.llm_agent import IMAGE_SCALE
from conveyor.painting.llm_agent import grid_margin
from conveyor.painting.llm_agent import grid_step
from conveyor.painting.llm_agent import gridded_png
from conveyor.painting.llm_agent import stroke_pressure


def test_grid_image_has_label_margins_and_lines():
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=64)
    im = Image.open(io.BytesIO(gridded_png(blank(target)))).convert("RGB")
    s, step = IMAGE_SCALE, grid_step(64)
    m = grid_margin(64 * s)
    assert im.size == (64 * s + 2 * m, 64 * s + 2 * m)
    arr = np.asarray(im).astype(int)
    on_line = arr[m + 5, m + step * s]
    between = arr[m + 5, m + step * s + 5]
    assert np.abs(on_line - between).sum() > 20  # the line is visible on blank paper
    assert (arr[: m // 2, m:-m] < 200).any()  # labels drawn in the top margin


def test_grid_spacing_and_margins_scale_with_the_canvas():
    # About eight divisions across at any size, so the labels don't crowd each other.
    assert (grid_step(64), grid_step(128), grid_step(256)) == (8, 16, 32)
    assert 128 / grid_step(128) == 64 / grid_step(64)
    assert grid_margin(128 * IMAGE_SCALE) > grid_margin(64 * IMAGE_SCALE)
    big = load_target(TARGETS_DIR / "self_portrait.jpg", width=128)
    im = Image.open(io.BytesIO(gridded_png(blank(big)))).convert("RGB")
    assert im.size[0] == 128 * IMAGE_SCALE + 2 * grid_margin(128 * IMAGE_SCALE)


def test_pressure_parsing():
    assert stroke_pressure({}) == 1.0
    assert stroke_pressure({"pressure": 0.3}) == 0.3
    assert stroke_pressure({"pressure": 5}) == 1.0
    assert stroke_pressure({"pressure": 0}) == 0.1
    assert stroke_pressure({"pressure": "hard"}) is None
