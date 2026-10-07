"""
The whole-canvas picture an agent gets from `overview`: everything painted so far, shrunk to fit, with a grid
labelled in canvas coordinates and the agent's own viewport outlined. Nothing else is marked, so other agents
stay unseen except through their paint.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from PIL import ImageDraw

from conveyor.commons.tiles import TILE
from conveyor.painting.canvas import _font

OVERVIEW_SIDE = 512  # the longer side of the picture, in pixels, however big the canvas grows
BOX_COLOR = (230, 0, 160)  # magenta: the agent's viewport


def grid_step(scale: float) -> int:
    """Canvas pixels between grid lines: a power-of-two number of tiles, at least 64 picture pixels apart."""
    step = TILE
    while step * scale < 64:
        step *= 2
    return step


def overview_png(img: np.ndarray, window: tuple[int, int, int, int], scale: float,
                 viewport: tuple[int, int, int, int]) -> bytes:
    x0, y0, x1, y1 = window
    h, w = img.shape[:2]
    pic = Image.fromarray(img).convert("RGBA")
    lines = Image.new("RGBA", pic.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lines)
    step = grid_step(scale)
    xs = range(-(-x0 // step) * step, x1 + 1, step)
    ys = range(-(-y0 // step) * step, y1 + 1, step)
    for v in xs:
        d.line([((v - x0) * scale, 0), ((v - x0) * scale, h - 1)], fill=(30, 70, 170, 90), width=1)
    for v in ys:
        d.line([(0, (v - y0) * scale), (w - 1, (v - y0) * scale)], fill=(30, 70, 170, 90), width=1)
    vx0, vy0, vx1, vy1 = ((c - o) * scale for c, o in zip(viewport, (x0, y0, x0, y0)))
    d.rectangle([vx0, vy0, max(vx1 - 1, vx0 + 2), max(vy1 - 1, vy0 + 2)], outline=(*BOX_COLOR, 255), width=2)
    pic = Image.alpha_composite(pic, lines).convert("RGB")
    font = _font(12)
    m = max(30, 8 + max(d.textbbox((0, 0), str(v), font=font)[2] for v in [*xs, *ys, 0]))  # room for the widest label
    out = Image.new("RGB", (w + 2 * m, h + 2 * m), (255, 255, 255))
    out.paste(pic, (m, m))
    d = ImageDraw.Draw(out)

    def label(text: str, cx: float, cy: float) -> None:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        d.text((cx - (right - left) / 2 - left, cy - (bottom - top) / 2 - top), text, font=font, fill=(30, 40, 60))

    for v in xs:
        label(str(v), m + (v - x0) * scale, m / 2)
        label(str(v), m + (v - x0) * scale, out.height - m / 2)
    for v in ys:
        label(str(v), m / 2, m + (v - y0) * scale)
        label(str(v), out.width - m / 2, m + (v - y0) * scale)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
