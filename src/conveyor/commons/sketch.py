"""
The sketch: lines a person draws on the canvas from its page, to show the agents what they want.

A sketch line is an op, like an agent's message, not pixels, so it never touches the tiles. Every picture an agent
is shown has the lines drawn over the paint and under the text, outlined in a contrasting colour so they stand out
on any paint, and paint never covers them. Erasing lines is an op too, so replay shows each line from when it was
drawn until it was erased.
"""

from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
from PIL import Image
from PIL import ImageDraw

from conveyor.commons.lettering import halo_color
from conveyor.painting.canvas import CanvasError
from conveyor.painting.canvas import parse_color

MAX_POINTS = 2000  # in one line; a longer one is sent as several
WIDTH_RANGE = (1, 64)  # canvas pixels
DEFAULT_SKETCH = "#0a84ff"
HALO = 2  # canvas pixels of outline on each side of a line
SMOOTH = 3  # times larger the lines are drawn before they're shrunk onto the picture
FAR = 10_000_000  # no point lies farther from the origin than this


class SketchError(ValueError):
    pass


def parse_line(body: object) -> dict:
    """A line as the page sends it, {points: [[x, y], ...], color, width} in canvas pixels, checked and rounded."""
    if not isinstance(body, dict) or set(body) - {"points", "color", "width"}:
        raise SketchError("Expected points, color and width.")
    points = body.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS:
        raise SketchError(f"points must be a list of 1 to {MAX_POINTS} [x, y] pairs.")
    out = []
    for p in points:
        if (not isinstance(p, list) or len(p) != 2 or any(isinstance(c, bool) or not isinstance(c, (int, float))
                                                          or not math.isfinite(c) or abs(c) > FAR for c in p)):
            raise SketchError("Each point must be [x, y] in canvas pixels.")
        out.append([round(float(p[0]), 1), round(float(p[1]), 1)])
    width = body.get("width", 6)
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not WIDTH_RANGE[0] <= width <= WIDTH_RANGE[1]:
        raise SketchError(f"width must be {WIDTH_RANGE[0]} to {WIDTH_RANGE[1]} pixels.")
    color = body.get("color") or DEFAULT_SKETCH
    try:
        parse_color(color)
    except (CanvasError, TypeError, ValueError):
        raise SketchError("color must be a hex colour like #0a84ff.") from None
    return {"points": out, "color": color, "width": round(float(width), 1)}


def bounds(line: dict) -> tuple[float, float, float, float]:
    """Everything the line covers, its outline included: (x0, y0, x1, y1) in canvas pixels."""
    xs, ys = [p[0] for p in line["points"]], [p[1] for p in line["points"]]
    pad = line["width"] / 2 + HALO
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def lines(conn: sqlite3.Connection, canvas_id: str) -> list[dict]:
    """Every line on the canvas now, oldest first: {seq, ts, points, color, width}."""
    drawn, erased = [], set()
    for row in conn.execute("SELECT seq, ts, tool, args FROM canvas_ops WHERE canvas_id=? AND tool IN "
                            "('sketch', 'erase_sketch') ORDER BY seq", (canvas_id,)):
        args = json.loads(row[3] or "{}")
        if row[2] == "sketch":
            drawn.append({"seq": row[0], "ts": row[1], **args})
        else:
            erased.update(args.get("seqs", []))
    return [line for line in drawn if line["seq"] not in erased]


def draw_sketch(img: np.ndarray, x0: float, y0: float, sketch: list[dict], scale: float = 1.0) -> np.ndarray:
    """`img` (float RGB in 0..1) showing the canvas from (x0, y0) at `scale` picture px per canvas px, with the
    lines drawn on top, oldest first, each outlined. Returns a new array."""
    if not sketch:
        return img.copy()
    h, w = img.shape[:2]
    layer = Image.new("RGBA", (w * SMOOTH, h * SMOOTH), (0, 0, 0, 0))  # drawn large and shrunk, for smooth edges
    draw = ImageDraw.Draw(layer)
    for line in sketch:
        lx0, ly0, lx1, ly1 = bounds(line)
        if lx1 < x0 or ly1 < y0 or lx0 > x0 + w / scale or ly0 > y0 + h / scale:
            continue
        ink = parse_color(line["color"])
        pts = [((px - x0) * scale * SMOOTH, (py - y0) * scale * SMOOTH) for px, py in line["points"]]
        core = max(1.0, line["width"] * scale) * SMOOTH
        for color, width in ((halo_color(ink), core + 2 * max(1.0, HALO * scale) * SMOOTH), (ink, core)):
            fill = tuple(round(c * 255) for c in color) + (255,)
            if len(pts) > 1:
                draw.line(pts, fill=fill, width=max(1, round(width)), joint="curve")
            r = width / 2  # round ends, and a dot for a single point
            for px, py in (pts[0], pts[-1]):
                draw.ellipse((px - r, py - r, px + r, py + r), fill=fill)
    rgba = np.asarray(layer.resize((w, h), Image.Resampling.BOX), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:]
    return img * (1 - alpha) + rgba[..., :3] * alpha
