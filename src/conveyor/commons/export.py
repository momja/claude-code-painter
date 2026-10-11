"""
A canvas as one PNG, for keeping or comparing the result.

The picture is the frame on a framed canvas, and on a canvas with no edges the reach of the paint, tile-aligned.
It's the paint alone: no sketch, no viewports. It comes at full size, one picture pixel per canvas pixel, unless
its longest side would pass `side`, and then it's shrunk to fit. `seq` gives the canvas as it stood after that op,
so replay can export the moment it shows.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3

from PIL import Image

from conveyor.commons.tiles import TILE
from conveyor.commons.tiles import compose

DEFAULT_SIDE = 4096  # the longest side of an export unless asked otherwise: 48 MB of pixels at most while it's made
MAX_SIDE = 8192


def window(conn: sqlite3.Connection, canvas_id: str, config: dict, seq: int | None = None) -> tuple[int, int, int, int]:
    """What an export shows: the frame, or the paint's reach (as of `seq`), or one blank viewport at the origin."""
    if config.get("frame"):
        return tuple(config["frame"])
    if seq is None:
        lo = conn.execute("SELECT min(tx), min(ty), max(tx), max(ty) FROM canvas_heads WHERE canvas_id=?",
                          (canvas_id,)).fetchone()
    else:
        lo = conn.execute("SELECT min(tx), min(ty), max(tx), max(ty) FROM canvas_tiles WHERE canvas_id=? AND seq<=?",
                          (canvas_id, seq)).fetchone()
    if lo[0] is None:
        side = int(config.get("viewport", 512))
        return 0, 0, side, side
    return lo[0] * TILE, lo[1] * TILE, (lo[2] + 1) * TILE, (lo[3] + 1) * TILE


def export_png(conn: sqlite3.Connection, canvas_id: str, side: int | None = None, seq: int | None = None) -> bytes:
    """The canvas as a PNG whose longest side is at most `side` (DEFAULT_SIDE, at most MAX_SIDE). KeyError for an
    unknown canvas, ValueError for a bad size."""
    row = conn.execute("SELECT config FROM canvases WHERE id=?", (canvas_id,)).fetchone() if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='canvases'").fetchone() else None
    if row is None:
        raise KeyError(canvas_id)
    side = DEFAULT_SIDE if side is None else int(side)
    if not 16 <= side <= MAX_SIDE:
        raise ValueError(f"side must be 16 to {MAX_SIDE} pixels")
    box = window(conn, canvas_id, json.loads(row[0]), seq)
    scale = min(1.0, side / max(box[2] - box[0], box[3] - box[1]))
    out = io.BytesIO()
    Image.fromarray(compose(conn, canvas_id, box, scale, seq)).save(out, format="PNG")
    return out.getvalue()


def filename(name: str, seq: int | None = None) -> str:
    """A file name for an export of the canvas called `name`."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:60] or "canvas"
    return f"{stem}{f'-op{seq}' if seq is not None else ''}.png"
