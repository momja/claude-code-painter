"""
The shared canvas in the database: an unbounded plane of tiles.

A tile is TILE x TILE pixels, stored as a PNG. A tile nobody has painted is blank paper and has no row. Every tool
call an agent makes is one row in `canvas_ops`, and its `seq` orders the whole canvas's history. A call that
changes pixels also writes a new version of each tile it changed, keyed by that seq, and moves the tile's head.
The canvas as it stood after any op is, tile by tile, the newest version at or before that op, so replay reads
stored pixels and never runs instrument code again.

Several agents paint at once, each from its own MCP server process. A paint call reads the tiles under the
agent's viewport, applies the call, and writes back what changed inside one `BEGIN IMMEDIATE` transaction, so
calls are serialized and no agent's marks are lost to a stale read. An instrument call is held to one second and
usually takes milliseconds, so the lock is short; a waiting writer gets the store's busy timeout.

Pixels are stored as 8-bit RGB. Paint composites in floating point within a call and is rounded when the call
commits, which is plenty for paint that layers.
"""

from __future__ import annotations

import io
import json
import math
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
from PIL import Image

from conveyor.commons.lettering import GLYPH_W
from conveyor.commons.lettering import LINE_H
from conveyor.commons.lettering import wrap
from conveyor.painting.canvas import PAPER
from conveyor.painting.canvas import parse_color
from conveyor.store import BUSY_TIMEOUT
from conveyor.store import dumps
from conveyor.store import init_db
from conveyor.store import new_id

TILE = 128
PAPER_RGB = np.asarray([round(c * 255) for c in PAPER], dtype=np.uint8)
DEFAULT_VIEWPORT = 512
DEFAULT_MAX_CALLS = 100
VIEWPORT_RANGE = (64, 1024)  # a canvas's default viewport, and any one agent's
MAX_TASK = 2000  # characters in a canvas's task
DEFAULT_INK = "#1d2a3a"
MOVE_SHARE = 0.75  # the farthest one move goes, as a share of the viewport side; consecutive views always overlap


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def encode_tile(arr: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return out.getvalue()


def decode_tile(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def blank_tile() -> np.ndarray:
    return np.broadcast_to(PAPER_RGB, (TILE, TILE, 3)).copy()


def message(seq: int, vx: int, vy: int, args: dict, viewport: int) -> dict:
    """A `write_message` op as the text layer draws it, in canvas pixels. (vx, vy) is the writer's viewport corner;
    the text was wrapped at the viewport's right edge and cut at its bottom, and stays that way. Agents can have
    viewports of their own size, so the op records the writer's; older ops use the canvas's, passed as `viewport`."""
    viewport = int(args.get("viewport") or viewport)
    ax, ay, scale = int(round(float(args.get("x", 0)))), int(round(float(args.get("y", 0)))), int(args.get("scale", 2))
    width, height = viewport - ax, viewport - ay
    lines = wrap(args["text"], width, scale)
    return {"seq": seq, "x": vx + ax, "y": vy + ay, "text": args["text"], "scale": scale, "width": width,
            "height": height, "color": parse_color(args.get("color") or DEFAULT_INK), "lines": lines,
            "w": min(width, max(len(line) for line in lines) * GLYPH_W * scale),
            "h": min(height, len(lines) * LINE_H * scale)}


def create_canvas(db: str | Path, name: str, viewport: int = DEFAULT_VIEWPORT,
                  max_calls: int = DEFAULT_MAX_CALLS, task: str | None = None) -> dict:
    """A new canvas. `task`, when given, is the one image every agent on it is told to make together."""
    init_db(db)
    row = {"id": new_id(), "name": name, "created": time.time(),
           "config": {"viewport": int(viewport), "max_calls": int(max_calls), "tile": TILE}}
    if task:
        row["config"]["task"] = task
    conn = sqlite3.connect(db, timeout=BUSY_TIMEOUT)
    try:
        conn.execute("INSERT INTO canvases (id, name, created, config) VALUES (?, ?, ?, ?)",
                     (row["id"], name, row["created"], dumps(row["config"])))
        conn.commit()
    finally:
        conn.close()
    return row


@dataclass
class Region:
    """The pixels of a window on the canvas, read from the tiles under it."""

    x: int
    y: int
    w: int
    h: int
    tiles: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)  # every tile the window touches

    @property
    def _origin(self) -> tuple[int, int]:
        return (self.x // TILE) * TILE, (self.y // TILE) * TILE

    def _mosaic(self) -> np.ndarray:
        ox, oy = self._origin
        tx0, ty0 = ox // TILE, oy // TILE
        cols = (self.x + self.w - 1) // TILE - tx0 + 1
        rows = (self.y + self.h - 1) // TILE - ty0 + 1
        big = np.empty((rows * TILE, cols * TILE, 3), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                big[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = self.tiles[(tx0 + c, ty0 + r)]
        return big

    def _crop(self) -> tuple[slice, slice]:
        ox, oy = self._origin
        return slice(self.y - oy, self.y - oy + self.h), slice(self.x - ox, self.x - ox + self.w)

    def image(self) -> np.ndarray:
        """The window as float32 RGB in 0..1, the way `Canvas` holds paint."""
        return self._mosaic()[self._crop()].astype(np.float32) / 255.0

    def changed(self, img: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        """The tiles that `img`, the window after a call, changes, each as its whole new tile."""
        before = self._mosaic()
        after = before.copy()
        after[self._crop()] = to_uint8(img)
        ox, oy = self._origin
        out = {}
        for (tx, ty) in self.tiles:
            r, c = ty * TILE - oy, tx * TILE - ox
            new = after[r:r + TILE, c:c + TILE]
            if not np.array_equal(new, before[r:r + TILE, c:c + TILE]):
                out[(tx, ty)] = new.copy()
        return out


class SharedCanvas:
    """One process's connection to one canvas. Not thread-safe: a process paints from one thread."""

    def __init__(self, db: str | Path, canvas_id: str) -> None:
        self.db = Path(db)
        self.canvas_id = canvas_id
        self.conn = sqlite3.connect(self.db, timeout=BUSY_TIMEOUT, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        row = self.conn.execute("SELECT * FROM canvases WHERE id=?", (canvas_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown canvas: {canvas_id}")
        self.name = row["name"]
        self.config = json.loads(row["config"])

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """A write transaction: the read, the call and the write-back of one tool call, as one step."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def read(self, x: int, y: int, w: int, h: int) -> Region:
        tx0, ty0, tx1, ty1 = x // TILE, y // TILE, (x + w - 1) // TILE, (y + h - 1) // TILE
        found = {}
        for row in self.conn.execute(
                "SELECT t.tx, t.ty, t.data FROM canvas_heads h JOIN canvas_tiles t ON t.canvas_id=h.canvas_id "
                "AND t.tx=h.tx AND t.ty=h.ty AND t.seq=h.seq WHERE h.canvas_id=? AND h.tx BETWEEN ? AND ? "
                "AND h.ty BETWEEN ? AND ?", (self.canvas_id, tx0, tx1, ty0, ty1)):
            found[(row["tx"], row["ty"])] = decode_tile(row["data"])
        tiles = {(tx, ty): found.get((tx, ty)) if (tx, ty) in found else blank_tile()
                 for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)}
        return Region(x, y, w, h, tiles)

    def extent(self) -> tuple[int, int, int, int] | None:
        """How far the paint reaches, tile-aligned (x0, y0, x1, y1), or None on blank paper."""
        lo = self.conn.execute("SELECT min(tx), min(ty), max(tx), max(ty) FROM canvas_heads WHERE canvas_id=?",
                               (self.canvas_id,)).fetchone()
        return None if lo[0] is None else (lo[0] * TILE, lo[1] * TILE, (lo[2] + 1) * TILE, (lo[3] + 1) * TILE)

    def shrunk(self, window: tuple[int, int, int, int], scale: float) -> np.ndarray:
        """The window x0..x1, y0..y1 at `scale` picture px per canvas px (at most 1), as uint8 RGB. Tiles are
        shrunk one at a time, so a wide window is never held at full size."""
        x0, y0, x1, y1 = window
        w, h = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
        out = np.broadcast_to(PAPER_RGB, (h, w, 3)).copy()
        for row in self.conn.execute(
                "SELECT t.tx, t.ty, t.data FROM canvas_heads h JOIN canvas_tiles t ON t.canvas_id=h.canvas_id "
                "AND t.tx=h.tx AND t.ty=h.ty AND t.seq=h.seq WHERE h.canvas_id=? AND h.tx BETWEEN ? AND ? "
                "AND h.ty BETWEEN ? AND ?", (self.canvas_id, x0 // TILE, (x1 - 1) // TILE, y0 // TILE, (y1 - 1) // TILE)):
            left, top = math.floor((row["tx"] * TILE - x0) * scale), math.floor((row["ty"] * TILE - y0) * scale)
            right = max(left + 1, math.floor(((row["tx"] + 1) * TILE - x0) * scale))
            bottom = max(top + 1, math.floor(((row["ty"] + 1) * TILE - y0) * scale))
            ys, xs = slice(max(0, top), min(h, bottom)), slice(max(0, left), min(w, right))
            if ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            with Image.open(io.BytesIO(row["data"])) as im:
                small = np.asarray(im.convert("RGB").resize((right - left, bottom - top), Image.Resampling.BOX))
            out[ys, xs] = small[ys.start - top:ys.stop - top, xs.start - left:xs.stop - left]
        return out

    def messages(self, x0: int, y0: int, x1: int, y1: int) -> list[dict]:
        """Every message that reaches into the window x0..x1, y0..y1, oldest first. A message is an op, not
        pixels, so paint never covers one."""
        size, out = int(self.config["viewport"]), []
        for row in self.conn.execute("SELECT seq, x, y, args FROM canvas_ops WHERE canvas_id=? AND tool='write_message' "
                                     "AND status IN ('painted', 'written') ORDER BY seq", (self.canvas_id,)):
            m = message(row["seq"], row["x"], row["y"], json.loads(row["args"]), size)
            pad = m["scale"]  # the outline
            if m["x"] - pad < x1 and m["x"] + m["w"] + pad > x0 and m["y"] - pad < y1 and m["y"] + m["h"] + pad > y0:
                out.append(m)
        return out

    def record(self, *, agent_id: str | None, tool: str, status: str, x: int, y: int, args: dict | None = None,
               note: str = "", tiles: dict[tuple[int, int], np.ndarray] | None = None) -> int:
        """Log one tool call and the tiles it changed. Call inside `transaction()` when tiles are written."""
        cur = self.conn.execute(
            "INSERT INTO canvas_ops (canvas_id, agent_id, ts, tool, status, args, note, x, y, tiles) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self.canvas_id, agent_id, time.time(), tool, status, dumps(args) if args is not None else None,
             note[:2000], x, y, len(tiles or {})))
        seq = int(cur.lastrowid)
        for (tx, ty), arr in (tiles or {}).items():
            self.conn.execute("INSERT INTO canvas_tiles (canvas_id, tx, ty, seq, data) VALUES (?, ?, ?, ?, ?)",
                              (self.canvas_id, tx, ty, seq, encode_tile(arr)))
            self.conn.execute("INSERT OR REPLACE INTO canvas_heads (canvas_id, tx, ty, seq) VALUES (?, ?, ?, ?)",
                              (self.canvas_id, tx, ty, seq))
        return seq

    def update_agent(self, agent_id: str, **fields) -> None:
        allowed = {"x", "y", "calls_used", "last_ts", "status", "ended", "session_id", "error"}
        if set(fields) - allowed:
            raise ValueError("unknown agent fields")
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE canvas_agents SET {cols} WHERE id=?", (*fields.values(), agent_id))
