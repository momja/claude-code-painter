"""
Export a canvas's paint as a still PNG, an animated GIF, or a PNG sheet of its progress.

History exports start on blank paper and sample tool-call counts on a cubic curve, including the last call. They read
stored tile versions, never rerun instruments, and keep the frame or the final paint's reach as one fixed crop.
`seq` stops the history at that op, so replay can export only the work it shows. Sketches and viewports stay out.
"""

from __future__ import annotations

import io
import json
import math
import re
import sqlite3

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL.PngImagePlugin import PngInfo

from conveyor.commons.tiles import TILE
from conveyor.commons.tiles import compose

DEFAULT_SIDE = 4096
MAX_SIDE = 8192
ANIMATION_SIDE = 1024
ANIMATION_FRAMES = 60
MAX_ANIMATION_SIDE = 2048
MAX_ANIMATION_FRAMES = 240
SHEET_SIDE = 512
SHEET_FRAMES = 9
MAX_SHEET_SIDE = 1024
MAX_SHEET_FRAMES = 24
MAX_HISTORY_PIXELS = 64_000_000  # bound both the GIF's retained palette frames and the sheet's RGB allocation


def _canvas(conn: sqlite3.Connection, canvas_id: str) -> tuple[str, dict]:
    row = conn.execute("SELECT name, config FROM canvases WHERE id=?", (canvas_id,)).fetchone() if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='canvases'").fetchone() else None
    if row is None:
        raise KeyError(canvas_id)
    return row[0], json.loads(row[1])


def _bounded(value: int | None, default: int, low: int, high: int, name: str) -> int:
    value = default if value is None else int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be {low} to {high}")
    return value


def _seq(seq: int | None) -> int | None:
    return None if seq is None else _bounded(seq, 0, 0, 2**63 - 1, "seq")


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


def _scale(box: tuple[int, int, int, int], side: int) -> float:
    return min(1.0, side / max(box[2] - box[0], box[3] - box[1]))


def _dimensions(box: tuple[int, int, int, int], scale: float) -> tuple[int, int]:
    return max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale))


def export_png(conn: sqlite3.Connection, canvas_id: str, side: int | None = None, seq: int | None = None) -> bytes:
    """The canvas as one PNG. KeyError for an unknown canvas, ValueError for a bad size or sequence."""
    _, config = _canvas(conn, canvas_id)
    side = _bounded(side, DEFAULT_SIDE, 16, MAX_SIDE, "side")
    seq = _seq(seq)
    box = window(conn, canvas_id, config, seq)
    out = io.BytesIO()
    Image.fromarray(compose(conn, canvas_id, box, _scale(box, side), seq)).save(out, format="PNG")
    return out.getvalue()


def stages(conn: sqlite3.Connection, canvas_id: str, frames: int, seq: int | None = None) -> list[tuple[int, int]]:
    """Early-weighted (op sequence, call count) pairs, from blank paper to the last op at or before `seq`.

    Sequences are database-wide, not per-canvas counters. Sample ranks within this canvas's calls instead of
    numeric sequence gaps. Pin the last op once, so agents can keep painting without changing this export.
    Read the sequence column in one pass, retaining only the chosen stages, even for a long history.
    """
    frames = _bounded(frames, ANIMATION_FRAMES, 2, MAX_ANIMATION_FRAMES, "frames")
    seq = _seq(seq)
    end = conn.execute("SELECT coalesce(max(seq), 0) FROM canvas_ops WHERE canvas_id=?"
                       + (" AND seq<=?" if seq is not None else ""),
                       (canvas_id, seq) if seq is not None else (canvas_id,)).fetchone()[0]
    count = conn.execute("SELECT count(*) FROM canvas_ops WHERE canvas_id=? AND seq<=?",
                         (canvas_id, end)).fetchone()[0]
    if not count:
        return [(0, 0)]
    frames = min(frames, count + 1)
    # Cubic spacing gives the first 10% of calls roughly 46% of the stages in a long history.
    # Keep ranks distinct after rounding, leaving room for every remaining stage and the final call.
    ranks = set()
    previous = 0
    denominator = (frames - 1) ** 3
    for i in range(1, frames):
        target = (i ** 3 * count + denominator // 2) // denominator
        rank = min(count - (frames - 1 - i), max(previous + 1, target))
        ranks.add(rank)
        previous = rank
    result = [(0, 0)]
    for rank, (op,) in enumerate(conn.execute("SELECT seq FROM canvas_ops WHERE canvas_id=? AND seq<=? ORDER BY seq",
                                            (canvas_id, end)), 1):
        if rank in ranks:
            result.append((op, rank))
    return result


def export_gif(conn: sqlite3.Connection, canvas_id: str, side: int | None = None, seq: int | None = None,
               frames: int | None = None, fps: int = 10) -> bytes:
    """A looping paint-history GIF. Hold blank paper for at least half a second and the result for 1.5 seconds.
    `frames` is a maximum, including blank paper and the result; short histories need fewer frames.
    """
    _, config = _canvas(conn, canvas_id)
    side = _bounded(side, ANIMATION_SIDE, 16, MAX_ANIMATION_SIDE, "side")
    frames = _bounded(frames, ANIMATION_FRAMES, 2, MAX_ANIMATION_FRAMES, "frames")
    fps = _bounded(fps, 10, 1, 30, "fps")
    sampled = stages(conn, canvas_id, frames, seq)
    box = window(conn, canvas_id, config, sampled[-1][0])
    scale = _scale(box, side)
    width, height = _dimensions(box, scale)
    if width * height * len(sampled) > MAX_HISTORY_PIXELS:
        raise ValueError("Animation is too large; use fewer frames or a smaller side")
    images = [Image.fromarray(compose(conn, canvas_id, box, scale, op)).quantize(
        colors=256, dither=Image.Dither.NONE) for op, _ in sampled]
    # Round cumulative time to GIF's hundredths of a second, rather than rounding every frame to the same
    # delay. At 30 fps, for example, alternate 30 and 40 ms delays instead of accidentally playing at 33 fps.
    durations = [10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps)) for i in range(len(images))]
    durations[0] = max(durations[0], 500)
    durations[-1] = max(durations[-1], 1500)
    out = io.BytesIO()
    images[0].save(out, format="GIF", save_all=True, append_images=images[1:], loop=0, duration=durations,
                   disposal=2, optimize=False)
    return out.getvalue()


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow 10.0 predates the scalable default font
            return ImageFont.load_default()


def _clip(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> str:
    text = " ".join(text.split())
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..."


def export_sheet(conn: sqlite3.Connection, canvas_id: str, side: int | None = None, seq: int | None = None,
                 frames: int | None = None, columns: int = 3) -> bytes:
    """A small-multiple PNG of early-weighted stages. `side` bounds each painting, not the whole sheet.
    Each panel uses the same crop and scale and names its percentage of calls and exact op sequence.
    """
    name, config = _canvas(conn, canvas_id)
    side = _bounded(side, SHEET_SIDE, 128, MAX_SHEET_SIDE, "side")
    frames = _bounded(frames, SHEET_FRAMES, 2, MAX_SHEET_FRAMES, "frames")
    columns = _bounded(columns, 3, 1, 6, "columns")
    sampled = stages(conn, canvas_id, frames, seq)
    box = window(conn, canvas_id, config, sampled[-1][0])
    scale = _scale(box, side)
    cell_w, cell_h = _dimensions(box, scale)
    columns = min(columns, len(sampled))
    rows = math.ceil(len(sampled) / columns)
    pad, caption, head = 16, 32, 64
    cell_w = max(128, cell_w)  # leave room for captions even on a narrow canvas
    width = pad + columns * (cell_w + pad)
    height = head + rows * (cell_h + caption + pad)
    if max(width, height) > MAX_SIDE or width * height > MAX_HISTORY_PIXELS:
        raise ValueError("Progress sheet is too large; use fewer stages, a smaller side or more columns")
    sheet = Image.new("RGB", (width, height), (250, 247, 240))
    draw = ImageDraw.Draw(sheet)
    title_font, font = _font(20), _font(14)
    draw.text((pad, 12), _clip(draw, name, title_font, width - 2 * pad), fill=(31, 35, 40), font=title_font)
    draw.text((pad, 38), _clip(draw, f"Painting progress, {len(sampled)} stages by tool call", font, width - 2 * pad),
              fill=(106, 110, 117), font=font)
    total = sampled[-1][1]
    for i, (op, rank) in enumerate(sampled):
        x, y = pad + (i % columns) * (cell_w + pad), head + (i // columns) * (cell_h + caption + pad)
        image = Image.fromarray(compose(conn, canvas_id, box, scale, op))
        sheet.paste(image, (x + (cell_w - image.width) // 2, y))
        draw.rectangle((x - 1, y - 1, x + cell_w, y + cell_h), outline=(221, 212, 196))
        label = f"{rank / total:.0%} / op {op}" if total else "Start / no calls yet"
        draw.text((x, y + cell_h + 8), _clip(draw, label, font, cell_w), fill=(31, 35, 40), font=font)
    info = PngInfo()
    info.add_text("canvas", canvas_id)
    info.add_text("op_sequences", json.dumps([op for op, _ in sampled]))
    info.add_text("canvas_window", json.dumps(box))
    out = io.BytesIO()
    sheet.save(out, format="PNG", pnginfo=info)
    return out.getvalue()


def filename(name: str, seq: int | None = None, kind: str = "png") -> str:
    """A safe file name for a still, animation or progress sheet."""
    if kind not in {"png", "gif", "sheet"}:
        raise ValueError("format must be png, gif or sheet")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:60] or "canvas"
    suffix = "-progress" if kind == "sheet" else ""
    extension = "gif" if kind == "gif" else "png"
    return f"{stem}{f'-op{seq}' if seq is not None else ''}{suffix}.{extension}"
