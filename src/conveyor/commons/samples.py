"""
A close-up of an instrument's demo sheet, for choosing painters.

A demo sheet (`labelled_sheet` in painting/canvas.py) is a white page of panels, each a blank canvas the size of a
painting with a caption above it naming the calls it ran. The calls mark a small part of each panel, so the sheet
shrunk to a card shows a few specks on paper. The close-up crops each panel to its marks, keeps the caption over
it, and sets the crops side by side, so what the tools draw fills the picture.

The panels are found from the pixels, not from how the sheet was made, so sheets already in the database work.
"""

from __future__ import annotations

import functools
import io

import numpy as np
from PIL import Image

from conveyor.painting.canvas import PAPER

PAPER_RGB = np.array([round(c * 255) for c in PAPER], dtype=np.int16)
CAPTION = 22  # the caption band above each panel
GAP = 8  # between panels on a sheet, and between crops on a close-up
PAD = 14  # paper kept around the marks
INK = 24  # how far a pixel must be from the paper colour to count as a mark
MIN_PANEL = 32  # pixels; a shorter run of non-white is a caption's text, not a panel


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Each run of True as (start, end), at least MIN_PANEL long."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= MIN_PANEL]


def panels(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """The panels of a sheet as (x0, y0, x1, y1), row by row."""
    filled = (arr < 250).any(axis=-1)
    out = []
    for y0, y1 in _runs(filled.mean(axis=1) > 0.02):
        for x0, x1 in _runs(filled[y0:y1].mean(axis=0) > 0.5):
            if filled[y0:y1, x0:x1].mean() > 0.6:  # a blank slot in the last row is white
                out.append((x0, y0, x1, y1))
    return out


@functools.lru_cache(maxsize=128)
def closeup(png: bytes) -> bytes:
    """The sheet with each panel cropped to its marks under its caption; the sheet itself when nothing is found."""
    sheet = Image.open(io.BytesIO(png)).convert("RGB")
    arr = np.asarray(sheet)
    tiles = []
    for x0, y0, x1, y1 in panels(arr):
        cell = arr[y0:y1, x0:x1].astype(np.int16)
        marks = (np.abs(cell - PAPER_RGB).max(axis=-1) > INK)
        if not marks.any():
            continue
        ys, xs = np.flatnonzero(marks.any(axis=1)), np.flatnonzero(marks.any(axis=0))
        crop = sheet.crop((x0 + max(0, xs[0] - PAD), y0 + max(0, ys[0] - PAD),
                           x0 + min(x1 - x0, xs[-1] + 1 + PAD), y0 + min(y1 - y0, ys[-1] + 1 + PAD)))
        band = arr[max(0, y0 - CAPTION):y0, x0:x1]
        text = np.flatnonzero((band < 200).any(axis=-1).any(axis=0))
        caption = sheet.crop((x0, max(0, y0 - CAPTION), x0 + (text[-1] + 2 if text.size else 1), y0))
        tiles.append((caption, crop))
    if not tiles:
        return png
    width = sum(max(c.width, m.width) for c, m in tiles) + GAP * (len(tiles) + 1)
    height = max(c.height + m.height for c, m in tiles) + 2 * GAP
    out = Image.new("RGB", (width, height), (255, 255, 255))
    x = GAP
    for caption, crop in tiles:
        out.paste(caption, (x, GAP))
        out.paste(crop, (x, GAP + caption.height))
        x += max(caption.width, crop.width) + GAP
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
