"""
Messages: ASCII text set in Pillow's built-in 6 x 11 bitmap font, scaled up by whole pixels.

Messages are painted into the canvas like any other mark, so paint can cover them. Each glyph gets an outline one
font pixel wide in a colour that contrasts with the ink, so dark ink stays readable on dark paint.
"""

from __future__ import annotations

import textwrap

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

GLYPH_W = 6
LINE_H = 13  # the font's 11 px plus 2 px of leading
MAX_MESSAGE = 200
MAX_SCALE = 4

_FONT = ImageFont.load_default_imagefont()


def clean(text: str) -> str:
    """Printable ASCII and newlines; anything else becomes '?', since the bitmap font has nothing else."""
    return "".join(c if c == "\n" or 32 <= ord(c) < 127 else "?" for c in str(text))


def wrap(text: str, max_width: int, scale: int) -> list[str]:
    cols = max(1, max_width // (GLYPH_W * scale))
    lines: list[str] = []
    for paragraph in clean(text).split("\n"):
        lines += textwrap.wrap(paragraph, width=cols, break_long_words=True, replace_whitespace=False) or [""]
    return lines


def _font_mask(text: str, max_width: int, scale: int) -> np.ndarray:
    lines = wrap(text, max_width, scale)
    width = max(1, max(len(line) for line in lines) * GLYPH_W)
    image = Image.new("1", (width, len(lines) * LINE_H), 0)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((0, i * LINE_H), line, font=_FONT, fill=1)
    return np.asarray(image, dtype=bool)


def text_mask(text: str, max_width: int, scale: int) -> np.ndarray:
    """A boolean mask of the text, wrapped to `max_width` px, `scale` px per font pixel."""
    return _font_mask(text, max_width, scale).repeat(scale, axis=0).repeat(scale, axis=1)


def glyphs(text: str, max_width: int, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """The ink and its outline as boolean masks, `scale` px per font pixel. Both are one font pixel larger than
    the text on every side, so the text itself starts at (scale, scale)."""
    ink = np.pad(_font_mask(text, max_width, scale), 1)
    halo = ink.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            halo |= np.roll(np.roll(ink, dy, axis=0), dx, axis=1)
    big = lambda m: m.repeat(scale, axis=0).repeat(scale, axis=1)  # noqa: E731
    return big(ink), big(halo & ~ink)


def halo_color(ink: tuple[float, float, float]) -> tuple[float, float, float]:
    """Near-black around light ink, paper-white around dark ink."""
    luminance = 0.2126 * ink[0] + 0.7152 * ink[1] + 0.0722 * ink[2]
    return (0.08, 0.08, 0.1) if luminance > 0.5 else (0.97, 0.95, 0.9)
