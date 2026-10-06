"""Messages as pixels: ASCII text set in Pillow's built-in 6 x 11 bitmap font, scaled up by whole pixels."""

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


def text_mask(text: str, max_width: int, scale: int) -> np.ndarray:
    """A boolean mask of the text, wrapped to `max_width` px, `scale` px per font pixel."""
    lines = wrap(text, max_width, scale)
    width = max(1, max(len(line) for line in lines) * GLYPH_W)
    image = Image.new("1", (width, len(lines) * LINE_H), 0)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((0, i * LINE_H), line, font=_FONT, fill=1)
    mask = np.asarray(image, dtype=bool)
    return mask.repeat(scale, axis=0).repeat(scale, axis=1)
