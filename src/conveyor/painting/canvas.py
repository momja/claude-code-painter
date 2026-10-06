"""
Canvas physics, targets, and image helpers.

Instruments draw only through `Canvas`: `dab` (a soft round mark), `stamp` (an arbitrary small mask), `smudge`
(drag paint already on the canvas), and `pick` (read a colour off the canvas). `view` is the read side: it cuts
a window out of the canvas and returns it as a `View`, which the paint server renders as a labelled picture. Paint is translucent and layers;
nothing is erased except by painting over it.

The one physical limit that matters is per call: a single tool call may touch at most `area_cap` pixels. That
is what keeps a tool a brush. Without it an evolved tool could flood the canvas in one call, and the instrument
would be a printer. The limit says nothing about the call's parameters, whether the tool keeps state, or what
shape the mark takes, so it leaves the interface free to evolve. Instruments never see the target.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

PAPER = (0.94, 0.91, 0.84)
TARGETS_DIR = Path(__file__).parent / "targets"
BASE_WIDTH = 64  # the canvas width the size limits below were written for

# Per call, as a share of the canvas: at 128 x 128 (the self-portrait) this is 1310 px, about a 50 x 26 px
# stroke, roughly what the old toolkit's widest wash could cover in one call.
CALL_AREA_SHARE = 0.08
MAX_DABS_PER_CALL = 3000
# A footprint pixel counts as touched from this alpha up, so a faint halo doesn't eat the area budget.
TOUCH_ALPHA = 0.02
MAX_VIEW_SIDE = 1024  # image pixels a rendered view may reach on one side


def brush_limit(width: int, height: int) -> float:
    """The biggest radius one call can lay at this canvas size: the nominal scale, kept inside the area budget.
    A dab at radius r with a feathered edge counts footprint out to about 1.24 r, so on a small canvas the area
    binds first. The two limits have to agree: a radius the canvas advertised but every call refused would be a
    brush the painter can't use."""
    area_cap = int(CALL_AREA_SHARE * height * width)
    return min(12.0 * max(1.0, width / BASE_WIDTH), math.sqrt(area_cap / math.pi) / 1.25)


class CanvasError(ValueError):
    """An instrument asked the canvas for something it can't do (bad colour, bad mask)."""


@dataclass
class View:
    """A window on the canvas: its rectangle in canvas pixels, the zoom it was asked for, and the pixels in it."""

    x0: int
    y0: int
    x1: int
    y1: int
    scale: int  # image pixels per canvas pixel
    img: np.ndarray

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def span(self) -> int:
        return self.x1 - self.x0


def parse_color(value) -> tuple[float, float, float]:
    """Hex string, or three numbers in 0..1 (or 0..255). Raises CanvasError otherwise."""
    if isinstance(value, str):
        s = value.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) == 6:
            try:
                return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                pass
        raise CanvasError(f"bad colour {value!r}: use hex like #8a6d4f")
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as e:
        raise CanvasError(f"bad colour {value!r}") from e
    if arr.size != 3 or not np.isfinite(arr).all():
        raise CanvasError(f"bad colour {value!r}: need three numbers")
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return float(arr[0]), float(arr[1]), float(arr[2])


def to_hex(color) -> str:
    r, g, b = (int(round(float(c) * 255)) for c in np.clip(np.asarray(color, dtype=np.float64), 0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"


class Canvas:
    """
    What an instrument's tools draw on. Coordinates are canvas pixels, x right and y down, origin top left.
    `begin_call()` resets the per-call area budget; the paint server calls it before every tool call.
    """

    def __init__(self, height: int, width: int, image: np.ndarray | None = None) -> None:
        self.height = int(height)
        self.width = int(width)
        self.img = (np.broadcast_to(np.asarray(PAPER, dtype=np.float32), (height, width, 3)).copy()
                    if image is None else image.astype(np.float32).copy())
        self.scale = max(1.0, self.width / BASE_WIDTH)
        self.area_cap = int(CALL_AREA_SHARE * self.height * self.width)
        self.max_radius = brush_limit(self.width, self.height)
        self._touched = np.zeros((height, width), dtype=bool)
        self.area_used = 0
        self.dabs_used = 0
        self.dry = False  # set when the call hit its area or dab limit and further marks were dropped

    # ---- per-call accounting --------------------------------------------------------------------------

    def begin_call(self) -> None:
        self._touched[:] = False
        self.area_used = 0
        self.dabs_used = 0
        self.dry = False

    @property
    def area_left(self) -> int:
        return max(0, self.area_cap - self.area_used)

    def _admit(self, ys: slice, xs: slice, alpha: np.ndarray) -> bool:
        """Charge a mark's footprint against the call's area budget. False (and dry) when it doesn't fit."""
        if self.dry:
            return False
        self.dabs_used += 1
        if self.dabs_used > MAX_DABS_PER_CALL:
            self.dry = True
            return False
        new = (alpha > TOUCH_ALPHA) & ~self._touched[ys, xs]
        n = int(new.sum())
        if self.area_used + n > self.area_cap:
            self.dry = True
            return False
        self._touched[ys, xs] |= new
        self.area_used += n
        return True

    # ---- the instrument API -----------------------------------------------------------------------------

    def dab(self, x, y, radius, color, opacity=1.0, hardness=0.5) -> bool:
        """
        A soft round mark centred on (x, y). `hardness` 0 is a feathered edge, 1 a crisp one. Radius is clamped
        to the canvas's brush limit. Returns False when the call is out of area and the dab was dropped.
        """
        r = float(np.clip(float(radius), 0.35, self.max_radius))
        x, y = float(x), float(y)
        opacity = float(np.clip(float(opacity), 0.0, 1.0))
        hardness = float(np.clip(float(hardness), 0.0, 1.0))
        rgb = np.asarray(parse_color(color), dtype=np.float32)
        pad = r + 1.5
        x0, x1 = max(0, int(math.floor(x - pad))), min(self.width, int(math.ceil(x + pad)))
        y0, y1 = max(0, int(math.floor(y - pad))), min(self.height, int(math.ceil(y + pad)))
        if x1 <= x0 or y1 <= y0 or opacity <= 0.0:
            return True
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
        d = np.hypot(xx - x, yy - y)
        ramp = max(r * (1.0 - hardness), 0.5)
        alpha = np.clip((r - d) / ramp + 0.5, 0.0, 1.0) * opacity
        return self._composite(slice(y0, y1), slice(x0, x1), alpha, rgb)

    def stamp(self, x, y, mask, color, opacity=1.0) -> bool:
        """
        Press an arbitrary mask (a 2-D array of 0..1, at most 2 * max_radius + 1 on a side) centred on (x, y).
        This is how an instrument gets a flat tip, a rake, or any shape a round dab can't make.
        """
        m = np.asarray(mask, dtype=np.float32)
        side = int(2 * self.max_radius + 1)
        if m.ndim != 2 or m.shape[0] < 1 or m.shape[1] < 1:
            raise CanvasError("stamp mask must be a 2-D array")
        if m.shape[0] > side or m.shape[1] > side:
            raise CanvasError(f"stamp mask is {m.shape[1]}x{m.shape[0]}, the limit is {side}x{side}")
        m = np.nan_to_num(np.clip(m, 0.0, 1.0)) * float(np.clip(float(opacity), 0.0, 1.0))
        rgb = np.asarray(parse_color(color), dtype=np.float32)
        h, w = m.shape
        top, left = int(round(float(y) - h / 2)), int(round(float(x) - w / 2))
        y0, y1 = max(0, top), min(self.height, top + h)
        x0, x1 = max(0, left), min(self.width, left + w)
        if x1 <= x0 or y1 <= y0:
            return True
        alpha = m[y0 - top : y1 - top, x0 - left : x1 - left]
        return self._composite(slice(y0, y1), slice(x0, x1), alpha, rgb)

    def smudge(self, x, y, radius, dx, dy, strength=0.5) -> bool:
        """
        Drag the paint under a round footprint at (x, y) toward (x + dx, y + dy), blending it in with `strength`.
        The drag is at most 2 * radius in each direction. Moves paint that's already there; adds none.
        """
        r = float(np.clip(float(radius), 0.5, self.max_radius))
        dx = int(round(float(np.clip(float(dx), -2 * r, 2 * r))))
        dy = int(round(float(np.clip(float(dy), -2 * r, 2 * r))))
        strength = float(np.clip(float(strength), 0.0, 1.0))
        cx, cy = float(x), float(y)
        pad = int(math.ceil(r + 1))
        sx0, sy0 = int(math.floor(cx)) - pad, int(math.floor(cy)) - pad
        size = 2 * pad + 1
        # Source and destination boxes, both clipped to the canvas, then to each other's valid part.
        ox0 = max(0, -sx0, -(sx0 + dx))
        oy0 = max(0, -sy0, -(sy0 + dy))
        ox1 = min(size, self.width - sx0, self.width - (sx0 + dx))
        oy1 = min(size, self.height - sy0, self.height - (sy0 + dy))
        if ox1 <= ox0 or oy1 <= oy0 or strength <= 0.0:
            return True
        yy, xx = np.mgrid[oy0:oy1, ox0:ox1].astype(np.float32)
        d = np.hypot(xx + sx0 + 0.5 - cx, yy + sy0 + 0.5 - cy)
        alpha = np.clip((r - d) / max(r * 0.5, 0.5) + 0.5, 0.0, 1.0) * strength
        src = self.img[sy0 + oy0 : sy0 + oy1, sx0 + ox0 : sx0 + ox1].copy()
        ys = slice(sy0 + dy + oy0, sy0 + dy + oy1)
        xs = slice(sx0 + dx + ox0, sx0 + dx + ox1)
        if not self._admit(ys, xs, alpha):
            return False
        a = alpha[..., None]
        self.img[ys, xs] = self.img[ys, xs] * (1.0 - a) + src * a
        return True

    def pick(self, x, y) -> tuple[float, float, float]:
        """The colour on the canvas at (x, y), as three floats in 0..1."""
        xi = int(np.clip(int(math.floor(float(x))), 0, self.width - 1))
        yi = int(np.clip(int(math.floor(float(y))), 0, self.height - 1))
        r, g, b = self.img[yi, xi]
        return float(r), float(g), float(b)

    def _composite(self, ys: slice, xs: slice, alpha: np.ndarray, rgb: np.ndarray) -> bool:
        if not self._admit(ys, xs, alpha):
            return False
        a = alpha[..., None]
        self.img[ys, xs] = self.img[ys, xs] * (1.0 - a) + rgb * a
        return True

    # ---- state for undo (the greedy painter tries a move and takes it back) -------------------------------

    def view(self, x, y, span, scale=4) -> View:
        """
        A square window `span` canvas pixels across, centred on (x, y) and kept inside the canvas, rendered at
        `scale` image pixels per canvas pixel (1 to 8, so the picture never exceeds 1024 px a side). Returned by
        a view tool for the painter to look at.
        """
        side = int(min(max(round(float(span)), 8), self.width, self.height))
        x0 = int(min(max(round(float(x) - side / 2), 0), self.width - side))
        y0 = int(min(max(round(float(y) - side / 2), 0), self.height - side))
        zoom = int(min(max(round(float(scale)), 1), max(1, min(8, MAX_VIEW_SIDE // side))))
        return View(x0, y0, x0 + side, y0 + side, zoom, self.img[y0 : y0 + side, x0 : x0 + side].copy())

    def snapshot(self) -> np.ndarray:
        return self.img.copy()

    def restore(self, img: np.ndarray) -> None:
        self.img[:] = img


# ---- targets ------------------------------------------------------------------------------------------------


@dataclass
class Target:
    name: str
    image: np.ndarray  # H x W x 3 float32 in 0..1
    patch: int

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def grid(self) -> tuple[int, int]:
        return self.height // self.patch, self.width // self.patch

    def patch_slice(self, row: int, col: int) -> tuple[slice, slice]:
        p = self.patch
        return slice(row * p, (row + 1) * p), slice(col * p, (col + 1) * p)


def load_target(name_or_path: str | Path, width: int = 128, patch: int = 16) -> Target:
    """A target resized to `width`, cropped to a whole number of patches. Bare names look in `targets/`."""
    path = Path(name_or_path)
    if not path.suffix:
        path = TARGETS_DIR / f"{path.name}.jpg"
    im = Image.open(path).convert("RGB")
    full_height = round(im.height * width / im.width)
    height = full_height - full_height % patch
    im = im.resize((width, full_height), Image.LANCZOS)
    top = (full_height - height) // 2
    img = np.asarray(im, dtype=np.float32)[top : top + height] / 255.0
    return Target(name=path.stem, image=img, patch=patch)


def region_errors(img: np.ndarray, target: Target) -> np.ndarray:
    """RMSE per patch, rows x cols."""
    rows, cols = target.grid
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            ys, xs = target.patch_slice(r, c)
            out[r, c] = np.sqrt(((img[ys, xs] - target.image[ys, xs]) ** 2).mean())
    return out


def error_table(img: np.ndarray, target: Target, rect: tuple[int, int, int, int] | None = None,
                patch: int | None = None) -> str:
    """
    Per-region error as a table labelled with pixel ranges, so nobody has to count rows. `rect` (x0, y0, x1, y1)
    restricts the table to a window, and `patch` overrides the cell size; a view asks for cells suited to its
    window rather than the whole canvas.
    """
    p = int(patch or target.patch)
    x0, y0, x1, y1 = (0, 0, target.width, target.height) if rect is None else tuple(int(v) for v in rect)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(target.width, max(x1, x0 + 1)), min(target.height, max(y1, y0 + 1))
    cols = [c for c in range(-(-target.width // p)) if c * p < x1 and (c + 1) * p > x0]
    rows = [r for r in range(-(-target.height // p)) if r * p < y1 and (r + 1) * p > y0]

    def cell(r: int, c: int) -> float:
        ys = slice(r * p, min((r + 1) * p, target.height))
        xs = slice(c * p, min((c + 1) * p, target.width))
        return float(np.sqrt(((img[ys, xs] - target.image[ys, xs]) ** 2).mean()))

    row_labels = [f"y {r * p}-{min((r + 1) * p, target.height)}" for r in rows]
    col_labels = [f"x {c * p}-{min((c + 1) * p, target.width)}" for c in cols]
    label_w = max(len(s) for s in row_labels) + 1
    cell_w = max(6, max(len(s) for s in col_labels) + 1)
    head = " " * label_w + "".join(s.ljust(cell_w) for s in col_labels)
    lines = [(row_labels[i].ljust(label_w) + "".join(f"{cell(r, c):.2f}".ljust(cell_w) for c in cols)).rstrip()
             for i, r in enumerate(rows)]
    where = "" if rect is None else f" in x {x0}-{x1}, y {y0}-{y1}"
    return f"Error per {p} x {p} px region{where} (0 is a perfect match):\n" + "\n".join([head.rstrip(), *lines])


def view_patch(span: int) -> int:
    """Cell size for a view's error table: about eight cells across the window, a power of two, at least 8 px."""
    return max(8, 2 ** int(math.log2(max(int(span), 8) / 8)))


def worst_regions(img: np.ndarray, target: Target, k: int = 4) -> list[dict]:
    errs = region_errors(img, target)
    out = []
    for i in np.argsort(errs, axis=None)[::-1][:k]:
        r, c = np.unravel_index(i, errs.shape)
        ys, xs = target.patch_slice(int(r), int(c))
        out.append({"x": [xs.start, xs.stop], "y": [ys.start, ys.stop], "error": round(float(errs[r, c]), 3)})
    return out


# ---- image helpers ------------------------------------------------------------------------------------------


def to_png(img: np.ndarray, scale: int = 1) -> bytes:
    arr = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    if scale > 1:
        arr = arr.repeat(scale, axis=0).repeat(scale, axis=1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def heatmap_png(err: np.ndarray, scale: int = 1, vmax: float = 0.35) -> bytes:
    """Per-pixel error on paper, vermilion where it's high."""
    t = np.clip(err / vmax, 0, 1)[..., None]
    hot = np.array([0.76, 0.23, 0.13], dtype=np.float32)
    return to_png(np.asarray(PAPER, dtype=np.float32) * (1 - t) + hot * t, scale)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType
        return ImageFont.load_default()


def grid_step(width: int) -> int:
    """Canvas pixels between grid lines: about eight divisions across."""
    return max(8, 8 * round(width / 64))


def gridded_png(img: np.ndarray, scale: int | None = None, x0: int = 0, y0: int = 0,
                lx0: int | None = None, ly0: int | None = None) -> bytes:
    """
    An image as the model sees it: upscaled, a faint line every `grid_step` canvas pixels, and the lines
    labelled in canvas pixels in a white margin, so labels never cover the picture. `(x0, y0)` is where the
    image sits on the canvas, so a window is labelled in the same coordinates as the whole thing. An omitted
    scale picks one that keeps the picture near 512 px a side however big the canvas is. `(lx0, ly0)`
    overrides the labels' origin: a scoped window keeps the canvas grid lines but numbers them from its own
    corner, so the painter reads local coordinates straight off the picture.
    """
    h, w = img.shape[:2]
    if scale is None:
        scale = max(1, min(4, 512 // max(w, 1)))
    ox, oy = x0 if lx0 is None else lx0, y0 if ly0 is None else ly0
    step = grid_step(w)
    xs = list(range(-((-x0) // step) * step, x0 + w + 1, step))
    ys = list(range(-((-y0) // step) * step, y0 + h + 1, step))
    arr = (np.clip(img, 0, 1) * 255).round().astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)
    pic = Image.fromarray(arr).convert("RGBA")
    lines = Image.new("RGBA", pic.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lines)
    for v in xs:
        if x0 < v < x0 + w:
            d.line([((v - x0) * scale, 0), ((v - x0) * scale, h * scale - 1)], fill=(30, 70, 170, 110), width=1)
    for v in ys:
        if y0 < v < y0 + h:
            d.line([(0, (v - y0) * scale), (w * scale - 1, (v - y0) * scale)], fill=(30, 70, 170, 110), width=1)
    pic = Image.alpha_composite(pic, lines).convert("RGB")
    m = max(22, round(0.055 * pic.width))
    out = Image.new("RGB", (pic.width + 2 * m, pic.height + 2 * m), (255, 255, 255))
    out.paste(pic, (m, m))
    d = ImageDraw.Draw(out)
    font = _font(max(12, round(m * 0.6)))

    def label(text: str, cx: float, cy: float) -> None:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        d.text((cx - (right - left) / 2 - left, cy - (bottom - top) / 2 - top), text, font=font, fill=(30, 40, 60))

    for v in xs:
        label(str(v - ox), m + (v - x0) * scale, m / 2)
        label(str(v - ox), m + (v - x0) * scale, out.height - m / 2)
    for v in ys:
        label(str(v - oy), m / 2, m + (v - y0) * scale)
        label(str(v - oy), out.width - m / 2, m + (v - y0) * scale)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def view_png(view: View) -> bytes:
    """A view as the painter sees it: its window, gridded and labelled in canvas coordinates."""
    return gridded_png(view.img, scale=view.scale, x0=view.x0, y0=view.y0)


def labelled_sheet(cells: list[tuple[str, np.ndarray]], scale: int | None = None, cols: int = 3) -> bytes:
    """Images side by side, each with a caption above it. Used for an instrument's demo sheet. An omitted scale
    keeps the sheet about the same size however big the canvas is."""
    if not cells:
        cells = [("(nothing to show)", np.broadcast_to(np.asarray(PAPER, dtype=np.float32), (32, 32, 3)))]
    if scale is None:
        scale = max(1, min(2, 512 // max(c[1].shape[1] for c in cells)))
    h = max(c[1].shape[0] for c in cells) * scale
    w = max(c[1].shape[1] for c in cells) * scale
    cap, gap = 22, 8
    cols = max(1, min(cols, len(cells)))
    rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * (w + gap) + gap, rows * (h + cap + gap) + gap), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = _font(13)
    for i, (caption, img) in enumerate(cells):
        r, c = divmod(i, cols)
        x0, y0 = gap + c * (w + gap), gap + r * (h + cap + gap)
        while caption and d.textlength(caption, font=font) > w:  # long call sequences ran into the next cell
            caption = caption[:-2].rstrip() + "…"
        d.text((x0, y0 + 3), caption, font=font, fill=(30, 40, 60))
        arr = (np.clip(img, 0, 1) * 255).round().astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)
        sheet.paste(Image.fromarray(arr), (x0, y0 + cap))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def side_by_side(*imgs: np.ndarray, scale: int = 2, gap: int = 2) -> bytes:
    h = imgs[0].shape[0]
    spacer = np.ones((h, gap, 3), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, im in enumerate(imgs):
        if i:
            parts.append(spacer)
        parts.append(im)
    return to_png(np.concatenate(parts, axis=1), scale)
