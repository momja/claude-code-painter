"""Canvas physics, target loading, and image helpers."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel
from pydantic import Field

from conveyor.painting.brushcode import BrushCodeError
from conveyor.painting.brushcode import compile_brush

PAPER = np.array([0.94, 0.91, 0.84], dtype=np.float32)
TARGETS_DIR = Path(__file__).parent / "targets"

# Physical limits. A toolkit that breaks these is not viable. Without them a code-writing mutator will
# eventually invent a "brush" that covers the whole canvas at full opacity, which is a stamp, not a brush.
# The rest of what used to be limited here (softness, opacity, bleed, granulation) now lives inside each
# brush's own program, so it is checked by rendering the brush instead of by reading its fields.
LIMITS = {
    "radius": (0.5, 12.0),
    "length": (0.0, 24.0),
}
MAX_BRUSHES = 8
BASE_WIDTH = 64  # the canvas width these limits were written for


def set_canvas_scale(width: int) -> float:
    """
    Scale the size limits to the canvas and return the factor. Brush sizes are in canvas pixels, so the same
    radius covers a quarter of the area at 128 px that it does at 64. Mutates LIMITS in place, because the
    mutators and the physics check read it by reference.
    """
    factor = max(1.0, width / BASE_WIDTH)
    LIMITS["radius"] = (0.5, round(12.0 * factor, 1))
    LIMITS["length"] = (0.0, round(24.0 * factor, 1))
    return factor


class Brush(BaseModel):
    """
    A brush is a name, a reach, and a program. `source` draws the mark (see `brushcode`), which is what lets
    a mutator invent a dotted line or a rake instead of turning the same six knobs.

    `radius` and `length` stay out of the program because the caller needs them before it runs: they set the
    bounding box the mark is drawn into, they are what the size limits bite on, and they are what the
    painter sorts by when it picks a brush for a patch.

    `template` and `params` are provenance, not truth. `source` is always what renders. When a scripted
    mutator wrote the brush from a template it records which one and with what numbers, so a later targeted
    edit can change "softness" by name. Brushes an LLM wrote have neither, and get edited as code.
    """

    name: str
    radius: float
    source: str
    length: float = 0.0
    doc: str = ""
    template: str | None = None
    params: dict[str, float] = Field(default_factory=dict)


def physics_violations(brushes: list[Brush]) -> list[str]:
    """
    Everything that makes a toolkit unusable: too many brushes, a brush that reaches too far, or a brush
    whose program will not compile or does not draw a mark. The code check runs the program once on a small
    grid, so a brush that would have crashed mid-painting is caught here and the toolkit is marked
    non-viable, which is the answer the dashboard can explain.
    """
    problems = []
    if not brushes:
        problems.append("toolkit has no brushes")
    if len(brushes) > MAX_BRUSHES:
        problems.append(f"{len(brushes)} brushes, limit is {MAX_BRUSHES}")
    for b in brushes:
        for attr, (lo, hi) in LIMITS.items():
            v = getattr(b, attr)
            if not lo <= v <= hi:
                problems.append(f"{b.name}.{attr} = {v:.2f}, allowed {lo} to {hi}")
        try:
            compile_brush(b.source)
        except BrushCodeError as e:
            problems.append(f"{b.name}: {e}")
    return problems


def stroke_alpha(
    brush: Brush, x: float, y: float, angle: float, height: int, width: int, rng: np.random.Generator
) -> tuple[slice, slice, np.ndarray] | None:
    """
    Alpha mask of one stroke, cropped to its bounding box.

    The caller places the stroke; the brush's own program decides what the mark looks like. The program sees
    the box in stroke-local pixels: `u` runs along the path from the start point, `v` runs across it. That
    is the whole reason a brush can be a dotted line (modulate on u) or a rake (modulate on v).

    Returns None when the stroke misses the canvas, and also when the program misbehaves on a box shape the
    probe in `physics_violations` did not hit. A dead brush laying nothing scores badly and gets selected
    out, which beats taking a whole run down with it.
    """
    r = brush.radius
    x2 = x + math.cos(angle) * brush.length
    y2 = y + math.sin(angle) * brush.length
    pad = r + 1.5
    x0 = max(0, int(math.floor(min(x, x2) - pad)))
    x1 = min(width, int(math.ceil(max(x, x2) + pad)))
    y0 = max(0, int(math.floor(min(y, y2) - pad)))
    y1 = min(height, int(math.ceil(max(y, y2) + pad)))
    if x1 <= x0 or y1 <= y0:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dx, dy = xx - x, yy - y
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u = dx * cos_a + dy * sin_a   # along the path, 0 at the start point
    v = -dx * sin_a + dy * cos_a  # across it, signed

    fn = compile_brush(brush.source)
    try:
        with np.errstate(all="ignore"):
            alpha = np.asarray(fn(u, v, rng, float(r), float(brush.length)), dtype=np.float32)
    except Exception:  # noqa: BLE001 - the program is the mutator's, and a bad one must not end the run
        return None
    if alpha.shape != u.shape:
        return None
    np.clip(alpha, 0.0, 1.0, out=alpha)
    return slice(y0, y1), slice(x0, x1), alpha[..., None]


def composite(region: np.ndarray, alpha: np.ndarray, color: np.ndarray) -> np.ndarray:
    return region * (1.0 - alpha) + color * alpha


@dataclass
class Target:
    name: str
    image: np.ndarray  # H x W x 3 float32 in [0, 1]
    patch: int
    classes: dict[tuple[int, int], str] = field(default_factory=dict)
    edge_angle: np.ndarray | None = None  # angle along edges, per pixel

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


def _gray(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = _gray(img)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return gx, gy


def load_target(path: str | Path, width: int = 80, patch: int = 16) -> Target:
    path = Path(path)
    im = Image.open(path).convert("RGB")
    full_height = round(im.height * width / im.width)
    height = full_height - full_height % patch
    im = im.resize((width, full_height), Image.LANCZOS)
    top = (full_height - height) // 2
    img = np.asarray(im, dtype=np.float32)[top : top + height] / 255.0
    target = Target(name=path.stem, image=img, patch=patch)

    gx, gy = gradients(img)
    target.edge_angle = np.arctan2(gy, gx) + math.pi / 2
    mag = np.hypot(gx, gy)
    g = _gray(img)
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]

    rows, cols = target.grid
    hf = np.zeros((rows, cols))
    edge = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            ys, xs = target.patch_slice(r, c)
            hf[r, c] = lap[ys, xs].var()
            edge[r, c] = mag[ys, xs].mean()
    hf_cut = np.percentile(hf, 66)
    edge_cut = np.percentile(edge, 50)
    for r in range(rows):
        for c in range(cols):
            if hf[r, c] >= hf_cut:
                target.classes[(r, c)] = "fine_detail"
            elif edge[r, c] >= edge_cut:
                target.classes[(r, c)] = "hard_edge"
            else:
                target.classes[(r, c)] = "soft_wash"
    return target


def default_targets(width: int = 80, patch: int = 16) -> tuple[list[Target], list[Target]]:
    """Self-Portrait (1889) to train on, The Starry Night (1889) held out."""
    return (
        [load_target(TARGETS_DIR / "self_portrait.jpg", width, patch)],
        [load_target(TARGETS_DIR / "starry_night.jpg", width, patch)],
    )


def blank(target: Target) -> np.ndarray:
    return np.broadcast_to(PAPER, target.image.shape).copy()


def to_png(img: np.ndarray, scale: int = 1) -> bytes:
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if scale > 1:
        arr = arr.repeat(scale, axis=0).repeat(scale, axis=1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def heatmap(err: np.ndarray, scale: int = 1, vmax: float = 0.35) -> bytes:
    """Per-pixel error on paper, with vermilion for high error."""
    t = np.clip(err / vmax, 0, 1)[..., None]
    hot = np.array([0.76, 0.23, 0.13], dtype=np.float32)
    img = PAPER * (1 - t) + hot * t
    return to_png(img, scale)


def triptych(*panels: np.ndarray, scale: int = 4, gap: int = 1) -> bytes:
    h = panels[0].shape[0]
    spacer = np.broadcast_to(PAPER, (h, gap, 3))
    parts = []
    for i, p in enumerate(panels):
        if i:
            parts.append(spacer)
        parts.append(p)
    return to_png(np.concatenate(parts, axis=1), scale)


def swatch_sheet(brushes: list[Brush], cell: int = 28) -> bytes:
    """One sample stroke per brush, side by side, in Prussian blue on paper."""
    n = max(1, len(brushes))
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    img = np.broadcast_to(PAPER, (rows * cell, cols * cell, 3)).copy()
    ink = np.array([0.15, 0.29, 0.48], dtype=np.float32)
    rng = np.random.default_rng(7)
    for i, b in enumerate(brushes):
        r, c = divmod(i, cols)
        cy, cx = r * cell + cell / 2, c * cell + cell / 2
        length = min(b.length, cell - 2 * b.radius - 2) if b.length else 0
        probe = b.model_copy(update={"length": max(0.0, length)})
        x = cx - math.cos(-0.5) * probe.length / 2
        y = cy - math.sin(-0.5) * probe.length / 2
        hit = stroke_alpha(probe, x, y, -0.5, img.shape[0], img.shape[1], rng)
        if hit:
            ys, xs, a = hit
            # keep each sample inside its own cell
            cell_ys = slice(max(ys.start, r * cell), min(ys.stop, (r + 1) * cell))
            cell_xs = slice(max(xs.start, c * cell), min(xs.stop, (c + 1) * cell))
            a = a[cell_ys.start - ys.start : cell_ys.stop - ys.start, cell_xs.start - xs.start : cell_xs.stop - xs.start]
            img[cell_ys, cell_xs] = composite(img[cell_ys, cell_xs], a, ink)
    return to_png(img, 2)
