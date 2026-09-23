"""
Component B as a real LLM painter.

Each turn the model sees the target and its canvas so far as images, plus the error per region, and answers
with brush calls. Every brush in the current toolkit is one tool, so a toolkit change changes the tools the
model gets. Turns are stateless: the canvas image carries the history, which keeps each request small.
"""

from __future__ import annotations

import io
import math

import numpy as np
from darwinian_evolver.problem import EvaluationFailureCase
from darwinian_evolver.problem import Organism
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from conveyor.events import current_trace
from conveyor.llm import LLMClient
from conveyor.llm import LLMError
from conveyor.llm import function_tool
from conveyor.llm import image_part
from conveyor.llm import text_part
from conveyor.painting.canvas import Brush
from conveyor.painting.canvas import Target
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import composite
from conveyor.painting.canvas import stroke_alpha
from conveyor.painting.canvas import to_png
from conveyor.painting.critic import Critic
from conveyor.painting.toolkit import Toolkit

INITIAL_PROMPT = """You are copying a painting onto a watercolor canvas, one brush stroke at a time.
Block in the large areas of color with the broadest brushes first, then work toward detail.
Put each stroke where the canvas differs most from the target.
Match colors to the target under the stroke."""

RULES = """The canvas is {w} by {h} pixels. x runs left to right from 0 to {w}, y runs top to bottom from 0 to {h}.
Each brush is a tool. One call lays one stroke starting at (x, y), heading in the direction of `angle`
(degrees, 0 points right, 90 points down), in `color` (hex like #8a6d4f). Watercolor is translucent, so
strokes layer over what's already there.
{pressure_rule}
{grid_rule}
Think briefly, then act. Make {per_turn} brush calls in this reply, one call per stroke.
You have {remaining} strokes left in total."""

# Thinking counts against max_tokens. A live run with 4096 spent all of it thinking and made no calls.
PAINT_MAX_TOKENS = 8192
LENGTH_NUDGE = (
    "Your last reply ran out of room while thinking and made no brush calls. "
    "Don't analyze further. Reply now with the brush calls."
)

IMAGE_SCALE = 4
PRESSURE_RANGE = (0.1, 1.0)

# Shared by both harnesses' rules. A live run placed a 4 px ear 10 px off and laid detail as dark bars, so the
# model gets a labelled grid to read positions from and a light touch for detail.
PRESSURE_RULE = (
    "`pressure` (0.1 to 1, default 1) sets how much pigment a stroke lays down: 1 is the brush's full opacity, "
    "0.3 a light touch. Use light pressure for detail and for corrections."
)


def grid_step(width: int) -> int:
    """Canvas pixels between grid lines: about eight divisions across, whatever the canvas size."""
    return max(8, 8 * round(width / 64))


def grid_margin(image_width: int) -> int:
    """Room around the picture for the labels, which grows with the image so the numbers stay readable."""
    return max(22, round(0.055 * image_width))


def grid_rule(step: int) -> str:
    return (f"The images have a coordinate grid: a line every {step} canvas pixels, labelled along the edges in "
            "canvas pixels, the same units as x and y. Read positions off the grid.")


def _grid_font(margin: int):
    size = max(12, round(margin * 0.6))
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType
        return ImageFont.load_default()


def gridded_png(img: np.ndarray, scale: int = IMAGE_SCALE, step: int | None = None) -> bytes:
    """
    An image as the model sees it: upscaled, with a faint line every `step` canvas pixels, and the lines
    labelled in canvas pixels in a margin, so the labels never cover the painting. Spacing, margin and font
    all scale with the canvas: at 64 px the labels were too cramped to read.
    """
    h, w = img.shape[:2]
    step = step or grid_step(w)
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)
    pic = Image.fromarray(arr).convert("RGBA")
    lines = Image.new("RGBA", pic.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lines)
    for v in range(step, w, step):
        d.line([(v * scale, 0), (v * scale, h * scale - 1)], fill=(30, 70, 170, 110), width=1)
    for v in range(step, h, step):
        d.line([(0, v * scale), (w * scale - 1, v * scale)], fill=(30, 70, 170, 110), width=1)
    pic = Image.alpha_composite(pic, lines).convert("RGB")

    m = grid_margin(pic.width)
    out = Image.new("RGB", (pic.width + 2 * m, pic.height + 2 * m), (255, 255, 255))
    out.paste(pic, (m, m))
    d = ImageDraw.Draw(out)
    font = _grid_font(m)

    def label(text: str, cx: float, cy: float) -> None:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        d.text((cx - (right - left) / 2 - left, cy - (bottom - top) / 2 - top), text, font=font, fill=(30, 40, 60))

    for v in range(0, w + 1, step):
        label(str(v), m + v * scale, m / 2)
        label(str(v), m + v * scale, out.height - m / 2)
    for v in range(0, h + 1, step):
        label(str(v), m / 2, m + v * scale)
        label(str(v), out.width - m / 2, m + v * scale)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def stroke_pressure(args: dict) -> float | None:
    """The stroke's pressure: 1.0 when absent, clamped to PRESSURE_RANGE, None when it isn't a number."""
    value = args.get("pressure", 1.0)
    if value is None:
        return 1.0
    try:
        return float(np.clip(float(value), *PRESSURE_RANGE))
    except (TypeError, ValueError):
        return None


class PromptStrategy(Organism):
    prompt: str
    # A fixed cap, not evolved: more strokes nearly always lower pixel error, so a free budget only grows.
    n_strokes: int = 500
    strokes_per_turn: int = 40  # stateless harness only; Pi lets the model batch as it likes

    def render_text(self) -> str:
        return f"Budget: {self.n_strokes} strokes, up to {self.strokes_per_turn} per turn.\n\n{self.prompt.strip()}\n"


PromptStrategy.model_rebuild(_types_namespace={"EvaluationFailureCase": EvaluationFailureCase})


def brush_description(b: Brush) -> str:
    """
    One line per brush for the painter's tool list. The brush's own `doc` says what the mark looks like,
    since its code can draw anything and there is no fixed set of numbers left to recite.
    """
    shape = f"a {b.length:.0f}px stroke" if b.length >= 1 else "a dab"
    reach = f"Lays {shape}, up to {b.radius:.1f}px from the path."
    return f"{reach} {b.doc.strip()}" if b.doc.strip() else reach


def brush_tools(toolkit: Toolkit, width: int, height: int) -> list[dict]:
    tools = []
    for b in toolkit.brushes:
        tools.append(
            function_tool(
                b.name,
                brush_description(b),
                {
                    "x": {"type": "number", "description": f"Start x, 0 to {width}"},
                    "y": {"type": "number", "description": f"Start y, 0 to {height}"},
                    "angle": {"type": "number", "description": "Direction in degrees. 0 points right, 90 down."},
                    "color": {"type": "string", "description": "Hex color, for example #8a6d4f"},
                    "pressure": {"type": "number", "description": "0.1 to 1, default 1. Lower lays less pigment."},
                },
                ["x", "y", "angle", "color"],
            )
        )
    return tools


def parse_color(value) -> np.ndarray | None:
    if isinstance(value, str):
        s = value.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) == 6:
            try:
                return np.array([int(s[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
            except ValueError:
                return None
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
        arr = np.array(value, dtype=np.float32)
        return np.clip(arr / 255.0 if arr.max() > 1 else arr, 0, 1)
    return None


def _hex(color: np.ndarray) -> str:
    r, g, b = (np.clip(color, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def region_label(target: Target, r: int, c: int) -> str:
    ys, xs = target.patch_slice(r, c)
    return f"x {xs.start}-{xs.stop}, y {ys.start}-{ys.stop}"


def error_grid_text(canvas: np.ndarray, target: Target) -> str:
    """The per-region error as a table labelled with pixel ranges, so nobody has to count rows from 0 or 1."""
    errs = Critic.patch_errors(canvas, target)
    rows, cols = errs.shape
    p = target.patch
    # Widths come from the labels themselves: at 128 px "y 112-128" is wider than a fixed 8 and ran into the
    # first number.
    row_labels = [f"y {r * p}-{(r + 1) * p}" for r in range(rows)]
    col_labels = [f"x {c * p}-{(c + 1) * p}" for c in range(cols)]
    label_w = max(len(s) for s in row_labels) + 1
    cell_w = max(6, max(len(s) for s in col_labels) + 1)
    head = " " * label_w + "".join(s.ljust(cell_w) for s in col_labels)
    lines = [row_labels[r].ljust(label_w) + "".join(f"{errs[r, c]:.2f}".ljust(cell_w) for c in range(cols))
             for r in range(rows)]
    lines = [line.rstrip() for line in lines]
    return f"Error per {p} × {p} px region (0 is a perfect match):\n" + "\n".join([head, *lines])


def worst_regions_text(canvas: np.ndarray, target: Target, k: int = 3) -> str:
    errs = Critic.patch_errors(canvas, target)
    cells = [np.unravel_index(i, errs.shape) for i in np.argsort(errs, axis=None)[::-1][:k]]
    return "Worst regions: " + "; ".join(f"{region_label(target, r, c)}: {errs[r, c]:.2f}" for r, c in cells) + "."


def llm_paint(
    strategy: PromptStrategy,
    toolkit: Toolkit,
    target: Target,
    seed: int,
    record: bool = False,
    *,
    client: LLMClient,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = target.image.shape[:2]
    canvas = blank(target)
    tools = brush_tools(toolkit, width, height)
    brushes = {b.name: b for b in toolkit.brushes}
    target_png = gridded_png(target.image)
    trace = current_trace() if record else None
    thumb_every = 6

    used = 0
    turn = 0
    # Models often return fewer calls than a turn allows, so leave room for three times the minimum turns.
    max_turns = 3 * math.ceil(strategy.n_strokes / strategy.strokes_per_turn) + 2
    history: list[str] = []
    while used < strategy.n_strokes and turn < max_turns:
        turn += 1
        remaining = strategy.n_strokes - used
        per_turn = min(strategy.strokes_per_turn, remaining)
        status = (
            f"{used} strokes used, {remaining} left.\n{error_grid_text(canvas, target)}"
        )
        if history:
            status += "\nYour last strokes and how each changed total error (negative is better):\n" + "\n".join(
                history[-per_turn:]
            )
        system = strategy.prompt.strip() + "\n\n" + RULES.format(
            w=width, h=height, per_turn=per_turn, remaining=remaining, pressure_rule=PRESSURE_RULE,
            grid_rule=grid_rule(grid_step(width)))
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    text_part("Target:"),
                    image_part(target_png),
                    text_part("Your canvas so far:"),
                    image_part(gridded_png(canvas)),
                    text_part(status),
                ],
            },
        ]
        try:
            reply = client.chat(messages, purpose="paint", tools=tools, tool_choice="required",
                                max_tokens=PAINT_MAX_TOKENS, record_span=record)
            if not reply.tool_calls and reply.finish_reason == "length":
                # It spent the whole reply thinking. Ask once more with minimal thinking and a plain nudge.
                messages[1]["content"].append(text_part(LENGTH_NUDGE))
                reply = client.chat(messages, purpose="paint retry", tools=tools, tool_choice="required",
                                    max_tokens=PAINT_MAX_TOKENS, reasoning_effort="minimal", record_span=record)
        except LLMError:
            break  # recorded by the client; keep whatever is on the canvas
        if not reply.tool_calls:
            break

        # What happened to each tool call, in reply order, recorded on the call's row for the transcript view.
        results: list[dict] = []
        for call in reply.tool_calls[:per_turn]:
            used += 1
            b = brushes.get(call.name)
            args = call.arguments or {}
            color = parse_color(args.get("color"))
            try:
                x = float(np.clip(float(args["x"]), 0, width))
                y = float(np.clip(float(args["y"]), 0, height))
                angle = math.radians(float(args["angle"]))
            except (KeyError, TypeError, ValueError):
                x = y = angle = None
            pressure = stroke_pressure(args)
            if b is None or color is None or x is None or pressure is None:
                problem = "unknown brush" if b is None else "bad arguments"
                results.append({"status": "rejected", "error": problem})
                history.append(f"{call.name}: rejected, {problem}")
                if trace:
                    trace.span(call.name or "unknown", args={"raw": call.raw[:200]}, result={"kept": False, "error": problem})
                continue
            hit = stroke_alpha(b, x, y, angle, height, width, rng)
            delta = 0.0
            if hit is not None:
                ys, xs, a = hit
                a = a * pressure
                tgt = target.image[ys, xs]
                new = composite(canvas[ys, xs], a, color)
                delta = float(((new - tgt) ** 2).sum() - ((canvas[ys, xs] - tgt) ** 2).sum())
                canvas[ys, xs] = new
            results.append({"status": "applied", "delta_error": round(delta, 4)})
            touch = f" pressure {pressure:.1f}" if pressure < 1 else ""
            history.append(f"{b.name} at ({x:.0f}, {y:.0f}) {_hex(color)}{touch}: {delta:+.3f}")
            if trace:
                last = used >= strategy.n_strokes
                image = to_png(canvas) if (used % thumb_every == 0 or last) else None
                trace.span(
                    b.name,
                    args={"x": round(x, 1), "y": round(y, 1), "angle": round(math.degrees(angle) % 360), "color": _hex(color),
                          "pressure": round(pressure, 2)},
                    result={"delta_error": round(delta, 4), "kept": True},
                    image=image,
                )
        ignored = len(reply.tool_calls) - per_turn
        if ignored > 0:
            results += [{"status": "ignored", "error": f"over the {per_turn} calls allowed this turn"}] * ignored
            history.append(f"({ignored} extra calls ignored, {per_turn} allowed per turn)")
        client.annotate(reply.call_id, tool_results=results)
    return canvas
