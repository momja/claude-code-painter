"""
One painting, in its own process.

    python -m conveyor.painting.paintserver <session_dir>            # MCP server; Claude Code launches it
    python -m conveyor.painting.paintserver <session_dir> --greedy   # offline painter, no model

The session directory holds `job.json` (the instrument's source, the target, the action budget) and receives
everything the painting produces: `calls.jsonl` (one line per tool call: arguments, what happened, the change in
score and in pixel error, the pen's state), `snaps/` (canvas PNGs every few calls), `canvas.npy` (the canvas after every call, so
the host has it even if the process is killed), and `finish.json` (the painter's closing note).

Instrument code runs here and nowhere near the evolution process, on this process's main thread, where the
per-call time limit can interrupt it.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from conveyor.mcp import StdioServer
from conveyor.mcp import ToolFailure
from conveyor.mcp import image
from conveyor.mcp import text
from conveyor.painting.canvas import Canvas
from conveyor.painting.canvas import gridded_png
from conveyor.painting.canvas import load_target
from conveyor.painting.canvas import to_hex
from conveyor.painting.canvas import to_png
from conveyor.painting.canvas import view_png
from conveyor.painting.critic import Critic
from conveyor.painting.instrument import Instrument
from conveyor.painting.instrument import ToolError

MAX_REJECTS = 60  # calls refused for bad arguments before the painting is ended, so a confused model can't loop
MAX_BATCH_CALLS = 40
MAX_PLAN_CHARS = 1200


def _rounded(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def batch_calls(inst: Instrument, args: dict) -> list[tuple[str, dict]]:
    """A paint_batch's calls as (tool, arguments), with shared defaults filled in. Raises ToolFailure."""
    calls, defaults = args.get("calls"), args.get("defaults", {})
    if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_BATCH_CALLS:
        raise ToolFailure(f"paint_batch needs 1 to {MAX_BATCH_CALLS} calls. Nothing was painted.")
    if not isinstance(defaults, dict):
        raise ToolFailure("defaults must be an object. Nothing was painted.")
    specs = {t.name: {p.name for p in t.params} for t in inst.spec.tools}
    prepared = []
    for i, call in enumerate(calls, 1):
        tool = args.get("tool")
        if isinstance(call, list) and len(call) == 2:
            tool, call = call
        if not isinstance(tool, str) or tool not in specs or not isinstance(call, dict):
            raise ToolFailure(f"Call {i}: use an argument object with tool set, or [paint_tool, arguments]. "
                              "Only the instrument's paint tools can go in a batch: no views, scope, finish or "
                              "nested batches. Nothing was painted.")
        shared = {k: v for k, v in defaults.items() if k in specs[tool]}
        prepared.append((tool, {**shared, **call}))
    known = set().union(*(specs[tool] for tool, _ in prepared))
    if set(defaults) - known:
        raise ToolFailure("Unknown shared parameters: " + ", ".join(sorted(set(defaults) - known)) + ". Nothing was painted.")
    return prepared


class PaintSession:
    def __init__(self, session_dir: Path) -> None:
        self.dir = Path(session_dir)
        job = json.loads((self.dir / "job.json").read_text())
        self.job = job
        self.target = (load_target(job["target"], width=job.get("width", 128), patch=job.get("patch", 16))
                       if job.get("target") else None)
        self.inst = Instrument(job["source"])
        self.canvas = Canvas(self.target.height if self.target else job.get("height", job.get("width", 128)),
                             self.target.width if self.target else job.get("width", 128))
        self.pen = self.inst.new_state()
        self.rng = np.random.default_rng(job.get("seed", 0))
        self.actions_left = int(job.get("actions", 200))
        self.looks_left = int(job.get("looks", 8))
        self.scope_rect: tuple[int, int, int, int] | None = None  # active window (x0, y0, x1, y1), or whole-canvas
        self.scope_enabled = bool(job.get("scope", False))
        self.batch_enabled = bool(job.get("paint_batch", True))
        self.plan = ""
        self._batch_call: int | None = None
        self.snapshot_every = int(job.get("snapshot_every", 5))
        self.applied = 0
        self.rejected = 0
        self.index = 0
        self.finished = False
        (self.dir / "snaps").mkdir(exist_ok=True)
        self._log = open(self.dir / "calls.jsonl", "a", buffering=1)
        self.critic = Critic()
        self.error = self.total_error()
        self.scores = self.critic.score(self.canvas.img, self.target) if self.target else {}
        self._save_canvas()

    @property
    def score(self) -> float | None:
        return self.scores.get("total")

    # ---- measuring ----------------------------------------------------------------------------------------

    def total_error(self) -> float | None:
        """Whole-canvas RMSE, or no metric when painting from text."""
        if self.target is None:
            return None
        return float(np.sqrt(((self.canvas.img - self.target.image) ** 2).mean()) * 1000.0)

    def _save_canvas(self) -> None:
        tmp = self.dir / "canvas.tmp.npy"
        np.save(tmp, self.canvas.img)
        os.replace(tmp, self.dir / "canvas.npy")

    def _record(self, **entry) -> None:
        self.index += 1
        if self.target is None:
            entry = {k: v for k, v in entry.items() if not k.startswith(("score_", "error_"))}
        entry = {"i": self.index, "t": round(time.time(), 3), **entry}
        if self._batch_call is not None:
            entry["batch_call"] = self._batch_call
        self._log.write(json.dumps(entry, default=str) + "\n")

    def _pen_state(self) -> dict:
        def tidy(v):
            if isinstance(v, float):
                return round(v, 3)
            if isinstance(v, dict):
                return {k: tidy(x) for k, x in v.items()}
            if isinstance(v, list | tuple):
                return [tidy(x) for x in v]
            return v
        return json.loads(json.dumps(tidy(self.pen), default=str))

    def _pen_text(self) -> str:
        return json.dumps(self._pen_state())[:600]

    def working_state(self) -> dict:
        """Painter-visible state. Quality metrics stay in diagnostics, not the working feedback."""
        return {"actions_used": self.applied, "actions_left": self.actions_left, "looks_left": self.looks_left,
                "pen": self._pen_state(),
                "scope": list(self.scope_rect) if self.scope_rect else None, "plan": self.plan,
                "finished": self.finished}

    def _prepare_batch(self, args: dict) -> tuple[list[tuple[str, dict]], str]:
        plan = args.get("plan", self.plan)
        if not isinstance(plan, str) or len(plan) > MAX_PLAN_CHARS:
            raise ToolFailure(f"plan must be at most {MAX_PLAN_CHARS} characters. Nothing was painted.")
        return batch_calls(self.inst, args), plan

    def batch(self, args: dict, tool_use_id: str | None = None) -> str:
        """Ordered paint calls with shared arguments and one result. Stop on the first failed call."""
        if self.finished:
            raise ToolFailure("The painting is finished. Don't call any more tools.")
        try:
            prepared, plan = self._prepare_batch(args)
        except ToolFailure as e:
            self.rejected += 1
            self._record(tool="paint_batch", tool_use_id=tool_use_id, args=args, status="rejected", error=str(e))
            if self.rejected >= MAX_REJECTS:
                self.finished = True
                raise ToolFailure(f"{e} Too many refused calls; the painting has been ended.") from e
            raise
        self.plan = plan
        before, dry = self.applied, 0
        error = None
        try:
            for i, (tool, call) in enumerate(prepared, 1):
                if self.actions_left <= 0:
                    break
                self._batch_call = i
                try:
                    self.apply(tool, call, tool_use_id)
                    dry += int(self.canvas.dry)
                except ToolFailure as e:
                    error = f"Call {i} ({tool}) failed: {e}"
                    break
        finally:
            self._batch_call = None
        used = self.applied - before
        result = (f"Applied {used}/{len(prepared)} calls; {dry} hit the area limit. "
                  f"Skipped {len(prepared) - used - int(error is not None)} calls.\n"
                  + "State: " + json.dumps(self.working_state(), separators=(",", ":")))
        if error:
            raise ToolFailure(error + " Earlier successful calls remain painted.\n" + result)
        if self.actions_left == 0:
            result += "\nNo actions left. Call finish with your note."
        return result

    # ---- the three kinds of call --------------------------------------------------------------------------

    def apply(self, tool: str, args: dict, tool_use_id: str | None = None) -> str:
        """Run one instrument tool call. Returns the text the painter sees; raises ToolFailure for errors."""
        if self.finished:
            raise ToolFailure("The painting is finished. Don't call any more tools.")
        if self.actions_left <= 0:
            raise ToolFailure("No actions left. Call finish with your note.")
        before = self.canvas.snapshot()
        pen_before = copy.deepcopy(self.pen)
        started = time.perf_counter()
        try:
            note = self.inst.call(tool, args, self.pen, self.canvas, self.rng, scope=self.scope_rect)
        except ToolError as e:
            # Nothing a failed call did should stick: not half a stroke, not a half-updated pen.
            self.canvas.restore(before)
            self.pen = pen_before
            self.rejected += 1
            self._record(tool=tool, tool_use_id=tool_use_id, args=args, status="rejected", error=str(e),
                         ms=round((time.perf_counter() - started) * 1000, 1))
            if self.rejected >= MAX_REJECTS:
                self.finished = True
                raise ToolFailure(f"{e}. Too many refused calls; the painting has been ended.")
            raise ToolFailure(f"{e}. Nothing was painted and no action was used.")
        self.actions_left -= 1
        self.applied += 1
        # Keep metrics for diagnostics and final evaluation, but don't make each stroke a numeric reward.
        # Pixel-based feedback can discourage finishing shapes and brushwork that improve perceived likeness.
        old, self.error = self.error, self.total_error()
        old_score, self.scores = self.score, self.critic.score(self.canvas.img, self.target) if self.target else {}
        snap = None
        last = self.actions_left == 0
        if self.applied % self.snapshot_every == 0 or last:
            snap = f"snaps/{self.applied:04d}.png"
            (self.dir / snap).write_bytes(to_png(self.canvas.img))
        self._save_canvas()
        self._record(tool=tool, tool_use_id=tool_use_id, args=args, status="applied", note=note,
                     error_before=_rounded(old, 2), error_after=_rounded(self.error, 2), score_before=_rounded(old_score, 4),
                     score_after=_rounded(self.score, 4), area=self.canvas.area_used,
                     dry=self.canvas.dry, actions_left=self.actions_left, pen=self._pen_text(), snapshot=snap,
                     scope=list(self.scope_rect) if self.scope_rect else None,
                     ms=round((time.perf_counter() - started) * 1000, 1))
        parts = []
        if note:
            parts.append(note[:1].upper() + note[1:])
        if self.canvas.dry and "dry" not in note:
            parts.append("The call hit its area limit and stopped early")
        if self.scope_rect is not None:
            x0, y0, x1, y1 = self.scope_rect
            parts.append(f"Scope is x {x0}-{x1}, y {y0}-{y1}: that call's coordinates were local")
        parts.append("That was your last action. Call finish with your note" if last
                     else f"{self.actions_left} actions left")
        return ". ".join(p.rstrip(". ") for p in parts) + "."

    def status(self) -> str:
        scope = "" if self.scope_rect is None else f" Scope is x {self.scope_rect[0]}-{self.scope_rect[2]}, y {self.scope_rect[1]}-{self.scope_rect[3]}."
        return f"{self.applied} actions used, {self.actions_left} left, {self.looks_text()}.{scope}"

    def looks_text(self) -> str:
        return "unlimited looks" if self.looks_left < 0 else f"{self.looks_left} looks left"

    def look(self, tool_use_id: str | None = None) -> tuple[str, bytes]:
        if self.looks_left == 0:
            raise ToolFailure("No looks left. Use the instrument's free views if available, or continue from your last view and plan.")
        if self.looks_left > 0:
            self.looks_left -= 1
        self._record(tool="look", tool_use_id=tool_use_id, status="applied", looks_left=self.looks_left,
                     error_after=_rounded(self.error, 2), score_after=_rounded(self.score, 4))
        return self.status(), gridded_png(self.canvas.img)

    def view(self, tool: str, args: dict, tool_use_id: str | None = None) -> tuple[str, list[bytes]]:
        """
        One of the instrument's viewing calls. Free: no action, no look, and the painting cannot change under
        one. The text says where the window was; the pictures are the views.
        """
        started = time.perf_counter()
        try:
            views, note = self.inst.view(tool, args, self.pen, self.canvas, self.rng)
        except ToolError as e:
            self.rejected += 1
            self._record(tool=tool, tool_use_id=tool_use_id, args=args, status="rejected", error=str(e),
                         ms=round((time.perf_counter() - started) * 1000, 1))
            if self.rejected >= MAX_REJECTS:
                self.finished = True
                raise ToolFailure(f"{e}. Too many refused calls; the painting has been ended.")
            raise ToolFailure(f"{e}. No action was used.")
        self._record(tool=tool, tool_use_id=tool_use_id, args=args, status="view", note=note,
                     views=[list(v.rect) for v in views], error_after=_rounded(self.error, 2),
                     score_after=_rounded(self.score, 4), actions_left=self.actions_left,
                     ms=round((time.perf_counter() - started) * 1000, 1))
        parts = []
        if note:
            parts.append(note)
        for v in views:
            parts.append(f"Window x {v.x0}-{v.x1}, y {v.y0}-{v.y1}, at {v.scale} image px per canvas px.")
        parts.append("Free view: no action and no look used. "
                     f"{self.actions_left} actions left, {self.looks_text()}.")
        return "\n\n".join(parts), [view_png(v) for v in views]

    def scope(self, args: dict, tool_use_id: str | None = None) -> tuple[str, list[bytes]]:
        """
        Set or clear the painting scope, the window later paint calls address in local coordinates.
        Free: no action, no look, and the painting cannot change under one. `clear` drops the scope;
        otherwise (x, y, span) centres a square window like a view. The picture comes back labelled from
        the window's own corner, so the painter reads local coordinates straight off it.
        """
        started = time.perf_counter()
        ms = lambda: round((time.perf_counter() - started) * 1000, 1)
        if args.get("clear") in (True, "true", "1", 1):
            self.scope_rect = None
            self._record(tool="scope", tool_use_id=tool_use_id, args=args, status="scope", scope=None,
                         error_after=_rounded(self.error, 2), score_after=_rounded(self.score, 4),
                         actions_left=self.actions_left, ms=ms())
            return (f"Scope cleared: coordinates are canvas pixels again. {self.actions_left} actions left.", [])
        try:
            x = float(args.get("x", self.canvas.width / 2))
            y = float(args.get("y", self.canvas.height / 2))
            span = float(args.get("span", 64))
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(span)):
                raise ValueError
        except (TypeError, ValueError):
            raise ToolFailure("scope needs numbers for x, y and span, or clear for the whole canvas.")
        v = self.canvas.view(x, y, span)
        self.scope_rect = v.rect
        x0, y0, x1, y1 = v.rect
        side = x1 - x0
        self._record(tool="scope", tool_use_id=tool_use_id, args=args, status="scope", scope=list(v.rect),
                     error_after=_rounded(self.error, 2), score_after=_rounded(self.score, 4),
                     actions_left=self.actions_left, ms=ms())
        png = gridded_png(v.img, scale=v.scale, x0=x0, y0=y0, lx0=x0, ly0=y0)
        return (f"Scope is x {x0}-{x1}, y {y0}-{y1}: until cleared, paint calls take local coordinates 0-{side}, "
                f"where local (0, 0) is canvas ({x0}, {y0}). Deltas, sizes and angles are unchanged.\n\n"
                "Free: no action and no look used. "
                f"{self.actions_left} actions left, {self.looks_text()}.", [png])

    def finish(self, note: str, tool_use_id: str | None = None) -> str:
        if self.finished:
            return "Already finished."
        self.finished = True
        record = {"note": str(note)[:4000], "actions_used": self.applied, "actions_left": self.actions_left,
                  "at": time.time()}
        if self.target is not None:
            record.update(error=round(self.error, 2), score=round(self.score, 4))
        (self.dir / "finish.json").write_text(json.dumps(record))
        self._record(tool="finish", tool_use_id=tool_use_id, status="applied", note=record["note"][:500])
        self._save_canvas()
        return "Finished. The painting is done: stop now and don't call any more tools."


class PaintServer(StdioServer):
    name = "canvas"

    def __init__(self, session: PaintSession) -> None:
        self.s = session

    def tools(self) -> list[dict]:
        w, h = self.s.canvas.width, self.s.canvas.height
        tools = self.s.inst.mcp_tools(w, h) + self.s.inst.mcp_views(w, h)
        if self.s.scope_enabled:
            tools.append({
                "name": "scope",
                "description": "Paint in a window: set a scope centred on (x, y), `span` canvas px across, "
                "and later paint calls take local coordinates from its top-left corner until cleared. "
                "Free: no action, no look, and the painting cannot change under one.",
                "inputSchema": {"type": "object", "properties": {
                    "x": {"type": "number", "description": "Centre x in canvas pixels."},
                    "y": {"type": "number", "description": "Centre y in canvas pixels."},
                    "span": {"type": "number", "description": "Window width in canvas px."},
                    "clear": {"type": "boolean", "description": "Drop the scope; coordinates are "
                    "canvas pixels again."}}},
            })
        if self.s.batch_enabled:
            tools.append({
                "name": "paint_batch",
                "description": "Paint up to 40 calls in order, with one summary result. Set tool for argument-only "
                "calls, or use [tool_name, arguments] pairs to mix tools. Shared defaults apply only to matching "
                "parameters; each call overrides them. Each call costs one action and has its own area limit. "
                "Stops at the first failure or when actions run out; earlier successful calls remain painted. "
                "Keep a short working plan here so it survives history trimming.",
                "inputSchema": {"type": "object", "properties": {
                    "tool": {"type": "string", "enum": [t.name for t in self.s.inst.spec.tools]},
                    "defaults": {"type": "object", "description": "Shared instrument arguments, overridden per call."},
                    "calls": {"type": "array", "minItems": 1, "maxItems": MAX_BATCH_CALLS, "items": {"anyOf": [
                        {"type": "object"},
                        {"type": "array", "minItems": 2, "maxItems": 2,
                         "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}},
                    ]}},
                    "plan": {"type": "string", "maxLength": MAX_PLAN_CHARS,
                             "description": "What is done, what remains, useful colours/settings and mistakes to avoid."},
                }, "required": ["calls"], "additionalProperties": False},
            })
        return tools + [
            {"name": "look", "description": "See the whole canvas as it is now, gridded in canvas pixels. "
             "Usually limited: the status line says how many looks are left.",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "finish", "description": "End the painting. Say what the target needed that these tools "
             "could not do, and which tool behaviour was hard to control. The instrument's designer reads it.",
             "inputSchema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}},
        ]

    def structured_content(self, meta: dict) -> dict | None:
        # Pi asks for this separately from model-visible text. Other clients keep their usual MCP result.
        return {"painting_state": self.s.working_state()} if meta.get("conveyor/paintingState") else None

    def call(self, name: str, args: dict, meta: dict) -> list[dict]:
        tool_use_id = meta.get("claudecode/toolUseId")
        if name == "paint_batch":
            if not self.s.batch_enabled:
                raise ToolFailure("No tool named paint_batch.")
            return [text(self.s.batch(args, tool_use_id))]
        if name == "look":
            status, png = self.s.look(tool_use_id)
            return [text(status), image(png)]
        if name == "finish":
            return [text(self.s.finish(args.get("note", ""), tool_use_id))]
        if name in {v.name for v in self.s.inst.spec.views}:
            status, pngs = self.s.view(name, args, tool_use_id)
            return [text(status), *(image(p) for p in pngs)]
        if name == "scope":
            if not self.s.scope_enabled:
                raise ToolFailure("No tool named scope.")
            status, pngs = self.s.scope(args, tool_use_id)
            return [text(status), *(image(p) for p in pngs)]
        if name not in {t.name for t in self.s.inst.spec.tools}:
            raise ToolFailure(f"No tool named {name}.")
        return [text(self.s.apply(name, args, tool_use_id))]


# ---- the offline painter -----------------------------------------------------------------------------------


def _jitter(inst: Instrument, tool: str, args: dict, rng: np.random.Generator, w: int, h: int) -> dict:
    """An example call with its numbers moved: resampled within bounds where there are bounds, scaled otherwise."""
    out = dict(args)
    for p in inst.spec.tool(tool).params:
        v = out.get(p.name, p.default if p.has_default else None)
        lo, hi = p.bound(p.min, w, h), p.bound(p.max, w, h)
        if p.type in ("number", "integer"):
            if lo is not None and hi is not None and rng.random() < 0.8:
                v = float(rng.uniform(lo, hi))
            elif v is not None:
                v = float(v) * float(rng.uniform(0.5, 1.6))
            out[p.name] = v
        elif p.type == "points" and v:
            dx, dy = rng.uniform(-w / 2, w / 2), rng.uniform(-h / 2, h / 2)
            out[p.name] = [[float(x + dx + rng.normal(0, 3)), float(y + dy + rng.normal(0, 3))] for x, y in v]
        elif p.type == "choice" and rng.random() < 0.3:
            out[p.name] = p.options[int(rng.integers(len(p.options)))]
    return out


def greedy_paint(s: PaintSession, tries_per_action: int = 6) -> None:
    """
    Paint with any instrument and no model: take an example call sequence, move its numbers around, pick each
    colour to match the target under wherever the marks landed, and keep the sequence only if it raises the
    score. It knows nothing about what the tools mean, which makes it a fair stand-in for tests and a measure of
    what an interface allows when driven blindly.
    """
    inst, canvas, rng = s.inst, s.canvas, s.rng
    w, h = canvas.width, canvas.height
    probe_color = (1.0, 0.0, 1.0)
    attempts = 0
    while s.actions_left > 0 and attempts < s.job.get("actions", 200) * tries_per_action:
        attempts += 1
        seq = inst.spec.examples[int(rng.integers(len(inst.spec.examples)))]
        if len(seq) > s.actions_left:
            continue
        calls = [(tool, _jitter(inst, tool, args, rng, w, h)) for tool, args in seq]
        colored = {(i, p.name) for i, (tool, _) in enumerate(calls) for p in inst.spec.tool(tool).params
                   if p.type == "color"}
        before, pen0 = canvas.snapshot(), copy.deepcopy(s.pen)

        def with_color(i: int, args: dict, color) -> dict:
            a = dict(args)
            if color is not None:
                a.update({name: color for j, name in colored if j == i})
            return a

        def run(color) -> bool:
            s.pen = copy.deepcopy(pen0)
            canvas.restore(before)
            try:
                for i, (tool, args) in enumerate(calls):
                    inst.call(tool, with_color(i, args, color), s.pen, canvas, rng)
            except ToolError:
                return False
            return True

        if colored:
            if not run(probe_color):
                continue
            moved = np.abs(canvas.img - before).sum(axis=-1)
            reach = np.abs(np.asarray(probe_color, dtype=np.float32) - before).sum(axis=-1) + 1e-6
            alpha = np.clip(moved / reach, 0, 1)
            if alpha.sum() < 1e-3:
                continue
            color = tuple(float(c) for c in (s.target.image * alpha[..., None]).sum(axis=(0, 1)) / alpha.sum())
        else:
            color = None
        if not run(color):
            continue
        new_error = s.total_error()
        new_scores = s.critic.score(canvas.img, s.target)
        if new_scores["total"] > s.score + 1e-5:
            for i, (tool, args) in enumerate(calls):
                shown = with_color(i, args, to_hex(color) if color is not None else None)
                s.actions_left -= 1
                s.applied += 1
                s._record(tool=tool, args=shown, status="applied", error_after=round(new_error, 2),
                          score_after=round(new_scores["total"], 4), greedy=True)
            s.error, s.scores = new_error, new_scores
            if s.applied % s.snapshot_every < len(calls):
                (s.dir / f"snaps/{s.applied:04d}.png").write_bytes(to_png(canvas.img))
        else:
            canvas.restore(before)
            s.pen = pen0
    s._save_canvas()
    s.finish(f"greedy painter: {s.applied} actions kept from {attempts} tries")


def main(argv: list[str]) -> int:
    session = PaintSession(Path(argv[0]))
    if "--greedy" in argv[1:]:
        greedy_paint(session)
        return 0
    PaintServer(session).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
