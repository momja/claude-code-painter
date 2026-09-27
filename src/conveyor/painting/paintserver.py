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
from conveyor.painting.canvas import error_table
from conveyor.painting.canvas import gridded_png
from conveyor.painting.canvas import load_target
from conveyor.painting.canvas import to_hex
from conveyor.painting.canvas import to_png
from conveyor.painting.critic import Critic
from conveyor.painting.instrument import Instrument
from conveyor.painting.instrument import ToolError

MAX_REJECTS = 60  # calls refused for bad arguments before the painting is ended, so a confused model can't loop


class PaintSession:
    def __init__(self, session_dir: Path) -> None:
        self.dir = Path(session_dir)
        job = json.loads((self.dir / "job.json").read_text())
        self.job = job
        self.target = load_target(job["target"], width=job.get("width", 128), patch=job.get("patch", 16))
        self.inst = Instrument(job["source"])
        self.canvas = Canvas(self.target.height, self.target.width)
        self.pen = self.inst.new_state()
        self.rng = np.random.default_rng(job.get("seed", 0))
        self.actions_left = int(job.get("actions", 200))
        self.looks_left = int(job.get("looks", 8))
        self.snapshot_every = int(job.get("snapshot_every", 5))
        self.applied = 0
        self.rejected = 0
        self.index = 0
        self.finished = False
        (self.dir / "snaps").mkdir(exist_ok=True)
        self._log = open(self.dir / "calls.jsonl", "a", buffering=1)
        self.critic = Critic()
        self.error = self.total_error()
        self.scores = self.critic.score(self.canvas.img, self.target)
        self._save_canvas()

    @property
    def score(self) -> float:
        return self.scores["total"]

    # ---- measuring ----------------------------------------------------------------------------------------

    def total_error(self) -> float:
        """Whole-canvas RMSE against the target, times 1000 so the numbers read easily."""
        return float(np.sqrt(((self.canvas.img - self.target.image) ** 2).mean()) * 1000.0)

    def _save_canvas(self) -> None:
        tmp = self.dir / "canvas.tmp.npy"
        np.save(tmp, self.canvas.img)
        os.replace(tmp, self.dir / "canvas.npy")

    def _record(self, **entry) -> None:
        self.index += 1
        entry = {"i": self.index, "t": round(time.time(), 3), **entry}
        self._log.write(json.dumps(entry, default=str) + "\n")

    def _pen_text(self) -> str:
        def tidy(v):
            if isinstance(v, float):
                return round(v, 3)
            if isinstance(v, dict):
                return {k: tidy(x) for k, x in v.items()}
            if isinstance(v, list | tuple):
                return [tidy(x) for x in v]
            return v
        return json.dumps(tidy(self.pen), default=str)[:600]

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
            note = self.inst.call(tool, args, self.pen, self.canvas, self.rng)
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
        # The painter steers by what each call returns, so it gets the critic's score, the number the painting is
        # judged by, and not only pixel error. With pixel error alone, a painter watched its swirl texture push the
        # error up and stopped with 28 actions left, on the painting that turned out to score best of its run.
        old, self.error = self.error, self.total_error()
        old_score, self.scores = self.score, self.critic.score(self.canvas.img, self.target)
        snap = None
        last = self.actions_left == 0
        if self.applied % self.snapshot_every == 0 or last:
            snap = f"snaps/{self.applied:04d}.png"
            (self.dir / snap).write_bytes(to_png(self.canvas.img))
        self._save_canvas()
        self._record(tool=tool, tool_use_id=tool_use_id, args=args, status="applied", note=note,
                     error_before=round(old, 2), error_after=round(self.error, 2), score_before=round(old_score, 4),
                     score_after=round(self.score, 4), area=self.canvas.area_used,
                     dry=self.canvas.dry, actions_left=self.actions_left, pen=self._pen_text(), snapshot=snap,
                     ms=round((time.perf_counter() - started) * 1000, 1))
        parts = [f"score {self.score:.4f} ({self.score - old_score:+.4f}), pixel error {self.error:.1f} ({self.error - old:+.1f})"]
        if note:
            parts.append(note[:1].upper() + note[1:])
        if self.canvas.dry and "dry" not in note:
            parts.append("The call hit its area limit and stopped early")
        parts.append("That was your last action. Call finish with your note" if last
                     else f"{self.actions_left} actions left")
        return ". ".join(p.rstrip(". ") for p in parts) + "."

    def status(self) -> str:
        sc = self.scores
        style = ", ".join(f"{k} {v:.2f}" for k, v in sc["style_distance"].items())
        return (f"{self.applied} actions used, {self.actions_left} left, {self.looks_left} looks left.\n"
                f"Score {sc['total']:.4f}: pixel match {sc['pixel']:.3f} (60%), texture and palette match "
                f"{sc['style']:.3f} (40%; distances from the target, 0 is a match: {style}). "
                f"Pixel error {self.error:.1f}.\n" + error_table(self.canvas.img, self.target))

    def look(self, tool_use_id: str | None = None) -> tuple[str, bytes]:
        if self.looks_left <= 0:
            raise ToolFailure("No looks left. Keep painting from the score each call returns.")
        self.looks_left -= 1
        self._record(tool="look", tool_use_id=tool_use_id, status="applied", looks_left=self.looks_left,
                     error_after=round(self.error, 2), score_after=round(self.score, 4))
        return self.status(), gridded_png(self.canvas.img)

    def finish(self, note: str, tool_use_id: str | None = None) -> str:
        if self.finished:
            return "Already finished."
        self.finished = True
        record = {"note": str(note)[:4000], "actions_used": self.applied, "actions_left": self.actions_left,
                  "error": round(self.error, 2), "score": round(self.score, 4), "at": time.time()}
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
        return self.s.inst.mcp_tools(w, h) + [
            {"name": "look", "description": "See the canvas as it is now, with the score's parts and the pixel error "
             "for each region. Limited: the status line says how many looks are left.",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "finish", "description": "End the painting. Say what the target needed that these tools "
             "could not do, and which tool behaviour was hard to control. The instrument's designer reads it.",
             "inputSchema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}},
        ]

    def call(self, name: str, args: dict, meta: dict) -> list[dict]:
        tool_use_id = meta.get("claudecode/toolUseId")
        if name == "look":
            status, png = self.s.look(tool_use_id)
            return [text(status), image(png)]
        if name == "finish":
            return [text(self.s.finish(args.get("note", ""), tool_use_id))]
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
