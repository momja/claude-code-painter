"""
One agent on the shared canvas, as an MCP server in its own process.

    python -m conveyor.commons.server <session_dir>

The session directory holds `job.json`: the database, the canvas and agent ids, the instrument's source, the
viewport's starting corner, and whether batches and successors are on. The server gives the agent the
instrument's tools on its viewport, plus `look`, `overview`, `move_viewport`, `write_message`, `paint_batch`
and `spawn_successor`, and counts every call it receives against the canvas's tool-call budget, refused calls
included. Past the budget, or once the agent has handed off to a successor, every call is refused.

`spawn_successor` only writes `successor.json` with where the agent's viewport stands. The host
process queues the successor once the session has closed (see agent.py).

Each call reads the viewport fresh from the database, so the agent sees what others painted since its last call.
A call that paints runs inside one write transaction (see tiles.py). The instrument sees the viewport as its
whole canvas, so any instrument from any run works here unchanged, and its marks are clipped at the viewport's
edge. The pen persists across calls and across moves; its coordinates are the viewport's.

Like the paint server, it also writes `calls.jsonl` and viewport snapshots to the session directory, which the
host copies into the session's stroke log, so the dashboard's session drawer can replay one agent's view.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from conveyor.commons import prompts
from conveyor.commons.lettering import MAX_MESSAGE
from conveyor.commons.lettering import MAX_SCALE
from conveyor.commons.lettering import draw_messages
from conveyor.commons.lettering import text_mask
from conveyor.commons.lettering import wrap
from conveyor.commons.overview import OVERVIEW_SIDE
from conveyor.commons.overview import overview_png
from conveyor.commons.tiles import DEFAULT_INK
from conveyor.commons.tiles import MOVE_SHARE
from conveyor.commons.tiles import SharedCanvas
from conveyor.commons.tiles import encode_tile
from conveyor.commons.tiles import to_uint8
from conveyor.mcp import StdioServer
from conveyor.mcp import ToolFailure
from conveyor.mcp import image
from conveyor.mcp import text
from conveyor.painting.canvas import Canvas
from conveyor.painting.canvas import CanvasError
from conveyor.painting.canvas import gridded_png
from conveyor.painting.canvas import parse_color
from conveyor.painting.canvas import view_png
from conveyor.painting.instrument import Instrument
from conveyor.painting.instrument import ToolError
from conveyor.painting.paintserver import MAX_BATCH_CALLS
from conveyor.painting.paintserver import batch_calls

HARNESS_TOOLS = ("look", "overview", "move_viewport", "write_message", "paint_batch", "spawn_successor")


class AgentSession:
    def __init__(self, session_dir: Path) -> None:
        self.dir = Path(session_dir)
        job = json.loads((self.dir / "job.json").read_text())
        self.canvas = SharedCanvas(job["db"], job["canvas_id"])
        self.agent_id = job["agent_id"]
        self.size = int(self.canvas.config["viewport"])
        self.max_calls = int(job.get("max_calls") or self.canvas.config["max_calls"])
        self.x, self.y = int(job["x"]), int(job["y"])
        self.inst = Instrument(job["source"])
        taken = {t.name for t in [*self.inst.spec.tools, *self.inst.spec.views]} & set(HARNESS_TOOLS)
        if taken:
            raise ValueError(f"the instrument's tool names clash with the canvas's: {sorted(taken)}")
        self.batch_enabled = bool(job.get("paint_batch", True))
        self.successors = bool(job.get("successors", False))
        self.handed_off = False
        self.pen = self.inst.new_state()
        self.rng = np.random.default_rng(job.get("seed", 0))
        self.calls_used = 0
        self.index = 0
        (self.dir / "snaps").mkdir(exist_ok=True)
        self._log = open(self.dir / "calls.jsonl", "a", buffering=1)

    # ---- bookkeeping ----------------------------------------------------------------------------------------

    @property
    def calls_left(self) -> int:
        return max(0, self.max_calls - self.calls_used)

    @property
    def area_cap(self) -> int:
        return Canvas(self.size, self.size).area_cap

    def status(self) -> str:
        where = (f"Your viewport is x {self.x} to {self.x + self.size}, y {self.y} to {self.y + self.size} "
                 "on the shared canvas.")
        if self.handed_off:
            return prompts.HANDED_OFF
        if not self.calls_left:
            return prompts.LAST_CALL + " " + where
        reminder = " " + prompts.SUCCESSOR_REMINDER if self.successors and self.calls_left <= 3 else ""
        return f"{self.calls_used} of {self.max_calls} tool calls used, {self.calls_left} left.{reminder} {where}"

    @property
    def over(self) -> bool:
        """No more calls: the budget is spent, or the work went to a successor."""
        return self.handed_off or self.calls_used >= self.max_calls

    def _op(self, tool: str, status: str, args: dict | None, note: str = "", tiles=None,
            snapshot: np.ndarray | None = None, tool_use_id: str | None = None, **extra) -> None:
        """Log the call on the canvas (the replay) and in calls.jsonl (this agent's session)."""
        self.canvas.record(agent_id=self.agent_id, tool=tool, status=status, x=self.x, y=self.y, args=args,
                           note=note, tiles=tiles)
        self.canvas.update_agent(self.agent_id, calls_used=self.calls_used, x=self.x, y=self.y, last_ts=time.time())
        self.index += 1
        entry = {"i": self.index, "t": round(time.time(), 3), "tool": tool, "tool_use_id": tool_use_id, "args": args,
                 "status": "applied" if status == "painted" else status, "note": note[:500], "x": self.x, "y": self.y,
                 "tiles": len(tiles or {}), "calls_left": self.calls_left, "pen": json.dumps(self.pen, default=str)[:600],
                 **extra}
        if snapshot is not None:  # the viewport after the call, for this agent's replay in the session drawer
            entry["snapshot"] = f"snaps/{self.index:04d}.png"
            (self.dir / entry["snapshot"]).write_bytes(encode_tile(to_uint8(snapshot)))
        self._log.write(json.dumps(entry, default=str) + "\n")

    def _seen(self) -> np.ndarray:
        """The viewport as the agent sees it: the paint, with the text layer on top. Paint calls work on the paint
        alone, so an instrument that reads the canvas never picks up lettering."""
        x, y, s = self.x, self.y, self.size
        return draw_messages(self.canvas.read(x, y, s, s).image(), x, y, self.canvas.messages(x, y, x + s, y + s))

    def _viewport(self) -> Canvas:
        return Canvas(self.size, self.size, image=self._seen())

    # ---- the tools ------------------------------------------------------------------------------------------

    def call(self, name: str, args: dict, tool_use_id: str | None = None) -> list[dict]:
        """Run one tool call against the budget. Every call counts, including the ones refused."""
        if self.handed_off:
            raise ToolFailure(prompts.HANDED_OFF)
        if self.calls_used >= self.max_calls:
            raise ToolFailure(prompts.OUT_OF_CALLS.format(max_calls=self.max_calls))
        self.calls_used += 1
        try:
            blocks = self._dispatch(name, args, tool_use_id)
        except ToolFailure as e:
            raise ToolFailure(f"{e}\n\n{self.status()}") from e
        blocks[0] = text((blocks[0]["text"] + "\n\n" + self.status()).strip())
        return blocks

    def _dispatch(self, name: str, args: dict, tool_use_id: str | None) -> list[dict]:
        if name == "look":
            return self.look(tool_use_id)
        if name == "overview":
            return self.overview(tool_use_id)
        if name == "move_viewport":
            return self.move(args, tool_use_id)
        if name == "write_message":
            return [text(self.write_message(args, tool_use_id))]
        if name == "spawn_successor" and self.successors:
            return [text(self.spawn_successor(args, tool_use_id))]
        if name == "paint_batch" and self.batch_enabled:
            return [text(self.batch(args, tool_use_id))]
        if name in {v.name for v in self.inst.spec.views}:
            return self.view(name, args, tool_use_id)
        if name in {t.name for t in self.inst.spec.tools}:
            return [text(self.paint(name, args, tool_use_id))]
        self._op(name, "rejected", args, "no such tool", tool_use_id=tool_use_id)
        raise ToolFailure(f"No tool named {name}.")

    def look(self, tool_use_id: str | None = None) -> list[dict]:
        img = self._seen()
        self._op("look", "view", {}, tool_use_id=tool_use_id)
        return [text("Your viewport as it is now, labelled in viewport pixels."), image(gridded_png(img))]

    def overview(self, tool_use_id: str | None = None) -> list[dict]:
        viewport = (self.x, self.y, self.x + self.size, self.y + self.size)
        img, (x0, y0, x1, y1), scale = self.canvas.overview(viewport, OVERVIEW_SIDE)
        img = to_uint8(draw_messages(img.astype(np.float32) / 255.0, x0, y0, self.canvas.messages(x0, y0, x1, y1), scale))
        self._op("overview", "view", {}, f"x {x0} to {x1}, y {y0} to {y1}", tool_use_id=tool_use_id)
        size = "at full size" if scale >= 1 else f"shrunk so one picture pixel is {1 / scale:.3g} canvas pixels"
        return [text(f"The whole canvas, x {x0} to {x1} and y {y0} to {y1}: everything painted so far and your "
                     f"viewport, {size}. The labels are canvas coordinates. Your viewport is the magenta box."),
                image(overview_png(img, (x0, y0, x1, y1), scale, viewport))]

    def move(self, args: dict, tool_use_id: str | None = None) -> list[dict]:
        limit = MOVE_SHARE * self.size
        try:
            angle = float(args.get("angle"))
            distance = float(args.get("distance", limit))
            if not (math.isfinite(angle) and math.isfinite(distance)):
                raise ValueError
        except (TypeError, ValueError):
            self._op("move_viewport", "rejected", args, "needs numbers", tool_use_id=tool_use_id)
            raise ToolFailure("move_viewport needs `angle` in degrees and `distance` in pixels. You didn't move.")
        distance = min(max(distance, 0.0), limit)
        dx, dy = round(distance * math.cos(math.radians(angle))), round(distance * math.sin(math.radians(angle)))
        self.x, self.y = self.x + dx, self.y + dy
        img = self._seen()
        self._op("move_viewport", "moved", {"angle": angle, "distance": round(distance, 1)}, f"moved by ({dx}, {dy})",
                 snapshot=img, tool_use_id=tool_use_id)
        return [text(f"Moved {distance:.0f} px toward {angle:g} degrees: by {dx} in x and {dy} in y. "
                     "Your new viewport, labelled in viewport pixels:"), image(gridded_png(img))]

    def view(self, tool: str, args: dict, tool_use_id: str | None = None) -> list[dict]:
        try:
            views, note = self.inst.view(tool, args, self.pen, self._viewport(), self.rng)
        except ToolError as e:
            self._op(tool, "rejected", args, str(e), tool_use_id=tool_use_id)
            raise ToolFailure(f"{e}.")
        self._op(tool, "view", args, note, tool_use_id=tool_use_id, views=[list(v.rect) for v in views])
        parts = ([note] if note else []) + [f"Window x {v.x0}-{v.x1}, y {v.y0}-{v.y1} of your viewport, "
                                            f"at {v.scale} image px per pixel." for v in views]
        return [text("\n\n".join(parts)), *(image(view_png(v)) for v in views)]

    def paint(self, tool: str, args: dict, tool_use_id: str | None = None) -> str:
        error = None
        with self.canvas.transaction():
            region = self.canvas.read(self.x, self.y, self.size, self.size)
            canvas = Canvas(self.size, self.size, image=region.image())
            pen_before = copy.deepcopy(self.pen)
            try:
                note = self.inst.call(tool, args, self.pen, canvas, self.rng)
            except ToolError as e:
                self.pen, error = pen_before, str(e)
                self._op(tool, "rejected", args, error, tool_use_id=tool_use_id)
            else:
                tiles = region.changed(canvas.img)
                if canvas.dry and "dry" not in note:
                    note = (note + "; " if note else "") + "the call hit its area limit and stopped early"
                self._op(tool, "painted", args, note, tiles=tiles, snapshot=canvas.img if tiles else None,
                         tool_use_id=tool_use_id, dry=canvas.dry, area=canvas.area_used)
        if error is not None:
            raise ToolFailure(f"{error}. Nothing was painted.")
        return (note[:1].upper() + note[1:] + "." if note else "Painted.") if tiles else (
            (note + ". " if note else "") + "Nothing visible changed.")

    def batch(self, args: dict, tool_use_id: str | None = None) -> str:
        try:
            prepared = batch_calls(self.inst, args)
        except ToolFailure as e:
            self._op("paint_batch", "rejected", args, str(e), tool_use_id=tool_use_id)
            raise
        error, used, dry = None, 0, 0
        with self.canvas.transaction():
            region = self.canvas.read(self.x, self.y, self.size, self.size)
            canvas = Canvas(self.size, self.size, image=region.image())
            for i, (tool, call) in enumerate(prepared, 1):
                pen_before = copy.deepcopy(self.pen)
                try:
                    self.inst.call(tool, call, self.pen, canvas, self.rng)
                except ToolError as e:
                    self.pen, error = pen_before, f"Call {i} ({tool}) failed: {str(e).rstrip('.')}."
                    break
                used += 1
                dry += int(canvas.dry)
            tiles = region.changed(canvas.img)
            summary = f"Applied {used} of {len(prepared)} calls; {dry} hit the area limit."
            self._op("paint_batch", "painted" if used else "rejected", args, (error + " " if error else "") + summary,
                     tiles=tiles, snapshot=canvas.img if tiles else None, tool_use_id=tool_use_id, applied=used)
        if error:
            raise ToolFailure(f"{error} The calls before it stay painted. {summary}")
        return summary

    def spawn_successor(self, args: dict, tool_use_id: str | None = None) -> str:
        # No note: a successor inherits only the canvas, so what it should know has to be there for anyone to see.
        (self.dir / "successor.json").write_text(json.dumps({"x": self.x, "y": self.y, "at": time.time()}))
        self.handed_off = True
        self._op("spawn_successor", "handed_off", {}, tool_use_id=tool_use_id)
        return (f"Your successor starts here, with its viewport at canvas ({self.x}, {self.y}) and {self.max_calls} "
                "fresh tool calls, once this session closes. It gets the canvas and nothing of this conversation.")

    def write_message(self, args: dict, tool_use_id: str | None = None) -> str:
        try:
            message = args.get("text")
            if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
                raise ToolFailure(f"text must be 1 to {MAX_MESSAGE} characters.")
            x, y = int(round(float(args.get("x", 0)))), int(round(float(args.get("y", 0))))
            scale = int(args.get("scale", 2))
            if not 1 <= scale <= MAX_SCALE:
                raise ToolFailure(f"scale must be 1 to {MAX_SCALE}.")
            if not (0 <= x < self.size and 0 <= y < self.size):
                raise ToolFailure(f"x and y must be inside the viewport, 0 to {self.size}.")
            color = str(args.get("color") or DEFAULT_INK)
            parse_color(color)
        except (TypeError, ValueError, CanvasError) as e:
            self._op("write_message", "rejected", args, str(e), tool_use_id=tool_use_id)
            raise ToolFailure(f"{e}. Nothing was written.") from None
        except ToolFailure as e:
            self._op("write_message", "rejected", args, str(e), tool_use_id=tool_use_id)
            raise ToolFailure(f"{e} Nothing was written.") from None
        mask = text_mask(message, self.size - x, scale)[: self.size - y, : self.size - x]
        h, w = mask.shape
        covered = int(mask.sum())
        if covered > self.area_cap:
            self._op("write_message", "rejected", args, "over the area limit", tool_use_id=tool_use_id)
            raise ToolFailure(f"That message's letters would cover {covered} pixels, past the area limit of "
                              f"{self.area_cap} one call may cover. Write less or use a smaller scale. Nothing was written.")
        # The message is the op itself: the text layer reads it back from the log, and no tile changes.
        self._op("write_message", "written", {"text": message, "x": x, "y": y, "scale": scale, "color": color},
                 message, tool_use_id=tool_use_id)
        lines = len(wrap(message, self.size - x, scale))
        return (f"Wrote {len(message)} characters in {lines} line{'s' if lines != 1 else ''}, {w} x {h} px from "
                f"({x}, {y}), above the paint. Paint won't cover it.")


class CommonsServer(StdioServer):
    name = "commons"

    def __init__(self, session: AgentSession) -> None:
        self.s = session

    def tools(self) -> list[dict]:
        s = self.s.size
        tools = self.s.inst.mcp_tools(s, s) + self.s.inst.mcp_views(s, s)
        if self.s.batch_enabled:
            tools.append({
                "name": "paint_batch",
                "description": f"Run up to {MAX_BATCH_CALLS} of the instrument's paint calls in order, for one "
                "tool call of your budget. Set tool for argument-only calls, or use [tool_name, arguments] pairs "
                "to mix tools. Shared defaults apply only to matching parameters; each call overrides them. Stops "
                "at the first failure; earlier calls stay painted.",
                "inputSchema": {"type": "object", "properties": {
                    "tool": {"type": "string", "enum": [t.name for t in self.s.inst.spec.tools]},
                    "defaults": {"type": "object", "description": "Shared instrument arguments, overridden per call."},
                    "calls": {"type": "array", "minItems": 1, "maxItems": MAX_BATCH_CALLS, "items": {"anyOf": [
                        {"type": "object"},
                        {"type": "array", "minItems": 2, "maxItems": 2,
                         "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}},
                    ]}},
                }, "required": ["calls"], "additionalProperties": False},
            })
        if self.s.successors:
            tools.append({"name": "spawn_successor", "description": "End your session now and start a new session "
                          f"of you where your viewport is, with {self.s.max_calls} fresh tool calls. It sees only the "
                          "canvas, nothing of this conversation. Use it to keep working past your budget.",
                          "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}})
        return tools + [
            {"name": "look", "description": "See your viewport as it is now, gridded in viewport pixels. Others "
             "may have painted in it since you last looked.", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "overview", "description": f"See the whole canvas, everything anyone has painted, shrunk to "
             f"at most {OVERVIEW_SIDE} px a side and labelled in canvas coordinates, with your viewport outlined in "
             "magenta.", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "move_viewport", "description": f"Slide your viewport `distance` px toward `angle` degrees "
             f"(0 right, 90 down, 180 left, 270 up), at most {int(MOVE_SHARE * s)} px, so the new view overlaps the "
             "old. Shows you the new view.",
             "inputSchema": {"type": "object", "properties": {
                 "angle": {"type": "number", "description": "Direction in degrees: 0 right, 90 down, 180 left, 270 up."},
                 "distance": {"type": "number", "minimum": 0, "maximum": int(MOVE_SHARE * s),
                              "description": f"Pixels to move. Default {int(MOVE_SHARE * s)}."}},
                 "required": ["angle"], "additionalProperties": False}},
            {"name": "write_message", "description": "Set ASCII text in your viewport, wrapped at the viewport's "
             "right edge. It floats above the paint: every agent whose view takes in that spot sees it, paint never "
             "covers it, and it can't be erased.",
             "inputSchema": {"type": "object", "properties": {
                 "text": {"type": "string", "maxLength": MAX_MESSAGE, "description": "Printable ASCII; newlines break lines."},
                 "x": {"type": "number", "minimum": 0, "maximum": s, "description": "Left edge of the text."},
                 "y": {"type": "number", "minimum": 0, "maximum": s, "description": "Top edge of the text."},
                 "color": {"type": "string", "description": "Hex colour of the letters. Default #1d2a3a."},
                 "scale": {"type": "integer", "minimum": 1, "maximum": MAX_SCALE,
                           "description": "Pixels per font pixel: a letter is 6 x 11 font pixels. Default 2."}},
                 "required": ["text", "x", "y"], "additionalProperties": False}},
        ]

    def structured_content(self, meta: dict) -> dict | None:
        # The Pi sidecar ends the session on this, so it doesn't spend a turn on a call that would be refused.
        return {"stop": True} if self.s.over else None

    def call(self, name: str, args: dict, meta: dict) -> list[dict]:
        return self.s.call(name, args, meta.get("claudecode/toolUseId"))


def main(argv: list[str]) -> int:
    CommonsServer(AgentSession(Path(argv[0]))).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
