"""
One agent on the shared canvas, as an MCP server in its own process.

    python -m conveyor.commons.server <session_dir>

The session directory holds `job.json`: the database, the canvas and agent ids, the instrument's source, the
viewport's starting corner, and whether batches and successors are on. The server gives the agent the
instrument's tools on its viewport, plus `look`, `overview`, `move_viewport`, `write_message`, `broadcast`,
`paint_batch` and `spawn_successor`, and counts every call it receives against the canvas's tool-call budget, refused calls
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
from conveyor.commons.lettering import glyphs
from conveyor.commons.lettering import halo_color
from conveyor.commons.lettering import wrap
from conveyor.commons.overview import OVERVIEW_SIDE
from conveyor.commons.overview import REGION
from conveyor.commons.overview import overview_png
from conveyor.commons.overview import region
from conveyor.commons.sketch import bounds
from conveyor.commons.sketch import lines as sketch_lines
from conveyor.commons.tiles import DEFAULT_INK
from conveyor.commons.tiles import MOVE_SHARE
from conveyor.commons.tiles import SharedCanvas
from conveyor.commons.tiles import clamp_to_frame
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

MAX_BROADCAST = 280  # characters in one broadcast
BROADCASTS_SHOWN = 5  # the most broadcasts one result carries; the newest are kept
SUCCESSOR_WINDOW = 10  # spawn_successor works only in an agent's last this-many calls, so a session stays put
OUTSIDE = (38, 38, 46)  # how the overview shows what lies outside a canvas's frame
JUDGE_TOOLS = ("look", "overview", "move_viewport", "write_message")  # a judge looks, moves and writes; it never paints
HARNESS_TOOLS = ("look", "overview", "move_viewport", "write_message", "broadcast", "paint_batch", "spawn_successor")


class AgentSession:
    def __init__(self, session_dir: Path) -> None:
        self.dir = Path(session_dir)
        job = json.loads((self.dir / "job.json").read_text())
        self.canvas = SharedCanvas(job["db"], job["canvas_id"])
        self.agent_id = job["agent_id"]
        self.size = int(job.get("viewport") or self.canvas.config["viewport"])  # the agent's own, or the canvas's
        self.max_calls = int(job.get("max_calls") or self.canvas.config["max_calls"])
        self.frame = self.canvas.config.get("frame")  # (x0, y0, x1, y1), or None for a canvas with no edges
        self.x, self.y = clamp_to_frame(self.frame, int(job["x"]), int(job["y"]), self.size)
        self.judge = job.get("kind") == "judge"
        self.inst = None if self.judge else Instrument(job["source"])
        taken = {t.name for t in [*self.inst.spec.tools, *self.inst.spec.views]} & set(HARNESS_TOOLS) if self.inst else set()
        if taken:
            raise ValueError(f"the instrument's tool names clash with the canvas's: {sorted(taken)}")
        self.batch_enabled = bool(job.get("paint_batch", True)) and not self.judge  # a judge has nothing to batch
        self.successors = bool(job.get("successors", False)) and not self.judge
        self.handed_off = False
        self.pen = self.inst.new_state() if self.inst else None
        self.rng = np.random.default_rng(job.get("seed", 0))
        self.calls_used = 0
        self.index = 0
        self.heard = 0  # the newest broadcast seq this agent has been given
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
        """The viewport as the agent sees it: the paint, with the sketch on top. Paint calls work on the paint
        alone, so an instrument that reads the canvas never picks up sketch lines."""
        x, y, s = self.x, self.y, self.size
        return self.canvas.overlay(self.canvas.read(x, y, s, s).image(), x, y, x + s, y + s)

    def _sketch_here(self) -> str:
        x, y, s = self.x, self.y, self.size
        return (" Lines of the sketch by the person running this canvas cross it."
                if self.canvas.sketch(x, y, x + s, y + s) else "")

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
            raise ToolFailure("\n\n".join(p for p in (str(e), self._sketch_news(), self._news(), self.status())
                                            if p)) from e
        blocks[0] = text("\n\n".join(p for p in (blocks[0]["text"], self._sketch_news(), self._news(), self.status())
                                       if p).strip())
        return blocks

    def _news(self) -> str:
        """Broadcasts from other agents that this one hasn't been given yet, the newest few, with where each
        sender's viewport was. An agent's first call brings the newest ones sent before it started."""
        rows = self.canvas.conn.execute(
            "SELECT seq, x, y, note, args FROM canvas_ops WHERE canvas_id=? AND tool='broadcast' AND status='sent' "
            "AND seq>? AND agent_id IS NOT ? ORDER BY seq", (self.canvas.canvas_id, self.heard, self.agent_id)).fetchall()
        if not rows:
            return ""
        first, self.heard = not self.heard and self.index <= 1, rows[-1]["seq"]
        shown = rows[-BROADCASTS_SHOWN:]
        head = "The newest broadcasts on this canvas" if first else "Broadcasts since your last call"
        more = f" ({len(rows) - len(shown)} older ones not shown)" if len(rows) > len(shown) else ""
        lines = []
        for r in shown:  # placed at the middle of the sender's viewport, whatever its size
            half = int(json.loads(r["args"] or "{}").get("viewport") or self.canvas.config["viewport"]) // 2
            lines.append(f'- from around canvas ({r["x"] + half}, {r["y"] + half}): "{r["note"]}"')
        return f"{head}, oldest first{more}:\n" + "\n".join(lines)

    def _sketch_news(self) -> str:
        """On an agent's first call, where the sketch is, so it can go and see it. After that nothing: new lines
        show up in its pictures, and a note under each result whenever lines were drawn would be noise."""
        if self.index > 1 or not (live := sketch_lines(self.canvas.conn, self.canvas.canvas_id)):
            return ""
        box = [bounds(line) for line in live]
        return (f"The person running this canvas has sketched {len(live)} line{'s' if len(live) != 1 else ''} on it, "
                f"over canvas x {round(min(b[0] for b in box))} to {round(max(b[2] for b in box))}, "
                f"y {round(min(b[1] for b in box))} to {round(max(b[3] for b in box))}.")

    def _dispatch(self, name: str, args: dict, tool_use_id: str | None) -> list[dict]:
        if name == "look":
            return self.look(tool_use_id)
        if name == "overview":
            return self.overview(tool_use_id)
        if name == "move_viewport":
            return self.move(args, tool_use_id)
        if name == "write_message":
            return [text(self.write_message(args, tool_use_id))]
        if name == "broadcast" and not self.judge:
            return [text(self.broadcast(args, tool_use_id))]
        if name == "spawn_successor" and self.successors:
            return [text(self.spawn_successor(args, tool_use_id))]
        if name == "paint_batch" and self.batch_enabled:
            return [text(self.batch(args, tool_use_id))]
        if self.inst and name in {v.name for v in self.inst.spec.views}:
            return self.view(name, args, tool_use_id)
        if self.inst and name in {t.name for t in self.inst.spec.tools}:
            return [text(self.paint(name, args, tool_use_id))]
        self._op(name, "rejected", args, "no such tool", tool_use_id=tool_use_id)
        raise ToolFailure(f"No tool named {name}.")

    def look(self, tool_use_id: str | None = None) -> list[dict]:
        img = self._seen()
        self._op("look", "view", {}, tool_use_id=tool_use_id)
        return [text(f"Your viewport as it is now, labelled in viewport pixels.{self._sketch_here()}"),
                image(gridded_png(img))]

    def overview(self, tool_use_id: str | None = None) -> list[dict]:
        viewport = (self.x, self.y, self.x + self.size, self.y + self.size)
        window, scale = region(self.x, self.y, self.size)
        x0, y0, x1, y1 = window
        img = self.canvas.shrunk(window, scale).astype(np.float32) / 255.0
        img = to_uint8(self.canvas.overlay(img, x0, y0, x1, y1, scale))
        if self.frame:  # nothing exists outside the frame: shade it so the edge is plain
            fx0, fy0, fx1, fy1 = ((c - o) * scale for c, o in zip(self.frame, (x0, y0, x0, y0)))
            rows, cols = np.arange(img.shape[0])[:, None] + 0.5, np.arange(img.shape[1])[None, :] + 0.5
            outside = (cols < fx0) | (cols > fx1) | (rows < fy0) | (rows > fy1)
            img[outside] = OUTSIDE
        self._op("overview", "view", {}, f"x {x0} to {x1}, y {y0} to {y1}", tool_use_id=tool_use_id)
        extent = self.canvas.extent()
        reach = (f"Paint on the whole canvas reaches from x {extent[0]} to {extent[2]} and y {extent[1]} to {extent[3]}."
                 if extent else "Nothing is painted anywhere yet.")
        if self.frame:
            reach += (f" The frame runs x {self.frame[0]} to {self.frame[2]} and y {self.frame[1]} to {self.frame[3]}; "
                      "the dark area is outside it.")
        moves = (REGION - 1) / 2 / MOVE_SHARE
        return [text(f"The canvas around you, x {x0} to {x1} and y {y0} to {y1}: {REGION} viewports across, with "
                     f"yours, the magenta box, in the middle, shrunk so one picture pixel is {1 / scale:g} canvas "
                     f"pixels. It reaches {moves:g} moves out in every direction. The labels are canvas coordinates. "
                     f"{reach}"),
                image(overview_png(img, window, scale, viewport))]

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
        x, y = clamp_to_frame(self.frame, self.x + dx, self.y + dy, self.size)
        stopped = (x, y) != (self.x + dx, self.y + dy)
        dx, dy = x - self.x, y - self.y
        self.x, self.y = x, y
        img = self._seen()
        self._op("move_viewport", "moved", {"angle": angle, "distance": round(distance, 1)}, f"moved by ({dx}, {dy})"
                 + (", stopped by the frame" if stopped else ""), snapshot=img, tool_use_id=tool_use_id)
        edge = " The frame's edge stopped you there." if stopped else ""
        return [text(f"Asked to move {distance:.0f} px toward {angle:g} degrees; moved by {dx} in x and {dy} in y."
                     f"{edge}{self._sketch_here()} Your new viewport, labelled in viewport pixels:"),
                image(gridded_png(img))]

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
        # Agents handed off after a handful of calls, each session painting one viewport and moving on, so the
        # canvas filled with small repeats. Holding the hand-off to the end keeps a session working one region.
        if self.calls_left >= SUCCESSOR_WINDOW:
            self._op("spawn_successor", "rejected", {}, "before the last calls", tool_use_id=tool_use_id)
            raise ToolFailure(f"spawn_successor only works in your last {SUCCESSOR_WINDOW} tool calls, and you have "
                              f"{self.calls_left} left. Keep working until then. Your session goes on.")
        # No note: a successor inherits only the canvas, so what it should know has to be there for anyone to see.
        (self.dir / "successor.json").write_text(json.dumps({"x": self.x, "y": self.y, "at": time.time()}))
        self.handed_off = True
        self._op("spawn_successor", "handed_off", {}, tool_use_id=tool_use_id)
        return (f"Your successor starts here, with its viewport at canvas ({self.x}, {self.y}) and {self.max_calls} "
                "fresh tool calls, once this session closes. It gets the canvas and nothing of this conversation.")

    def broadcast(self, args: dict, tool_use_id: str | None = None) -> str:
        message = args.get("text")
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_BROADCAST:
            self._op("broadcast", "rejected", args, "bad text", tool_use_id=tool_use_id)
            raise ToolFailure(f"text must be 1 to {MAX_BROADCAST} characters. Nothing was sent.")
        message = message.strip()
        self._op("broadcast", "sent", {"text": message, "viewport": self.size}, message, tool_use_id=tool_use_id)
        return ("Sent to every other agent on the canvas, with where your viewport is now. Each gets it with its next "
                "tool result, and agents that start later get it among the newest broadcasts.")

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
        # The glyphs carry an outline one font pixel wide all round: place them so the letters start at (x, y),
        # and cut them at the viewport's edges, like any paint.
        ink_mask, halo_mask = glyphs(message, self.size - x, scale)
        left, top = x - scale, y - scale
        ys = slice(max(0, top), min(self.size, top + ink_mask.shape[0]))
        xs = slice(max(0, left), min(self.size, left + ink_mask.shape[1]))
        cut = (slice(ys.start - top, ys.stop - top), slice(xs.start - left, xs.stop - left))
        ink_mask, halo_mask = ink_mask[cut], halo_mask[cut]
        # Letters are held to the area limit of the canvas's own viewport, not the agent's: an agent with a small
        # viewport paints finer, but should still be able to say something. Wrapping keeps the text inside its view.
        limit = max(self.area_cap, Canvas(*(2 * [int(self.canvas.config["viewport"])])).area_cap)
        covered = int((ink_mask | halo_mask).sum())
        if covered > limit:
            self._op("write_message", "rejected", args, "over the area limit", tool_use_id=tool_use_id)
            raise ToolFailure(f"That message's letters and outline would cover {covered} pixels, past the area limit "
                              f"of {limit} a message may cover. Write less or use a smaller scale. Nothing was written.")
        ink = parse_color(color)
        with self.canvas.transaction():
            region = self.canvas.read(self.x, self.y, self.size, self.size)
            img = region.image()
            window = img[ys, xs]
            window[halo_mask] = halo_color(ink)
            window[ink_mask] = ink
            tiles = region.changed(img)
            self._op("write_message", "painted", {"text": message, "x": x, "y": y, "scale": scale, "color": color,
                                                  "viewport": self.size},
                     message, tiles=tiles, snapshot=img, tool_use_id=tool_use_id)
        lines = len(wrap(message, self.size - x, scale))
        return (f"Painted {len(message)} characters in {lines} line{'s' if lines != 1 else ''} from ({x}, {y}), "
                "outlined. It's paint now, and anyone can paint over it.")


class CommonsServer(StdioServer):
    name = "commons"

    def __init__(self, session: AgentSession) -> None:
        self.s = session

    def tools(self) -> list[dict]:
        s = self.s.size
        tools = self.s.inst.mcp_tools(s, s) + self.s.inst.mcp_views(s, s) if self.s.inst else []
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
            tools.append({"name": "spawn_successor", "description": "End your session now and start a new painter "
                          f"where your viewport is, with {self.s.max_calls} fresh tool calls and a randomly drawn "
                          "instrument and instructions. It sees only the canvas, nothing of this conversation. Use it "
                          f"to keep the work going past your budget. It only works in your last {SUCCESSOR_WINDOW} "
                          "tool calls.",
                          "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}})
        return tools + [t for t in [
            {"name": "look", "description": "See your viewport as it is now, gridded in viewport pixels. Others "
             "may have painted in it since you last looked.", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "overview", "description": f"See the canvas around you, {REGION} viewports across with yours "
             f"outlined in magenta in the middle, shrunk to {OVERVIEW_SIDE} px and labelled in canvas coordinates. "
             "It also says how far the paint on the whole canvas reaches.",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "move_viewport", "description": f"Slide your viewport `distance` px toward `angle` degrees "
             f"(0 right, 90 down, 180 left, 270 up), at most {int(MOVE_SHARE * s)} px, so the new view overlaps the "
             "old. Shows you the new view.",
             "inputSchema": {"type": "object", "properties": {
                 "angle": {"type": "number", "description": "Direction in degrees: 0 right, 90 down, 180 left, 270 up."},
                 "distance": {"type": "number", "minimum": 0, "maximum": int(MOVE_SHARE * s),
                              "description": f"Pixels to move. Default {int(MOVE_SHARE * s)}."}},
                 "required": ["angle"], "additionalProperties": False}},
            {"name": "broadcast", "description": "Send a short text to every other agent on the canvas, wherever "
             "they are. Each gets it with its next tool result, with where your viewport is but not who sent it; "
             "agents that start later get the newest few. It isn't on the canvas.",
             "inputSchema": {"type": "object", "properties": {
                 "text": {"type": "string", "maxLength": MAX_BROADCAST, "description": "What to say."}},
                 "required": ["text"], "additionalProperties": False}},
            {"name": "write_message", "description": "Paint ASCII text into your viewport, wrapped at the "
             "viewport's right edge and outlined in a contrasting colour so it reads on any paint. It is paint like "
             "any other: anyone can paint over it.",
             "inputSchema": {"type": "object", "properties": {
                 "text": {"type": "string", "maxLength": MAX_MESSAGE, "description": "Printable ASCII; newlines break lines."},
                 "x": {"type": "number", "minimum": 0, "maximum": s, "description": "Left edge of the text."},
                 "y": {"type": "number", "minimum": 0, "maximum": s, "description": "Top edge of the text."},
                 "color": {"type": "string", "description": "Hex colour of the letters. Default #1d2a3a."},
                 "scale": {"type": "integer", "minimum": 1, "maximum": MAX_SCALE,
                           "description": "Pixels per font pixel: a letter is 6 x 11 font pixels. Default 2."}},
                 "required": ["text", "x", "y"], "additionalProperties": False}},
        ] if not self.s.judge or t["name"] in JUDGE_TOOLS]

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
