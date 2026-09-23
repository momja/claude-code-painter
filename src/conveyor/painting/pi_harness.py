"""
Component B as one ongoing conversation, run by Pi.

`pi-painter/painter.mjs` runs Pi's Agent: it keeps the whole conversation, streams each turn, and executes
tool calls by asking this module over stdio. This module owns the canvas, applies strokes, answers `look`,
and records every turn as a model call in the same conversation, so the dashboard shows the thinking, the
tool calls, what each call did, and how many input tokens came from the provider's cache.

Pi sends a session id upstream, which keeps every turn of one painting on the same provider so its prompt
cache can hit. The history is never trimmed: a prefix cache only reuses an unchanged prefix, so dropping old
canvas images would throw the cache away from that point on.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

from conveyor.catalog import model_info
from conveyor.events import current_trace
from conveyor.llm import LLMClient
from conveyor.llm import LLMReply
from conveyor.llm import ToolCall
from conveyor.painting.canvas import Target
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import composite
from conveyor.painting.canvas import stroke_alpha
from conveyor.painting.canvas import to_png
from conveyor.painting.llm_agent import PRESSURE_RULE
from conveyor.painting.llm_agent import brush_description
from conveyor.painting.llm_agent import PromptStrategy
from conveyor.painting.llm_agent import error_grid_text
from conveyor.painting.llm_agent import grid_rule
from conveyor.painting.llm_agent import grid_step
from conveyor.painting.llm_agent import gridded_png
from conveyor.painting.llm_agent import parse_color
from conveyor.painting.llm_agent import stroke_pressure
from conveyor.painting.llm_agent import worst_regions_text
from conveyor.painting.toolkit import Toolkit

PI_DIR = Path(__file__).resolve().parents[3] / "pi-painter"
PI_SCRIPT = PI_DIR / "painter.mjs"

RULES = """The canvas is {w} by {h} pixels. x runs left to right from 0 to {w}, y runs top to bottom from 0 to {h}.
Each brush is a tool. One call lays one stroke starting at (x, y), heading in the direction of `angle`
(degrees, 0 points right, 90 points down), in `color` (hex like #8a6d4f). Watercolor is translucent, so
strokes layer over what's already there, and nothing can be erased.
{pressure_rule}
{grid_rule}
Each stroke's result tells you how it changed the total error. Negative is better.
`look` shows you the canvas as it is now, with the error for each region. You can look {looks} times, so
use it to check your work after a batch of strokes.
You have {strokes} strokes. Put many brush calls in each reply, up to {batch}, rather than a few at a time.
{finish_rule} Think briefly and act."""

# Every turn re-sends the whole conversation, so fewer, fuller turns cost less. The model may still batch less.
BATCH_HINT = 40
STROKES_PER_LOOK = 40
# The finish floor (`min_stroke_fraction`) is off by default. One live 500-stroke run quit at stroke 88 while its
# strokes still helped; with the floor on, the next spent its last ~150 required strokes on beige filler that
# washed the painting out, because none of its brushes could make detail. Worth turning on once fine brushes
# exist. After this many refusals, finish is accepted anyway, so a model that won't paint can't loop forever.
MAX_FINISH_REFUSALS = 3

STOP_REASONS = {"toolUse": "tool_calls", "stop": "stop", "length": "length", "error": "error", "aborted": "aborted"}


class PiUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=16)
def pi_model_def(provider: str, model_id: str) -> dict:
    """Pi's model definition from the shared catalog: modalities, context, and $/million prices. Pi knows both
    providers already; this is what it falls back to for a model the installed pi-ai predates."""
    info = model_info(provider, model_id)
    return {
        "id": model_id,
        "name": info.name,
        "reasoning": info.reasoning,
        "input": [x for x in info.inputs if x in ("text", "image")],
        "cost": {"input": info.cost.input, "output": info.cost.output,
                 "cacheRead": info.cost.cache_read, "cacheWrite": info.cost.cache_write},
        "contextWindow": info.context,
        "maxTokens": info.max_tokens,
    }


def _b64png(png: bytes) -> str:
    return base64.b64encode(png).decode()


def _hex(color: np.ndarray) -> str:
    r, g, b = (np.clip(color, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


class PiHarness:
    """Painter callable with the same signature as `llm_paint`, backed by a Pi sidecar per painting."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_looks: int | None = None,  # default: one per STROKES_PER_LOOK strokes, at least 6
        min_stroke_fraction: float = 0.0,  # share of the budget used before finish is accepted; off by default
        model: str | None = None,  # paint with this instead of the client's model
        max_tokens: int = 8192,
        heap_mb: int = 256,  # V8 heap per sidecar; they run several at a time
        cache_retention: str = "short",
        timeout: float = 1800.0,
        faux: list[dict] | None = None,
        node: str | None = None,
        script: Path = PI_SCRIPT,
    ) -> None:
        self.client = client
        # The painter can run a different model from the mutators. Pi dispatches on the model's own API, so a
        # responses-API model paints fine here even though `LLMClient` only speaks chat completions.
        self.model = model or client.model
        self.max_looks = max_looks
        self.min_stroke_fraction = min_stroke_fraction
        self.max_tokens = max_tokens
        self.heap_mb = heap_mb
        self.cache_retention = cache_retention
        self.timeout = timeout
        self.faux = faux
        self.node = node or shutil.which("node")
        self.script = script
        if self.node is None:
            raise PiUnavailable("node is not on PATH. Install Node 20+ to use the Pi harness.")
        if not (script.parent / "node_modules" / "@earendil-works" / "pi-agent-core").is_dir():
            raise PiUnavailable(f"Pi isn't installed. Run `npm install` in {script.parent}.")
        self.model_def = None if faux is not None else pi_model_def(client.provider.key, self.model)

    def __call__(self, strategy: PromptStrategy, toolkit: Toolkit, target: Target, seed: int, record: bool = False):
        return _PaintSession(self, strategy, toolkit, target, seed, record).run()


class _PaintSession:
    def __init__(self, harness: PiHarness, strategy, toolkit, target: Target, seed: int, record: bool) -> None:
        self.h = harness
        self.client = harness.client
        self.strategy = strategy
        self.target = target
        self.canvas = blank(target)
        self.rng = np.random.default_rng(seed)
        self.brushes = {b.name: b for b in toolkit.brushes}
        self.strokes_left = strategy.n_strokes
        # Looks are not capped by the provider's image limit: the sidecar trims what it sends instead, keeping
        # the target and the newest views. Old canvas views are stale anyway, and cutting looks cost more than
        # it bought. In one run 15 of 21 paintings sent 10 to 15 images fine while 7 were rejected at the 9th.
        self.max_looks = harness.max_looks or max(6, math.ceil(strategy.n_strokes / STROKES_PER_LOOK))
        self.looks_left = self.max_looks
        self.floor = math.ceil(strategy.n_strokes * harness.min_stroke_fraction)
        self.finish_refusals = 0
        self.used = 0
        self.trace = current_trace() if record else None
        self.record = record
        self.conversation = uuid.uuid4().hex[:16]
        self.call_id: str | None = None
        self.call_started = 0.0
        self.first_token: float | None = None
        self.n_tools = 0
        self.tool_slots: dict[str, tuple[str, int]] = {}  # tool call id -> (model call id, index in its reply)
        self.results: dict[str, list] = {}
        self.error: str | None = None

    # ---- lifecycle ------------------------------------------------------------------------------------

    def run(self) -> np.ndarray:
        h, t = self.h, self.target
        height, width = t.image.shape[:2]
        tools = [{"name": b.name, "kind": "brush", "description": self._brush_description(b)} for b in self.brushes.values()]
        tools.append({"name": "look", "kind": "look",
                      "description": "See the canvas as it is now, with the error for each region."})
        tools.append({"name": "finish", "kind": "finish", "description": "End the painting."})
        system = self.strategy.prompt.strip() + "\n\n" + RULES.format(
            w=width, h=height, looks=self.max_looks, strokes=self.strategy.n_strokes, batch=BATCH_HINT,
            pressure_rule=PRESSURE_RULE, grid_rule=grid_rule(grid_step(width)),
            finish_rule=(f"`finish` ends the painting, but it's only accepted once you've used at least {self.floor} "
                         "strokes." if self.floor else "Call `finish` when you're done."))
        prompt = [
            {"type": "text", "text": "Target:"},
            {"type": "image", "data": _b64png(gridded_png(t.image)), "mimeType": "image/png"},
            {"type": "text", "text": "Your canvas, still blank paper:"},
            {"type": "image", "data": _b64png(gridded_png(self.canvas)), "mimeType": "image/png"},
            {"type": "text", "text": self._status()},
        ]
        start = {
            "type": "start",
            "model": h.model_def,
            "faux": h.faux,
            "systemPrompt": system,
            "prompt": prompt,
            "tools": tools,
            "width": width,
            "height": height,
            "thinkingLevel": _thinking_level(self.client.reasoning_effort),
            "maxTokens": h.max_tokens,
            "cacheRetention": h.cache_retention,
            "provider": self.client.provider.key,
            "baseUrl": self.client.provider.base_url,
            "sessionId": f"conveyor-{self.conversation}",
            # This pi-ai sends no session header of its own, and OpenCode Go rejects requests without one.
            "sessionHeader": self.client.provider.session_header,
            "maxImages": self.client.provider.max_images,
            "maxTurns": self.strategy.n_strokes + self.max_looks + 4,
        }
        env = {**os.environ}
        if self.client.api_key:
            env[self.client.provider.env_var] = self.client.api_key
        # One sidecar per painting, several at once, and node's default heap limit is 2 GB each. A run on an
        # 8 GB machine was killed for memory with 5 alive; a painting's whole conversation is a few MB.
        proc = subprocess.Popen(
            [h.node, f"--max-old-space-size={h.heap_mb}", str(h.script)], cwd=h.script.parent, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stderr: list[bytes] = []
        threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True).start()
        deadline = time.time() + h.timeout
        aborted = False
        try:
            self._send(proc, start)
            for raw in proc.stdout:  # readline splits on b"\n" only
                msg = json.loads(raw)
                kind = msg["type"]
                if kind == "llm_request":
                    self._on_request(msg)
                elif kind == "llm_params":
                    self._on_params(msg)
                elif kind == "llm_delta":
                    self._on_delta(msg)
                elif kind == "llm_end":
                    self._on_end(msg)
                elif kind == "tool":
                    self._send(proc, self._on_tool(msg))
                elif kind == "tool_error":
                    self._on_tool_error(msg)
                elif kind == "note" and self.trace is not None:
                    self.trace.span("harness note", result={"text": msg.get("text")})
                elif kind == "fatal":
                    self.error = msg.get("error")
                elif kind == "done":
                    if msg.get("error") and not self.error:
                        self.error = msg["error"]
                    break
                over_budget = self.client.over_hard_cap()
                if not aborted and (over_budget or time.time() > deadline):
                    aborted = True
                    self.error = "budget exhausted" if over_budget else f"timed out after {h.timeout:.0f}s"
                    self._send(proc, {"type": "abort"})
        finally:
            if self.call_id is not None:  # the sidecar died mid-call
                self._close_call(self.error or "sidecar exited during the call")
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc.returncode not in (0, None) and not self.error:
            self.error = b"".join(stderr).decode(errors="replace")[-500:]
        if self.error and self.trace is not None:
            self.trace.span("harness error", result={"error": self.error[:500]})
        return self.canvas

    def _send(self, proc: subprocess.Popen, msg: dict) -> None:
        try:
            proc.stdin.write((json.dumps(msg) + "\n").encode())
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    # ---- model calls ----------------------------------------------------------------------------------

    def _on_request(self, msg: dict) -> None:
        self.call_id = uuid.uuid4().hex[:16]
        self.call_started = time.perf_counter()
        self.first_token = None
        messages = [self._display_message(m) for m in msg.get("messages") or []]
        if msg.get("systemPrompt"):
            messages.insert(0, {"role": "system", "content": msg["systemPrompt"]})
        request = {"conversation": self.conversation, "turn": msg["turn"], "n_messages": msg["nMessages"],
                   "messages": messages}
        if msg.get("tools"):
            request["tools"] = [{"type": "function", "function": t} for t in msg["tools"]]
            self.n_tools = len(msg["tools"])
        self._request = request
        self.client.external_start(self.call_id, purpose="paint", request=request, model=self.h.model)

    def _on_params(self, msg: dict) -> None:
        if self.call_id is None:
            return
        params = msg.get("params") or {}
        self._request.update({k: v for k, v in params.items() if v is not None})
        self.client.annotate(self.call_id, request=self._request)

    def _on_delta(self, msg: dict) -> None:
        if self.call_id is None:
            return
        if self.first_token is None:
            self.first_token = time.time()
        self.client.annotate(
            self.call_id, first_token=self.first_token, reasoning=msg.get("thinking", ""), content=msg.get("text", ""),
            tool_calls=msg.get("toolCalls") or [],
        )

    def _on_end(self, msg: dict) -> None:
        if self.call_id is None:
            return
        u = msg.get("usage") or {}
        cache_read, cache_write = int(u.get("cacheRead") or 0), int(u.get("cacheWrite") or 0)
        usage = {
            "prompt_tokens": int(u.get("input") or 0) + cache_read + cache_write,
            "completion_tokens": int(u.get("output") or 0),
            "cost": (u.get("cost") or {}).get("total"),
            "prompt_tokens_details": {"cached_tokens": cache_read, "cache_write_tokens": cache_write},
            "completion_tokens_details": {"reasoning_tokens": u.get("reasoning")},
        }
        calls = []
        for c in msg.get("toolCalls") or []:
            args = c.get("arguments")
            calls.append(ToolCall(id=c.get("id", ""), name=c.get("name", ""),
                                  arguments=args if isinstance(args, dict) else None, raw=json.dumps(args)))
        stop = msg.get("stopReason")
        error = (msg.get("errorMessage") or stop) if stop in ("error", "aborted") else None
        reply = LLMReply(
            text=msg.get("text", ""), reasoning=msg.get("thinking", ""), tool_calls=calls,
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"], cost=usage["cost"],
            latency=time.perf_counter() - self.call_started, finish_reason=STOP_REASONS.get(stop, stop),
            model=msg.get("model") or self.h.model, call_id=self.call_id,
            reasoning_tokens=u.get("reasoning"), usage=usage,
        )
        for i, c in enumerate(calls):
            self.tool_slots[c.id] = (self.call_id, i)
        self.results[self.call_id] = [None] * len(calls)
        self.client.external_finish(
            self.call_id, reply, error, self.first_token, purpose="paint", latency=reply.latency,
            n_tools=self.n_tools, record_span=self.record,
        )
        self.call_id = None

    def _close_call(self, error: str) -> None:
        self.client.external_finish(
            self.call_id, None, error, self.first_token, purpose="paint",
            latency=time.perf_counter() - self.call_started, n_tools=self.n_tools, record_span=self.record,
        )
        self.call_id = None

    def _record_outcome(self, tool_call_id: str, outcome: dict) -> None:
        slot = self.tool_slots.get(tool_call_id)
        if slot is None:
            return
        call_id, index = slot
        self.results[call_id][index] = outcome
        self.client.annotate(call_id, tool_results=self.results[call_id])

    # ---- tools ----------------------------------------------------------------------------------------

    def _on_tool(self, msg: dict) -> dict:
        name, args, tool_id = msg.get("name"), msg.get("args") or {}, msg["id"]
        if name == "look":
            reply, outcome = self._look()
        elif name == "finish":
            reply, outcome = self._finish(args)
        elif name in self.brushes:
            reply, outcome = self._stroke(self.brushes[name], args)
        else:
            reply, outcome = {"isError": True, "text": f"Unknown tool {name}."}, {"status": "rejected", "error": "unknown tool"}
        self._record_outcome(tool_id, outcome)
        return {"type": "tool_result", "id": tool_id, **reply}

    def _finish(self, args: dict) -> tuple[dict, dict]:
        note = str(args.get("note", ""))[:200]
        if self.used < self.floor and self.finish_refusals < MAX_FINISH_REFUSALS:
            self.finish_refusals += 1
            if self.trace is not None:
                self.trace.span("finish refused", args={"note": note}, result={"kept": False, "used": self.used})
            text = (f"Not yet. You've used {self.used} of {self.strategy.n_strokes} strokes, and finish is accepted "
                    f"from {self.floor}. Keep painting. {self._worst_regions()}")
            return ({"isError": True, "text": text},
                    {"status": "rejected", "error": f"refused, {self.used} of the {self.floor} required strokes used"})
        if self.trace is not None:
            self.trace.span("finish", args={"note": note}, result={"kept": True, "used": self.used})
        forced = self.used < self.floor
        return ({"text": "Finished.", "stop": True},
                {"status": "applied", "note": f"ended the painting at {self.used} strokes"
                 + (f", accepted after {self.finish_refusals} refusals" if forced else "")})

    def _worst_regions(self, k: int = 3) -> str:
        return worst_regions_text(self.canvas, self.target, k)

    def _on_tool_error(self, msg: dict) -> None:
        slot = self.tool_slots.get(msg.get("id"))
        if slot and self.results[slot[0]][slot[1]] is None:
            self._record_outcome(msg["id"], {"status": "rejected", "error": (msg.get("error") or "tool error")[:200]})

    def _stroke(self, brush, args: dict) -> tuple[dict, dict]:
        if self.strokes_left <= 0:
            return ({"text": "No strokes left, so this did nothing. Call finish.", "stop": True},
                    {"status": "ignored", "error": "no strokes left"})
        height, width = self.target.image.shape[:2]
        color = parse_color(args.get("color"))
        try:
            x = float(np.clip(float(args["x"]), 0, width))
            y = float(np.clip(float(args["y"]), 0, height))
            angle = math.radians(float(args["angle"]))
        except (KeyError, TypeError, ValueError):
            color = None
        pressure = stroke_pressure(args)
        if color is None or pressure is None:
            return ({"isError": True, "text": "Bad arguments. Send numbers for x, y, angle and pressure, and a hex color."},
                    {"status": "rejected", "error": "bad arguments"})
        hit = stroke_alpha(brush, x, y, angle, height, width, self.rng)
        delta = 0.0
        if hit is not None:
            ys, xs, a = hit
            a = a * pressure
            tgt = self.target.image[ys, xs]
            new = composite(self.canvas[ys, xs], a, color)
            delta = float(((new - tgt) ** 2).sum() - ((self.canvas[ys, xs] - tgt) ** 2).sum())
            self.canvas[ys, xs] = new
        self.strokes_left -= 1
        self.used += 1
        last = self.strokes_left == 0
        # Terse on purpose: at 500 strokes this text is repeated in every later turn's context.
        text = f"Error {delta:+.3f}. " + ("That was the last stroke." if last else f"{self.strokes_left} left.")
        if self.trace is not None:
            image = to_png(self.canvas) if (self.used % 6 == 0 or last) else None
            self.trace.span(
                brush.name,
                args={"x": round(x, 1), "y": round(y, 1), "angle": round(math.degrees(angle) % 360), "color": _hex(color),
                      "pressure": round(pressure, 2)},
                result={"delta_error": round(delta, 4), "kept": True},
                image=image,
            )
        return {"text": text, "stop": last}, {"status": "applied", "delta_error": round(delta, 4)}

    def _look(self) -> tuple[dict, dict]:
        if self.looks_left <= 0:
            return ({"isError": True, "text": "No looks left. Keep painting from the stroke results."},
                    {"status": "rejected", "error": "no looks left"})
        self.looks_left -= 1
        png = gridded_png(self.canvas)
        if self.trace is not None:
            self.trace.span("look", args={"looks_left": self.looks_left}, result={"kept": True}, image=to_png(self.canvas))
        return ({"text": self._status(), "images": [_b64png(png)]},
                {"status": "applied", "note": f"{self.looks_left} looks left"})

    def _status(self) -> str:
        return (f"{self.used} strokes used, {self.strokes_left} left, {self.looks_left} looks left.\n"
                + error_grid_text(self.canvas, self.target))

    @staticmethod
    def _brush_description(b) -> str:
        return brush_description(b)

    def _display_message(self, m: dict) -> dict:
        """A Pi message as the transcript shows it, with images stored as artifacts."""
        role = "tool" if m.get("role") == "toolResult" else m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = []
            for block in content or []:
                if block.get("type") == "image" and block.get("data"):
                    sink = self.client.sink
                    name = sink.store_artifact(base64.b64decode(block["data"])) if sink else None
                    parts.append({"type": "image", "artifact": name} if name else {"type": "text", "text": "[image]"})
                elif block.get("type") == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
        out = {"role": role, "content": parts}
        if role == "tool":
            out["name"] = m.get("toolName")
            if m.get("isError"):
                out["is_error"] = True
        return out


def _thinking_level(effort: str | None) -> str:
    if not effort or effort == "none":
        return "off"
    return effort if effort in ("minimal", "low", "medium", "high", "xhigh", "max") else "low"
