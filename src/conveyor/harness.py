"""
Agent harnesses: what runs a model for conveyor, and how everything it does gets recorded.

Every model interaction (a painting, a mutation, a verdict) is one `Job` handed to a harness, which runs one
agent process to the end and returns an `Outcome`. Two harnesses exist:

  claude  Claude Code headless (`claude -p`), see claude.py
  pi      Pi's agent loop in a Node sidecar, on OpenCode Go, OpenRouter, or OpenAI Codex, see pi.py

Both speak the same protocol to conveyor. A job's tools come from one stdio MCP server of ours (the paint
server, the workbench), which the harness's process launches and talks to itself. The process prints Claude
Code's stream-json events on stdout: `init`, `assistant` messages with thinking, text and tool_use blocks, `user`
messages with tool results, `rate_limit_event`, and a final `result` with cost and usage. The Pi sidecar emits
the same shapes on purpose, so one parser here records either into the store, and the dashboard can't tell
them apart except by the model's name.

`Meter` tracks spend across every harness in a run, against one budget, and each harness's rate-limit windows
separately: Claude Code reports its plan's usage windows, Pi reports none, and an exhausted Claude window must
not stop a role that runs on Pi. A run can wait out a full window instead of stopping: new sessions block until
it resets, and a session the limit ends partway raises `SessionCutOff`, so the conductor can run that job again
rather than score half a painting.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from conveyor.store import Store
from conveyor.store import new_id

DEFAULT_STALL_TIMEOUT = 600.0  # seconds of silence from a session before it is killed as stalled
HARD_CAP_MULTIPLE = 1.25  # past budget x this, no new session starts
RESET_GRACE = 60.0  # seconds a waiting run gives a window past its reported reset, in case the clocks disagree


class BudgetExhausted(RuntimeError):
    pass


class RateLimited(RuntimeError):
    pass


class SessionCutOff(RateLimited):
    """A usage limit ended a session partway. Its work is incomplete, so its job should run again, not count."""


class SessionFailed(RuntimeError):
    """A session ended early for a reason that says nothing about the work: a stalled or dropped stream, a crash,
    a timeout. Its job should run again, and if it keeps failing it counts as no result, not as a bad one."""


@dataclass
class Job:
    purpose: str
    system_prompt: str
    content: list[dict]  # the first user message: {"type": "text", "text"} or {"type": "png", "data": bytes}
    cwd: Path
    mcp: dict | None = None  # {"name": "canvas", "command": ..., "args": [...]}
    tools: list[str] = field(default_factory=list)  # the server's tool names the model may call
    stop_tools: list[str] = field(default_factory=list)  # a successful call to one of these ends the session (Pi)
    json_schema: dict | None = None
    max_budget_usd: float | None = None
    timeout: float | None = None  # seconds; the harness default when None
    task_budget: int | None = None  # tokens the model may spend, where the harness supports it
    node: str | None = None
    organism_id: str | None = None
    on_event: Callable[[dict], None] | None = None  # every parsed stream-json message, after it's recorded
    on_start: Callable[[str], None] | None = None  # the session id, before the process starts


@dataclass
class Outcome:
    session_id: str
    ok: bool
    error: str | None = None
    cost: float = 0.0
    usage: dict = field(default_factory=dict)
    num_turns: int = 0
    result: str = ""
    structured: Any = None
    seconds: float = 0.0
    tool_calls: int = 0

    @property
    def interrupted(self) -> bool:
        """Cut off by something outside the work (a stall, a dropped stream, a crash, a timeout), as opposed to
        finishing, being stopped on purpose, or hitting the spend cap."""
        return not self.ok and bool(self.error) and self.error != "stopped" \
            and not self.error.startswith("error_max_budget_usd")


class Meter:
    """Spend shared by every harness in a run, and each harness's rate-limit state."""

    def __init__(self, budget_usd: float | None = None, on_budget: Callable[[float], None] | None = None,
                 max_usage: float | None = None, wait_hours: float | None = None,
                 on_wait: Callable[[str, float], None] | None = None, on_resume: Callable[[], None] | None = None) -> None:
        """
        `max_usage`: stop starting sessions on a harness once any of its usage windows (Claude's five-hour and
        weekly windows) reaches this share. A run shares those windows with everything else you do in Claude,
        so it should leave some behind.

        `wait_hours`: when a window or a rate limit that resets within this many hours blocks new sessions, they
        wait for the reset instead of the run stopping. `on_wait(reason, until)` announces each wait and
        `on_resume()` its end.
        """
        self.budget_usd = budget_usd
        self.on_budget = on_budget
        self.max_usage = max_usage
        self.wait_hours = wait_hours
        self.on_wait = on_wait
        self.on_resume = on_resume
        self.spent = 0.0
        self.sessions = 0
        self.rate_limits: dict[str, dict] = {}
        self.limited_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self._announced = False
        self._grace = RESET_GRACE if wait_hours is not None else 0.0
        self._waiting_until: float | None = None  # the reset sessions are waiting for, while they wait

    @property
    def rate_limit(self) -> dict:
        """The most recently reported rate-limit state of any harness (for display)."""
        return next(reversed(self.rate_limits.values()), {}) if self.rate_limits else {}

    def check(self, harness: str | None = None, stop: threading.Event | None = None) -> None:
        """
        Raise if no new session may start: past the hard budget cap, or `harness` (any, if None) limited. A
        limit this meter may wait out blocks instead, until it resets or `stop` is set; the caller then checks
        `stop` itself.
        """
        stop = stop or threading.Event()
        while True:
            with self._lock:
                if self.budget_usd is not None and self.spent >= self.budget_usd * HARD_CAP_MULTIPLE:
                    raise BudgetExhausted(f"spent ${self.spent:.2f}, hard cap ${self.budget_usd * HARD_CAP_MULTIPLE:.2f}")
                block = self._blocked(harness)
                if block is None:
                    resumed, self._waiting_until = self._waiting_until is not None, None
                    break
                reason, until = block
                if not self._can_wait(until):
                    raise RateLimited(reason + self._resets(until))
                announce, self._waiting_until = self._waiting_until != until, until
            if announce and self.on_wait:
                self.on_wait(reason, until)
            # Short naps, so a stop or a newer report from a session still running is noticed.
            if stop.wait(max(1.0, min(60.0, until + self._grace - time.time()))):
                return
        if resumed and self.on_resume:
            self.on_resume()

    def limited(self, harness: str) -> bool:
        """Whether `harness` has hit a usage limit that hasn't reset: a rejection, or a window reported full."""
        with self._lock:
            now = time.time()
            if self.limited_until.get(harness, 0.0) > now:
                return True
            windows = (self.rate_limits.get(harness) or {}).get("unifiedWindows") or {}
            return any((w.get("utilization") or 0.0) >= 1.0 and float(w.get("resetsAt") or 0) > now
                       for w in windows.values())

    def _blocked(self, harness: str | None = None) -> tuple[str, float | None] | None:
        """Why no session may start on `harness` (any, if None), and when that ends, if known."""
        now = time.time()
        for name, until in self.limited_until.items():
            if (harness is None or name == harness) and now < until + self._grace:
                return f"{name} hit its usage limit", until
        return self._window_over(harness)

    def _window_over(self, harness: str | None = None) -> tuple[str, float | None] | None:
        if self.max_usage is None:
            return None
        for name, info in self.rate_limits.items():
            if harness is not None and name != harness:
                continue
            for window, w in (info.get("unifiedWindows") or {}).items():
                used = w.get("utilization") or 0.0
                resets = float(w["resetsAt"]) if w.get("resetsAt") else None
                if used >= self.max_usage and (resets is None or time.time() < resets + self._grace):
                    return (f"the {name} {window.replace('_', '-')} usage window is at {used:.0%}, past the "
                            f"{self.max_usage:.0%} this run may use", resets)
        return None

    def _can_wait(self, until: float | None) -> bool:
        return self.wait_hours is not None and until is not None and until - time.time() <= self.wait_hours * 3600

    def _resets(self, until: float | None) -> str:
        """When a limit that stops the run resets, and why the run isn't waiting for it, if it waits for any."""
        if until is None:
            return "" if self.wait_hours is None else "; it gave no reset time to wait for"
        if self.wait_hours is None:
            return f"; it resets {_clock(until)}"
        return f"; it resets {_clock(until)}, past the {self.wait_hours:g}-hour wait this run allows"

    def add(self, cost: float) -> None:
        crossed = False
        with self._lock:
            self.spent += cost or 0.0
            self.sessions += 1
            if self.budget_usd is not None and self.spent >= self.budget_usd and not self._announced:
                self._announced = crossed = True
        if crossed and self.on_budget:
            self.on_budget(self.spent)

    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.spent >= self.budget_usd

    def should_stop(self) -> str | None:
        """Why to stop after the current iteration, if so: over budget, or a usage window past `max_usage` that
        this meter won't wait out."""
        with self._lock:
            if self.over_budget():
                return f"spent ${self.spent:.2f} of the ${self.budget_usd:.2f} budget"
            window = self._window_over()
            if window and not self._can_wait(window[1]):
                return window[0] + self._resets(window[1])
            return None

    def note_rate_limit(self, info: dict, harness: str = "claude") -> None:
        with self._lock:
            self.rate_limits[harness] = info
            if info.get("status") == "rejected" and info.get("resetsAt"):
                self.limited_until[harness] = float(info["resetsAt"])


class ProcessHarness:
    """
    A harness that runs each job as one child process printing stream-json. Subclasses say how to start the
    process (`command`); launching, feeding it, reading its stream, recording, time limits and kill-on-stop live
    here.
    """

    name = "process"

    def __init__(self, *, model: str, effort: str | None, store: Store | None, meter: Meter | None, lanes: int,
                 timeout: float, stall_timeout: float | None = DEFAULT_STALL_TIMEOUT) -> None:
        """`stall_timeout`: kill a session that prints nothing for this many seconds (None: only `timeout` applies).
        A provider can accept a request and then say nothing until the connection drops, which otherwise holds a
        lane for the whole `timeout`. The clock is monotonic, so a laptop asleep doesn't count against it."""
        self.model = model
        self.stall_timeout = stall_timeout
        self.effort = effort
        self.store = store
        self.meter = meter or Meter()
        self.timeout = timeout
        self._lanes = threading.BoundedSemaphore(max(1, lanes))
        self._procs: set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()
        self._stopping = threading.Event()

    # ---- what a subclass provides ------------------------------------------------------------------------

    def command(self, job: Job) -> tuple[list[str], bytes, dict[str, str]]:
        """argv, what to write on stdin before closing it, and the environment."""
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}: {self.model}" + (f", effort {self.effort}" if self.effort else "")

    # ---- public -------------------------------------------------------------------------------------------

    def run(self, job: Job) -> Outcome:
        self.meter.check(self.name, self._stopping)
        with self._lanes:
            if not self._stopping.is_set():
                self.meter.check(self.name, self._stopping)
            if self._stopping.is_set():
                return Outcome(session_id="", ok=False, error="stopped")
            outcome = self._attempt(job)
        self.meter.add(outcome.cost)
        if not outcome.ok and not self._stopping.is_set() and self.meter.limited(self.name):
            raise SessionCutOff(f"the {self.name} usage limit ended a {job.purpose} session partway")
        return outcome

    def _attempt(self, job: Job) -> Outcome:
        """One run of the job. Subclasses override to retry on flags a CLI version refuses."""
        return self._run_once(job)

    def stop_all(self) -> None:
        """Kill every running session, and refuse new ones."""
        self._stopping.set()
        with self._procs_lock:
            procs = list(self._procs)
        for p in procs:
            _kill(p)

    # ---- one process --------------------------------------------------------------------------------------

    def _request_record(self, job: Job, argv: list[str]) -> dict:
        content = []
        for b in job.content:
            if b["type"] == "png":
                name = self.store.artifact(b["data"]) if self.store else None
                content.append({"type": "image", "artifact": name})
            else:
                content.append({"type": "text", "text": b["text"]})
        shown = [a if len(a) < 300 else a[:120] + f"... ({len(a)} chars)" for a in argv]
        return {"harness": self.name, "system": job.system_prompt, "content": content, "argv": shown,
                "tools": job.tools, "json_schema": job.json_schema}

    def _run_once(self, job: Job) -> Outcome:
        sid = new_id()
        argv, stdin_bytes, env = self.command(job)
        job.cwd.mkdir(parents=True, exist_ok=True)
        if self.store:
            self.store.start_session(sid, node=job.node, organism_id=job.organism_id, purpose=job.purpose,
                                     model=self.model, effort=self.effort, request=self._request_record(job, argv),
                                     dir=str(job.cwd))
        if job.on_start:
            job.on_start(sid)
        started = time.time()
        out = Outcome(session_id=sid, ok=False)
        try:
            proc = subprocess.Popen(argv, cwd=job.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True, env=env)
        except OSError as e:
            out.error = f"could not start {argv[0]}: {e}"
            self._close(out, started)
            return out
        with self._procs_lock:
            self._procs.add(proc)
        stderr: list[bytes] = []
        threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True).start()
        timed_out = threading.Event()
        proc_done = threading.Event()
        limit = job.timeout or self.timeout
        stalled = threading.Event()
        last_seen = [time.monotonic()]
        tick = min(5.0, max(0.05, (self.stall_timeout or 20.0) / 4))

        def watchdog() -> None:
            deadline = time.monotonic() + limit
            while not proc_done.wait(tick):
                now = time.monotonic()
                if now >= deadline:
                    timed_out.set()
                    _kill(proc)
                    return
                if self.stall_timeout and now - last_seen[0] >= self.stall_timeout:
                    stalled.set()
                    _kill(proc)
                    return

        threading.Thread(target=watchdog, daemon=True).start()
        try:
            proc.stdin.write(stdin_bytes)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        got_result = False
        try:
            for raw in proc.stdout:
                last_seen[0] = time.monotonic()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                got_result |= self._handle(sid, msg, out)
                if job.on_event:
                    try:
                        job.on_event(msg)
                    except Exception as e:  # noqa: BLE001 - a recorder bug must not end the painting
                        self._event(sid, "harness_error", {"error": f"{type(e).__name__}: {e}"})
        finally:
            proc.wait()
            proc_done.set()
            with self._procs_lock:
                self._procs.discard(proc)
        tail = b"".join(stderr).decode(errors="replace")[-1500:]
        if timed_out.is_set():
            out.ok, out.error = False, f"timed out after {limit:.0f}s"
        elif stalled.is_set():
            out.ok, out.error = False, f"stalled: nothing from the model for {self.stall_timeout:.0f}s"
        elif self._stopping.is_set() and not got_result:
            out.ok, out.error = False, "stopped"
        elif not got_result:
            out.ok, out.error = False, (tail.strip() or f"{argv[0]} exited with {proc.returncode} and no result")
        if tail.strip() and self.store:
            self._event(sid, "stderr", {"text": tail})
        self._close(out, started)
        return out

    def _close(self, out: Outcome, started: float) -> None:
        out.seconds = time.time() - started
        if self.store:
            self.store.update_session(out.session_id, status="ok" if out.ok else "error", ended=time.time(),
                                      cost=out.cost, usage=out.usage, num_turns=out.num_turns,
                                      result=out.result[:20000], error=out.error)

    def _event(self, sid: str, kind: str, data: dict) -> None:
        if self.store:
            self.store.session_event(sid, kind, data)

    def _handle(self, sid: str, msg: dict, out: Outcome) -> bool:
        """Record one stream-json message. True when it's the final result."""
        kind, sub = msg.get("type"), msg.get("subtype")
        if kind == "system" and sub == "init":
            self._event(sid, "init", {k: msg.get(k) for k in ("model", "tools", "mcp_servers", "permissionMode",
                                                               "apiKeySource", "claude_code_version", "harness",
                                                               "provider")})
            failed = [m["name"] for m in msg.get("mcp_servers") or [] if m.get("status") != "connected"]
            if failed:
                self._event(sid, "harness_error", {"error": f"MCP server not connected: {failed}"})
        elif kind == "assistant":
            m = msg.get("message") or {}
            for block in m.get("content") or []:
                t = block.get("type")
                if t == "thinking":
                    if block.get("thinking"):
                        self._event(sid, "thinking", {"text": block["thinking"], "message": m.get("id")})
                elif t == "text":
                    if block.get("text", "").strip():
                        self._event(sid, "text", {"text": block["text"], "message": m.get("id")})
                elif t == "tool_use":
                    out.tool_calls += 1
                    self._event(sid, "tool_use", {"id": block.get("id"), "name": _short(block.get("name", "")),
                                                  "input": block.get("input"), "message": m.get("id")})
            if msg.get("error"):
                self._event(sid, "api_error", {"error": msg.get("error")})
        elif kind == "user":
            content = (msg.get("message") or {}).get("content")
            for block in content if isinstance(content, list) else []:
                if block.get("type") != "tool_result":
                    continue
                texts, images = [], []
                inner = block.get("content")
                for part in inner if isinstance(inner, list) else [{"type": "text", "text": str(inner or "")}]:
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image":
                        data = ((part.get("source") or {}).get("data")) or ""
                        if data and self.store:
                            images.append(self.store.artifact(base64.b64decode(data)))
                self._event(sid, "tool_result", {"tool_use_id": block.get("tool_use_id"), "text": "\n".join(texts)[:4000],
                                                 "images": images, "is_error": bool(block.get("is_error"))})
        elif kind == "rate_limit_event":
            info = msg.get("rate_limit_info") or {}
            self.meter.note_rate_limit(info, self.name)
            self._event(sid, "rate_limit", info)
        elif kind == "result":
            out.cost = float(msg.get("total_cost_usd") or 0.0)
            out.usage = msg.get("usage") or {}
            out.num_turns = int(msg.get("num_turns") or 0)
            out.result = msg.get("result") or ""
            out.structured = msg.get("structured_output")
            out.ok = not msg.get("is_error") and msg.get("subtype") == "success"
            if not out.ok:
                errors = msg.get("errors") or []
                out.error = f"{msg.get('subtype')}: " + ("; ".join(map(str, errors)) or out.result or "failed")
            self._event(sid, "result", {k: msg.get(k) for k in ("subtype", "is_error", "num_turns", "total_cost_usd",
                                                                "duration_ms", "duration_api_ms", "terminal_reason",
                                                                "stop_reason", "errors", "api_error_status")})
            return True
        elif kind == "system" and sub in ("thinking_tokens",):
            pass  # an estimate that streams many times a second; the result has the real count
        else:
            extra = {k: v for k, v in msg.items() if k not in ("type", "subtype")} if kind == "system" else {}
            self._event(sid, "other", {"type": kind, "subtype": sub, **extra})
        return False


def content_blocks(job: Job) -> list[dict]:
    """The first user message in Claude's content-block form, images as base64."""
    out = []
    for b in job.content:
        if b["type"] == "png":
            out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": base64.b64encode(b["data"]).decode()}})
        else:
            out.append({"type": "text", "text": b["text"]})
    return out


def _clock(ts: float) -> str:
    """A local time: 23:50 within the next day, Tue 09:00 past it."""
    return time.strftime("%H:%M" if ts - time.time() < 20 * 3600 else "%a %H:%M", time.localtime(ts))


def _short(name: str) -> str:
    """mcp__canvas__stroke -> stroke"""
    return name.split("__", 2)[-1] if name.startswith("mcp__") else name


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    # When conveyor itself runs inside a Claude Code session, don't let the children think they're nested in it.
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(k, None)
    return env


def _kill(proc: subprocess.Popen) -> None:
    """Kill the process and everything it started (the MCP server), which share its process group."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
