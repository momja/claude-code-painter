"""
Starts `conveyor run` from the dashboard.

A launch is a child process, `python -m conveyor run ... --no-serve --db <the served database>`. The run is the
same code the command line starts, so every option keeps one definition (the argparse parser) and one set of
defaults. The dashboard reads the run from the database like any other.

The form's values become argv through `ALLOWED`, an allowlist, then go through the real parser before anything
starts. Every value is passed as `--flag=value`, so no value can be read as another flag. The database, the
`claude` binary, the host and the port can't be set from here.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import io
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError

from conveyor.claude import DEFAULT_MODEL
from conveyor.store import new_id

LOG_LINES = 300
MAX_TARGET_BYTES = 6 * 1024 * 1024  # an uploaded target, before decoding
MAX_TARGET_SIDE = 2048  # an uploaded target is shrunk to fit this, however big it came
RUN_LINE = re.compile(r"^Run ([0-9a-f]{12}) writing to ")
# How long a stopping run gets before it's terminated, then killed. A stop is Ctrl+C: the run stops its
# sessions and writes its closing events, which takes a few seconds.
TERM_AFTER = 60.0
KILL_AFTER = 90.0

CLAUDE_MODELS = [DEFAULT_MODEL, "claude-sonnet-5-5", "claude-fable-5-1", "claude-haiku-4-5-20251001"]

# What the form may set, as `run` option -> kind. Anything else is refused.
#   text, int, float  the value as given      flag  --x when true      toggle  --x or --no-x
#   hours             --wait, with or without a number of hours        list  comma-joined
ALLOWED = {
    "name": "text", "offline": "flag", "cycles": "int", "budget": "float", "wait": "hours",
    "harness": "text", "model": "text", "effort": "text", "provider": "text",
    "paint_harness": "text", "paint_model": "text", "paint_effort": "text",
    "mutate_harness": "text", "mutate_model": "text", "mutate_effort": "text",
    "judge_harness": "text", "judge_model": "text", "judge_effort": "text",
    "target": "text", "width": "int", "actions": "int", "looks": "int",
    "judge": "toggle", "judge_weight": "float", "scope": "toggle", "paint_batch": "toggle",
    "seeds": "list", "parents": "int", "confirm": "int",
    "refine": "float", "invent": "float", "recombine": "float",
    "lanes": "int", "paint_cap": "float", "mutate_cap": "float", "max_usage": "float",
    "compact_every_looks": "int", "autocompact": "int", "paint_context_turns": "int",
}
# Sanity bounds, to catch a typo before it spends money. (low, high), inclusive.
BOUNDS = {
    "cycles": (1, 1000), "budget": (0.01, 10_000), "wait": (0.1, 168), "width": (64, 2048), "actions": (1, 5000),
    "looks": (-1, 500), "judge_weight": (0, 1), "parents": (1, 10), "confirm": (1, 10),
    "refine": (0, 100), "invent": (0, 100), "recombine": (0, 100), "lanes": (1, 8),
    "paint_cap": (0.01, 1000), "mutate_cap": (0.01, 1000), "max_usage": (0.05, 1),
    "compact_every_looks": (1, 500), "autocompact": (100_000, 1_000_000), "paint_context_turns": (0, 20),
}


class LaunchError(ValueError):
    """The form's values can't start a run; the message says why, in words for the person at the form."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise LaunchError(message)


class Launch:
    def __init__(self, argv: list[str], name: str | None, proc: subprocess.Popen) -> None:
        self.id = new_id()
        self.argv = argv
        self.name = name
        self.proc = proc
        self.started = time.time()
        self.ended: float | None = None
        self.run_id: str | None = None
        self.stop_requested = False
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        code = self.proc.poll()
        if code is None:
            return "stopping" if self.stop_requested else "running"
        if self.stop_requested:
            return "stopped"
        return "finished" if code == 0 else "failed"

    def view(self, log: bool = True) -> dict:
        with self._lock:
            lines = list(self.log)
        out = {"id": self.id, "state": self.state, "run_id": self.run_id, "name": self.name, "pid": self.proc.pid,
               "started": self.started, "ended": self.ended, "exit_code": self.proc.poll(),
               "command": "conveyor " + " ".join(self.argv)}
        if log:
            out["log"] = lines
        return out

    def _read(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\n")
            with self._lock:
                self.log.append(line)
                if self.run_id is None and (m := RUN_LINE.match(line)):
                    self.run_id = m.group(1)
        self.proc.wait()
        self.ended = time.time()


class Launcher:
    def __init__(self, db: Path, build_parser: Callable[..., argparse.ArgumentParser], python: str | None = None,
                 targets: list[str] | None = None, token: str | None = None) -> None:
        self.db = Path(db).resolve()
        self.token = token or None  # when set, every launch endpoint wants it in X-Conveyor-Token
        self.python = python or sys.executable
        self._parser = build_parser(_Parser)
        self._targets = targets
        self.targets_dir = self.db.parent / f"{self.db.stem}-targets"  # imported images, beside the database
        self._launches: dict[str, Launch] = {}
        self._agents: dict[str, tuple[str, Launch]] = {}  # canvas agent id -> (its canvas, its process)
        self._watching = threading.Event()  # set while the canvas watcher runs; cleared to stop it
        self._lock = threading.Lock()
        self._probe: tuple[float, dict] | None = None

    def authorized(self, presented: str | None) -> bool:
        return not self.token or hmac.compare_digest((presented or "").encode(), self.token.encode())

    # ---- what the form offers -----------------------------------------------------------------------------

    @property
    def _run(self) -> argparse.ArgumentParser:
        return self._parser.commands["run"]  # type: ignore[attr-defined]

    def targets(self) -> list[str]:
        """Names a run may paint: the bundled images, then any imported ones."""
        if self._targets is not None:
            base = list(self._targets)
        else:
            from conveyor.painting.canvas import TARGETS_DIR

            base = sorted(p.stem for p in Path(TARGETS_DIR).glob("*.jpg"))
        return base + [n for n in sorted(self._imported()) if n not in base]

    def _imported(self) -> dict[str, Path]:
        return {p.stem: p for p in self.targets_dir.glob("*.png")} if self.targets_dir.is_dir() else {}

    def import_target(self, body: dict) -> dict:
        """
        Keep an uploaded image as a target. It's decoded safely (size limits, EXIF rotation, transparency flattened
        onto white), shrunk to MAX_TARGET_SIDE and stored as PNG named after the file plus a content hash, so the
        same image is stored once and a name can't collide with a bundled one. The run scales it to its canvas.
        """
        from conveyor.painting.canvas import open_image

        image = body.get("image") if isinstance(body, dict) else None
        if not isinstance(image, str) or len(image) > MAX_TARGET_BYTES * 4 // 3 + 256:
            raise LaunchError("Image must be at most 6 MB.")
        try:
            data = base64.b64decode(image.split(",", 1)[1] if image.startswith("data:image/") else image, validate=True)
            if len(data) > MAX_TARGET_BYTES:
                raise LaunchError("Image must be at most 6 MB.")
            im = open_image(io.BytesIO(data))
        except LaunchError:
            raise
        except ValueError as error:  # too many pixels, or not base64
            raise LaunchError(str(error) if "pixels" in str(error) else "Upload a valid image.") from None
        except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise LaunchError("Upload a valid PNG, JPEG, WebP, GIF or BMP image.") from None
        if min(im.size) < 16:
            raise LaunchError("Image must be at least 16 pixels on each side.")
        im.thumbnail((MAX_TARGET_SIDE, MAX_TARGET_SIDE), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="PNG")
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(str(body.get("name") or "image")).stem).strip("-")[:40] or "image"
        name = f"{stem}-{hashlib.sha256(out.getvalue()).hexdigest()[:8]}"
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        path = self.targets_dir / f"{name}.png"
        if not path.exists():
            path.write_bytes(out.getvalue())
        return {"target": name, "width": im.width, "height": im.height, "targets": self.targets()}

    def options(self) -> dict:
        from conveyor.painting.seeds import SEEDS
        from conveyor.pi import PROVIDERS

        defaults, help_, choices = {}, {}, {}
        for action in self._run._actions:
            key = action.dest
            if key in ALLOWED:
                defaults[key] = action.default
                help_[key] = action.help
                if action.choices:
                    choices[key] = list(action.choices)
        harnesses = self.harnesses()
        if not harnesses["claude"]["ok"] and harnesses["pi"]["ok"]:
            defaults["harness"] = "pi"  # a host without Claude Code (a container) starts on the one it has
        return {
            "launch": True, "db": str(self.db), "defaults": defaults, "help": help_, "choices": choices,
            "seeds": list(SEEDS), "targets": self.targets(),
            "models": {"claude": CLAUDE_MODELS, "pi": {k: [p.default_model] for k, p in PROVIDERS.items()}},
            "harnesses": harnesses,
        }

    def harnesses(self) -> dict:
        """Whether each harness can run here. Cached briefly: the check starts `claude --version`."""
        if self._probe and time.time() - self._probe[0] < 30:
            return self._probe[1]
        from conveyor.claude import available as claude_available
        from conveyor.claude import logged_in
        from conveyor.pi import PROVIDERS
        from conveyor.pi import available as pi_available
        from conveyor.pi import provider_authenticated

        version = claude_available("claude")
        problem = pi_available()
        keys = {k: provider_authenticated(p) for k, p in PROVIDERS.items()}
        if version is None:
            claude = {"ok": False, "detail": "`claude --version` fails; install Claude Code"}
        elif logged_in("claude") is False:
            claude = {"ok": False, "detail": f"{version}, but not logged in; run `claude auth login`"}
        else:
            claude = {"ok": True, "detail": version}
        out = {"claude": claude,
               "pi": {"ok": problem is None, "detail": problem or "ready", "keys": keys,
                      "env_vars": {k: p.env_var for k, p in PROVIDERS.items() if p.env_var},
                      "login_commands": {k: p.login_command for k, p in PROVIDERS.items() if p.login_command}}}
        self._probe = (time.time(), out)
        return out

    # ---- validation ---------------------------------------------------------------------------------------

    def argv(self, values: dict) -> list[str]:
        """The `run` argv for the form's values, checked against the allowlist, the bounds and the parser."""
        from conveyor.painting.seeds import SEEDS

        if not isinstance(values, dict):
            raise LaunchError("Expected an object of options.")
        argv: list[str] = ["run"]
        for key, value in values.items():
            kind = ALLOWED.get(key)
            if kind is None:
                raise LaunchError(f"Unknown option: {key}")
            if value is None or value == "" or value == []:
                continue
            flag = "--" + key.replace("_", "-")
            if kind == "flag":
                if value is True:
                    argv.append(flag)
            elif kind == "toggle":
                if not isinstance(value, bool):
                    raise LaunchError(f"{key} must be on or off.")
                argv.append(flag if value else "--no-" + key.replace("_", "-"))
            elif kind == "hours":
                if value is True:
                    argv.append(flag)
                elif value is not False:
                    argv.append(f"{flag}={self._number(key, value, float)}")
            elif kind == "list":
                items = value if isinstance(value, list) else [value]
                bad = [s for s in items if s not in SEEDS]
                if bad:
                    raise LaunchError(f"Unknown seed instrument: {', '.join(map(str, bad))}")
                argv.append(f"{flag}={','.join(items)}")
            elif kind in ("int", "float"):
                argv.append(f"{flag}={self._number(key, value, int if kind == 'int' else float)}")
            else:
                if not isinstance(value, str) or len(value) > 200 or "\n" in value or value.lstrip().startswith("-"):
                    raise LaunchError(f"{key} must be a short line of text that doesn't start with a dash.")
                argv.append(f"{flag}={value.strip()}")
        target = values.get("target")
        if target and target not in self.targets():
            raise LaunchError(f"Unknown target: {target}. Choose one of {', '.join(self.targets())}.")
        weights = [values.get(k) for k in ("refine", "invent", "recombine")]
        if any(w is not None for w in weights) and not sum(float(w or 0) for w in weights) > 0:
            raise LaunchError("The operator weights can't all be zero.")
        if target in (imported := self._imported()):  # a run finds an imported image by path, not by name
            argv = [f"--target={imported[target]}" if a.startswith("--target=") else a for a in argv]
        argv += ["--no-serve", f"--db={self.db}"]
        self._parser.parse_args(argv)  # raises LaunchError with the parser's own message
        return argv

    @staticmethod
    def _number(key: str, value, kind: type):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise LaunchError(f"{key} must be a number.")
        try:
            n = kind(value)
        except ValueError:
            raise LaunchError(f"{key} must be {'a whole number' if kind is int else 'a number'}.") from None
        low, high = BOUNDS.get(key, (None, None))
        if low is not None and not low <= n <= high:
            raise LaunchError(f"{key} must be between {low} and {high}.")
        return n

    # ---- processes ----------------------------------------------------------------------------------------

    def start(self, values: dict) -> dict:
        argv = self.argv(values)
        # Start the child where this server started, so a relative --db or a .env file resolves the same way.
        proc = subprocess.Popen([self.python, "-m", "conveyor", *argv], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                start_new_session=True)  # Ctrl+C in this terminal reaches only the server
        launch = Launch(argv, values.get("name"), proc)
        with self._lock:
            self._launches[launch.id] = launch
        threading.Thread(target=launch._read, name=f"launch-{launch.id}", daemon=True).start()
        return launch.view()

    def start_painting(self, run: str, body: dict) -> dict:
        from conveyor.studio import create_request
        from conveyor.store import Store

        request = create_request(self.db, run, body)
        argv = ["studio", str(self.db), request["id"]]
        try:
            proc = subprocess.Popen([self.python, "-m", "conveyor.studio", str(self.db), request["id"]],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                    start_new_session=True)
        except OSError as error:
            with_store = Store(self.db, run_id=run)
            with_store.update_painting_request(request["id"], status="failed", ended=time.time(), error=str(error))
            with_store.close()
            raise LaunchError(f"Could not start painting: {error}") from error
        launch = Launch(argv, "New painting", proc)
        launch.run_id = run
        with self._lock:
            self._launches[launch.id] = launch
        threading.Thread(target=self._read_painting, args=(launch, request["id"], run),
                         name=f"painting-{launch.id}", daemon=True).start()
        return {**launch.view(), "request_id": request["id"]}

    def _read_painting(self, launch: Launch, request_id: str, run: str) -> None:
        from conveyor.store import Store, connect

        launch._read()
        conn = connect(self.db, readonly=True)
        try:
            row = conn.execute("SELECT status FROM painting_requests WHERE id=?", (request_id,)).fetchone()
        finally:
            conn.close()
        if row and row["status"] in ("queued", "running"):
            store = Store(self.db, run_id=run)
            store.update_painting_request(request_id, status="stopped" if launch.stop_requested else "failed",
                                          ended=time.time(), error="Painting process ended before saving its result.")
            store.close()

    # ---- the shared canvas --------------------------------------------------------------------------------

    def create_canvas(self, body: dict) -> dict:
        from conveyor.commons.tiles import DEFAULT_MAX_CALLS, DEFAULT_VIEWPORT, MAX_FRAME, MAX_TASK, VIEWPORT_RANGE
        from conveyor.commons.tiles import create_canvas

        if not isinstance(body, dict) or set(body) - {"name", "viewport", "max_calls", "task", "frame"}:
            raise LaunchError("Expected name, viewport, max_calls, task and frame.")
        task = body.get("task") or None
        if task is not None and (not isinstance(task, str) or len(task) > MAX_TASK):
            raise LaunchError(f"task must be text, at most {MAX_TASK} characters.")
        name = body.get("name") or "Commons"
        if not isinstance(name, str) or not name.strip() or len(name) > 60 or "\n" in name:
            raise LaunchError("name must be a line of text, at most 60 characters.")
        viewport = self._number("viewport", body.get("viewport", DEFAULT_VIEWPORT), int)
        if not VIEWPORT_RANGE[0] <= viewport <= VIEWPORT_RANGE[1]:
            raise LaunchError(f"viewport must be between {VIEWPORT_RANGE[0]} and {VIEWPORT_RANGE[1]} pixels.")
        max_calls = self._number("max_calls", body.get("max_calls", DEFAULT_MAX_CALLS), int)
        if not 1 <= max_calls <= 1000:
            raise LaunchError("max_calls must be between 1 and 1000.")
        frame = body.get("frame") or None
        if frame is not None:
            if not isinstance(frame, dict) or set(frame) != {"width", "height"}:
                raise LaunchError("frame must be {width, height} in pixels, or left out for a canvas with no edges.")
            frame = tuple(self._number(f"frame {k}", frame[k], int) for k in ("width", "height"))
            if not all(viewport <= side <= MAX_FRAME for side in frame):
                raise LaunchError(f"The frame's width and height must be from the viewport's {viewport} to {MAX_FRAME:,} pixels.")
        return create_canvas(self.db, name.strip(), viewport, max_calls, (task or "").strip() or None, frame)

    def _canvas(self, canvas_id: str):
        from conveyor.commons.tiles import SharedCanvas

        try:
            return SharedCanvas(self.db, canvas_id)
        except (ValueError, sqlite3.OperationalError):  # no such canvas, or no canvas tables yet
            return None

    def add_sketch(self, canvas_id: str, body: dict) -> dict | None:
        """Draw one line of a canvas's sketch, for its agents to see above the paint. None for an unknown canvas."""
        from conveyor.commons.sketch import SketchError, bounds, parse_line

        try:
            line = parse_line(body)
        except SketchError as e:
            raise LaunchError(str(e)) from None
        if (canvas := self._canvas(canvas_id)) is None:
            return None
        try:
            x0, y0, _, _ = bounds(line)
            seq = canvas.record(agent_id=None, tool="sketch", status="drawn", x=int(x0), y=int(y0), args=line,
                                note=f"{len(line['points'])} points")
        finally:
            canvas.close()
        return {"seq": seq, **line}

    def erase_sketch(self, canvas_id: str, body: dict) -> dict | None:
        """Erase lines of a canvas's sketch by their seqs. None for an unknown canvas."""
        from conveyor.commons.sketch import lines

        seqs = body.get("seqs") if isinstance(body, dict) and set(body) == {"seqs"} else None
        if not isinstance(seqs, list) or not seqs or any(isinstance(s, bool) or not isinstance(s, int) for s in seqs):
            raise LaunchError("Expected seqs: a list of the sketch lines to erase.")
        if (canvas := self._canvas(canvas_id)) is None:
            return None
        try:
            found = sorted(set(seqs) & {line["seq"] for line in lines(canvas.conn, canvas_id)})
            if not found:
                raise LaunchError("None of those lines is on the canvas.")
            seq = canvas.record(agent_id=None, tool="erase_sketch", status="erased", x=None, y=None,
                                args={"seqs": found}, note=f"{len(found)} lines")
        finally:
            canvas.close()
        return {"seq": seq, "erased": found}

    def start_agent(self, canvas_id: str, body: dict) -> dict:
        from conveyor.commons.agent import create_agent

        agent = create_agent(self.db, canvas_id, body, status="starting")
        return {**agent, "launch": self._spawn_agent(canvas_id, agent["id"], agent["name"])}

    def _spawn_agent(self, canvas_id: str, agent_id: str, name: str | None) -> dict:
        """Start the process for an agent already marked `starting`."""
        argv = ["canvas agent", agent_id]
        try:
            proc = subprocess.Popen([self.python, "-m", "conveyor.commons.agent", str(self.db), agent_id],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                    start_new_session=True)
        except OSError as error:
            self._close_agent(agent_id, "failed", f"Could not start the agent: {error}")
            raise LaunchError(f"Could not start the agent: {error}") from error
        launch = Launch(argv, f"Canvas agent {name or agent_id}", proc)
        with self._lock:
            self._launches[launch.id] = launch
            self._agents[agent_id] = (canvas_id, launch)
        threading.Thread(target=self._read_agent, args=(launch, agent_id), name=f"agent-{agent_id}",
                         daemon=True).start()
        return launch.view(log=False)

    def watch_canvas(self, interval: float = 2.0) -> None:
        """Start agents that wait in the database: the successors agents queue when they hand off."""
        if self._watching.is_set():
            return
        self._watching.set()

        def loop() -> None:
            while self._watching.is_set():
                try:
                    self.start_queued()
                except Exception as e:  # noqa: BLE001 - a bad pass must not end the watcher
                    print(f"[canvas] couldn't start queued agents: {e}", flush=True)
                time.sleep(interval)

        threading.Thread(target=loop, name="canvas-watcher", daemon=True).start()

    def start_queued(self) -> list[str]:
        """One pass: queue the judges that are due, then claim each queued agent (queued -> starting, so only one
        claim wins) and start it."""
        from conveyor.commons.agent import queue_judges
        from conveyor.store import connect

        try:
            queue_judges(self.db)
        except Exception as e:  # noqa: BLE001 - a judge that can't be queued mustn't stop the agents that are
            print(f"[canvas] couldn't queue judges: {e}", flush=True)

        conn = connect(self.db)
        try:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='canvas_agents'").fetchone() is None:
                return []
            queued = conn.execute("SELECT id, canvas_id, name FROM canvas_agents WHERE status='queued' "
                                  "ORDER BY created").fetchall()
            claimed = []
            for row in queued:
                cur = conn.execute("UPDATE canvas_agents SET status='starting' WHERE id=? AND status='queued'", (row["id"],))
                conn.commit()
                if cur.rowcount == 1:
                    claimed.append(row)
        finally:
            conn.close()
        for row in claimed:
            try:
                self._spawn_agent(row["canvas_id"], row["id"], row["name"])
            except LaunchError:
                pass  # _spawn_agent has marked it failed
        return [row["id"] for row in claimed]

    def _read_agent(self, launch: Launch, agent_id: str) -> None:
        launch._read()
        self._close_agent(agent_id, "stopped" if launch.stop_requested else "failed",
                          "The agent's process ended before it saved its result.\n" + "\n".join(list(launch.log)[-5:]))

    def _close_agent(self, agent_id: str, status: str, error: str) -> None:
        """Mark an agent ended if its process died without saying so itself."""
        from conveyor.store import connect

        conn = connect(self.db)
        try:
            conn.execute("UPDATE canvas_agents SET status=?, ended=?, error=? WHERE id=? AND status IN ('queued', 'starting', 'running')",
                         (status, time.time(), error.strip()[:2000], agent_id))
            conn.commit()
        finally:
            conn.close()

    def stop_agent(self, canvas_id: str, agent_id: str) -> dict | None:
        canvas, launch = self._agents.get(agent_id, (None, None))
        if launch is None:  # a successor still waiting for the watcher: it just never starts
            from conveyor.store import connect

            conn = connect(self.db)
            try:
                cur = conn.execute("UPDATE canvas_agents SET status='stopped', ended=?, error='Stopped before it started' "
                                   "WHERE id=? AND canvas_id=? AND status='queued'", (time.time(), agent_id, canvas_id))
                conn.commit()
            finally:
                conn.close()
            return {"id": agent_id, "state": "stopped"} if cur.rowcount else None
        if canvas != canvas_id:
            return None
        return self.stop(launch.id)

    def get(self, launch_id: str) -> dict | None:
        launch = self._launches.get(launch_id)
        return launch.view() if launch else None

    def list(self) -> list[dict]:
        with self._lock:
            launches = sorted(self._launches.values(), key=lambda x: -x.started)
        return [x.view(log=False) for x in launches]

    def stop(self, launch_id: str) -> dict | None:
        launch = self._launches.get(launch_id)
        if launch is None:
            return None
        if launch.proc.poll() is None and not launch.stop_requested:
            launch.stop_requested = True
            self._interrupt(launch.proc)
            threading.Thread(target=self._escalate, args=(launch.proc,), daemon=True).start()
        return launch.view()

    @staticmethod
    def _interrupt(proc: subprocess.Popen) -> None:
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

    @staticmethod
    def _escalate(proc: subprocess.Popen) -> None:
        """A run that ignores Ctrl+C gets SIGTERM, then SIGKILL for it and the sessions it started."""
        for wait, sig in ((TERM_AFTER, signal.SIGTERM), (KILL_AFTER - TERM_AFTER, signal.SIGKILL)):
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    return

    def shutdown(self, grace: float = 30.0) -> None:
        """The server is going away; stop its runs rather than leave them spending in the background."""
        self._watching.clear()
        live = [x for x in self._launches.values() if x.proc.poll() is None]
        if live:
            print(f"Stopping {len(live)} run{'s' if len(live) > 1 else ''} started from the dashboard...")
        for x in live:
            x.stop_requested = True
            self._interrupt(x.proc)
        deadline = time.time() + grace
        for x in live:
            try:
                x.proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(x.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
