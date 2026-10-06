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
import hmac
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from conveyor.claude import DEFAULT_MODEL
from conveyor.store import new_id

LOG_LINES = 300
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
    "judge": "toggle", "judge_weight": "float", "scope": "toggle",
    "seeds": "list", "parents": "int", "confirm": "int",
    "refine": "float", "invent": "float", "recombine": "float",
    "lanes": "int", "paint_cap": "float", "mutate_cap": "float", "max_usage": "float",
    "compact_every_looks": "int", "autocompact": "int",
}
# Sanity bounds, to catch a typo before it spends money. (low, high), inclusive.
BOUNDS = {
    "cycles": (1, 1000), "budget": (0.01, 10_000), "wait": (0.1, 168), "width": (64, 2048), "actions": (1, 5000),
    "looks": (-1, 500), "judge_weight": (0, 1), "parents": (1, 10), "confirm": (1, 10),
    "refine": (0, 100), "invent": (0, 100), "recombine": (0, 100), "lanes": (1, 8),
    "paint_cap": (0.01, 1000), "mutate_cap": (0.01, 1000), "max_usage": (0.05, 1),
    "compact_every_looks": (1, 500), "autocompact": (100_000, 1_000_000),
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
        self._launches: dict[str, Launch] = {}
        self._lock = threading.Lock()
        self._probe: tuple[float, dict] | None = None

    def authorized(self, presented: str | None) -> bool:
        return not self.token or hmac.compare_digest((presented or "").encode(), self.token.encode())

    # ---- what the form offers -----------------------------------------------------------------------------

    @property
    def _run(self) -> argparse.ArgumentParser:
        return self._parser.commands["run"]  # type: ignore[attr-defined]

    def targets(self) -> list[str]:
        if self._targets is not None:
            return self._targets
        from conveyor.painting.canvas import TARGETS_DIR

        return sorted(p.stem for p in Path(TARGETS_DIR).glob("*.jpg"))

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
        from conveyor.pi import load_api_key

        version = claude_available("claude")
        problem = pi_available()
        keys = {k: bool(load_api_key(p)) for k, p in PROVIDERS.items()}
        if version is None:
            claude = {"ok": False, "detail": "`claude --version` fails; install Claude Code"}
        elif logged_in("claude") is False:
            claude = {"ok": False, "detail": f"{version}, but not logged in; run `claude auth login`"}
        else:
            claude = {"ok": True, "detail": version}
        out = {"claude": claude,
               "pi": {"ok": problem is None, "detail": problem or "ready", "keys": keys,
                      "env_vars": {k: p.env_var for k, p in PROVIDERS.items()}}}
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
