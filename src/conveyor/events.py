"""
Append-only event log for a run.

Everything the dashboard shows is derived from three tables written here:
  * events: one row per thing that happened (an organism was created, evaluated, rescored, ...)
  * spans: tool calls inside one evaluation, grouped by trace id
  * artifacts: content-addressed blobs (PNGs) referenced from events and spans by name

All of it lives in one SQLite file. Writes go through a queue drained by one writer thread, because
evaluators run on thread pools and SQLite does not like concurrent writers. An artifact is always queued
before the event that references it, so readers never see a dangling reference.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT,
    started REAL NOT NULL,
    config TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    node TEXT,
    kind TEXT NOT NULL,
    organism_id TEXT,
    iteration INTEGER,
    ts REAL NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run_kind ON events(run_id, kind);
CREATE INDEX IF NOT EXISTS events_run_node_kind ON events(run_id, node, kind);
CREATE INDEX IF NOT EXISTS events_organism ON events(organism_id);
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    name TEXT NOT NULL,
    args TEXT,
    result TEXT,
    duration REAL,
    artifact TEXT,
    ts REAL NOT NULL,
    PRIMARY KEY (trace_id, idx)
);
CREATE TABLE IF NOT EXISTS artifacts (
    name TEXT PRIMARY KEY,
    data BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node TEXT,
    mutator TEXT,
    organism_id TEXT,
    trace_id TEXT,
    purpose TEXT,
    model TEXT,
    status TEXT NOT NULL,
    started REAL NOT NULL,
    first_token REAL,
    ended REAL,
    request TEXT,
    reasoning TEXT,
    content TEXT,
    tool_calls TEXT,
    tool_results TEXT,
    usage TEXT,
    finish_reason TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_run_status ON llm_calls(run_id, status);
CREATE INDEX IF NOT EXISTS llm_calls_run_node ON llm_calls(run_id, node);
CREATE INDEX IF NOT EXISTS llm_calls_trace ON llm_calls(trace_id);
CREATE INDEX IF NOT EXISTS llm_calls_organism ON llm_calls(organism_id);
"""

CALL_FIELDS = {
    "status", "first_token", "ended", "model", "reasoning", "content", "tool_calls", "tool_results", "usage",
    "finish_reason", "error", "request",
}
JSON_CALL_FIELDS = {"tool_calls", "tool_results", "usage", "request"}

_STOP = object()
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("conveyor_trace", default=None)
# Who is doing the work right now (node, mutator, organism). Lets code deep inside a mutator or evaluator,
# such as the LLM client, attribute what it records without being passed any of it.
_context: contextvars.ContextVar[dict | None] = contextvars.ContextVar("conveyor_context", default=None)


@contextmanager
def observing(**values: Any) -> Iterator[None]:
    token = _context.set({**(_context.get() or {}), **values})
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> dict:
    return _context.get() or {}


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def connect(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


class EventSink:
    def __init__(self, db_path: str | Path, run_name: str | None = None, config: dict | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]

        conn = connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO runs (id, name, started, config) VALUES (?, ?, ?, ?)",
            (self.run_id, run_name or self.run_id, time.time(), _dumps(config or {})),
        )
        conn.commit()
        conn.close()

        self._iterations: dict[str, int] = {}
        self._known_artifacts: set[str] = set()
        self._artifact_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, name="conveyor-event-writer", daemon=True)
        self._writer.start()
        self._closed = False

    def set_iteration(self, node: str, iteration: int) -> None:
        """Events emitted for `node` get stamped with this iteration until it changes."""
        self._iterations[node] = iteration

    def emit(self, node: str | None, kind: str, organism_id: Any = None, **data: Any) -> None:
        iteration = self._iterations.get(node) if node else None
        row = (
            self.run_id,
            node,
            kind,
            str(organism_id) if organism_id is not None else None,
            iteration,
            time.time(),
            _dumps(data),
        )
        self._queue.put(("event", row))

    def store_artifact(self, data: bytes, ext: str = "png") -> str:
        name = f"{hashlib.sha256(data).hexdigest()[:24]}.{ext}"
        with self._artifact_lock:
            if name in self._known_artifacts:
                return name
            self._known_artifacts.add(name)
        self._queue.put(("artifact", (name, data)))
        return name

    def start_call(
        self,
        call_id: str,
        *,
        node: str | None,
        mutator: str | None,
        organism_id: str | None,
        trace_id: str | None,
        purpose: str,
        model: str,
        request: dict,
    ) -> None:
        """Open a row for one model call. `update_call` fills it in as the reply streams and when it ends."""
        row = (call_id, self.run_id, node, mutator, organism_id, trace_id, purpose, model, "running", time.time(),
               _dumps(request))
        sql = ("INSERT INTO llm_calls (id, run_id, node, mutator, organism_id, trace_id, purpose, model, status, "
               "started, request) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
        self._queue.put(("call", (sql, row)))

    def update_call(self, call_id: str, **fields: Any) -> None:
        unknown = set(fields) - CALL_FIELDS
        if unknown:
            raise ValueError(f"unknown llm_calls fields: {sorted(unknown)}")
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        values = [_dumps(v) if k in JSON_CALL_FIELDS else v for k, v in fields.items()]
        self._queue.put(("call", (f"UPDATE llm_calls SET {cols} WHERE id=?", (*values, call_id))))

    @contextmanager
    def trace(self) -> Iterator[Trace]:
        """Open a trace. Code running inside can call `current_trace()` / `artifact()` without plumbing."""
        t = Trace(self, uuid.uuid4().hex[:16])
        token = _current_trace.set(t)
        try:
            yield t
        finally:
            _current_trace.reset(token)

    def flush(self, timeout: float = 10.0) -> None:
        done = threading.Event()
        self._queue.put(("flush", done))
        done.wait(timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._writer.join(timeout=10)

    def _write_loop(self) -> None:
        conn = connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        pending: dict[str, list[tuple]] = {"artifact": [], "call": [], "event": [], "span": []}
        flush_waiters: list[threading.Event] = []
        last_commit = time.time()
        stop = False
        while not stop:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                item = None
            if item is _STOP:
                stop = True
            elif item is not None:
                kind, payload = item
                if kind == "flush":
                    flush_waiters.append(payload)
                else:
                    pending[kind].append(payload)

            n = sum(len(v) for v in pending.values())
            due = stop or flush_waiters or n >= 500 or time.time() - last_commit > 0.5
            if due and (n or flush_waiters):
                # Artifacts first, so an event never commits ahead of the image it points to.
                if pending["artifact"]:
                    conn.executemany("INSERT OR IGNORE INTO artifacts (name, data) VALUES (?, ?)", pending["artifact"])
                # Call inserts and updates run in the order they were queued, so a row exists before it's updated.
                for sql, params in pending["call"]:
                    conn.execute(sql, params)
                if pending["event"]:
                    conn.executemany(
                        "INSERT INTO events (run_id, node, kind, organism_id, iteration, ts, data) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        pending["event"],
                    )
                if pending["span"]:
                    conn.executemany(
                        "INSERT OR REPLACE INTO spans (trace_id, idx, name, args, result, duration, artifact, ts) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        pending["span"],
                    )
                conn.commit()
                pending = {k: [] for k in pending}
                last_commit = time.time()
                for waiter in flush_waiters:
                    waiter.set()
                flush_waiters = []
        conn.close()


class Trace:
    """A sequence of spans (tool calls) recorded during one evaluation."""

    def __init__(self, sink: EventSink, trace_id: str) -> None:
        self.sink = sink
        self.id = trace_id
        self._idx = 0
        self._lock = threading.Lock()

    def span(
        self,
        name: str,
        args: dict | None = None,
        result: Any = None,
        duration: float | None = None,
        image: bytes | None = None,
    ) -> None:
        artifact_name = self.sink.store_artifact(image) if image is not None else None
        with self._lock:
            idx = self._idx
            self._idx += 1
        self.sink._queue.put(
            (
                "span",
                (self.id, idx, name, _dumps(args or {}), _dumps(result), duration, artifact_name, time.time()),
            )
        )

    @property
    def num_spans(self) -> int:
        return self._idx


def current_trace() -> Trace | None:
    return _current_trace.get()


def artifact(data: bytes, ext: str = "png") -> str | None:
    """Store an artifact against the active trace's sink. Returns None when nothing is observing."""
    t = _current_trace.get()
    if t is None:
        return None
    return t.sink.store_artifact(data, ext)
