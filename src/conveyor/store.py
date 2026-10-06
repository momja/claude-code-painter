"""
The run log. Everything the dashboard shows is read from one SQLite file written here.

  runs            one row per run, with its config
  events          things that happened: cycles, champion changes, node status, alarms
  organisms       every organism, with its genome, parent, mutator, and niche
  evaluations     every evaluation: score, partner, details, images, and the Claude session that painted it
  sessions        every Claude Code process: purpose, model, cost, usage, outcome, the request it was sent
  session_events  the stream-json events of each session, in order (thinking, text, tool calls, results)
  strokes         the paint server's per-call log for each painting session (with canvas snapshots)
  artifacts       content-addressed images, referenced from the rows above by name

Writes go through a queue drained by one writer thread, in the order they were queued, so a row that
references an artifact is never committed before the artifact. The dashboard opens the file read-only, so it
can serve a run in progress from another process.
"""

from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, name TEXT, started REAL NOT NULL, config TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, node TEXT, kind TEXT NOT NULL,
    organism_id TEXT, ts REAL NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run_kind ON events(run_id, kind);
CREATE TABLE IF NOT EXISTS organisms (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, node TEXT NOT NULL, parent_id TEXT, parent2_id TEXT,
    mutator TEXT, created REAL NOT NULL, summary TEXT, genome TEXT NOT NULL, text TEXT, niche TEXT,
    traits TEXT, sheet TEXT, session_id TEXT, viable INTEGER, note TEXT
);
CREATE INDEX IF NOT EXISTS organisms_run_node ON organisms(run_id, node);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, node TEXT NOT NULL, organism_id TEXT NOT NULL,
    partner_id TEXT, reason TEXT, score REAL, viable INTEGER, started REAL, ended REAL, details TEXT,
    artifacts TEXT, session_id TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS evaluations_organism ON evaluations(organism_id);
CREATE INDEX IF NOT EXISTS evaluations_run ON evaluations(run_id, node);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, node TEXT, organism_id TEXT, purpose TEXT, model TEXT,
    effort TEXT, status TEXT NOT NULL, started REAL NOT NULL, ended REAL, cost REAL, usage TEXT,
    num_turns INTEGER, result TEXT, error TEXT, request TEXT, dir TEXT, last_ts REAL
);
CREATE INDEX IF NOT EXISTS sessions_run ON sessions(run_id, status);
CREATE INDEX IF NOT EXISTS sessions_organism ON sessions(organism_id);
CREATE TABLE IF NOT EXISTS session_events (
    session_id TEXT NOT NULL, idx INTEGER NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS strokes (
    session_id TEXT NOT NULL, idx INTEGER NOT NULL, ts REAL NOT NULL, data TEXT NOT NULL, snapshot TEXT,
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS artifacts (name TEXT PRIMARY KEY, data BLOB NOT NULL);
"""

SESSION_FIELDS = {"status", "ended", "cost", "usage", "num_turns", "result", "error", "request", "model", "last_ts"}
JSON_FIELDS = {"usage", "request", "genome", "traits", "details", "artifacts"}
_STOP = object()


def dumps(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


# Several runs may write one file at once. SQLite takes one writer at a time, so a writer waits this long for
# the lock before a statement fails, and the batch is then retried rather than dropped.
BUSY_TIMEOUT = 30.0
LOCK_RETRIES = 20


def connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=BUSY_TIMEOUT)
    else:
        conn = sqlite3.connect(path, check_same_thread=False, timeout=BUSY_TIMEOUT)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path) -> None:
    """Create the file and its tables if they're missing, so a dashboard can open a database no run has written yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        _ensure_wal(conn)
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    def __init__(self, path: str | Path, run_name: str | None = None, config: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = new_id()
        conn = connect(self.path)
        _ensure_wal(conn)
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO runs (id, name, started, config) VALUES (?, ?, ?, ?)",
                     (self.run_id, run_name or self.run_id, time.time(), dumps(config or {})))
        conn.commit()
        conn.close()
        self._known: set[str] = set()
        self._known_lock = threading.Lock()
        self._idx: dict[str, int] = {}
        self._idx_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._loop, name="conveyor-store", daemon=True)
        self._writer.start()
        self._closed = False

    # ---- writing ------------------------------------------------------------------------------------------

    def _put(self, sql: str, params: tuple) -> None:
        self._queue.put((sql, params))

    def emit(self, kind: str, node: str | None = None, organism_id: str | None = None, **data: Any) -> None:
        self._put("INSERT INTO events (run_id, node, kind, organism_id, ts, data) VALUES (?, ?, ?, ?, ?, ?)",
                  (self.run_id, node, kind, organism_id, time.time(), dumps(data)))

    def artifact(self, data: bytes, ext: str = "png") -> str:
        name = f"{hashlib.sha256(data).hexdigest()[:24]}.{ext}"
        with self._known_lock:
            if name in self._known:
                return name
            self._known.add(name)
        self._put("INSERT OR IGNORE INTO artifacts (name, data) VALUES (?, ?)", (name, data))
        return name

    def organism(self, row: dict) -> None:
        cols = ["id", "node", "parent_id", "parent2_id", "mutator", "created", "summary", "genome", "text", "niche",
                "traits", "sheet", "session_id", "viable", "note"]
        values = [dumps(row.get(c)) if c in JSON_FIELDS else row.get(c) for c in cols]
        self._put(f"INSERT OR REPLACE INTO organisms (run_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
                  (self.run_id, *values))

    def update_organism(self, organism_id: str, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        values = [dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()]
        self._put(f"UPDATE organisms SET {cols} WHERE id=?", (*values, organism_id))

    def evaluation(self, row: dict) -> None:
        cols = ["id", "node", "organism_id", "partner_id", "reason", "score", "viable", "started", "ended", "details",
                "artifacts", "session_id", "error"]
        values = [dumps(row.get(c)) if c in JSON_FIELDS else row.get(c) for c in cols]
        self._put(f"INSERT OR REPLACE INTO evaluations (run_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
                  (self.run_id, *values))

    def start_session(self, session_id: str, *, node: str | None, organism_id: str | None, purpose: str, model: str,
                      effort: str | None, request: dict, dir: str | None) -> None:
        now = time.time()
        self._put("INSERT INTO sessions (id, run_id, node, organism_id, purpose, model, effort, status, started, request,"
                  " dir, last_ts) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)",
                  (session_id, self.run_id, node, organism_id, purpose, model, effort, now, dumps(request), dir, now))

    def update_session(self, session_id: str, **fields: Any) -> None:
        unknown = set(fields) - SESSION_FIELDS
        if unknown:
            raise ValueError(f"unknown session fields {sorted(unknown)}")
        cols = ", ".join(f"{k}=?" for k in fields)
        values = [dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()]
        self._put(f"UPDATE sessions SET {cols} WHERE id=?", (*values, session_id))

    def session_event(self, session_id: str, kind: str, data: dict) -> None:
        with self._idx_lock:
            idx = self._idx.get(session_id, 0)
            self._idx[session_id] = idx + 1
        now = time.time()
        self._put("INSERT INTO session_events (session_id, idx, ts, kind, data) VALUES (?, ?, ?, ?, ?)",
                  (session_id, idx, now, kind, dumps(data)))
        self._put("UPDATE sessions SET last_ts=? WHERE id=?", (now, session_id))

    def stroke(self, session_id: str, idx: int, data: dict, snapshot: str | None) -> None:
        self._put("INSERT OR REPLACE INTO strokes (session_id, idx, ts, data, snapshot) VALUES (?, ?, ?, ?, ?)",
                  (session_id, idx, data.get("t", time.time()), dumps(data), snapshot))

    # ---- lifecycle ----------------------------------------------------------------------------------------

    def flush(self, timeout: float = 10.0) -> None:
        done = threading.Event()
        self._queue.put(("__flush__", done))
        done.wait(timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._writer.join(timeout=15)

    def _loop(self) -> None:
        conn = connect(self.path)
        _ensure_wal(conn)
        conn.execute("PRAGMA synchronous=NORMAL")
        pending: list[tuple[str, tuple]] = []
        waiters: list[threading.Event] = []
        last = time.time()
        stop = False
        while not stop:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                item = None
            if item is _STOP:
                stop = True
            elif item is not None:
                if item[0] == "__flush__":
                    waiters.append(item[1])
                else:
                    pending.append(item)
            if pending and (stop or waiters or len(pending) >= 400 or time.time() - last > 0.3):
                self._commit(conn, pending)
                pending = []
                last = time.time()
            if waiters and not pending:
                for w in waiters:
                    w.set()
                waiters = []
        conn.close()

    @staticmethod
    def _commit(conn: sqlite3.Connection, pending: list[tuple[str, tuple]]) -> None:
        """
        Write a batch in one transaction. A locked database (another run writing the same file) rolls the batch
        back and retries it whole; any other error skips just that row, since one bad row must not take the log
        down. Nothing here may raise: an exception would end the writer thread, and every later write with it.
        """
        for attempt in range(LOCK_RETRIES):
            try:
                for sql, params in pending:
                    try:
                        conn.execute(sql, params)
                    except sqlite3.OperationalError as e:
                        if _locked(e):
                            raise
                        print(f"[store] {e}: {sql[:80]}", flush=True)
                    except sqlite3.Error as e:
                        print(f"[store] {e}: {sql[:80]}", flush=True)
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                if not _locked(e):
                    print(f"[store] {e}", flush=True)
                    return
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                time.sleep(min(2.0, 0.1 * (attempt + 1)))
        print(f"[store] gave up on {len(pending)} writes: the database stayed locked", flush=True)


def _ensure_wal(conn: sqlite3.Connection) -> None:
    """
    Put the file in WAL mode, once. The mode is stored in the file, so a run that finds it set leaves it alone.
    Switching needs an exclusive lock and fails at once, without waiting out the busy timeout, when another run
    is opening the same file at that moment; so it is retried here.
    """
    for attempt in range(LOCK_RETRIES * 5):
        try:
            if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal":
                return
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as e:
            if not _locked(e):
                raise
            time.sleep(min(1.0, 0.05 * (attempt + 1)))
    raise sqlite3.OperationalError("database is locked: could not switch it to WAL mode")


def _locked(e: sqlite3.OperationalError) -> bool:
    text = str(e).lower()
    return "locked" in text or "busy" in text
