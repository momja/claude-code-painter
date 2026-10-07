"""What the canvas page reads. Every function takes a read-only connection and returns plain data."""

from __future__ import annotations

import json
import sqlite3
import time

from conveyor.commons.tiles import message

STALE_SECONDS = 20 * 60  # an agent marked running with no call for this long died without closing its row
LIVE = ("queued", "starting", "running")


def _agent(row: sqlite3.Row) -> dict:
    agent = dict(row)
    config = json.loads(agent.pop("config") or "{}")
    agent["config"] = {k: v for k, v in config.items() if k not in ("source", "prompt")}  # the text is in the catalog
    if agent["status"] in LIVE and time.time() - (agent["last_ts"] or agent["created"]) > STALE_SECONDS:
        agent["status"] = "interrupted"
    return agent


def canvases(conn: sqlite3.Connection) -> list[dict]:
    # `conveyor serve` leaves an existing database alone, so its canvas tables appear with its first canvas.
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='canvases'").fetchone() is None:
        return []
    out = []
    for row in conn.execute("SELECT * FROM canvases ORDER BY created DESC"):
        c = dict(row)
        c["config"] = json.loads(c["config"])
        stats = conn.execute("SELECT count(*) AS agents, sum(status IN ('queued','starting','running')) AS running "
                             "FROM canvas_agents WHERE canvas_id=?", (c["id"],)).fetchone()
        ops = conn.execute("SELECT count(*) AS n, max(ts) AS last FROM canvas_ops WHERE canvas_id=?", (c["id"],)).fetchone()
        c.update(agents=stats["agents"], running=stats["running"] or 0, ops=ops["n"], last_ts=ops["last"])
        out.append(c)
    return out


def canvas(conn: sqlite3.Connection, canvas_id: str, since: int = 0) -> dict:
    """The canvas now: its agents, the tiles whose head moved after op `since` (all of them from 0), the messages,
    broadcasts and sketch lines after it, the sketch erasures after it, and the newest op, which the page passes back as `since` on its next poll. A message comes
    in canvas pixels with its wrapped lines, the way the text layer draws it."""
    row = conn.execute("SELECT * FROM canvases WHERE id=?", (canvas_id,)).fetchone()
    if row is None:
        raise KeyError(canvas_id)
    meta = dict(row)
    meta["config"] = json.loads(meta["config"])
    agents = [_agent(r) for r in conn.execute("SELECT * FROM canvas_agents WHERE canvas_id=? ORDER BY created", (canvas_id,))]
    heads = [list(r) for r in conn.execute("SELECT tx, ty, seq FROM canvas_heads WHERE canvas_id=? AND seq>?",
                                            (canvas_id, since))]
    seq = conn.execute("SELECT max(seq) FROM canvas_ops WHERE canvas_id=?", (canvas_id,)).fetchone()[0] or 0
    messages = []
    for r in conn.execute("SELECT seq, agent_id, ts, x, y, args FROM canvas_ops WHERE canvas_id=? AND seq>? AND "
                          "tool='write_message' AND status IN ('painted', 'written') ORDER BY seq", (canvas_id, since)):
        m = message(r["seq"], r["x"], r["y"], json.loads(r["args"]), int(meta["config"]["viewport"]))
        messages.append({**m, "agent_id": r["agent_id"], "ts": r["ts"],
                         "color": "#%02x%02x%02x" % tuple(round(c * 255) for c in m["color"])})
    broadcasts = []
    for r in conn.execute("SELECT seq, agent_id, ts, x, y, note, args FROM canvas_ops WHERE canvas_id=? AND seq>? "
                          "AND tool='broadcast' AND status='sent' ORDER BY seq", (canvas_id, since)):
        # placed at the middle of its sender's viewport
        half = int(json.loads(r["args"] or "{}").get("viewport") or meta["config"]["viewport"]) // 2
        broadcasts.append({"seq": r["seq"], "agent_id": r["agent_id"], "ts": r["ts"], "text": r["note"],
                           "x": r["x"] + half, "y": r["y"] + half})
    sketches, erasures = [], []  # lines drawn after `since`, and [erase seq, [line seqs]] for erasures after it
    for r in conn.execute("SELECT seq, ts, tool, args FROM canvas_ops WHERE canvas_id=? AND seq>? AND "
                          "tool IN ('sketch', 'erase_sketch') ORDER BY seq", (canvas_id, since)):
        args = json.loads(r["args"] or "{}")
        if r["tool"] == "sketch":
            sketches.append({"seq": r["seq"], "ts": r["ts"], **args})
        else:
            erasures.append([r["seq"], args.get("seqs", [])])
    return {"canvas": meta, "agents": agents, "heads": heads, "seq": seq, "since": since, "messages": messages,
            "broadcasts": broadcasts, "sketches": sketches, "erasures": erasures,
            "now": time.time()}


def history(conn: sqlite3.Connection, canvas_id: str) -> dict:
    """Everything replay needs: each tile version as [seq, tx, ty], and each op as
    [seq, agent_id, tool, status, x, y, ts], both in seq order."""
    if conn.execute("SELECT 1 FROM canvases WHERE id=?", (canvas_id,)).fetchone() is None:
        raise KeyError(canvas_id)
    versions = [list(r) for r in conn.execute("SELECT seq, tx, ty FROM canvas_tiles WHERE canvas_id=? ORDER BY seq",
                                               (canvas_id,))]
    ops = [list(r) for r in conn.execute("SELECT seq, agent_id, tool, status, x, y, ts FROM canvas_ops "
                                          "WHERE canvas_id=? ORDER BY seq", (canvas_id,))]
    return {"versions": versions, "ops": ops}


def op(conn: sqlite3.Connection, canvas_id: str, seq: int) -> dict:
    row = conn.execute("SELECT * FROM canvas_ops WHERE canvas_id=? AND seq=?", (canvas_id, seq)).fetchone()
    if row is None:
        raise KeyError(seq)
    out = dict(row)
    out["args"] = json.loads(out["args"] or "null")
    return out


def tile(conn: sqlite3.Connection, canvas_id: str, tx: int, ty: int, seq: int) -> bytes | None:
    row = conn.execute("SELECT data FROM canvas_tiles WHERE canvas_id=? AND tx=? AND ty=? AND seq=?",
                       (canvas_id, tx, ty, seq)).fetchone()
    return row[0] if row else None
