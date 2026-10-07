"""
Read-only HTTP server for the dashboard. Standard library only; every view is computed from the store on request,
so it works the same during a run and after it, and from another process.

  GET /                              the dashboard
  GET /api/runs                      runs in the database
  GET /api/runs/<run>                overview: nodes, champions, archive, spend, rate limit, live sessions, alarms
  GET /api/runs/<run>/history        every evaluation's score over time, and champion changes
  GET /api/runs/<run>/organisms      every organism, newest first (?node=)
  GET /api/runs/<run>/mutators       per node and mutator: tries, children, viable, beat parent, niche hits, cost
  GET /studio                       painter studio graph and request page
  GET /api/runs/<run>/studio         all prompts, instruments/tools, and unique painting sessions
  GET /api/organisms/<id>            one organism: source, parent's source, evaluations, children, who wrote it
  GET /api/sessions/<id>             one model session: request, events, strokes, result
  GET /artifacts/<name>              images
  GET /canvas                        the shared canvas: watch it live, replay it, spawn agents
  GET /api/canvases                  shared canvases
  GET /api/canvases/<id>             one canvas: agents, tile heads moved after ?since=<op seq>, recent messages
  GET /api/canvases/<id>/history     every tile version and every op, for replay
  GET /api/canvases/<id>/ops/<seq>   one op: the call's arguments and result note
  GET /api/canvases/<id>/tiles/<tx>/<ty>/<seq>  one version of one tile (PNG)
  GET /api/canvas-catalog            every instrument and painter prompt in any run, deduplicated, plus the seeds

With a launcher (`conveyor serve` on a loopback address) it can also start runs:

  GET  /api/options                  what the new-run form offers: defaults, help, models, seeds, targets, harnesses
  GET  /api/launches                 runs started from here, newest first
  GET  /api/launches/<id>            one launch: state, run id once it has one, the last lines of its output
  POST /api/targets                  import a target image from {name, image: base64}; answers its target name
  POST /api/launches                 start a run from {option: value}; answers 201 with the launch
  POST /api/launches/<id>/stop       stop it, as Ctrl+C would
  POST /api/runs/<run>/paintings      paint with saved instrument_id/prompt_id, text and optional base64 image
  POST /api/canvases                 create a shared canvas from {name, viewport, max_calls, task, frame: {width, height}}
  POST /api/canvases/<id>/agents     spawn an agent: kind (painter or judge), catalog pair_id (or "random"), harness, model, effort,
                                     provider, start x and y, name, cap, paint_batch, successors, task, viewport
  POST /api/canvases/<id>/agents/<agent>/stop   stop it, as Ctrl+C would
  POST /api/canvases/<id>/sketches   draw a sketch line from {points: [[x, y], ...], color, width} in canvas pixels
  POST /api/canvases/<id>/sketches/erase        erase sketch lines from {seqs: [...]}

On a loopback address launching needs nothing more. Served to a network, it needs a token: the launcher is given
one, and then these endpoints answer only to a request carrying it in X-Conveyor-Token. The dashboard's reads stay
open. Without the token /api/options says `{"launch": true, "locked": true}` and the rest answer 401.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse

from conveyor.launch import LaunchError
from conveyor.store import connect

MAX_BODY = 64 * 1024
MAX_PAINTING_BODY = 9 * 1024 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

DASHBOARD = Path(__file__).parent / "dashboard.html"
STUDIO = Path(__file__).parent / "studio.html"
COMMONS = Path(__file__).parent / "commons.html"
ARTIFACT_RE = re.compile(r"^[0-9a-f]{24}\.(png|jpg)$")
JSON_COLS = {"data", "config", "genome", "traits", "details", "artifacts", "usage", "request"}
STALL_ITERATIONS = 4
LIVE_EVENTS = 6
# A session still marked running with no event for this long, or in a run that has finished, died without
# closing its row (the process was killed). Opus at high effort can think for a couple of minutes between events.
STALE_SECONDS = 20 * 60


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS & d.keys():
        if isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


class Views:
    def __init__(self, db: Path) -> None:
        self.db = db
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = connect(self.db, readonly=True)
        return c

    def rows(self, sql: str, *args: Any) -> list[dict]:
        return [_row(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, *args: Any) -> dict | None:
        return _row(self.conn.execute(sql, args).fetchone())

    def last(self, run: str, kind: str, node: str | None = None) -> dict | None:
        if node is None:
            return self.one("SELECT * FROM events WHERE run_id=? AND kind=? ORDER BY seq DESC LIMIT 1", run, kind)
        return self.one("SELECT * FROM events WHERE run_id=? AND kind=? AND node=? ORDER BY seq DESC LIMIT 1", run, kind, node)

    # ---- runs ---------------------------------------------------------------------------------------------

    def runs(self) -> list[dict]:
        out = []
        for r in self.rows("SELECT * FROM runs ORDER BY started DESC"):
            stats = self.one("SELECT max(ts) AS last_ts, count(*) AS n FROM events WHERE run_id=?", r["id"])
            spend = self.one("SELECT coalesce(sum(cost), 0) AS cost, count(*) AS n FROM sessions WHERE run_id=?", r["id"])
            n_org = self.one("SELECT count(*) AS n FROM organisms WHERE run_id=?", r["id"])["n"]
            out.append({**r, "last_ts": stats["last_ts"], "n_events": stats["n"], "cost": spend["cost"],
                        "n_sessions": spend["n"], "n_organisms": n_org,
                        "finished": self.last(r["id"], "run_finished") is not None})
        return out

    def _latest_evals(self, run: str) -> dict[str, dict]:
        """Each organism's standing, as the conductor keeps it: the latest evaluation, scored with the mean of the
        paintings since its partner last changed. A first evaluation or a rescore starts a tally; each `confirm`
        repeat adds to it, and a failed repeat leaves it alone."""
        tally: dict[str, list[dict]] = {}
        for r in self.rows("SELECT id, organism_id, partner_id, reason, score, viable, ended, artifacts FROM evaluations "
                           "WHERE run_id=? ORDER BY ended", run):
            if r["reason"] != "confirm":
                tally[r["organism_id"]] = [r]
            elif r["viable"]:
                tally.setdefault(r["organism_id"], []).append(r)
        out = {}
        for oid, rs in tally.items():
            viable = [r for r in rs if r["viable"]]
            out[oid] = rs[-1] if not viable else {
                **viable[-1], "score": sum(r["score"] for r in viable) / len(viable), "samples": len(viable),
                "scores": [round(r["score"], 4) for r in viable]}
        return out

    def overview(self, run: str) -> dict:
        meta = self.one("SELECT * FROM runs WHERE id=?", run)
        if meta is None:
            raise KeyError(run)
        graph = self.last(run, "graph")
        cycle = self.last(run, "cycle")
        finished = self.last(run, "run_finished")
        stopping = self.last(run, "stopping")
        waiting = self.last(run, "waiting")
        resumed = self.last(run, "resumed")
        if finished or (waiting and resumed and resumed["seq"] > waiting["seq"]):
            waiting = None
        spend_ev = self.last(run, "spend")
        costs = self.one("SELECT coalesce(sum(cost),0) AS cost, count(*) AS n, sum(status='running') AS running, "
                         "sum(status='error') AS errors FROM sessions WHERE run_id=?", run)
        rate = self.one("SELECT se.data FROM session_events se JOIN sessions s ON s.id=se.session_id WHERE s.run_id=? "
                        "AND se.kind='rate_limit' ORDER BY se.ts DESC LIMIT 1", run)
        evals = self._latest_evals(run)
        live = self.live(run)
        organisms = {o["id"]: o for o in self.rows(
            "SELECT id, node, niche, parent_id, mutator, summary, sheet, viable, created, traits FROM organisms WHERE run_id=?", run)}

        nodes = []
        for n in (graph["data"]["nodes"] if graph else []):
            name = n["name"]
            status = self.last(run, "node_status", name)
            champ_ev = self.last(run, "champion_changed", name)
            champ = None
            if champ_ev:
                oid = champ_ev["organism_id"]
                ev = evals.get(oid)
                org = organisms.get(oid, {})
                champ = {"id": oid, "score": ev["score"] if ev else champ_ev["data"].get("score"), "niche": org.get("niche"),
                         "samples": (ev or {}).get("samples", 1), "scores": (ev or {}).get("scores"),
                         "sheet": org.get("sheet"), "painting": (ev or {}).get("artifacts", {}).get("painting"),
                         "since": champ_ev["ts"], "summary": org.get("summary")}
            its = self.one("SELECT count(*) AS n FROM events WHERE run_id=? AND kind='iteration' AND node=?", run, name)["n"]
            n_org = sum(1 for o in organisms.values() if o["node"] == name)
            nodes.append({**n, "status": (status or {}).get("data", {}).get("status", "idle"), "champion": champ,
                          "iterations": its, "organisms": n_org})

        # The best organism per niche, and how many landed there. Unpainted organisms count too (the `mutate`
        # command makes only those); a painted one beats an unpainted one for the cell.
        archive: dict[str, dict] = {}
        for oid, o in sorted(organisms.items(), key=lambda kv: kv[1]["created"]):
            ev = evals.get(oid)
            if not o.get("niche") or o.get("viable") == 0 or (ev and not ev["viable"]):
                continue
            score = ev["score"] if ev else None
            best = archive.get(o["niche"])
            count = (best["count"] if best else 0) + 1
            if best is None or (score is not None and (best["score"] is None or score > best["score"])):
                archive[o["niche"]] = {"id": oid, "score": score, "sheet": o.get("sheet"), "count": count,
                                       "painting": ev["artifacts"].get("painting") if ev else None,
                                       "summary": o.get("summary"), "mutator": o.get("mutator")}
            else:
                best["count"] = count
        all_niches = next((n.get("all_niches") for n in (graph["data"]["nodes"] if graph else []) if n.get("archive")), None)
        if all_niches is None and archive:
            from conveyor.painting.instrument import all_niches as niches_of_the_problem
            all_niches = niches_of_the_problem()
        return {
            "run": meta, "graph": graph["data"] if graph else None, "nodes": nodes,
            "cycle": cycle["data"] if cycle else None, "finished": finished["data"] if finished else None,
            "stopping": stopping["data"] if stopping else None,
            "waiting": waiting["data"] if waiting else None,
            "spend": {"cost": costs["cost"], "sessions": costs["n"], "running": len(live),
                      "errors": costs["errors"] or 0, "budget": meta["config"].get("budget")},
            "rate_limit": rate["data"] if rate else (spend_ev["data"].get("rate_limit") if spend_ev else None),
            "archive": {"niches": all_niches or sorted(archive), "cells": archive},
            "live": live, "alarms": self.alarms(run, nodes, costs, rate),
        }

    def live(self, run: str) -> list[dict]:
        if self.last(run, "run_finished") is not None:
            return []
        out = []
        for s in self.rows("SELECT id, node, organism_id, purpose, model, started, last_ts, "
                           "json_extract(request, '$.harness') AS harness FROM sessions "
                           "WHERE run_id=? AND status='running' AND coalesce(last_ts, started) > ? ORDER BY started",
                           run, time.time() - STALE_SECONDS):
            events = self.rows("SELECT idx, ts, kind, data FROM session_events WHERE session_id=? AND kind IN "
                               "('thinking','text','tool_use','tool_result') ORDER BY idx DESC LIMIT ?", s["id"], LIVE_EVENTS)
            n_tools = self.one("SELECT count(*) AS n FROM session_events WHERE session_id=? AND kind='tool_use'", s["id"])["n"]
            canvas = self.one("SELECT snapshot FROM strokes WHERE session_id=? AND snapshot IS NOT NULL ORDER BY idx DESC LIMIT 1", s["id"])
            out.append({**s, "events": list(reversed(events)), "tool_calls": n_tools, "now": time.time(),
                        "canvas": canvas["snapshot"] if canvas else None})
        return out

    def alarms(self, run: str, nodes: list[dict], costs: dict, rate: dict | None) -> list[dict]:
        out = []
        if rate:
            info = rate["data"]
            if info.get("status") == "rejected":
                out.append({"level": "bad", "text": "Claude Code hit its rate limit; new sessions can't start until "
                            + time.strftime("%H:%M", time.localtime(info.get("resetsAt") or 0))})
            else:
                for window, w in (info.get("unifiedWindows") or {}).items():
                    if (w.get("utilization") or 0) >= 0.8:
                        out.append({"level": "warn", "text": f"{window.replace('_', '-')} rate limit at {w['utilization']:.0%}"})
        if costs["n"] >= 4 and (costs["errors"] or 0) / costs["n"] > 0.25:
            out.append({"level": "warn", "text": f"{costs['errors']} of {costs['n']} model sessions ended in an error"})
        for m in self.mutator_stats(run):
            if m["tries"] >= 4 and m["errors"] / m["tries"] > 0.5:
                out.append({"level": "warn", "text": f"{m['node']} mutator {m['mutator']} failed {m['errors']} of {m['tries']} times"})
        for n in nodes:
            if n.get("fixed"):
                continue
            since = self.one("SELECT count(*) AS n FROM events WHERE run_id=? AND kind='iteration' AND node=? AND ts > ?",
                             run, n["name"], (n.get("champion") or {}).get("since") or 0)["n"]
            if since >= STALL_ITERATIONS:
                out.append({"level": "info", "text": f"{n['name']} champion unchanged for {since} iterations"})
        return out

    # ---- history and mutators -----------------------------------------------------------------------------

    def history(self, run: str) -> dict:
        evals = self.rows("SELECT e.id, e.node, e.organism_id, e.reason, e.score, e.viable, e.ended, e.partner_id, "
                          "o.mutator, o.niche FROM evaluations e LEFT JOIN organisms o ON o.id=e.organism_id "
                          "WHERE e.run_id=? ORDER BY e.ended", run)
        champions = self.rows("SELECT node, organism_id, ts, data FROM events WHERE run_id=? AND kind='champion_changed' "
                              "ORDER BY seq", run)
        return {"evaluations": evals, "champions": champions}

    def mutator_stats(self, run: str) -> list[dict]:
        evals = self._latest_evals(run)
        orgs = {o["id"]: o for o in self.rows("SELECT id, node, parent_id, mutator, niche, viable, traits, session_id "
                                              "FROM organisms WHERE run_id=?", run)}
        costs = {s["id"]: s["cost"] or 0.0 for s in self.rows("SELECT id, cost FROM sessions WHERE run_id=?", run)}
        stats: dict[tuple, dict] = defaultdict(lambda: {"tries": 0, "errors": 0, "children": 0, "viable": 0, "beat_parent": 0,
                                                       "compared": 0, "new_niche": 0, "asked_niche": 0, "hit_niche": 0, "cost": 0.0})
        for ev in self.rows("SELECT node, data FROM events WHERE run_id=? AND kind='mutation'", run):
            d = ev["data"]
            s = stats[(ev["node"], d["mutator"])]
            s["tries"] += 1
            s["errors"] += 1 if d.get("error") else 0
        for o in orgs.values():
            if not o["mutator"]:
                continue
            s = stats[(o["node"], o["mutator"])]
            s["children"] += 1
            s["cost"] += costs.get(o["session_id"], 0.0)
            ev, pev = evals.get(o["id"]), evals.get(o["parent_id"])
            if ev and ev["viable"]:
                s["viable"] += 1
                if pev and pev["viable"]:
                    s["compared"] += 1
                    s["beat_parent"] += 1 if ev["score"] > pev["score"] else 0
            parent = orgs.get(o["parent_id"])
            if o["niche"] and parent and o["niche"] != parent["niche"]:
                s["new_niche"] += 1
            wanted = (o.get("traits") or {}).get("wanted_niche")
            if wanted:
                s["asked_niche"] += 1
                s["hit_niche"] += 1 if o["niche"] == wanted else 0
        return [{"node": k[0], "mutator": k[1], **v} for k, v in sorted(stats.items())]

    def organisms(self, run: str, node: str | None) -> list[dict]:
        evals = self._latest_evals(run)
        sql = ("SELECT id, node, parent_id, parent2_id, mutator, created, summary, niche, viable, note, sheet, "
               "json_extract(traits, '$.wanted_niche') AS wanted_niche FROM organisms WHERE run_id=?")
        rows = self.rows(sql + (" AND node=?" if node else "") + " ORDER BY created DESC", *([run, node] if node else [run]))
        for r in rows:
            ev = evals.get(r["id"])
            r["score"] = ev["score"] if ev and ev["viable"] else None
            r["samples"], r["scores"] = (ev or {}).get("samples", 1), (ev or {}).get("scores")
            art = (ev or {}).get("artifacts") or {}
            r["painting"], r["pair"] = art.get("painting"), art.get("pair")  # the pair image holds the target too
        return rows

    # ---- one organism, one session ------------------------------------------------------------------------

    def organism(self, oid: str) -> dict:
        o = self.one("SELECT * FROM organisms WHERE id=?", oid)
        if o is None:
            raise KeyError(oid)
        parent = self.one("SELECT id, text, niche, summary FROM organisms WHERE id=?", o["parent_id"]) if o["parent_id"] else None
        parent2 = self.one("SELECT id, niche, summary FROM organisms WHERE id=?", o["parent2_id"]) if o["parent2_id"] else None
        evals = self.rows("SELECT * FROM evaluations WHERE organism_id=? ORDER BY ended DESC", oid)
        for ev in evals:
            p = self.one("SELECT id, node, niche, summary FROM organisms WHERE id=?", ev["partner_id"]) if ev["partner_id"] else None
            ev["partner"] = p
        children = self.rows("SELECT id, mutator, niche, summary, viable FROM organisms WHERE parent_id=? OR parent2_id=?", oid, oid)
        writer = self.one("SELECT id, purpose, cost, status, num_turns FROM sessions WHERE id=?", o["session_id"]) if o["session_id"] else None
        return {**o, "parent": parent, "parent2": parent2, "evaluations": evals, "children": children, "writer": writer}

    def session(self, sid: str) -> dict:
        s = self.one("SELECT * FROM sessions WHERE id=?", sid)
        if s is None:
            raise KeyError(sid)
        events = self.rows("SELECT idx, ts, kind, data FROM session_events WHERE session_id=? ORDER BY idx", sid)
        strokes = self.rows("SELECT idx, ts, data, snapshot FROM strokes WHERE session_id=? ORDER BY idx", sid)
        organism = self.one("SELECT id, node, niche FROM organisms WHERE id=?", s["organism_id"]) if s["organism_id"] else None
        has_requests = self.one("SELECT name FROM sqlite_master WHERE type='table' AND name='painting_requests'")
        is_request = has_requests and self.one("SELECT id FROM painting_requests WHERE session_id=?", sid)
        if s["status"] == "running" and (time.time() - (s["last_ts"] or s["started"]) > STALE_SECONDS
                                         or (not is_request and self.last(s["run_id"], "run_finished") is not None)):
            s["status"] = "interrupted"
        return {**s, "events": events, "strokes": strokes, "organism": organism, "now": time.time()}

    def studio(self, run: str) -> dict:
        meta = self.one("SELECT * FROM runs WHERE id=?", run)
        if meta is None:
            raise KeyError(run)
        organisms = self.rows("SELECT * FROM organisms WHERE run_id=? ORDER BY created", run)
        for org in organisms:
            org["tools"], org["views"] = [], []
            if org["node"] == "instrument":
                # Inspect source without executing an evolved instrument in the HTTP server.
                try:
                    tree = ast.parse(org["genome"].get("source", ""))
                    for statement in tree.body:
                        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Dict):
                            for target in statement.targets:
                                if isinstance(target, ast.Name) and target.id in ("TOOLS", "VIEWS"):
                                    key = "tools" if target.id == "TOOLS" else "views"
                                    org[key] = [k.value for k in statement.value.keys
                                                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                except (SyntaxError, ValueError):
                    pass
        by_id = {o["id"]: o for o in organisms}
        paintings = {}
        for ev in self.rows("SELECT * FROM evaluations WHERE run_id=? ORDER BY ended", run):
            details = ev.get("details") or {}
            instrument = details.get("instrument_id")
            prompt = details.get("prompt_id")
            if not instrument:
                org = by_id.get(ev["organism_id"], {})
                instrument = ev["organism_id"] if org.get("node") == "instrument" else ev["partner_id"]
                prompt = ev["partner_id"] if org.get("node") == "instrument" else ev["organism_id"]
            # The same session is evaluated for both organisms, and again on rescore. Show its painting once.
            key = ev["session_id"] or (instrument, prompt, (ev.get("artifacts") or {}).get("painting"),
                                       details.get("sample", ev["id"]))
            paintings[key] = {"id": ev["session_id"] or ev["id"], "session_id": ev["session_id"],
                              "instrument_id": instrument, "prompt_id": prompt, "created": ev["started"],
                              "status": "finished" if ev["viable"] else "failed", "score": ev["score"],
                              "artifacts": ev.get("artifacts") or {}, "details": details, "error": ev["error"]}
        has_requests = self.one("SELECT name FROM sqlite_master WHERE type='table' AND name='painting_requests'")
        requests = self.rows("SELECT * FROM painting_requests WHERE run_id=? ORDER BY created", run) if has_requests else []
        request_sessions = {r["session_id"] for r in requests}
        starts = {r["data"]["session_id"]: r["data"] for r in self.rows(
            "SELECT data FROM events WHERE run_id=? AND kind='painting_started'", run)}
        for session in self.rows("SELECT id, organism_id, started, status, last_ts, error FROM sessions "
                                 "WHERE run_id=? AND purpose LIKE 'paint%' ORDER BY started", run):
            sid = session["id"]
            if sid not in paintings:
                pair = starts.get(sid, {})
                status = session["status"]
                if status == "running" and (time.time() - (session["last_ts"] or session["started"]) > STALE_SECONDS
                                             or (sid not in request_sessions and self.last(run, "run_finished") is not None)):
                    status = "interrupted"
                paintings[sid] = {"id": sid, "session_id": sid, "created": session["started"], "status": status,
                                  "instrument_id": pair.get("instrument_id") or session["organism_id"],
                                  "prompt_id": pair.get("prompt_id"), "score": None, "artifacts": {},
                                  "details": {}, "error": session["error"]}
            snapshot = self.one("SELECT snapshot FROM strokes WHERE session_id=? AND snapshot IS NOT NULL "
                                "ORDER BY idx DESC LIMIT 1", sid)
            if snapshot and not paintings[sid]["artifacts"].get("painting"):
                paintings[sid]["artifacts"]["painting"] = snapshot["snapshot"]
        for request in requests:
            prior = paintings.pop(request["session_id"], {})
            artifacts = request.get("artifacts") or prior.get("artifacts") or {}
            details = request.get("details") or {}
            status = request["status"]
            if status == "running" and prior.get("status") == "interrupted":
                status = "interrupted"
            paintings[request["id"]] = {**request, "status": status, "artifacts": artifacts, "details": details,
                                        "score": details.get("score"), "request": True}
        settings = {r["node"]: r["data"] for r in self.rows(
            "SELECT node, data FROM events WHERE run_id=? AND kind='role_settings' ORDER BY seq", run)}
        return {"run": meta, "organisms": organisms,
                "paintings": sorted(paintings.values(), key=lambda p: p.get("created") or 0), "settings": settings}

    def artifact(self, name: str) -> bytes | None:
        r = self.conn.execute("SELECT data FROM artifacts WHERE name=?", (name,)).fetchone()
        return r[0] if r else None

    def canvas_get(self, parts: list[str], q: dict) -> tuple[Any, str] | None:
        """The shared canvas's reads, under /api/canvases and /api/canvas-catalog. (body, content type), or None."""
        from conveyor.commons import catalog
        from conveyor.commons import views as canvas_views

        if parts == ["api", "canvas-catalog"]:
            return catalog.catalog(self.conn), "json"
        if parts[:2] != ["api", "canvases"]:
            return None
        if len(parts) == 2:
            return canvas_views.canvases(self.conn), "json"
        cid = parts[2]
        if len(parts) == 3:
            return canvas_views.canvas(self.conn, cid, int(q.get("since") or 0)), "json"
        if parts[3:] == ["history"]:
            return canvas_views.history(self.conn, cid), "json"
        if len(parts) == 5 and parts[3] == "ops" and parts[4].isdigit():
            return canvas_views.op(self.conn, cid, int(parts[4])), "json"
        if len(parts) == 7 and parts[3] == "tiles" and all(re.fullmatch(r"-?\d+", p) for p in parts[4:]):
            data = canvas_views.tile(self.conn, cid, *map(int, parts[4:]))
            if data is None:
                raise KeyError("tile")
            return data, "png"
        return None


def _hostname(netloc: str) -> str:
    return netloc.rsplit(":", 1)[0] if not netloc.endswith("]") else netloc


def make_server(db: str | Path, host: str = "127.0.0.1", port: int = 8765, launcher=None) -> ThreadingHTTPServer:
    views = Views(Path(db))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=31536000, immutable" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, status: int = 200) -> None:
            self._send(status, json.dumps(value, default=str).encode(), "application/json")

        def _authorized(self) -> bool:
            return launcher is not None and launcher.authorized(self.headers.get("X-Conveyor-Token"))

        def _guard_post(self) -> str | None:
            """Why this POST can't start or stop a run, or None. A page on another site can make a browser send a
            request to 127.0.0.1, so a POST must come from this server's own pages: a loopback Host (a rebound DNS
            name fails here), no foreign Origin, and a JSON body (which a cross-site form can't send). With a token
            the Host can be anything, since a page that doesn't know the token can't send it."""
            if not launcher.token and _hostname(self.headers.get("Host", "")) not in LOCAL_HOSTS:
                return "Host not allowed"
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return "Origin not allowed"
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return "Send application/json"
            return None

        def do_POST(self) -> None:  # noqa: N802
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            try:
                painting_post = len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "paintings"
                canvas_post = parts[:2] == ["api", "canvases"]
                target_post = parts == ["api", "targets"]
                if launcher is None or (parts[:2] != ["api", "launches"] and not painting_post and not canvas_post
                                        and not target_post):
                    return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                if (why := self._guard_post()) is not None:
                    return self._json({"error": why}, HTTPStatus.FORBIDDEN)
                if not self._authorized():
                    return self._json({"error": "launch token needed"}, HTTPStatus.UNAUTHORIZED)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return self._json({"error": "invalid content length"}, HTTPStatus.BAD_REQUEST)
                if length < 0:
                    return self._json({"error": "invalid content length"}, HTTPStatus.BAD_REQUEST)
                if length > (MAX_PAINTING_BODY if painting_post or target_post else MAX_BODY):
                    return self._json({"error": "body too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return self._json({"error": "body isn't JSON"}, HTTPStatus.BAD_REQUEST)
                if target_post:
                    try:
                        self._json(launcher.import_target(body), HTTPStatus.CREATED)
                    except LaunchError as e:
                        self._json({"error": str(e)}, HTTPStatus.UNPROCESSABLE_ENTITY)
                elif canvas_post:
                    try:
                        if len(parts) == 2:
                            self._json(launcher.create_canvas(body), HTTPStatus.CREATED)
                        elif len(parts) == 4 and parts[3] == "agents":
                            self._json(launcher.start_agent(parts[2], body), HTTPStatus.CREATED)
                        elif parts[3:] in (["sketches"], ["sketches", "erase"]):
                            done = (launcher.add_sketch if len(parts) == 4 else launcher.erase_sketch)(parts[2], body)
                            self._json(done or {"error": "not found"},
                                       (HTTPStatus.CREATED if len(parts) == 4 else HTTPStatus.OK) if done
                                       else HTTPStatus.NOT_FOUND)
                        elif len(parts) == 6 and parts[3] == "agents" and parts[5] == "stop":
                            agent = launcher.stop_agent(parts[2], parts[4])
                            self._json(agent or {"error": "not found"}, HTTPStatus.OK if agent else HTTPStatus.NOT_FOUND)
                        else:
                            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    except LaunchError as e:
                        self._json({"error": str(e)}, HTTPStatus.UNPROCESSABLE_ENTITY)
                elif painting_post or len(parts) == 2:
                    try:
                        self._json(launcher.start_painting(parts[2], body) if painting_post else launcher.start(body),
                                   HTTPStatus.CREATED)
                    except LaunchError as e:
                        self._json({"error": str(e)}, HTTPStatus.UNPROCESSABLE_ENTITY)
                elif len(parts) == 4 and parts[3] == "stop":
                    launch = launcher.stop(parts[2])
                    self._json(launch or {"error": "not found"}, HTTPStatus.OK if launch else HTTPStatus.NOT_FOUND)
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if not parts:
                    self._send(200, DASHBOARD.read_bytes(), "text/html; charset=utf-8")
                elif parts == ["studio"]:
                    self._send(200, STUDIO.read_bytes(), "text/html; charset=utf-8")
                elif parts == ["canvas"]:
                    self._send(200, COMMONS.read_bytes(), "text/html; charset=utf-8")
                elif (found := views.canvas_get(parts, q)) is not None:
                    body, kind = found
                    if kind == "png":
                        self._send(200, body, "image/png", cache=True)
                    else:
                        self._json(body)
                elif parts[0] == "artifacts" and len(parts) == 2 and ARTIFACT_RE.match(parts[1]):
                    data = views.artifact(parts[1])
                    if data is None:
                        self._send(404, b"not found", "text/plain")
                    else:
                        self._send(200, data, "image/png" if parts[1].endswith("png") else "image/jpeg", cache=True)
                elif parts == ["api", "options"]:
                    if launcher is None:
                        self._json({"launch": False})
                    else:
                        self._json(launcher.options() if self._authorized() else {"launch": True, "locked": True})
                elif parts[:2] == ["api", "launches"] and launcher is not None and not self._authorized():
                    self._json({"error": "launch token needed"}, HTTPStatus.UNAUTHORIZED)
                elif parts[:2] == ["api", "launches"] and launcher is not None and len(parts) == 2:
                    self._json(launcher.list())
                elif parts[:2] == ["api", "launches"] and launcher is not None and len(parts) == 3:
                    launch = launcher.get(parts[2])
                    self._json(launch or {"error": "not found"}, HTTPStatus.OK if launch else HTTPStatus.NOT_FOUND)
                elif parts[:2] == ["api", "runs"] and len(parts) == 2:
                    self._json(views.runs())
                elif parts[:2] == ["api", "runs"] and len(parts) == 3:
                    self._json(views.overview(parts[2]))
                elif parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "studio":
                    self._json(views.studio(parts[2]))
                elif parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "history":
                    self._json(views.history(parts[2]))
                elif parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "organisms":
                    self._json(views.organisms(parts[2], q.get("node")))
                elif parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "mutators":
                    self._json(views.mutator_stats(parts[2]))
                elif parts[:2] == ["api", "organisms"] and len(parts) == 3:
                    self._json(views.organism(parts[2]))
                elif parts[:2] == ["api", "sessions"] and len(parts) == 3:
                    self._json(views.session(parts[2]))
                else:
                    self._send(404, b"not found", "text/plain")
            except KeyError as e:
                self._json({"error": f"not found: {e}"}, HTTPStatus.NOT_FOUND)
            except ValueError as e:
                self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
            except sqlite3.OperationalError as e:  # the run hasn't created the tables yet, or the file is busy
                self._json({"error": str(e)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except BrokenPipeError:
                pass

    return ThreadingHTTPServer((host, port), Handler)
