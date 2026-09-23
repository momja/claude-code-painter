"""
Read-only HTTP server for the dashboard. Standard library only.

Every view is computed from the event log on request, so the dashboard works the same during a run and
after it. Endpoints:

  GET /                                   dashboard
  GET /api/runs                           runs in the database
  GET /api/runs/<run>                     graph, node status, champions, blame split, alarms
  GET /api/runs/<run>/nodes/<node>        score percentiles, failure mix, mutator yield, lineage, learning log
  GET /api/runs/<run>/organisms/<id>      genome text, parent diff source, evaluations, failures, children
  GET /api/runs/<run>/calls               model calls, newest first (?node= &organism= &trace= &limit=)
  GET /api/runs/<run>/live                model calls still running, with the tail of their thinking
  GET /api/calls/<call>                   one model call in full: request, thinking, reply, tool calls, outcomes
  GET /api/traces/<trace>                 tool-call spans for one evaluation
  GET /artifacts/<name>                   content-addressed images
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
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

from conveyor.events import connect

DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"
ARTIFACT_RE = re.compile(r"^[0-9a-f]{24}\.(png|jpg|jpeg|webp)$")
CONTENT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}

STALL_ITERATIONS = 8
LOW_YIELD_MIN_TRIES = 12
LOW_YIELD_RATE = 0.05
LOW_VERIFY_MIN_TRIES = 10
LOW_VERIFY_RATE = 0.10
NONVIABLE_RATE = 0.30
RESCORE_DROP = 0.03
MIN_RESCORE_DROPS = 2
BLAME_WINDOW = 12


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    if "data" in d and isinstance(d["data"], str):
        d["data"] = json.loads(d["data"])
    return d


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = connect(self.db_path, readonly=True)
        return c

    def q(self, sql: str, *args: Any) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def rows(self, sql: str, *args: Any) -> list[dict]:
        return [_row(r) for r in self.q(sql, *args)]

    def one(self, sql: str, *args: Any) -> dict | None:
        r = self.conn.execute(sql, args).fetchone()
        return _row(r) if r else None

    def last_event(self, run: str, kind: str, node: str | None = None) -> dict | None:
        if node is None:
            return self.one(
                "SELECT * FROM events WHERE run_id=? AND kind=? ORDER BY seq DESC LIMIT 1", run, kind
            )
        return self.one(
            "SELECT * FROM events WHERE run_id=? AND kind=? AND node=? ORDER BY seq DESC LIMIT 1", run, kind, node
        )

    def latest_evaluation(self, run: str, organism_id: str) -> dict | None:
        return self.one(
            "SELECT * FROM events WHERE run_id=? AND kind='evaluated' AND organism_id=? ORDER BY seq DESC LIMIT 1",
            run,
            organism_id,
        )

    # ---- runs -------------------------------------------------------------------------------------------

    def runs(self) -> list[dict]:
        out = []
        for r in self.rows("SELECT id, name, started, config FROM runs ORDER BY started DESC"):
            stats = self.one("SELECT max(ts) AS last_ts, count(*) AS n FROM events WHERE run_id=?", r["id"])
            cycle = self.last_event(r["id"], "cycle")
            finished = self.last_event(r["id"], "run_finished")
            out.append(
                dict(
                    id=r["id"],
                    name=r["name"],
                    started=r["started"],
                    config=json.loads(r["config"]),
                    last_ts=stats["last_ts"],
                    n_events=stats["n"],
                    cycle=cycle["data"] if cycle else None,
                    finished=bool(finished),
                )
            )
        return out

    # ---- overview ---------------------------------------------------------------------------------------

    def overview(self, run: str) -> dict:
        meta = self.one("SELECT id, name, started, config FROM runs WHERE id=?", run)
        if meta is None:
            raise KeyError(run)
        meta["config"] = json.loads(meta["config"])
        graph = self.last_event(run, "graph")
        last_seq = self.one("SELECT max(seq) AS s FROM events WHERE run_id=?", run)["s"]
        cycle = self.last_event(run, "cycle")
        finished = self.last_event(run, "run_finished")
        base = dict(
            run=meta,
            last_seq=last_seq,
            cycle=cycle["data"] if cycle else None,
            finished=finished["data"] if finished else None,
        )
        llm, llm_total = self._llm_totals(run)
        comparison = [dict(node=r["node"], **r["data"]) for r in self.rows(
            "SELECT node, data FROM events WHERE run_id=? AND kind='comparison' ORDER BY seq", run)]
        for c in comparison:
            first = self.one("SELECT id FROM llm_calls WHERE trace_id=? ORDER BY started LIMIT 1", c.get("trace_id"))
            c["first_call"] = first["id"] if first else None
        base.update(llm_total=llm_total, comparison=comparison)
        if graph is None:
            return {**base, "graph": None, "alarms": [], "blame": None}

        nodes = graph["data"]["nodes"]
        status = {
            r["node"]: r["data"]
            for r in self.rows(
                "SELECT e.node, e.data FROM events e JOIN (SELECT node, max(seq) AS s FROM events "
                "WHERE run_id=? AND kind='node_status' GROUP BY node) m ON e.seq = m.s",
                run,
            )
        }
        champions = {
            r["node"]: r
            for r in self.rows(
                "SELECT e.* FROM events e JOIN (SELECT node, max(seq) AS s FROM events "
                "WHERE run_id=? AND kind='champion_changed' GROUP BY node) m ON e.seq = m.s",
                run,
            )
        }
        iterations = {
            r["node"]: dict(r)
            for r in self.q(
                "SELECT node, max(iteration) AS last, count(*) AS n FROM events "
                "WHERE run_id=? AND kind='iteration' GROUP BY node",
                run,
            )
        }
        evals = {
            r["node"]: dict(r)
            for r in self.q(
                "SELECT node, count(*) AS n, sum(json_extract(data, '$.duration')) AS seconds FROM events "
                "WHERE run_id=? AND kind='evaluated' GROUP BY node",
                run,
            )
        }
        total_evals = sum(e["n"] for e in evals.values())

        def champion_info(name: str, artifact_key: str) -> tuple[dict | None, str | None]:
            ch = champions.get(name)
            if ch is None:
                return None, None
            ev = self.latest_evaluation(run, ch["organism_id"])
            info = dict(
                id=ch["organism_id"],
                since_iteration=ch["iteration"],
                score=ev["data"]["score"] if ev else ch["data"]["new_score"],
                sub_scores=ev["data"].get("sub_scores", {}) if ev else {},
            )
            thumb = ev["data"].get("artifacts", {}).get(artifact_key) if ev else None
            return info, thumb

        out_nodes = []
        for n in nodes:
            name = n["name"]
            info = dict(n)
            info["status"] = status.get(name, {}).get("status", "idle" if not n["fixed"] else "fixed")
            info["status_detail"] = status.get(name, {})
            if n["fixed"]:
                info["evaluations"] = total_evals
                info["champion"], info["thumbnail"] = (
                    champion_info(n["mirror"], n["thumbnail_artifact"]) if n.get("mirror") else (None, None)
                )
            else:
                info["iterations"] = iterations.get(name, {}).get("last") or 0
                info["evaluations"] = evals.get(name, {}).get("n", 0)
                info["eval_seconds"] = evals.get(name, {}).get("seconds") or 0.0
                info["champion"], info["thumbnail"] = champion_info(name, n["thumbnail_artifact"])
                info["llm"] = llm.get(name)
            out_nodes.append(info)

        return {
            **base,
            "graph": {"nodes": out_nodes, "edges": graph["data"]["edges"], "schedule": graph["data"]["schedule"]},
            "blame": self.blame(run),
            "alarms": self.alarms(run, nodes, iterations),
        }

    def _llm_totals(self, run: str) -> tuple[dict[str, dict], dict | None]:
        """Model-call totals per node and for the whole run, from the llm_call events."""
        rows = [
            dict(r)
            for r in self.q(
                "SELECT node, count(*) AS calls, coalesce(sum(json_extract(data, '$.cost')), 0) AS cost, "
                "coalesce(sum(json_extract(data, '$.prompt_tokens')), 0) AS tokens_in, "
                "coalesce(sum(json_extract(data, '$.completion_tokens')), 0) AS tokens_out, "
                "coalesce(sum(json_extract(data, '$.cached_tokens')), 0) AS tokens_cached, "
                "sum(json_extract(data, '$.error') IS NOT NULL) AS errors "
                "FROM events WHERE run_id=? AND kind='llm_call' GROUP BY node",
                run,
            )
        ]
        keys = ("calls", "cost", "tokens_in", "tokens_out", "tokens_cached", "errors")
        total = {k: sum(r[k] for r in rows) for k in keys} if rows else None
        return {r["node"]: r for r in rows}, total

    def blame(self, run: str) -> dict:
        """Rolling average of how many patches each component was blamed for, across all evaluations."""
        rows = self.q(
            "SELECT json_extract(data, '$.sub_scores') AS s FROM events WHERE run_id=? AND kind='evaluated' "
            "AND json_extract(data, '$.viable') ORDER BY seq",
            run,
        )
        samples = []
        for r in rows:
            subs = json.loads(r["s"] or "{}")
            b = {k[len("blame_") :]: v for k, v in subs.items() if k.startswith("blame_")}
            if b:
                samples.append(b)
        keys = sorted({k for s in samples for k in s})
        points = []
        for start in range(0, len(samples), BLAME_WINDOW):
            window = samples[start : start + BLAME_WINDOW]
            points.append({k: sum(s.get(k, 0) for s in window) / len(window) for k in keys} | {"n": start + len(window)})
        return {"keys": keys, "points": points[-60:], "window": BLAME_WINDOW}

    # ---- node detail ------------------------------------------------------------------------------------

    def _evals_by_organism(self, run: str, node: str) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        for r in self.q(
            "SELECT seq, organism_id, json_extract(data, '$.score') AS s, json_extract(data, '$.viable') AS v, "
            "json_extract(data, '$.reason') AS reason, json_extract(data, '$.partners') AS p "
            "FROM events WHERE run_id=? AND node=? AND kind='evaluated' ORDER BY seq",
            run,
            node,
        ):
            out[r["organism_id"]].append(
                dict(seq=r["seq"], score=r["s"], viable=bool(r["v"]), reason=r["reason"], partners=r["p"])
            )
        return out

    @staticmethod
    def _compare_to_parent(evals: dict[str, list[dict]], child_id: str, parent_id: str | None) -> tuple[str, float | None]:
        """
        Child score minus parent score, but only when both were measured against the same partner versions.

        A child is always scored against the partners' current champions. Its parent's stored score may come from
        older, weaker partners, and comparing the two would credit the mutator for the partner's progress.
        """
        child = next((e for e in evals.get(child_id, []) if e["reason"] == "new"), None)
        if child is None:
            return "pending", None
        if not child["viable"]:
            return "nonviable", None
        prior = [e for e in evals.get(parent_id or "", []) if e["seq"] < child["seq"] and e["viable"]]
        if not prior:
            return "no_parent", None
        if prior[-1]["partners"] != child["partners"]:
            return "partner_changed", None
        return "ok", child["score"] - prior[-1]["score"]

    def mutator_table(self, run: str, node: str) -> list[dict]:
        orgs = self.q(
            "SELECT organism_id, json_extract(data, '$.mutator') AS m, json_extract(data, '$.parent_id') AS pid "
            "FROM events WHERE run_id=? AND node=? AND kind='organism' AND json_extract(data, '$.mutator') IS NOT NULL",
            run,
            node,
        )
        evals = self._evals_by_organism(run, node)
        verified = {
            r["organism_id"]: bool(r["p"])
            for r in self.q(
                "SELECT organism_id, json_extract(data, '$.passed') AS p FROM events "
                "WHERE run_id=? AND node=? AND kind='verified'",
                run,
                node,
            )
        }
        calls = {
            r["m"]: dict(r)
            for r in self.q(
                "SELECT json_extract(data, '$.mutator') AS m, count(*) AS calls, "
                "sum(json_extract(data, '$.error') IS NOT NULL) AS errors, "
                "sum(json_extract(data, '$.n_children')) AS children, "
                "sum(json_extract(data, '$.duration')) AS seconds "
                "FROM events WHERE run_id=? AND node=? AND kind='mutate_call' GROUP BY m",
                run,
                node,
            )
        }
        model = {
            r["m"]: dict(r)
            for r in self.q(
                "SELECT json_extract(data, '$.mutator') AS m, count(*) AS calls, "
                "coalesce(sum(json_extract(data, '$.cost')), 0) AS cost FROM events "
                "WHERE run_id=? AND node=? AND kind='llm_call' AND json_extract(data, '$.mutator') IS NOT NULL GROUP BY m",
                run,
                node,
            )
        }
        table: dict[str, dict] = defaultdict(
            lambda: dict(
                evaluated=0, nonviable=0, comparable=0, partner_changed=0, improved=0, deltas=[],
                verify_pass=0, verify_total=0,
            )
        )
        for r in orgs:
            t = table[r["m"]]
            oid = r["organism_id"]
            if oid in verified:
                t["verify_total"] += 1
                t["verify_pass"] += verified[oid]
            status, delta = self._compare_to_parent(evals, oid, r["pid"])
            if status == "pending":
                continue
            t["evaluated"] += 1
            if status == "nonviable":
                t["nonviable"] += 1
            elif status == "partner_changed":
                t["partner_changed"] += 1
            elif status == "ok":
                t["comparable"] += 1
                t["deltas"].append(delta)
                if delta > 1e-9:
                    t["improved"] += 1
        out = []
        for name in sorted(set(table) | set(calls)):
            t = table[name]
            c = calls.get(name, {})
            deltas = t.pop("deltas")
            out.append(
                dict(
                    mutator=name,
                    calls=c.get("calls", 0),
                    errors=c.get("errors", 0),
                    children=c.get("children") or 0,
                    seconds=c.get("seconds") or 0.0,
                    model_calls=model.get(name, {}).get("calls", 0),
                    model_cost=model.get(name, {}).get("cost", 0.0),
                    **t,
                    mean_delta=sum(deltas) / len(deltas) if deltas else None,
                    best_delta=max(deltas) if deltas else None,
                )
            )
        return out

    def node_detail(self, run: str, node: str) -> dict:
        iterations = [
            dict(iteration=r["iteration"], **r["data"])
            for r in self.rows(
                "SELECT iteration, data FROM events WHERE run_id=? AND node=? AND kind='iteration' ORDER BY seq",
                run,
                node,
            )
        ]
        initial = self.one(
            "SELECT * FROM events WHERE run_id=? AND node=? AND kind='champion_changed' ORDER BY seq LIMIT 1",
            run,
            node,
        )
        if initial:
            s = initial["data"]["new_score"]
            iterations.insert(
                0,
                dict(iteration=0, percentiles={k: s for k in ("0", "25", "50", "75", "90", "100")}, best_score=s),
            )

        mix: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self.q(
            "SELECT iteration, json_extract(data, '$.failure_counts') AS fc FROM events WHERE run_id=? AND node=? "
            "AND kind='evaluated' AND json_extract(data, '$.reason') IN ('new', 'initial') ORDER BY seq",
            run,
            node,
        ):
            for k, v in json.loads(r["fc"] or "{}").items():
                mix[r["iteration"] or 0][k] += v
        types = sorted({k for m in mix.values() for k in m})
        failure_mix = {
            "types": types,
            "points": [{"iteration": i, **{t: mix[i].get(t, 0) for t in types}} for i in sorted(mix)],
        }

        latest: dict[str, dict] = {}
        for r in self.q(
            "SELECT organism_id, json_extract(data, '$.score') AS s, json_extract(data, '$.viable') AS v, "
            "json_extract(data, '$.reason') AS reason FROM events WHERE run_id=? AND node=? AND kind='evaluated' "
            "ORDER BY seq",
            run,
            node,
        ):
            prev = latest.get(r["organism_id"])
            latest[r["organism_id"]] = dict(
                score=r["s"],
                viable=bool(r["v"]),
                first_score=prev["first_score"] if prev else r["s"],
                n_evals=(prev["n_evals"] + 1) if prev else 1,
            )
        verified = {
            r["organism_id"]: bool(r["p"])
            for r in self.q(
                "SELECT organism_id, json_extract(data, '$.passed') AS p FROM events "
                "WHERE run_id=? AND node=? AND kind='verified'",
                run,
                node,
            )
        }
        evals = self._evals_by_organism(run, node)
        lineage = []
        feed = []
        for r in self.q(
            "SELECT seq, organism_id, iteration, json_extract(data, '$.parent_id') AS parent_id, "
            "json_extract(data, '$.additional_parent_ids') AS extra, json_extract(data, '$.generation') AS gen, "
            "json_extract(data, '$.mutator') AS mutator, json_extract(data, '$.change_summary') AS summary, "
            "json_extract(data, '$.parent_score') AS parent_score, json_extract(data, '$.failure_type') AS ftype "
            "FROM events WHERE run_id=? AND node=? AND kind='organism' ORDER BY seq",
            run,
            node,
        ):
            oid = r["organism_id"]
            ev = latest.get(oid)
            if ev is not None:
                lineage.append(
                    dict(
                        id=oid,
                        parent_id=r["parent_id"],
                        additional_parent_ids=json.loads(r["extra"] or "[]"),
                        generation=r["gen"],
                        iteration=r["iteration"],
                        mutator=r["mutator"],
                        **ev,
                    )
                )
            if r["summary"]:
                compare, delta = self._compare_to_parent(evals, oid, r["parent_id"])
                feed.append(
                    dict(
                        id=oid,
                        seq=r["seq"],
                        iteration=r["iteration"],
                        mutator=r["mutator"],
                        failure_type=r["ftype"],
                        summary=r["summary"],
                        parent_id=r["parent_id"],
                        score=ev["first_score"] if ev else None,
                        viable=ev["viable"] if ev else None,
                        verified=verified.get(oid),
                        compare=compare,
                        delta=delta,
                    )
                )

        champion = self.last_event(run, "champion_changed", node)
        champions = [
            dict(id=r["organism_id"], iteration=r["iteration"], **r["data"])
            for r in self.rows(
                "SELECT organism_id, iteration, data FROM events WHERE run_id=? AND node=? AND kind='champion_changed' "
                "ORDER BY seq",
                run,
                node,
            )
        ]
        rescores = [
            dict(id=r["organism_id"], iteration=r["iteration"], **r["data"])
            for r in self.rows(
                "SELECT organism_id, iteration, data FROM events WHERE run_id=? AND node=? AND kind='rescored' "
                "ORDER BY seq DESC LIMIT 40",
                run,
                node,
            )
        ]
        return dict(
            node=node,
            iterations=iterations,
            failure_mix=failure_mix,
            mutators=self.mutator_table(run, node),
            lineage=lineage,
            champion_id=champion["organism_id"] if champion else None,
            champions=champions,
            feed=list(reversed(feed[-60:])),
            rescores=rescores,
        )

    # ---- organism ---------------------------------------------------------------------------------------

    def organism(self, run: str, oid: str) -> dict:
        org = self.one(
            "SELECT * FROM events WHERE run_id=? AND kind='organism' AND organism_id=? ORDER BY seq LIMIT 1", run, oid
        )
        if org is None:
            raise KeyError(oid)
        parent = None
        if org["data"].get("parent_id"):
            p = self.one(
                "SELECT * FROM events WHERE run_id=? AND kind='organism' AND organism_id=? ORDER BY seq LIMIT 1",
                run,
                org["data"]["parent_id"],
            )
            if p:
                pe = self.latest_evaluation(run, p["organism_id"])
                parent = dict(id=p["organism_id"], text=p["data"]["text"], score=pe["data"]["score"] if pe else None)
        evaluations = [
            dict(seq=r["seq"], iteration=r["iteration"], **r["data"])
            for r in self.rows(
                "SELECT seq, iteration, data FROM events WHERE run_id=? AND kind='evaluated' AND organism_id=? "
                "ORDER BY seq",
                run,
                oid,
            )
        ]
        holdouts = [
            dict(seq=r["seq"], iteration=r["iteration"], **r["data"])
            for r in self.rows(
                "SELECT seq, iteration, data FROM events WHERE run_id=? AND kind='holdout' AND organism_id=? "
                "ORDER BY seq",
                run,
                oid,
            )
        ]
        verified = self.one(
            "SELECT data FROM events WHERE run_id=? AND kind='verified' AND organism_id=? ORDER BY seq DESC LIMIT 1",
            run,
            oid,
        )
        children = []
        for r in self.q(
            "SELECT organism_id, json_extract(data, '$.mutator') AS m FROM events WHERE run_id=? AND kind='organism' "
            "AND json_extract(data, '$.parent_id')=? ORDER BY seq",
            run,
            oid,
        ):
            ev = self.latest_evaluation(run, r["organism_id"])
            children.append(
                dict(
                    id=r["organism_id"],
                    mutator=r["m"],
                    score=ev["data"]["score"] if ev else None,
                    viable=ev["data"]["viable"] if ev else None,
                )
            )
        champion = self.last_event(run, "champion_changed", org["node"])
        return dict(
            id=oid,
            node=org["node"],
            iteration=org["iteration"],
            is_champion=bool(champion and champion["organism_id"] == oid),
            **{k: org["data"].get(k) for k in ("parent_id", "additional_parent_ids", "generation", "mutator",
                                               "failure_type", "failure_ids", "change_summary", "parent_score",
                                               "genome", "text", "llm_call_ids")},
            parent=parent,
            evaluations=evaluations,
            holdouts=holdouts,
            verified=verified["data"]["passed"] if verified else None,
            children=children,
        )

    # ---- model calls ------------------------------------------------------------------------------------

    CALL_SUMMARY = (
        "id, node, mutator, organism_id, trace_id, purpose, model, status, started, first_token, ended, usage, "
        "finish_reason, error, substr(coalesce(reasoning, ''), 1, 280) AS reasoning_head, "
        "length(coalesce(reasoning, '')) AS reasoning_chars, "
        "coalesce(json_array_length(tool_calls), 0) AS n_tool_calls"
    )

    def calls(
        self, run: str, node: str | None = None, organism: str | None = None, trace: str | None = None, limit: int = 50
    ) -> list[dict]:
        where, args = ["run_id=?"], [run]
        for col, value in (("node", node), ("organism_id", organism), ("trace_id", trace)):
            if value:
                where.append(f"{col}=?")
                args.append(value)
        rows = self.q(
            f"SELECT {self.CALL_SUMMARY} FROM llm_calls WHERE {' AND '.join(where)} ORDER BY started DESC LIMIT ?",
            *args,
            max(1, min(limit, 500)),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["usage"] = json.loads(d["usage"] or "{}")
            out.append(d)
        return out

    def call(self, call_id: str) -> dict:
        r = self.conn.execute("SELECT * FROM llm_calls WHERE id=?", (call_id,)).fetchone()
        if r is None:
            raise KeyError(call_id)
        d = dict(r)
        for key in ("request", "tool_calls", "tool_results", "usage"):
            d[key] = json.loads(d[key]) if d.get(key) else None
        conversation = (d["request"] or {}).get("conversation")
        if conversation:
            d["conversation_calls"] = [
                dict(c)
                for c in self.q(
                    "SELECT id, json_extract(request, '$.turn') AS turn, status FROM llm_calls "
                    "WHERE run_id=? AND json_extract(request, '$.conversation')=? ORDER BY started",
                    d["run_id"],
                    conversation,
                )
            ]
        return d

    def live(self, run: str) -> dict:
        now = time.time()
        running = []
        for r in self.q(
            "SELECT id, node, mutator, organism_id, purpose, model, started, first_token, reasoning, content, "
            "coalesce(json_array_length(tool_calls), 0) AS n_tool_calls FROM llm_calls "
            "WHERE run_id=? AND status='running' AND started > ? ORDER BY started",
            run,
            now - 900,  # a row still "running" after 15 minutes belongs to a process that died
        ):
            d = dict(r)
            reasoning, content = d.pop("reasoning") or "", d.pop("content") or ""
            d.update(reasoning_tail=reasoning[-1500:], reasoning_chars=len(reasoning), content_tail=content[-400:])
            running.append(d)
        recent = [c for c in self.calls(run, limit=10) if c["status"] != "running"][:5]
        return {"now": now, "running": running, "recent": recent}

    def artifact(self, name: str) -> bytes | None:
        r = self.conn.execute("SELECT data FROM artifacts WHERE name=?", (name,)).fetchone()
        return r["data"] if r else None

    def trace(self, trace_id: str) -> dict:
        spans = []
        for r in self.q(
            "SELECT idx, name, args, result, duration, artifact FROM spans WHERE trace_id=? ORDER BY idx", trace_id
        ):
            spans.append(
                dict(
                    idx=r["idx"],
                    name=r["name"],
                    args=json.loads(r["args"] or "{}"),
                    result=json.loads(r["result"] or "null"),
                    duration=r["duration"],
                    artifact=r["artifact"],
                )
            )
        return {"trace_id": trace_id, "spans": spans}

    # ---- alarms -----------------------------------------------------------------------------------------

    def alarms(self, run: str, nodes: list[dict], iterations: dict[str, dict]) -> list[dict]:
        out: list[dict] = []
        for n in nodes:
            if n["fixed"]:
                continue
            name = n["name"]
            last_iter = iterations.get(name, {}).get("last") or 0

            finished = self.last_event(run, "iteration_error", name)
            if finished and (finished["iteration"] or 0) >= last_iter:
                out.append(
                    dict(
                        level="info",
                        node=name,
                        title=f"{name} has nothing left to fix",
                        detail="No organism has trainable failures, so the node stopped sampling parents. "
                        "Tighten the blame thresholds or add harder targets to keep it working.",
                    )
                )

            champ = self.last_event(run, "champion_changed", name)
            since = last_iter - (champ["iteration"] or 0) if champ else last_iter
            if since >= STALL_ITERATIONS:
                out.append(
                    dict(
                        level="warn",
                        node=name,
                        title=f"No new {name} champion for {since} iterations",
                        detail=f"Best score has held at {self._score_of(run, champ):.3f} since iteration "
                        f"{champ['iteration'] or 0}.",
                    )
                )

            for m in self.mutator_table(run, name):
                tried = m["comparable"]
                if tried >= LOW_YIELD_MIN_TRIES and m["improved"] / tried < LOW_YIELD_RATE:
                    out.append(
                        dict(
                            level="warn",
                            node=name,
                            title=f"{m['mutator']} rarely beats its parent",
                            detail=f"{m['improved']} of {tried} children scored higher than their parent did "
                            "against the same partners.",
                        )
                    )
                if m["verify_total"] >= LOW_VERIFY_MIN_TRIES and m["verify_pass"] / m["verify_total"] < LOW_VERIFY_RATE:
                    out.append(
                        dict(
                            level="warn",
                            node=name,
                            title=f"{m['mutator']} fails verification",
                            detail=f"{m['verify_pass']} of {m['verify_total']} proposals improved the patches "
                            "they were meant to fix.",
                        )
                    )
                if m["errors"] and m["errors"] / max(1, m["calls"]) > 0.2:
                    out.append(
                        dict(
                            level="warn",
                            node=name,
                            title=f"{m['mutator']} is throwing errors",
                            detail=f"{m['errors']} of {m['calls']} calls raised.",
                        )
                    )

            calls = self.one(
                "SELECT count(*) AS n, coalesce(sum(json_extract(data, '$.error') IS NOT NULL), 0) AS errors "
                "FROM events WHERE run_id=? AND node=? AND kind='llm_call'",
                run,
                name,
            )
            if calls["n"] >= 10 and calls["errors"] / calls["n"] > 0.1:
                latest = self.one(
                    "SELECT json_extract(data, '$.error') AS e FROM events WHERE run_id=? AND node=? AND kind='llm_call' "
                    "AND json_extract(data, '$.error') IS NOT NULL ORDER BY seq DESC LIMIT 1",
                    run,
                    name,
                )
                out.append(
                    dict(
                        level="alarm",
                        node=name,
                        title=f"Model calls are failing on {name}",
                        detail=f"{calls['errors']} of {calls['n']} calls failed. Latest: {(latest['e'] or '')[:240]}",
                    )
                )

            cut_off = self.one(
                "SELECT count(*) AS n, coalesce(sum(json_extract(data, '$.reasoning_tokens')), 0) AS thinking "
                "FROM events WHERE run_id=? AND node=? AND kind='llm_call' "
                "AND json_extract(data, '$.finish_reason')='length' AND json_extract(data, '$.n_tool_calls')=0",
                run,
                name,
            )
            if cut_off["n"] >= 3:
                out.append(
                    dict(
                        level="warn",
                        node=name,
                        title=f"The model keeps running out of room on {name}",
                        detail=f"{cut_off['n']} calls hit their token limit before making a single tool call, "
                        f"with {cut_off['thinking']:,} tokens spent thinking between them. Lower --reasoning or "
                        "raise max_tokens.",
                    )
                )

            last = self.last_event(run, "iteration", name)
            if last and last["data"].get("n_new", 0) >= 3:
                rate = last["data"]["n_nonviable"] / last["data"]["n_new"]
                if rate > NONVIABLE_RATE:
                    out.append(
                        dict(
                            level="warn",
                            node=name,
                            title=f"{last['data']['n_nonviable']} of {last['data']['n_new']} new {name} organisms "
                            "were not viable",
                            detail=f"Iteration {last['iteration']}. Open the lineage and look for hollow rings.",
                        )
                    )

            champs = self.q(
                "SELECT organism_id FROM events WHERE run_id=? AND node=? AND kind='champion_changed' "
                "ORDER BY seq DESC LIMIT 6",
                run,
                name,
            )
            # Held-out scores come from the checks the runner makes when an organism becomes champion.
            checked = [(c["organism_id"], self._holdout_score(run, c["organism_id"])) for c in champs]
            checked = [(oid, h) for oid, h in checked if h is not None]
            if len(champs) >= 3 and len(checked) >= 2:
                (new_id, hn), (old_id, ho) = checked[0], checked[-1]
                newest, oldest = self.latest_evaluation(run, new_id), self.latest_evaluation(run, old_id)
                if newest and oldest:
                    tn, to = newest["data"]["score"], oldest["data"]["score"]
                    if tn - to > 0.01 and hn - ho < -0.01:
                        out.append(
                            dict(
                                level="alarm",
                                node=name,
                                title=f"{name} is overfitting the training painting",
                                detail=f"Over the last {len(champs)} champions the training score went "
                                f"{to:.3f} to {tn:.3f} while the held-out score went {ho:.3f} to {hn:.3f}.",
                            )
                        )

            drops = self.rows(
                "SELECT organism_id, data FROM events WHERE run_id=? AND node=? AND kind='rescored' "
                "AND json_extract(data, '$.new_score') - json_extract(data, '$.old_score') <= ? "
                "ORDER BY seq DESC LIMIT 20",
                run,
                name,
                -RESCORE_DROP,
            )
            # One drop proves nothing: repainting varies on its own with an LLM painter, so wait for a pattern.
            if len(drops) >= MIN_RESCORE_DROPS:
                worst = min(drops, key=lambda r: r["data"]["new_score"] - r["data"]["old_score"])
                d = worst["data"]
                out.append(
                    dict(
                        level="alarm",
                        node=name,
                        title=f"{len(drops)} {name} organisms scored lower after {d['because']} changed",
                        detail=f"Each dropped by at least {RESCORE_DROP} when repainted. The worst went "
                        f"{d['old_score']:.3f} to {d['new_score']:.3f}. Repainting varies by itself, so check "
                        f"whether these organisms only worked with the old {d['because']} champion.",
                        organism_id=worst["organism_id"],
                    )
                )
        order = {"alarm": 0, "warn": 1, "info": 2}
        return sorted(out, key=lambda a: order[a["level"]])

    def _holdout_score(self, run: str, organism_id: str) -> float | None:
        r = self.one(
            "SELECT json_extract(data, '$.score') AS s FROM events WHERE run_id=? AND kind='holdout' "
            "AND organism_id=? ORDER BY seq DESC LIMIT 1",
            run,
            organism_id,
        )
        return r["s"] if r else None

    def _score_of(self, run: str, champ: dict | None) -> float:
        if champ is None:
            return 0.0
        ev = self.latest_evaluation(run, champ["organism_id"])
        return ev["data"]["score"] if ev else champ["data"]["new_score"]


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path = url.path
        try:
            if path in ("/", "/index.html"):
                return self._bytes(DASHBOARD.read_bytes(), "text/html; charset=utf-8", cache=False)
            if m := re.fullmatch(r"/artifacts/([^/]+)", path):
                name = m.group(1)
                if not ARTIFACT_RE.match(name):
                    return self._error(HTTPStatus.BAD_REQUEST, "bad artifact name")
                data = self.store.artifact(name)
                if data is None:
                    return self._error(HTTPStatus.NOT_FOUND, "artifact not found")
                return self._bytes(data, CONTENT_TYPES[name.rsplit(".", 1)[1]], cache=True)
            if path == "/api/runs":
                return self._json(self.store.runs())
            if m := re.fullmatch(r"/api/runs/([\w-]+)", path):
                return self._json(self.store.overview(m.group(1)))
            if m := re.fullmatch(r"/api/runs/([\w-]+)/nodes/([\w.-]+)", path):
                return self._json(self.store.node_detail(m.group(1), m.group(2)))
            if m := re.fullmatch(r"/api/runs/([\w-]+)/organisms/([\w-]+)", path):
                return self._json(self.store.organism(m.group(1), m.group(2)))
            if m := re.fullmatch(r"/api/runs/([\w-]+)/calls", path):
                query = parse_qs(url.query)

                def one(key: str) -> str | None:
                    return (query.get(key) or [None])[0]

                return self._json(
                    self.store.calls(
                        m.group(1), node=one("node"), organism=one("organism"), trace=one("trace"),
                        limit=int(one("limit") or 50),
                    )
                )
            if m := re.fullmatch(r"/api/runs/([\w-]+)/live", path):
                return self._json(self.store.live(m.group(1)))
            if m := re.fullmatch(r"/api/calls/([\w-]+)", path):
                return self._json(self.store.call(m.group(1)))
            if m := re.fullmatch(r"/api/traces/([\w-]+)", path):
                return self._json(self.store.trace(m.group(1)))
            return self._error(HTTPStatus.NOT_FOUND, f"no route for {path}")
        except KeyError as e:
            return self._error(HTTPStatus.NOT_FOUND, f"not found: {e}")
        except sqlite3.OperationalError as e:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, f"database not readable yet: {e}")

    def _json(self, obj: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _bytes(self, body: bytes, content_type: str, cache: bool) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A browser that navigates away mid-response is normal. Anything else still gets a traceback.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def make_server(db_path: str | Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    store = Store(db_path)
    handler = type("BoundHandler", (Handler,), {"store": store})
    return DashboardServer((host, port), handler)
