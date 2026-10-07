"""
The catalog an agent is spawned from: painters, each an instrument together with the prompt it painted with.

A prompt is evolved alongside an instrument and judged on paintings made with it, so the two travel as a pair.
Pairs come from what actually painted together (every evaluation and every studio request in every run), and only
each run's best are offered: the TOP_PER_RUN pairs with the highest painting scores in that run. A pair repainted
to confirm a score takes one place, not several, and an unscored painting (a text-only request) can't place. The
seed instruments with the starting strategy are offered only when no run has a scored painting, so a new database
still has painters.

Runs copy the same seed into their own organisms and a pair often paints many times, so a pair is keyed by a hash
of its two texts, `c-<hash>`: the same key in every run, stable as new runs are added. Each pair lists its best
score, how many paintings it made, its best painting, the runs it came from, and its place in each run whose top
it made.

Instrument source is read with `ast`, never run: the HTTP server lists the catalog, and evolved code only runs in
an agent's own processes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3

from conveyor.painting.prompts import INITIAL_STRATEGY
from conveyor.painting.seeds import SEEDS

MAX_ORIGINS = 12  # runs listed per pair; the count of paintings covers the rest
TOP_PER_RUN = 5  # pairs each run contributes: its best-scoring ones


def pair_key(source: str, prompt: str) -> str:
    return "c-" + hashlib.sha256((source + "\0" + prompt).encode()).hexdigest()[:12]


def tool_names(source: str) -> tuple[list[str], list[str]]:
    """The names in an instrument's TOOLS and VIEWS, read without running it."""
    tools: list[str] = []
    views: list[str] = []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return tools, views
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Dict):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in ("TOOLS", "VIEWS"):
                    names = [k.value for k in statement.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                    (tools if target.id == "TOOLS" else views).extend(names)
    return tools, views


def _first_line(text: str, limit: int = 140) -> str:
    try:
        doc = ast.get_docstring(ast.parse(text)) or ""
    except (SyntaxError, ValueError):
        doc = ""
    line = " ".join((doc or text).strip().split("\n\n")[0].split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _side(text: str, label: str, sheet: str | None = None, instrument: bool = False) -> dict:
    side = {"label": label or _first_line(text), "summary": _first_line(text), "text": text}
    if instrument:
        side["tools"], side["views"] = tool_names(text)
        side["sheet"] = sheet
    return side


def _obj(value: str | None) -> dict:
    """A JSON column that holds an object, or null when the row had none."""
    out = json.loads(value) if value else None
    return out if isinstance(out, dict) else {}


def catalog(conn: sqlite3.Connection) -> dict:
    pairs: dict[str, dict] = {}
    seed_labels = {source: f"seed: {name}" for name, source in SEEDS.items()}
    seed_labels[INITIAL_STRATEGY] = "starting strategy"

    def pair(source: str, prompt: str, labels: dict, sheets: dict) -> dict:
        k = pair_key(source, prompt)
        if k not in pairs:
            pairs[k] = {"id": k, "instrument": _side(source, labels.get(source, ""), sheets.get(source), True),
                        "prompt": _side(prompt, labels.get(prompt, "")), "best_score": None, "paintings": 0,
                        "painting": None, "last": 0.0, "origins": [], "ranks": [], "_seen": set(), "_runs": {}}
        pairs[k]["instrument"]["sheet"] = pairs[k]["instrument"]["sheet"] or sheets.get(source)
        return pairs[k]

    def painted(p: dict, key: str, score: float | None, artifact: str | None, when: float, origin: dict) -> None:
        if key in p["_seen"]:  # one painting is evaluated for both organisms, and again on rescore
            return
        p["_seen"].add(key)
        if score is not None:  # the pair's best score within each run, for that run's ranking
            p["_runs"][origin["run_id"]] = max(score, p["_runs"].get(origin["run_id"], score))
        p["paintings"] += 1
        better = score is not None and (p["best_score"] is None or score > p["best_score"])
        if artifact and (better or (p["best_score"] is None and when >= p["last"])):
            p["painting"] = artifact
        if better:
            p["best_score"] = score
        p["last"] = max(p["last"], when or 0.0)
        if len(p["origins"]) < MAX_ORIGINS and origin not in p["origins"]:
            p["origins"].append(origin)

    seeds = {pair(source, INITIAL_STRATEGY, seed_labels, {})["id"] for source in SEEDS.values()}
    run_names: dict[str, str] = {}

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if {"runs", "organisms", "evaluations"} <= tables:  # absent in a new file, or one from before the rewrite
        runs = run_names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM runs")}
        orgs: dict[str, tuple[str, str]] = {}  # organism id -> (node, text), viable ones only
        labels, sheets = dict(seed_labels), {}
        for row in conn.execute("SELECT id, node, genome, summary, sheet FROM organisms "
                                "WHERE viable IS NULL OR viable != 0 ORDER BY created"):
            try:
                genome = json.loads(row["genome"] or "{}")
            except json.JSONDecodeError:
                continue
            text = genome.get("source" if row["node"] == "instrument" else "prompt")
            if row["node"] not in ("instrument", "painter") or not isinstance(text, str) or not text.strip():
                continue
            orgs[row["id"]] = (row["node"], text)
            if not labels.get(text) and (row["summary"] or "").strip():
                labels[text] = row["summary"].strip()[:140]
            if row["node"] == "instrument" and row["sheet"]:
                sheets.setdefault(text, row["sheet"])

        def texts(instrument_id: str | None, prompt_id: str | None) -> tuple[str, str] | None:
            i, p = orgs.get(instrument_id or ""), orgs.get(prompt_id or "")
            return (i[1], p[1]) if i and p and i[0] == "instrument" and p[0] == "painter" else None

        for ev in conn.execute("SELECT id, run_id, organism_id, partner_id, score, details, artifacts, session_id, "
                               "ended FROM evaluations WHERE viable=1"):
            details = _obj(ev["details"])
            ids = (details.get("instrument_id"), details.get("prompt_id"))
            if not all(ids):
                ids = ((ev["organism_id"], ev["partner_id"]) if orgs.get(ev["organism_id"], ("",))[0] == "instrument"
                       else (ev["partner_id"], ev["organism_id"]))
            both = texts(*ids)
            if both:
                painted(pair(*both, labels, sheets), ev["session_id"] or ev["id"], ev["score"],
                        _obj(ev["artifacts"]).get("painting"), ev["ended"] or 0.0,
                        {"run_id": ev["run_id"], "run": runs.get(ev["run_id"], ev["run_id"])})
        if "painting_requests" in tables:
            for req in conn.execute("SELECT id, run_id, instrument_id, prompt_id, created, artifacts, details, session_id "
                                    "FROM painting_requests WHERE status='finished'"):
                both = texts(req["instrument_id"], req["prompt_id"])
                if both:
                    painted(pair(*both, labels, sheets), req["session_id"] or req["id"],
                            _obj(req["details"]).get("score"),
                            _obj(req["artifacts"]).get("painting"), req["created"] or 0.0,
                            {"run_id": req["run_id"], "run": runs.get(req["run_id"], req["run_id"])})
    by_run: dict[str, list[tuple[float, str]]] = {}
    for k, p in pairs.items():
        for run, score in p["_runs"].items():
            by_run.setdefault(run, []).append((score, k))
    keep: set[str] = set()
    for run, scored in by_run.items():
        for place, (score, k) in enumerate(sorted(scored, key=lambda t: (-t[0], t[1]))[:TOP_PER_RUN], 1):
            keep.add(k)
            pairs[k]["ranks"].append({"run_id": run, "run": run_names.get(run, run), "place": place, "score": score})
    offered = [pairs[k] for k in (keep or seeds)]
    for p in offered:
        del p["_seen"], p["_runs"]
        p["ranks"].sort(key=lambda r: (r["place"], -r["score"]))
    # Best score first; seeds (offered only on a database without scores) after.
    return {"pairs": sorted(offered, key=lambda p: (p["best_score"] is None, -(p["best_score"] or 0), -p["last"]))}


def entry(conn: sqlite3.Connection, pair_id: str) -> dict | None:
    return next((p for p in catalog(conn)["pairs"] if p["id"] == pair_id), None)
