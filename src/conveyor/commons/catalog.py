"""
The catalog an agent is spawned from: every instrument and painter prompt saved in any run of the database, plus
the seeds, each listed once.

Runs copy the same seed into their own organisms, and a prompt or instrument often reappears unchanged, so entries
are keyed by a hash of their text: `i-<hash>` for an instrument, `p-<hash>` for a prompt. A key means the same
text in every run and survives new runs being added. Each entry lists the runs and organisms it came from and the
best score any of them got.

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


def key(kind: str, text: str) -> str:
    return ("i-" if kind == "instrument" else "p-") + hashlib.sha256(text.encode()).hexdigest()[:12]


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


def catalog(conn: sqlite3.Connection) -> dict:
    entries: dict[str, dict] = {}

    def add(kind: str, text: str, label: str, created: float, origin: dict | None = None, sheet: str | None = None,
            score: float | None = None) -> None:
        k = key(kind, text)
        entry = entries.get(k)
        if entry is None:
            entry = entries[k] = {"id": k, "kind": kind, "label": label, "summary": _first_line(text),
                                  "text": text, "sheet": None, "best_score": None, "created": created, "origins": []}
            if kind == "instrument":
                entry["tools"], entry["views"] = tool_names(text)
        if origin:
            entry["origins"].append(origin)
        if not entry["label"]:  # the first organism had no summary; a later copy may
            entry["label"] = label
        entry["sheet"] = entry["sheet"] or sheet
        if score is not None and (entry["best_score"] is None or score > entry["best_score"]):
            entry["best_score"] = score

    for name, source in SEEDS.items():
        add("instrument", source, f"seed: {name}", 0.0)
    add("prompt", INITIAL_STRATEGY, "starting strategy", 0.0)

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"runs", "organisms", "evaluations"} <= tables:  # a new file, or one from before the rewrite: seeds only
        return _ranked(entries)
    runs ={r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM runs")}
    scores = {r[0]: r[1] for r in conn.execute(
        "SELECT organism_id, max(score) FROM evaluations WHERE viable=1 AND score IS NOT NULL GROUP BY organism_id")}
    for row in conn.execute("SELECT id, run_id, node, genome, summary, sheet, created, mutator FROM organisms "
                            "WHERE viable IS NULL OR viable != 0 ORDER BY created"):
        try:
            genome = json.loads(row["genome"] or "{}")
        except json.JSONDecodeError:
            continue
        kind = "instrument" if row["node"] == "instrument" else "prompt" if row["node"] == "painter" else None
        text = genome.get("source" if kind == "instrument" else "prompt") if kind else None
        if not isinstance(text, str) or not text.strip():
            continue
        origin = {"run_id": row["run_id"], "run": runs.get(row["run_id"], row["run_id"]), "organism_id": row["id"],
                  "mutator": row["mutator"], "score": scores.get(row["id"])}
        add(kind, text, (row["summary"] or "").strip()[:140], row["created"], origin, row["sheet"], scores.get(row["id"]))
    return _ranked(entries)


def _ranked(entries: dict[str, dict]) -> dict:
    """Best score first, then unscored entries newest first."""
    ranked = sorted(entries.values(), key=lambda e: (e["best_score"] is None, -(e["best_score"] or 0), -e["created"]))
    return {"instruments": [e for e in ranked if e["kind"] == "instrument"],
            "prompts": [e for e in ranked if e["kind"] == "prompt"]}


def entry(conn: sqlite3.Connection, catalog_id: str) -> dict | None:
    want = "instruments" if catalog_id.startswith("i-") else "prompts"
    return next((e for e in catalog(conn)[want] if e["id"] == catalog_id), None)
