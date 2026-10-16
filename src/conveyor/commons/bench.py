"""
Benchmarks: one shared-canvas setup run with different models, so the results can be set side by side.

A spec says what canvas to make, where its agents spawn and with what settings, and which models to try. Each
variant gets a canvas of its own with exactly the same settings and agents; only the models differ. A variant is
one model for every agent, or a list that agents take in turn (agent i gets item i mod the list's length), which is
how a mix is written.

    {
      "name": "Harbour",
      "canvas": {"viewport": 512, "max_calls": 60, "frame": {"width": 1536, "height": 1024},
                 "task": "A harbour at dusk, boats coming in."},
      "defaults": {"cap": 3, "effort": "high", "max_successors": 2},
      "agents": [{"x": 0, "y": 0, "task": "The sky and the far hills."}, {"x": 1024, "y": 512}],
      "variants": {"claude": "claude", "gpt": "gpt", "muse": "muse", "mix": ["claude", "gpt", "muse"]}
    }

Models are named by PRESETS or by the spec's own `models` ({"name": {harness, model, effort, provider}}), or
written out in place. An agent takes any field a spawn request takes except the model's (harness, model,
provider); `defaults` fills what an agent leaves out, and a model's own effort beats both. `label` names the
agent's slot in the results. `max_successors` defaults to DEFAULT_MAX_SUCCESSORS here, so a benchmark always ends.

Every run has to start from the same place to compare, so painters are fixed per slot: an agent with no `pair_id`
(or "random") draws one from a generator seeded by the benchmark's `seed` and its slot, which hands it the same
painter in every variant and repeat. Successors draw the same way, by slot and generation. `canvas.painters`
narrows the draws to some catalog pairs, as on the canvas page.

`order` is "sequential" (the default) or "parallel". Sequential runs one variant's canvas at a time: the next
canvas's agents wait as `held` until every agent on the one before has stopped, so a bad batch can be stopped after
one variant and only one model's rate limit is in use at once. `repeats` runs each variant that many times.

Automatic judges start on benchmark canvases like any other, with the model of the canvas's newest agent. Set the
spec's `judge` to a model to give every canvas the same judge, or `canvas.auto_judges` to false for none.

Nothing is created until the whole spec checks out, every agent of every run included, so a typo never leaves half
a benchmark spending money. `plan` does the checking and is also the dry run.
"""

from __future__ import annotations

import io
import json
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from conveyor.commons import catalog as catalog_module
from conveyor.commons.agent import SPAWN_FIELDS
from conveyor.commons.agent import agent_row
from conveyor.commons.agent import draw_rng
from conveyor.commons.agent import halted
from conveyor.commons.agent import insert_agent
from conveyor.commons.agent import session_scope
from conveyor.commons.export import export_png
from conveyor.commons.tiles import TILE
from conveyor.commons.tiles import canvas_record
from conveyor.commons.tiles import insert_canvas
from conveyor.commons.views import STALE_SECONDS
from conveyor.launch import LaunchError
from conveyor.launch import canvas_settings
from conveyor.store import connect
from conveyor.store import dumps
from conveyor.store import init_db
from conveyor.store import new_id


def _presets() -> dict[str, dict]:
    from conveyor.claude import DEFAULT_MODEL
    from conveyor.pi import PROVIDERS

    return {
        "claude": {"harness": "claude", "model": DEFAULT_MODEL},
        "sonnet": {"harness": "claude", "model": "claude-sonnet-5-5"},
        "haiku": {"harness": "claude", "model": "claude-haiku-4-5-20251001"},
        "gpt": {"harness": "pi", "provider": "openai-codex", "model": PROVIDERS["openai-codex"].default_model},
        "muse": {"harness": "pi", "provider": "opencode-go", "model": "muse-spark-1.3-contributor"},
        "glm": {"harness": "pi", "provider": "opencode-go", "model": PROVIDERS["opencode-go"].default_model},
    }


PRESETS = _presets()
ORDERS = ("sequential", "parallel")
DEFAULT_MAX_SUCCESSORS = 3
MAX_AGENTS = 32
MAX_VARIANTS = 12
MAX_REPEATS = 10
MAX_RUNS = 24  # canvases in one benchmark: variants times repeats
TOP_KEYS = frozenset({"name", "canvas", "defaults", "agents", "models", "variants", "repeats", "order", "seed", "judge"})
MODEL_KEYS = frozenset({"harness", "model", "effort", "provider"})
AGENT_KEYS = (SPAWN_FIELDS - {"harness", "model", "provider"}) | {"label"}
DEFAULT_KEYS = AGENT_KEYS - {"x", "y", "label", "name"}
LIVE = ("queued", "starting", "running")

EXAMPLE = {
    "name": "Harbour at dusk",
    "order": "sequential",
    "repeats": 1,
    "canvas": {"viewport": 512, "max_calls": 60, "frame": {"width": 1536, "height": 1024},
               "task": "A harbour at dusk: fishing boats coming in, gulls, lamps on the quay."},
    "defaults": {"cap": 3, "effort": "high", "max_successors": 2},
    "agents": [{"label": "sky", "x": 0, "y": 0, "task": "The sky and the far hills."},
               {"label": "boats", "x": 512, "y": 512},
               {"label": "quay", "x": 1024, "y": 512}],
    "variants": {"claude": "claude", "gpt": "gpt", "muse": "muse", "mix": ["claude", "gpt", "muse"]},
}


# ---- the spec ---------------------------------------------------------------------------------------------------


def model_spec(value, models: dict, where: str) -> dict:
    """A model as {harness, model?, effort?, provider?}, from a name in `models` or PRESETS, or written out."""
    from conveyor.pi import PROVIDERS

    if isinstance(value, str):
        named = models.get(value) or PRESETS.get(value)
        if named is None:
            raise LaunchError(f"{where}: no model called {value!r}. Define it under models, or use one of "
                              f"{', '.join(PRESETS)}.")
        value = named
    if not isinstance(value, dict) or set(value) - MODEL_KEYS:
        raise LaunchError(f"{where} must be a model's name or an object of harness, model, effort and provider.")
    harness = value.get("harness") or "claude"
    if harness not in ("claude", "pi"):
        raise LaunchError(f"{where}: harness must be claude or pi.")
    out = {"harness": harness, **{k: value[k] for k in ("model", "effort", "provider") if value.get(k) is not None}}
    if harness == "claude":
        out.pop("provider", None)
    elif out.get("provider", "opencode-go") not in PROVIDERS:
        raise LaunchError(f"{where}: provider must be one of {', '.join(PROVIDERS)}.")
    return out


def describe(config: dict) -> str:
    """An agent's model in a few words: `claude-opus-5-5 high`, `gpt-6.1-sol (openai-codex) medium`."""
    where = f" ({config['provider']})" if config.get("harness") == "pi" and config.get("provider") else ""
    return f"{config.get('model') or 'unknown model'}{where} {config.get('effort') or ''}".strip()


def plan(conn: sqlite3.Connection, spec: dict) -> dict:
    """Check a spec and lay out its runs, each with its canvas row and its agents' rows, none of them saved. Raises
    LaunchError saying what's wrong."""
    if not isinstance(spec, dict) or set(spec) - TOP_KEYS:
        raise LaunchError("A benchmark spec takes " + ", ".join(sorted(TOP_KEYS)) + ".")
    name = spec.get("name") or "Benchmark"
    if not isinstance(name, str) or not name.strip() or len(name) > 60 or "\n" in name:
        raise LaunchError("name must be a line of text, at most 60 characters.")
    name = name.strip()
    canvas_body = spec.get("canvas") or {}
    settings = canvas_settings(canvas_body, extra=frozenset({"painters", "auto_judges"}))
    settings.pop("name")
    painters = canvas_body.get("painters")
    if painters is not None:
        usable = {p["id"] for p in catalog_module.catalog(conn)["pairs"]}
        if not isinstance(painters, list) or not painters or not all(isinstance(p, str) for p in painters):
            raise LaunchError("canvas.painters must be a list of catalog painters, or left out for all of them.")
        if unknown := sorted(set(painters) - usable):
            raise LaunchError(f"canvas.painters: not in the catalog: {', '.join(unknown)}.")
        painters = list(dict.fromkeys(painters))
    auto_judges = canvas_body.get("auto_judges", True)
    if not isinstance(auto_judges, bool):
        raise LaunchError("canvas.auto_judges must be true or false.")
    order = spec.get("order") or "sequential"
    if order not in ORDERS:
        raise LaunchError("order must be sequential or parallel.")
    repeats = spec.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        raise LaunchError(f"repeats must be a whole number from 1 to {MAX_REPEATS}.")
    seed = spec.get("seed")
    if seed is None:
        seed = random.randrange(1 << 31)
    elif isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 31:
        raise LaunchError("seed must be a whole number from 0 to 2,147,483,647.")

    raw_models = spec.get("models") or {}
    if not isinstance(raw_models, dict):
        raise LaunchError("models must be an object of {name: {harness, model, effort, provider}}.")
    models = {k: model_spec(v, {}, f"models.{k}") for k, v in raw_models.items()}
    judge = model_spec(spec["judge"], models, "judge") if spec.get("judge") is not None else None

    defaults = spec.get("defaults") or {}
    if not isinstance(defaults, dict) or set(defaults) - DEFAULT_KEYS:
        raise LaunchError("defaults takes " + ", ".join(sorted(DEFAULT_KEYS)) + ".")
    defaults = dict(defaults)
    defaults.setdefault("max_successors", DEFAULT_MAX_SUCCESSORS)
    agents = spec.get("agents")
    if not isinstance(agents, list) or not 1 <= len(agents) <= MAX_AGENTS:
        raise LaunchError(f"agents must be a list of 1 to {MAX_AGENTS} agents.")
    for i, a in enumerate(agents):
        if not isinstance(a, dict) or set(a) - AGENT_KEYS:
            raise LaunchError(f"agents[{i}] takes " + ", ".join(sorted(AGENT_KEYS)) + ".")
        if a.get("label") is not None and (not isinstance(a["label"], str) or len(a["label"]) > 40):
            raise LaunchError(f"agents[{i}].label must be text, at most 40 characters.")

    variants = spec.get("variants")
    if not isinstance(variants, dict) or not 1 <= len(variants) <= MAX_VARIANTS:
        raise LaunchError(f"variants must be an object of 1 to {MAX_VARIANTS} {{name: model or [models]}}.")
    if len(variants) * repeats > MAX_RUNS:
        raise LaunchError(f"{len(variants)} variants times {repeats} repeats is {len(variants) * repeats} canvases; "
                          f"a benchmark makes at most {MAX_RUNS}.")
    lineups = {}
    for v, value in variants.items():
        if not isinstance(v, str) or not v.strip() or len(v) > 40 or "\n" in v:
            raise LaunchError("Each variant's name must be a line of text, at most 40 characters.")
        items = value if isinstance(value, list) else [value]
        if not items:
            raise LaunchError(f"variants.{v} needs at least one model.")
        lineups[v] = [model_spec(m, models, f"variants.{v}" + (f"[{k}]" if isinstance(value, list) else ""))
                      for k, m in enumerate(items)]

    bench_id = new_id()
    shared = {"draw_seed": seed, **({"pairs": painters} if painters else {}),
              **({} if auto_judges else {"auto_judges": False}), **({"judge_model": judge} if judge else {})}
    runs = []
    for v, lineup in lineups.items():
        for r in range(repeats):
            label = v if repeats == 1 else f"{v} #{r + 1}"
            row = canvas_record(f"{name} · {label}"[:60], extra={**shared, "benchmark": {
                "id": bench_id, "variant": v, "repeat": r + 1, "label": label}}, **settings)
            made = []
            for i, a in enumerate(agents):
                model = lineup[i % len(lineup)]
                body = {**defaults, **{k: x for k, x in a.items() if k != "label"}, **model}
                if body.get("kind", "painter") == "painter" and not body.get("pair_id"):
                    body["pair_id"] = "random"
                slot = a.get("label") or f"#{i + 1}"
                try:
                    made.append(agent_row(conn, row["config"], body, {"slot": i, "slot_label": slot, "bench": bench_id},
                                          rng=draw_rng(row["config"], i, 1)))
                except LaunchError as e:
                    raise LaunchError(f"agents[{i}] ({slot}) in {label}: {e}") from None
            runs.append({"variant": v, "repeat": r + 1, "label": label, "canvas": row, "agents": made,
                         "models": [describe(m["config"]) for m in made]})
    normal = {**spec, "name": name, "order": order, "repeats": repeats, "seed": seed}
    return {"id": bench_id, "name": name, "order": order, "repeats": repeats, "seed": seed, "spec": normal,
            "runs": runs}


def ceiling(config: dict) -> float | None:
    """The most one agent's chain can spend at list price: its cap for each session it may have. None without a
    cap on hand-offs, since then the chain has no end of its own."""
    if not config.get("successors"):
        return float(config["cap"])
    cap = config.get("max_successors")
    return None if cap is None else float(config["cap"]) * (1 + cap)


def preview(p: dict) -> dict:
    """What a plan will do, for the dry run and the page: each run's agents with their models and painters, and
    the most it can spend (None when some chain has no cap), automatic judges aside."""
    ceilings = [ceiling(m["config"]) for r in p["runs"] for m in r["agents"]]
    return {"name": p["name"], "order": p["order"], "seed": p["seed"], "canvases": len(p["runs"]),
            "agents": sum(len(r["agents"]) for r in p["runs"]),
            "ceiling": None if None in ceilings else round(sum(ceilings), 2),
            "runs": [{"label": r["label"], "agents": [{
                "slot": m["config"]["slot_label"], "kind": m["config"]["kind"], "model": describe(m["config"]),
                "x": m["x"], "y": m["y"], "cap": m["config"]["cap"], "max_successors": m["config"].get("max_successors"),
                "painter": m["config"].get("instrument_label") and
                f"{m['config']['instrument_label']} / {m['config']['prompt_label']}"} for m in r["agents"]]}
                for r in p["runs"]]}


def create(db: Path, spec: dict) -> dict:
    """Check the spec and save the benchmark, its canvases and their agents, in one transaction. A sequential
    benchmark's first canvas starts at once (its agents are queued for the launcher's watcher); the others hold."""
    init_db(db)
    conn = connect(db)
    try:
        p = plan(conn, spec)
        config = {"spec": p["spec"], "order": p["order"], "seed": p["seed"],
                  "runs": [{"canvas_id": r["canvas"]["id"], "variant": r["variant"], "repeat": r["repeat"],
                            "label": r["label"], "models": r["models"]} for r in p["runs"]]}
        try:
            conn.execute("INSERT INTO benchmarks (id, name, created, status, config) VALUES (?, ?, ?, 'running', ?)",
                         (p["id"], p["name"], time.time(), dumps(config)))
            for k, run in enumerate(p["runs"]):
                insert_canvas(conn, run["canvas"])
                status = "queued" if p["order"] == "parallel" or k == 0 else "held"
                for made in run["agents"]:
                    insert_agent(conn, run["canvas"]["id"], made, status)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return {"id": p["id"], "name": p["name"], **preview(p)}
    finally:
        conn.close()


# ---- running ------------------------------------------------------------------------------------------------------


def _live(row, now: float) -> bool:
    """Queued or starting, or running with a call in the last while: a running row gone quiet for long died."""
    return row["status"] in ("queued", "starting") or (
        row["status"] == "running" and now - (row["last_ts"] or row["created"]) < STALE_SECONDS)


def advance(db: Path) -> list[str]:
    """One pass of the launcher's watcher over running benchmarks: start the next canvas of a sequential one when
    the canvas before it is done, drop held agents on a stopped canvas, and mark a benchmark done once nothing on
    any of its canvases is left to run. Answers the agents it released."""
    conn = connect(db)
    released: list[str] = []
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='benchmarks'").fetchone() is None:
            return []
        now = time.time()
        for b in conn.execute("SELECT * FROM benchmarks WHERE status='running'").fetchall():
            config = json.loads(b["config"])
            live, waiting = False, []
            for run in config["runs"]:
                rows = conn.execute("SELECT id, status, created, last_ts FROM canvas_agents WHERE canvas_id=?",
                                    (run["canvas_id"],)).fetchall()
                held = [r["id"] for r in rows if r["status"] == "held"]
                if held and halted(conn, run["canvas_id"]):
                    conn.execute(f"UPDATE canvas_agents SET status='stopped', ended=?, error='Stopped before it started' "
                                 f"WHERE id IN ({','.join('?' * len(held))})", (now, *held))
                    held = []
                live = live or any(_live(r, now) for r in rows)
                if held:
                    waiting.append(held)
            if not live and waiting:  # the canvas before is done: start the next one
                ids = waiting[0] if config["order"] == "sequential" else [i for w in waiting for i in w]
                conn.execute(f"UPDATE canvas_agents SET status='queued' WHERE status='held' AND id IN "
                             f"({','.join('?' * len(ids))})", ids)
                released += ids
            elif not live:
                conn.execute("UPDATE benchmarks SET status='done', ended=? WHERE id=? AND status='running'", (now, b["id"]))
        conn.commit()
    finally:
        conn.close()
    return released


def stop(db: Path, bench_id: str) -> list[str] | None:
    """Mark a benchmark stopped and its held agents with it. Answers its canvases, for the launcher to stop the
    agents running on them, or None for an unknown benchmark."""
    conn = connect(db)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='benchmarks'").fetchone() is None:
            return None
        row = conn.execute("SELECT config FROM benchmarks WHERE id=?", (bench_id,)).fetchone()
        if row is None:
            return None
        canvases = [r["canvas_id"] for r in json.loads(row["config"])["runs"]]
        now = time.time()
        conn.execute("UPDATE benchmarks SET status='stopped', ended=? WHERE id=? AND status='running'", (now, bench_id))
        conn.execute(f"UPDATE canvas_agents SET status='stopped', ended=?, error='Stopped before it started' "
                     f"WHERE status='held' AND canvas_id IN ({','.join('?' * len(canvases))})", (now, *canvases))
        conn.commit()
        return canvases
    finally:
        conn.close()


# ---- reading ------------------------------------------------------------------------------------------------------


def listing(conn: sqlite3.Connection) -> list[dict]:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='benchmarks'").fetchone() is None:
        return []
    out = []
    for b in conn.execute("SELECT * FROM benchmarks ORDER BY created DESC"):
        config = json.loads(b["config"])
        out.append({"id": b["id"], "name": b["name"], "created": b["created"], "ended": b["ended"],
                    "status": b["status"], "order": config["order"],
                    "variants": list(dict.fromkeys(r["variant"] for r in config["runs"])), "canvases": len(config["runs"])})
    return out


def _tokens(usage: dict) -> tuple[int, int]:
    """Input (cache reads and writes included) and output tokens from a session's usage."""
    usage = usage or {}
    taken = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return taken, int(usage.get("output_tokens") or 0)


def run_stats(conn: sqlite3.Connection, canvas_id: str) -> dict:
    """What happened on one canvas: its agents and their ends, calls, paint, coverage, words, spend and time, in
    all and per model."""
    canvas = conn.execute("SELECT config FROM canvases WHERE id=?", (canvas_id,)).fetchone()
    config = json.loads(canvas[0])
    now = time.time()
    agents = [dict(r) for r in conn.execute("SELECT * FROM canvas_agents WHERE canvas_id=? ORDER BY created", (canvas_id,))]
    sessions = {r["id"]: dict(r) for r in conn.execute("SELECT id, cost, usage FROM sessions WHERE run_id=?",
                                                       (session_scope(canvas_id),))}
    ops = {r["agent_id"]: dict(r) for r in conn.execute(
        "SELECT agent_id, count(*) AS calls, sum(tiles > 0 AND tool != 'write_message') AS paints, "
        "sum(status = 'rejected') AS refused FROM canvas_ops WHERE canvas_id=? AND agent_id IS NOT NULL "
        "GROUP BY agent_id", (canvas_id,))}
    tools = dict(conn.execute("SELECT tool, count(*) FROM canvas_ops WHERE canvas_id=? AND status IN "
                              "('painted', 'sent', 'generated', 'handed_off') GROUP BY tool", (canvas_id,)).fetchall())
    span = conn.execute("SELECT min(ts), max(ts), count(*) FROM canvas_ops WHERE canvas_id=? AND agent_id IS NOT NULL",
                        (canvas_id,)).fetchone()
    heads = conn.execute("SELECT tx, ty FROM canvas_heads WHERE canvas_id=?", (canvas_id,)).fetchall()
    frame = config.get("frame")
    if frame:
        inside = {(tx, ty) for tx, ty in heads if frame[0] <= (tx + 1) * TILE and tx * TILE < frame[2]
                  and frame[1] <= (ty + 1) * TILE and ty * TILE < frame[3]}
        possible = (-(-frame[2] // TILE) - frame[0] // TILE) * (-(-frame[3] // TILE) - frame[1] // TILE)
        coverage = round(len(inside) / possible, 3) if possible else None
    else:
        coverage = None
    by_model: dict[str, dict] = {}
    statuses: Counter = Counter()
    total = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0}
    for a in agents:
        c = json.loads(a["config"])
        live = _live(a, now)
        status = "interrupted" if a["status"] == "running" and not live else a["status"]
        statuses[status] += 1
        s = sessions.get(a["session_id"]) or {}
        cost = float(s.get("cost") or 0)
        taken, given = _tokens(json.loads(s["usage"]) if s.get("usage") else {})
        o = ops.get(a["id"]) or {}
        m = by_model.setdefault(describe(c), {"model": describe(c), "harness": c.get("harness"), "sessions": 0,
                                              "calls": 0, "paints": 0, "refused": 0, "cost": 0.0,
                                              "input_tokens": 0, "output_tokens": 0, "failed": 0})
        m["sessions"] += 1
        m["calls"] += a["calls_used"]
        m["paints"] += int(o.get("paints") or 0)
        m["refused"] += int(o.get("refused") or 0)
        m["cost"] += cost
        m["input_tokens"] += taken
        m["output_tokens"] += given
        m["failed"] += status == "failed"
        total["cost"] += cost
        total["input_tokens"] += taken
        total["output_tokens"] += given
    if not agents:
        state = "empty"
    elif statuses["held"] == len(agents):
        state = "waiting"
    elif any(_live(a, now) for a in agents):
        state = "running"
    elif config.get("halted"):
        state = "stopped"
    else:
        state = "done"
    started = min((a["created"] for a in agents), default=None)
    ended = max((a["ended"] or 0 for a in agents), default=None) if state in ("done", "stopped") else None
    return {"canvas_id": canvas_id, "state": state, "agents": len(agents), "statuses": dict(statuses),
            "handoffs": statuses.get("handed_off", 0), "calls": sum(a["calls_used"] for a in agents),
            "ops": span[2], "paints": sum(int(o.get("paints") or 0) for o in ops.values()),
            "refused": sum(int(o.get("refused") or 0) for o in ops.values()),
            "messages": tools.get("write_message", 0), "broadcasts": tools.get("broadcast", 0),
            "references": tools.get("generate_reference", 0), "tiles": len(heads), "coverage": coverage,
            "cost": round(total["cost"], 4), "input_tokens": total["input_tokens"],
            "output_tokens": total["output_tokens"], "started": started, "ended": ended,
            "seconds": round((ended or now) - started, 1) if started and state != "waiting" else None,
            "first_op": span[0], "last_op": span[1],
            "by_model": [{**m, "cost": round(m["cost"], 4)} for m in by_model.values()]}


def results(conn: sqlite3.Connection, bench_id: str) -> dict:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='benchmarks'").fetchone() is None:
        raise KeyError(bench_id)
    row = conn.execute("SELECT * FROM benchmarks WHERE id=?", (bench_id,)).fetchone()
    if row is None:
        raise KeyError(bench_id)
    config = json.loads(row["config"])
    runs = [{**r, **run_stats(conn, r["canvas_id"])} for r in config["runs"]]
    return {"id": row["id"], "name": row["name"], "created": row["created"], "ended": row["ended"],
            "status": row["status"], "order": config["order"], "seed": config["seed"], "spec": config["spec"],
            "runs": runs, "cost": round(sum(r["cost"] for r in runs), 4)}


# ---- pictures -----------------------------------------------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def sheet_png(conn: sqlite3.Connection, bench_id: str, cell: int = 640) -> bytes:
    """Every run's canvas side by side, a column per variant and a row per repeat, each captioned with its
    models, spend and how far it got, under the benchmark's name and task."""
    res = results(conn, bench_id)
    cell = max(128, min(1600, int(cell)))
    variants = list(dict.fromkeys(r["variant"] for r in res["runs"]))
    rows = max(r["repeat"] for r in res["runs"])
    pad, caption, head = 16, 86, 64
    width = len(variants) * (cell + pad) + pad
    height = head + rows * (cell + caption + pad) + pad
    sheet = Image.new("RGB", (width, height), (250, 247, 240))
    draw = ImageDraw.Draw(sheet)
    task = (res["spec"].get("canvas") or {}).get("task") or "No task"
    draw.text((pad, 14), res["name"], fill=(31, 35, 40), font=_font(24))
    draw.text((pad, 42), _clip(draw, task, _font(14), width - 2 * pad), fill=(106, 110, 117), font=_font(14))
    for r in res["runs"]:
        x = pad + variants.index(r["variant"]) * (cell + pad)
        y = head + (r["repeat"] - 1) * (cell + caption + pad)
        with Image.open(io.BytesIO(export_png(conn, r["canvas_id"], side=cell))) as im:
            im = im.convert("RGB")
            sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        draw.rectangle((x - 1, y - 1, x + cell, y + cell), outline=(221, 212, 196))
        models = ", ".join(dict.fromkeys(r["models"]))
        lines = [(r["label"], 18, (31, 35, 40)), (models, 13, (106, 110, 117)),
                 (f"{r['agents']} sessions · {r['paints']} paint calls · ${r['cost']:.2f}"
                  + (f" · {r['coverage']:.0%} covered" if r["coverage"] is not None else "") + f" · {r['state']}",
                  13, (106, 110, 117))]
        ty = y + cell + 8
        for text, size, fill in lines:
            draw.text((x, ty), _clip(draw, text, _font(size), cell), fill=fill, font=_font(size))
            ty += size + 8
    out = io.BytesIO()
    sheet.save(out, format="PNG")
    return out.getvalue()


def _clip(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> str:
    """`text` cut with an ellipsis to fit `width` pixels."""
    text = " ".join(str(text).split())
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def write_results(conn: sqlite3.Connection, bench_id: str, out: Path, side: int | None = None) -> list[Path]:
    """Each run's canvas as a PNG, the comparison sheet and results.json, into `out`. Answers the files."""
    from conveyor.commons.export import filename

    res = results(conn, bench_id)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for r in res["runs"]:
        path = out / filename(r["label"])
        path.write_bytes(export_png(conn, r["canvas_id"], side=side))
        written.append(path)
    (out / "comparison.png").write_bytes(sheet_png(conn, bench_id))
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    return written + [out / "comparison.png", out / "results.json"]


# ---- from a canvas ------------------------------------------------------------------------------------------------


def spec_from_canvas(conn: sqlite3.Connection, canvas_id: str) -> dict:
    """A benchmark spec that sets a canvas up again: its settings, and every agent spawned on it by hand, where it
    started and with what settings. The variant `as spawned` gives each agent the model it had; the others put
    one preset on every agent. Painters drawn at random are drawn again, by slot."""
    row = conn.execute("SELECT * FROM canvases WHERE id=?", (canvas_id,)).fetchone()
    if row is None:
        raise KeyError(canvas_id)
    config = json.loads(row["config"])
    canvas = {"viewport": config["viewport"], "max_calls": config["max_calls"]}
    if config.get("task"):
        canvas["task"] = config["task"]
    if config.get("frame"):
        canvas["frame"] = {"width": config["frame"][2] - config["frame"][0], "height": config["frame"][3] - config["frame"][1]}
    if config.get("pairs"):
        canvas["painters"] = config["pairs"]
    agents, lineup = [], []
    for r in conn.execute("SELECT * FROM canvas_agents WHERE canvas_id=? ORDER BY created", (canvas_id,)):
        c = json.loads(r["config"])
        if c.get("auto") or int(c.get("generation", 1)) != 1:
            continue  # judges that started by themselves, and successors, come again by themselves
        start = c.get("start") or [r["x"], r["y"]]
        a = {"label": c.get("slot_label") or r["name"], "x": start[0], "y": start[1], "cap": c.get("cap", 5.0)}
        if c.get("kind") == "judge":
            a["kind"] = "judge"
        else:
            a["pair_id"] = "random" if c.get("random") else c.get("pair_id")
            a.update(paint_batch=c.get("paint_batch", True), successors=bool(c.get("successors")),
                     max_successors=c.get("max_successors"), references=bool(c.get("references")))
            if c.get("viewport") and c["viewport"] != config["viewport"]:
                a["viewport"] = c["viewport"]
        if c.get("agent_task"):
            a["task"] = c["agent_task"]
        agents.append(a)
        lineup.append({k: c[k] for k in ("harness", "model", "effort", "provider") if c.get(k) is not None})
    if not agents:
        raise ValueError("No agent was spawned on this canvas by hand, so there is nothing to set up again.")
    return {"name": row["name"][:60], "order": "sequential", "repeats": 1, "canvas": canvas, "agents": agents,
            "variants": {"as spawned": lineup, "claude": "claude", "gpt": "gpt", "muse": "muse"}}
