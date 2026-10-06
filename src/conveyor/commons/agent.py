"""
Spawning an agent on the shared canvas, and the process that runs it.

    python -m conveyor.commons.agent <db> <agent_id>

`create_agent` checks a spawn request (an instrument and a prompt from the catalog, a harness, model, effort and
provider, a starting corner) and saves the agent with the full text of its instrument and prompt, so a later run
or catalog change never alters it. The launcher then starts this module in a child process, so stopping an agent
and shutting the server down work like they do for runs and studio paintings.

An agent's model sessions are recorded under `canvas-<canvas id>` instead of a run, so the dashboard's session
drawer shows its transcript and replays its viewport, and the run list doesn't fill up with agents.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conveyor.commons import catalog as catalog_module
from conveyor.commons import prompts
from conveyor.commons.server import HARNESS_TOOLS
from conveyor.commons.tiles import MOVE_SHARE
from conveyor.commons.tiles import SharedCanvas
from conveyor.launch import LaunchError
from conveyor.store import Store
from conveyor.store import connect
from conveyor.store import dumps
from conveyor.store import init_db
from conveyor.store import new_id

EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_COORD = 10_000_000
SESSION_TIMEOUT = 3 * 3600.0  # seconds; a hundred calls at high effort can outlast the 45 minutes a painting gets
GRACE_CALLS = 5  # calls past the budget a model may make, refused, before its session is killed


def session_scope(canvas_id: str) -> str:
    return f"canvas-{canvas_id}"


def _text(body: dict, key: str, limit: int, default: str | None = None) -> str | None:
    value = body.get(key, default)
    if value is None or value == "":
        return default
    if not isinstance(value, str) or len(value) > limit or "\n" in value or value.lstrip().startswith("-"):
        raise LaunchError(f"{key} must be a short line of text that doesn't start with a dash.")
    return value.strip()


def create_agent(db: Path, canvas_id: str, body: dict) -> dict:
    from conveyor.pi import PROVIDERS

    allowed = {"instrument_id", "prompt_id", "harness", "model", "effort", "provider", "x", "y", "name", "cap",
               "paint_batch"}
    if not isinstance(body, dict) or set(body) - allowed:
        raise LaunchError("Expected " + ", ".join(sorted(allowed)) + ".")
    harness = body.get("harness") or "claude"
    if harness not in ("claude", "pi"):
        raise LaunchError("harness must be claude or pi.")
    model = _text(body, "model", 200)
    effort = body.get("effort") or "high"
    if effort not in EFFORTS:
        raise LaunchError(f"effort must be one of {', '.join(EFFORTS)}.")
    provider = body.get("provider") or "opencode-go"
    if provider not in PROVIDERS:
        raise LaunchError(f"provider must be one of {', '.join(PROVIDERS)}.")
    name = _text(body, "name", 60)
    model = model or default_model(harness, provider)  # saved resolved, so the agent list names the real model
    try:
        x, y = int(body.get("x", 0)), int(body.get("y", 0))
        cap = float(body.get("cap", 5.0))
    except (TypeError, ValueError):
        raise LaunchError("x and y must be whole numbers and cap a number.") from None
    if isinstance(body.get("x"), bool) or isinstance(body.get("y"), bool) or not (abs(x) <= MAX_COORD and abs(y) <= MAX_COORD):
        raise LaunchError(f"x and y must be whole numbers within {MAX_COORD:,} of the origin.")
    if not 0.01 <= cap <= 100:
        raise LaunchError("cap must be between 0.01 and 100 dollars.")
    paint_batch = body.get("paint_batch", True)
    if not isinstance(paint_batch, bool):
        raise LaunchError("paint_batch must be on or off.")
    init_db(db)
    conn = connect(db)
    try:
        canvas = conn.execute("SELECT * FROM canvases WHERE id=?", (canvas_id,)).fetchone()
        if canvas is None:
            raise LaunchError("Canvas not found.")
        found = {}
        for key, kind in (("instrument_id", "instrument"), ("prompt_id", "prompt")):
            value = body.get(key)
            found[kind] = catalog_module.entry(conn, value) if isinstance(value, str) else None
            if found[kind] is None or found[kind]["kind"] != kind:
                raise LaunchError(f"Choose a {kind} from the catalog.")
        clash = set(found["instrument"]["tools"] + found["instrument"]["views"]) & set(HARNESS_TOOLS)
        if clash:
            raise LaunchError(f"That instrument has a tool named {', '.join(sorted(clash))}, which the canvas uses.")
        config = {"instrument_id": found["instrument"]["id"], "prompt_id": found["prompt"]["id"],
                  "instrument_label": found["instrument"]["label"], "prompt_label": found["prompt"]["label"],
                  "source": found["instrument"]["text"], "prompt": found["prompt"]["text"],
                  "sheet": found["instrument"]["sheet"], "harness": harness, "model": model, "effort": effort,
                  "provider": provider if harness == "pi" else None, "cap": cap, "paint_batch": paint_batch,
                  "start": [x, y]}
        agent_id = new_id()
        name = name or f"{re.sub(r'^claude-', '', model.split('/')[-1])} {agent_id[:4]}"
        max_calls = int(json.loads(canvas["config"]).get("max_calls", 100))
        conn.execute("INSERT INTO canvas_agents (id, canvas_id, name, created, status, config, x, y, calls_used, "
                     "max_calls) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, 0, ?)",
                     (agent_id, canvas_id, name, time.time(), dumps(config), x, y,
                      max_calls))
        conn.commit()
        return {"id": agent_id, "canvas_id": canvas_id, "name": name, "x": x, "y": y, "max_calls": max_calls}
    finally:
        conn.close()


def default_model(harness: str, provider: str) -> str:
    from conveyor.claude import DEFAULT_MODEL
    from conveyor.pi import PROVIDERS

    return DEFAULT_MODEL if harness == "claude" else PROVIDERS[provider].default_model


def _args(config: dict, db: Path) -> argparse.Namespace:
    """The `run` options an agent's harness is built from: the parser's defaults with the agent's choices."""
    from conveyor.__main__ import build_parser

    args = build_parser().parse_args(["run"])
    args.harness, args.model, args.effort = config["harness"], config["model"], config["effort"]
    args.provider = config.get("provider") or args.provider
    args.lanes = 1
    args.db = str(db)
    return args


def system_prompt(config: dict, size: int, x: int, y: int, max_calls: int) -> str:
    from conveyor.painting.canvas import CALL_AREA_SHARE, Canvas
    from conveyor.painting.instrument import Instrument

    inst = Instrument(config["source"])
    rules = prompts.AGENT_RULES.format(
        s=size, x=x, y=y, area_cap=Canvas(size, size).area_cap, share=CALL_AREA_SHARE, max_move=int(MOVE_SHARE * size),
        max_calls=max_calls, batch=prompts.BATCH_RULE.format(max_calls=max_calls) if config.get("paint_batch", True) else "",
        views=prompts.VIEWS_RULE if inst.spec.views else "", reference=inst.reference(size, size))
    return config["prompt"].strip() + "\n\n" + prompts.STRATEGY_BRIDGE + "\n\n" + rules


def run_agent(db: Path, agent_id: str) -> None:
    from conveyor.__main__ import _harness
    from conveyor.harness import Job
    from conveyor.harness import Meter
    from conveyor.painting.canvas import gridded_png
    from conveyor.painting.instrument import Instrument
    from conveyor.painting.problem import _Ingest
    from conveyor.painting.problem import probe_source

    db = Path(db).resolve()
    conn = connect(db, readonly=True)
    try:
        row = dict(conn.execute("SELECT * FROM canvas_agents WHERE id=?", (agent_id,)).fetchone())
        sheet_row = None
        config = json.loads(row["config"])
        if config.get("sheet"):
            sheet_row = conn.execute("SELECT data FROM artifacts WHERE name=?", (config["sheet"],)).fetchone()
    finally:
        conn.close()
    canvas = SharedCanvas(db, row["canvas_id"])
    store = Store(db, run_id=session_scope(row["canvas_id"]), check_run=False)
    harness = None
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        canvas.update_agent(agent_id, status="running", last_ts=time.time())
        args = _args(config, db)
        harness = _harness(config["harness"], config["model"], config["effort"], args, store,
                           Meter(max_usage=args.max_usage), {})
        size, max_calls = int(canvas.config["viewport"]), int(row["max_calls"])
        x, y = int(row["x"]), int(row["y"])
        work = db.parent / f"{db.stem}-sessions"
        d = work / session_scope(row["canvas_id"]) / f"agent-{agent_id}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps({
            "db": str(db), "canvas_id": row["canvas_id"], "agent_id": agent_id, "source": config["source"],
            "x": x, "y": y, "max_calls": max_calls, "paint_batch": config.get("paint_batch", True),
            "seed": random.randrange(1 << 30)}))
        sheet = sheet_row[0] if sheet_row else probe_source(config["source"], size, size, work)[1]
        view = gridded_png(canvas.read(x, y, size, size).image())
        first, second, third = prompts.FIRST_MESSAGE
        content = [{"type": "text", "text": first.format(x=x, y=y)}, {"type": "png", "data": view}]
        if sheet:
            content += [{"type": "text", "text": second}, {"type": "png", "data": sheet}]
        content.append({"type": "text", "text": third})
        inst = Instrument(config["source"])
        tools = [t.name for t in [*inst.spec.tools, *inst.spec.views]] + ["look", "move_viewport", "write_message"]
        if config.get("paint_batch", True):
            tools.append("paint_batch")
        ingest = _Ingest(store, d)
        tool_uses = [0]

        def started(sid: str) -> None:
            ingest.start(sid)
            canvas.update_agent(agent_id, session_id=sid)

        def on_event(msg: dict) -> None:
            if msg.get("type") == "user":
                ingest.pull()
            elif msg.get("type") == "assistant":
                tool_uses[0] += sum(1 for b in (msg.get("message") or {}).get("content") or [] if b.get("type") == "tool_use")
                if tool_uses[0] > max_calls + GRACE_CALLS:  # it kept calling after being told it was out
                    harness.stop_all()

        job = Job(purpose="canvas agent", system_prompt=system_prompt(config, size, x, y, max_calls), content=content,
                  cwd=d, mcp={"name": "commons", "command": sys.executable, "args": ["-m", "conveyor.commons.server", str(d)]},
                  tools=tools, max_budget_usd=config["cap"], timeout=SESSION_TIMEOUT, node="canvas",
                  organism_id=agent_id, on_start=started, on_event=on_event)
        # The harness's stream reader blocks; keep it off the main thread so Ctrl+C (Stop) can kill the session.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(harness.run, job)
            try:
                outcome = future.result()
            except KeyboardInterrupt:
                harness.stop_all()
                raise
        ingest.pull()
        used = canvas.conn.execute("SELECT calls_used FROM canvas_agents WHERE id=?", (agent_id,)).fetchone()[0]
        if used >= max_calls:
            status, error = "finished", None
        elif outcome.ok:
            status, error = "ended", f"The model ended its turn with {max_calls - used} tool calls left."
        else:
            status, error = "failed", outcome.error or "the session failed"
        canvas.update_agent(agent_id, status=status, ended=time.time(), error=error, session_id=outcome.session_id)
        print(f"Agent {agent_id}: {status}, {used} of {max_calls} tool calls" + (f" ({error})" if error else ""))
    except BaseException as error:
        stopped = isinstance(error, KeyboardInterrupt)
        canvas.update_agent(agent_id, status="stopped" if stopped else "failed", ended=time.time(),
                            error="Stopped" if stopped else (str(error) or type(error).__name__))
        raise
    finally:
        if harness is not None:
            harness.stop_all()
        store.close()
        canvas.close()


if __name__ == "__main__":
    run_agent(Path(sys.argv[1]), sys.argv[2])
