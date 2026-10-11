"""Human-directed instrument design. No evolutionary organisms, evaluations or scores.

HTTP handlers only read source as text/AST. A launcher's child process runs the designer,
probes drafts, renders example replays and runs optional text-only painting auditions.
"""
from __future__ import annotations

import ast
import json
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conveyor.launch import LaunchError
from conveyor.store import Store, connect, dumps, init_db, new_id

PAINTER_PROMPT = """Study the instrument's documentation and examples before painting. Explore its distinctive
behaviour deliberately, including state carried between calls and interactions with existing paint.
Make an original composition from the brief. Do not smooth away unusual marks just to make the image
conventional. Use the instrument's own controls, look at the result, and adapt your next gestures.
Finish with a concrete note about which behaviours were useful, surprising or hard to control."""
DIRECTIONS = (
    "Explore memory and history: make earlier gestures affect later marks.",
    "Explore interaction with paint already on the canvas, such as pickup, dragging or edge response.",
    "Explore unusual geometry and gesture response, such as lag, splitting, orbiting or broken contact.",
    "Explore changing material behaviour, such as depletion, accumulation, damage or explicit growth steps.",
)
ACTIVE = ("queued", "running")


def _object(body: dict, keys: set[str]) -> None:
    if not isinstance(body, dict) or set(body) - keys:
        raise LaunchError("Expected an object with only: " + ", ".join(sorted(keys)) + ".")


def _text(body: dict, key: str, limit: int, default: str = "", required: bool = False) -> str:
    value = body.get(key, default)
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise LaunchError(f"{key} must be {'non-empty ' if required else ''}text, at most {limit} characters.")
    return value.strip()


def _int(body: dict, key: str, default: int, low: int, high: int) -> int:
    value = body.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise LaunchError(f"{key} must be a whole number from {low} to {high}.")
    return value


def model_config(body: dict) -> dict:
    from conveyor.commons.agent import EFFORTS, default_model
    from conveyor.pi import PROVIDERS

    _object(body, {"harness", "model", "provider", "effort", "cap"})
    kind = _text(body, "harness", 20, "claude")
    provider = _text(body, "provider", 40, "opencode-go")
    effort = _text(body, "effort", 20, "high")
    if kind not in ("claude", "pi") or provider not in PROVIDERS or effort not in EFFORTS:
        raise LaunchError("Choose a supported harness, provider and thinking effort.")
    cap = body.get("cap", 4.0)
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0.01 <= cap <= 100:
        raise LaunchError("cap must be from 0.01 to 100 dollars per session.")
    model = _text(body, "model", 200) or default_model(kind, provider)
    if model.startswith("-") or "\n" in model:
        raise LaunchError("model must be one line and cannot start with a dash.")
    return {"harness": kind, "model": model, "provider": provider, "effort": effort, "cap": cap}


def _decode(row) -> dict:
    out = dict(row)
    for key in ("config", "request", "result", "report"):
        if out.get(key):
            out[key] = json.loads(out[key])
    return out


def listing(conn) -> list[dict]:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='brush_labs'").fetchone():
        return []
    return [dict(r) for r in conn.execute(
        "SELECT l.*, (SELECT count(*) FROM brush_candidates c WHERE c.lab_id=l.id) AS candidates "
        "FROM brush_labs l ORDER BY created DESC")]


def detail(conn, lab_id: str) -> dict:
    row = conn.execute("SELECT * FROM brush_labs WHERE id=?", (lab_id,)).fetchone()
    if row is None:
        raise KeyError(lab_id)
    out = dict(row)
    out["candidates"] = [_decode(r) for r in conn.execute(
        "SELECT * FROM brush_candidates WHERE lab_id=? ORDER BY created, ordinal", (lab_id,))]
    out["jobs"] = [_decode(r) for r in conn.execute(
        "SELECT j.*, (SELECT coalesce(sum(s.cost),0) FROM sessions s "
        "WHERE s.run_id='brush-' || j.id) AS cost FROM brush_jobs j WHERE lab_id=? ORDER BY created", (lab_id,))]
    return out


def sources(conn) -> list[dict]:
    from conveyor.commons.catalog import tool_names
    from conveyor.painting.seeds import SEEDS

    out = [{"id": "seed:" + name, "label": "Seed: " + name, "source": source} for name, source in SEEDS.items()]
    seen = set(SEEDS.values())
    for row in conn.execute("SELECT id, summary, genome FROM organisms WHERE node='instrument' "
                            "AND (viable IS NULL OR viable!=0) ORDER BY created DESC"):
        source = json.loads(row["genome"]).get("source")
        if isinstance(source, str) and source and source not in seen:
            out.append({"id": "organism:" + row["id"], "label": row["summary"] or row["id"], "source": source})
            seen.add(source)
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='brush_candidates'").fetchone():
        for row in conn.execute("SELECT c.id,c.source,c.summary,l.name FROM brush_candidates c "
                                "JOIN brush_labs l ON l.id=c.lab_id WHERE c.published=1 AND c.status='ready'"):
            if row["source"] not in seen:
                out.append({"id": "brush:" + row["id"], "label": row["name"] + ": " + (row["summary"] or "Published brush"),
                            "source": row["source"]})
                seen.add(row["source"])
    for item in out:
        item["tools"] = tool_names(item.pop("source"))[0]
    return out


def _source(conn, source_id: str) -> str:
    from conveyor.painting.seeds import SEEDS

    if source_id.startswith("seed:") and source_id[5:] in SEEDS:
        return SEEDS[source_id[5:]]
    if source_id.startswith("brush:"):
        row = conn.execute("SELECT source FROM brush_candidates WHERE id=? AND status='ready' AND published=1",
                           (source_id[6:],)).fetchone()
        if row:
            return row["source"]
    if source_id.startswith("organism:"):
        row = conn.execute("SELECT genome FROM organisms WHERE id=? AND node='instrument' "
                           "AND (viable IS NULL OR viable!=0)", (source_id[9:],)).fetchone()
        if row:
            source = json.loads(row["genome"]).get("source")
            if isinstance(source, str) and source:
                return source
    raise LaunchError("Choose a saved instrument or seed.")


def _candidate(conn, lab_id: str, candidate_id: str) -> dict:
    row = conn.execute("SELECT * FROM brush_candidates WHERE id=? AND lab_id=? AND status='ready'",
                       (candidate_id, lab_id)).fetchone()
    if row is None:
        raise LaunchError("Choose a ready candidate from this lab.")
    return _decode(row)


def create_lab(db: Path, body: dict) -> dict:
    _object(body, {"name", "brief"})
    name = _text(body, "name", 100, required=True)
    brief = _text(body, "brief", 12000, required=True)
    init_db(db)
    conn = connect(db)
    try:
        row = {"id": new_id(), "name": name, "brief": brief, "created": time.time()}
        conn.execute("INSERT INTO brush_labs (id,name,brief,created) VALUES (:id,:name,:brief,:created)", row)
        conn.commit()
        return row
    finally:
        conn.close()


def create_job(db: Path, lab_id: str, kind: str, body: dict) -> dict:
    allowed = {"brief", "config", "candidate_id", "actions", "prompt"} if kind == "audition" else {
        "brief", "config", "parent_id", "parent2_id", "source_id", "locks", "lock_interface", "count"}
    if kind not in ("design", "audition"):
        raise LaunchError("Unknown Brush lab job.")
    _object(body, allowed)
    config = model_config(body.get("config", {}))
    brief = _text(body, "brief", 12000)
    conn = connect(db)
    try:
        # One batch or audition at a time per lab, including racing HTTP requests.
        conn.execute("BEGIN IMMEDIATE")
        lab = conn.execute("SELECT * FROM brush_labs WHERE id=?", (lab_id,)).fetchone()
        if lab is None:
            raise LaunchError("Lab not found.")
        if conn.execute("SELECT 1 FROM brush_jobs WHERE lab_id=? AND status IN ('queued','running')", (lab_id,)).fetchone():
            raise LaunchError("This lab already has work running. Stop it or wait for it to finish.")
        request = {"brief": brief or lab["brief"], "lab_brief": lab["brief"]}
        count = 0
        if kind == "audition":
            candidate = _candidate(conn, lab_id, _text(body, "candidate_id", 40, required=True))
            request.update(candidate_id=candidate["id"], source=candidate["source"],
                           prompt=_text(body, "prompt", 12000, candidate["prompt"], required=True),
                           actions=_int(body, "actions", 40, 5, 120))
        else:
            count = _int(body, "count", 4, 1, 4)
            parent_id = _text(body, "parent_id", 40)
            parent2_id = _text(body, "parent2_id", 40)
            parent = _candidate(conn, lab_id, parent_id) if parent_id else None
            inherited = json.loads(conn.execute("SELECT request FROM brush_jobs WHERE id=?", (parent["job_id"],)).fetchone()[0]) if parent else {}
            locks = body.get("locks", inherited.get("locks", []))
            if (not isinstance(locks, list) or len(locks) > 12
                    or any(not isinstance(s, str) or not s.strip() or len(s) > 1000 for s in locks)):
                raise LaunchError("locks must be up to 12 non-empty descriptions, each at most 1000 characters.")
            lock_interface = body.get("lock_interface", inherited.get("lock_interface", False))
            if type(lock_interface) is not bool:
                raise LaunchError("lock_interface must be on or off.")
            if parent2_id and (not parent_id or parent_id == parent2_id):
                raise LaunchError("Combining needs two different ready candidates.")
            other = _candidate(conn, lab_id, parent2_id) if parent2_id else None
            source_id = _text(body, "source_id", 80, "seed:round")
            source = parent["source"] if parent else _source(conn, source_id)
            request.update(parent_id=parent_id or None, parent2_id=parent2_id or None, source_id=source_id,
                           source=source, other_source=other["source"] if other else None, locks=locks,
                           lock_interface=lock_interface, count=count,
                           prompt=parent["prompt"] if parent else PAINTER_PROMPT,
                           history=(inherited.get("history", []) + [inherited["brief"]])[-8:] if parent else [])
        jid, now = new_id(), time.time()
        conn.execute("INSERT INTO brush_jobs (id,lab_id,kind,created,status,config,request) "
                     "VALUES (?,?,?,?,'queued',?,?)", (jid, lab_id, kind, now, dumps(config), dumps(request)))
        for i in range(count):
            conn.execute("INSERT INTO brush_candidates (id,lab_id,job_id,ordinal,parent_id,parent2_id,created,status,prompt) "
                         "VALUES (?,?,?,?,?,?,?,'queued',?)", (new_id(), lab_id, jid, i, request["parent_id"],
                                                              request["parent2_id"], now, request["prompt"]))
        conn.commit()
        return {"id": jid, "lab_id": lab_id, "kind": kind, "status": "queued", "config": config,
                "session_cap_total": config["cap"] * (count or 1)}
    finally:
        conn.close()


def publish(db: Path, lab_id: str, cid: str, body: dict) -> dict:
    from conveyor.commons.catalog import pair_key

    _object(body, {"prompt", "confirm_traits"})
    conn = connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        candidate = _candidate(conn, lab_id, cid)
        prompt = _text(body, "prompt", 12000, candidate["prompt"], required=True)
        request = json.loads(conn.execute("SELECT request FROM brush_jobs WHERE id=?", (candidate["job_id"],)).fetchone()[0])
        if request["locks"] and body.get("confirm_traits") is not True:
            raise LaunchError("Review the free-form trait locks and confirm them before publishing.")
        if candidate["published"] and candidate["prompt"] != prompt:
            raise LaunchError("Published pairs are immutable. Branch this candidate to change its painter prompt.")
        conn.execute("UPDATE brush_candidates SET published=1,prompt=? WHERE id=?", (prompt, cid))
        conn.commit()
        return {"id": cid, "published": True, "pair_id": pair_key(candidate["source"], prompt)}
    finally:
        conn.close()


def interface_contract(source: str) -> dict:
    """Read literal tool/parameter declarations without executing the module. Docs may change."""
    values = {"TOOLS": {}, "VIEWS": {}}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in values:
                    values[name.id] = ast.literal_eval(node.value)
    for tools in values.values():
        for tool in tools.values():
            tool.pop("doc", None)
            for param in tool.get("params", {}).values():
                param.pop("doc", None)
    return values


def close_job(db: Path, jid: str, status: str, error: str | None = None, result: dict | None = None) -> None:
    """Terminal states are sticky, so process-exit reconciliation can't overwrite a saved result."""
    conn = connect(db)
    try:
        conn.execute("UPDATE brush_candidates SET status=?,error=? WHERE job_id=? AND status IN ('queued','running')",
                     ("stopped" if status == "stopped" else "failed", error or "Designer did not return a result.", jid))
        conn.execute("UPDATE brush_jobs SET status=?,ended=?,error=?,result=coalesce(?,result) "
                     "WHERE id=? AND status IN ('queued','running')",
                     (status, time.time(), error, dumps(result) if result is not None else None, jid))
        conn.commit()
    finally:
        conn.close()


def recover_jobs(db: Path) -> None:
    """Release abandoned work after a server crash. Never interrupt a still-live worker.

    A just-created row may not have a PID yet. Give it a minute to be claimed by its worker.
    This runs at launcher startup, not from the read-only HTTP endpoints.
    """
    if not db.exists():
        return
    conn = connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='brush_jobs'").fetchone():
            return
        rows = list(conn.execute("SELECT id,worker_pid,created FROM brush_jobs WHERE status IN ('queued','running')"))
        for row in rows:
            pid = row["worker_pid"]
            if pid:
                try:
                    os.kill(pid, 0)
                    continue
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue
            elif time.time() - row["created"] < 60:
                continue
            error = "Worker ended without saving a result. Start a new round to continue."
            conn.execute("UPDATE brush_candidates SET status='failed',error=? WHERE job_id=? "
                         "AND status IN ('queued','running')", (error, row["id"]))
            conn.execute("UPDATE brush_jobs SET status='failed',ended=?,error=? WHERE id=?",
                         (time.time(), error, row["id"]))
        conn.commit()
    finally:
        conn.close()


def _update_candidate(db: Path, cid: str, **fields) -> None:
    allowed = {"status", "source", "summary", "sheet", "report", "session_id", "error"}
    if set(fields) - allowed:
        raise ValueError("Unknown candidate fields")
    conn = connect(db)
    try:
        conn.execute("UPDATE brush_candidates SET " + ",".join(k + "=?" for k in fields) + " WHERE id=?",
                     (*[dumps(v) if k == "report" else v for k, v in fields.items()], cid))
        conn.commit()
    finally:
        conn.close()


def design_prompt(request: dict, ordinal: int, prior: list[str]) -> str:
    direction = DIRECTIONS[ordinal] if request["count"] > 1 else "Follow the requested revision precisely."
    locks = "\n".join("- " + s for s in request["locks"]) or "None."
    text = f"""Lab's original brief:
{request['lab_brief']}

Human's request for this round:
{request['brief']}

Candidate {ordinal + 1} of {request['count']}. {direction}
These directions are alternatives, not requirements that override the human's brief or locks.
Change the mechanism, not merely numeric settings or surface noise. Make it controllable enough to explore.
When revising, preserve what the human likes. Do not replace it just to make a conventional brush.

Traits to preserve, to be reviewed by the human:
{locks}

Interface lock: {'ON. TOOLS and VIEWS names and parameter declarations must stay exactly the same. Documentation may change.' if request['lock_interface'] else 'OFF. You may redesign the interface.'}

Starting instrument:
```python
{request['source']}
```
"""
    if request.get("history"):
        text += "\nEarlier human requests in this branch, oldest first:\n" + "\n".join(request["history"])
    if request.get("other_source"):
        text += f"\nSecond parent to combine with the first:\n```python\n{request['other_source']}\n```\n"
    if prior:
        text += "\nEarlier candidates in this batch; take a different direction:\n" + "\n".join(prior)
    return text + "\nInclude examples that reveal repeated use, state changes and interaction with existing paint."


def _design_content(db, request: dict, ordinal: int, prior: list[str]) -> list[dict]:
    """Show the actual parent marks and the latest audition, not just its source code."""
    content = [{"type": "text", "text": design_prompt(request, ordinal, prior)}]
    conn = connect(db, readonly=True)
    try:
        for parent_id in (request["parent_id"], request["parent2_id"]):
            if not parent_id:
                continue
            parent = conn.execute("SELECT sheet FROM brush_candidates WHERE id=?", (parent_id,)).fetchone()
            evidence = [(parent["sheet"], "Parent " + parent_id + " example sheet")]
            audition = conn.execute("SELECT request,result FROM brush_jobs WHERE kind='audition' AND status='finished' "
                                    "AND json_extract(request,'$.candidate_id')=? ORDER BY created DESC LIMIT 1",
                                    (parent_id,)).fetchone()
            if audition:
                result = json.loads(audition["result"])
                note = result.get("details", {}).get("note", "")
                brief = json.loads(audition["request"])["brief"]
                evidence.append((result.get("artifacts", {}).get("painting"),
                                 f"Parent {parent_id}, latest audition. Brief: {brief}\nPainter's note: {note}"))
            for artifact, caption in evidence:
                data = conn.execute("SELECT data FROM artifacts WHERE name=?", (artifact,)).fetchone()
                if data:
                    content.extend([{"type": "text", "text": caption}, {"type": "png", "data": data[0]}])
    finally:
        conn.close()
    return content


def _replays(source: str, store: Store) -> list[list[dict]]:
    """Only called in the worker's main thread, where instrument time limits are effective."""
    import numpy as np

    from conveyor.painting.canvas import Canvas, to_png
    from conveyor.painting.instrument import Instrument

    inst = Instrument(source)
    sequences = []
    for i, calls in enumerate(inst.spec.examples):
        canvas, pen, rng = Canvas(512, 512), inst.new_state(), np.random.default_rng(i)
        frames = [{"image": store.artifact(to_png(canvas.img)), "tool": "Blank canvas", "args": {}, "state": pen.copy()}]
        for tool, args in calls:
            note = inst.call(tool, args, pen, canvas, rng)
            frames.append({"image": store.artifact(to_png(canvas.img)), "tool": tool, "args": args, "note": note,
                           "state": json.loads(json.dumps(pen, default=str))})
        sequences.append(frames)
    return sequences


def run_job(db: Path, jid: str) -> None:
    from conveyor.__main__ import _harness
    from conveyor.commons.agent import _args
    from conveyor.harness import BudgetExhausted, Job, Meter, RateLimited
    from conveyor.painting.problem import ClaudeInstrumentMutator, Setup, probe_source

    db = db.resolve()
    conn = connect(db)
    try:
        row = conn.execute("SELECT * FROM brush_jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise ValueError("Unknown Brush lab job")
        job = _decode(row)
        claimed = conn.execute("UPDATE brush_jobs SET status='running',worker_pid=? WHERE id=? AND status='queued'",
                               (os.getpid(), jid)).rowcount
        conn.commit()
        candidates = [dict(r) for r in conn.execute("SELECT * FROM brush_candidates WHERE job_id=? ORDER BY ordinal", (jid,))]
    finally:
        conn.close()
    if not claimed:
        return
    signal.signal(signal.SIGINT, signal.default_int_handler)
    config, request = job["config"], job["request"]
    store = Store(db, run_id="brush-" + jid, check_run=False)
    harness = None
    try:
        args = _args(config, db)
        harness = _harness(config["harness"], config["model"], config["effort"], args, store,
                           Meter(budget_usd=config["cap"] * max(1, len(candidates))), {})
        root = db.parent / f"{db.stem}-sessions" / ("brush-" + jid)
        setup = Setup(target=None, width=512, height=512, work_dir=root, judge=False,
                      paint_budget_usd=config["cap"], mutate_budget_usd=config["cap"])
        if job["kind"] == "audition":
            result = _audition(db, jid, request, setup, store, harness)
            close_job(db, jid, "finished", result=result)
            return
        # Reuse the existing physics and sandbox instructions, but replace the copying objective.
        system = ClaudeInstrumentMutator("invent", harness, setup, store, 512, 1)._system()
        system = system.replace("The painter is another Claude model. It copies a target painting onto a ",
                                "The painter is another model. It explores original images on a ")
        system = system.replace("It sees the target, its canvas when it looks, your docstring, and your tool definitions.",
                                "It sees its brief, its canvas when it looks, your docstring, and your tool definitions.")
        system = system.replace("What decides your instrument's fate is the painting made with it.",
                                "The human decides what to keep. There is no likeness score or automated ranking.")
        prior = []
        for candidate in candidates:
            cid = candidate["id"]
            _update_candidate(db, cid, status="running")
            try:
                folder = root / cid
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "job.json").write_text(dumps({"width": 512, "height": 512, "max_tries": 5,
                    "locked_interface": interface_contract(request["source"]) if request["lock_interface"] else None}))
                task = Job(purpose="brush lab design", system_prompt=system,
                           content=_design_content(db, request, candidate["ordinal"], prior),
                           cwd=folder, mcp={"name": "bench", "command": sys.executable,
                                           "args": ["-m", "conveyor.painting.workbench", str(folder)]},
                           tools=["try_instrument", "submit_instrument"], stop_tools=["submit_instrument"],
                           max_budget_usd=config["cap"], task_budget=80_000, timeout=20 * 60, node="brush",
                           on_start=lambda sid, cid=cid: _update_candidate(db, cid, session_id=sid))
                outcome = _run_interruptible(harness, harness.run, task)
                submitted = folder / "submitted.json"
                if not submitted.exists():
                    raise RuntimeError(outcome.error or "Designer ended without submitting an instrument.")
                record = json.loads(submitted.read_text())
                source = record["source"]
                if request["lock_interface"] and interface_contract(source) != interface_contract(request["source"]):
                    raise ValueError("Interface lock violated.")
                report, sheet = probe_source(source, 512, 512, root)
                if not report["ok"] or not sheet:
                    raise ValueError("; ".join(report["errors"]) or "No demo sheet.")
                report["interface_lock"] = "passed" if request["lock_interface"] else "off"
                report["trait_locks_to_review"] = request["locks"]
                report["replays"] = _replays(source, store)
                sheet_name = store.artifact(sheet)
                store.flush()  # Readers must never see a ready candidate whose images aren't saved yet.
                _update_candidate(db, cid, status="ready", source=source, summary=record.get("summary", ""),
                                  sheet=sheet_name, report=report, session_id=outcome.session_id)
                prior.append(record.get("summary", ""))
            except (BudgetExhausted, RateLimited):
                raise
            except Exception as error:  # noqa: BLE001 - a failed candidate must not discard its siblings
                _update_candidate(db, cid, status="failed", error=str(error)[:2000])
        conn = connect(db, readonly=True)
        try:
            failed = conn.execute("SELECT count(*) FROM brush_candidates WHERE job_id=? AND status!='ready'", (jid,)).fetchone()[0]
        finally:
            conn.close()
        close_job(db, jid, "finished" if not failed else "failed",
                  f"{failed} candidate(s) failed. Ready candidates are still available." if failed else None)
    except BaseException as error:
        close_job(db, jid, "stopped" if isinstance(error, KeyboardInterrupt) else "failed", str(error) or "Interrupted")
        raise
    finally:
        if harness:
            harness.stop_all()
        store.close()


def _run_interruptible(harness, fn, *args):
    # Keep signal handling outside the model stream reader, as in studio.run_request.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        try:
            return future.result()
        except KeyboardInterrupt:
            harness.stop_all()
            raise


def _audition(db, jid, request, setup, store, harness) -> dict:
    from conveyor.evolve import Organism
    from conveyor.painting.problem import Painter, Roles

    setup.brief, setup.actions = request["brief"], request["actions"]
    painter = Painter(setup, store, Roles(paint=harness))
    original_start = store.start_session

    def start_session(sid, **fields):
        original_start(sid, **fields)
        if fields["purpose"].startswith("paint"):
            conn = connect(db)
            try:
                conn.execute("UPDATE brush_jobs SET result=? WHERE id=? AND status='running'",
                             (dumps({"session_id": sid}), jid))
                conn.commit()
            finally:
                conn.close()

    store.start_session = start_session
    instrument = Organism(node="instrument", id=request["candidate_id"], genome={"source": request["source"]})
    strategy = Organism(node="painter", genome={"prompt": request["prompt"]})
    painting = _run_interruptible(harness, painter.paint, instrument, strategy)
    if not painting.viable:
        raise RuntimeError(painting.error or "Audition failed.")
    store.flush()
    return {"artifacts": painting.artifacts, "details": painting.details, "session_id": painting.session_id}


if __name__ == "__main__":
    run_job(Path(sys.argv[1]), sys.argv[2])
