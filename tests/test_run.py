"""Whole runs: offline, and in Claude mode against the fake CLI, then read back through the dashboard API."""

import json
import threading
import urllib.request

import pytest

from conveyor.claude import ClaudeCode
from conveyor.claude import Meter
from conveyor.claude import Settings
from conveyor.evolve import Conductor
from conveyor.painting.instrument import probe
from conveyor.painting.problem import Setup
from conveyor.painting.problem import build
from conveyor.painting.seeds import PEN
from conveyor.server import make_server
from conveyor.store import Store


def serve(db):
    server = make_server(db, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}{path}") as r:
            return json.loads(r.read()) if r.headers["Content-Type"] == "application/json" else r.read()
    return server, get


def test_offline_run_and_dashboard(tmp_path):
    store = Store(tmp_path / "off.db", run_name="offline", config={"budget": None})
    setup = Setup(width=64, actions=15, mode="offline", seeds=["round", "pen"], work_dir=tmp_path / "s", parents=2)
    nodes, _ = build(setup, store, None)
    conductor = Conductor(nodes, store, lanes=2, schedule=[("painter", 1), ("instrument", 1)])
    conductor.run(2)
    store.close()
    assert len(conductor.pops["instrument"].organisms) == 2 + 4

    server, get = serve(tmp_path / "off.db")
    try:
        [run] = get("/api/runs")
        ov = get(f"/api/runs/{run['id']}")
        assert {n["name"] for n in ov["nodes"]} == {"instrument", "painter"}
        assert probe(PEN, 64, 80)["niche"] in ov["archive"]["cells"] and len(ov["archive"]["niches"]) == 12
        assert get(f"/api/runs/{run['id']}/history")["evaluations"]
        assert get(f"/api/runs/{run['id']}/mutators")[0]["mutator"] == "jitter"
        orgs = get(f"/api/runs/{run['id']}/organisms")
        painted = [o for o in orgs if o["painting"]]
        assert painted and all(o["pair"] for o in painted)  # the lineage view crops the target out of a pair image
        assert all(o["parent_id"] in {x["id"] for x in orgs} for o in orgs if o["mutator"])  # every child's parent is listed
        detail = get(f"/api/organisms/{orgs[0]['id']}")
        assert detail["genome"]["source"] and "evaluations" in detail
        ev = next(e for o in orgs for e in get(f"/api/organisms/{o['id']}")["evaluations"] if e["session_id"])
        session = get(f"/api/sessions/{ev['session_id']}")
        assert session["strokes"] and session["purpose"] == "paint (greedy)"
        assert get(f"/artifacts/{ev['artifacts']['painting']}").startswith(b"\x89PNG")
        assert b"conveyor" in get("/")
    finally:
        server.shutdown()


def test_claude_mode_run_with_the_fake_cli(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", PEN)
    edits = [{"old": "Block in the large areas of colour first", "new": "Darks first, then lights"}]
    monkeypatch.setenv("FAKE_CLAUDE_STRUCTURED", json.dumps({"edits": edits, "summary": "order"}))
    store = Store(tmp_path / "c.db", config={"budget": 5})
    meter = Meter(budget_usd=5)
    claude = ClaudeCode(Settings(binary=str(fake_claude)), store, meter, lanes=2)
    setup = Setup(width=64, actions=10, work_dir=tmp_path / "s", parents=2,
                  operator_weights={"refine": 0.0, "invent": 1.0, "recombine": 0.0})
    nodes, painter = build(setup, store, claude)
    conductor = Conductor(nodes, store, lanes=2, schedule=[("painter", 1), ("instrument", 1)],
                          should_stop=meter.over_budget)
    conductor.run(1)
    store.close()
    inst = conductor.pops["instrument"]
    assert probe(PEN, 64, 80)["niche"] in inst.niches()  # the invented pen got painted and filed in its niche
    assert any(o.genome["prompt"].startswith("You are copying") and "Darks first, then lights" in o.genome["prompt"]
               for o in conductor.pops["painter"].organisms.values())
    assert meter.sessions >= 5 and meter.spent == pytest.approx(0.01 * meter.sessions)

    server, get = serve(tmp_path / "c.db")
    try:
        run_id = get("/api/runs")[0]["id"]
        muts = {m["mutator"]: m for m in get(f"/api/runs/{run_id}/mutators")}
        # Each invention was aimed at an empty niche; the fake submits the same pen whatever it's asked for.
        assert muts["claude:invent"]["asked_niche"] == 2 and muts["claude:invent"]["viable"] == 2
        assert get(f"/api/runs/{run_id}")["rate_limit"]["status"] == "allowed"
    finally:
        server.shutdown()


def test_a_run_waits_out_a_usage_limit(tmp_path, fake_claude, monkeypatch):
    """The limit ends the first painting partway: the run waits for the reset and paints it again, unscored."""
    import sqlite3

    from conveyor import harness

    monkeypatch.setattr(harness, "RESET_GRACE", 0.0)
    flag = tmp_path / "cut-off"
    flag.touch()
    monkeypatch.setenv("FAKE_CLAUDE_CUT_OFF", str(flag))
    monkeypatch.setenv("FAKE_CLAUDE_RESETS_IN", "2")
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", PEN)
    edits = [{"old": "Block in the large areas of colour first", "new": "Darks first, then lights"}]
    monkeypatch.setenv("FAKE_CLAUDE_STRUCTURED", json.dumps({"edits": edits, "summary": "order"}))
    store = Store(tmp_path / "w.db", config={"budget": 5})
    waits, resumes = [], []

    def waiting(reason, until):
        waits.append(until)
        store.emit("waiting", reason=reason, until=until)

    def resumed():
        resumes.append(True)
        store.emit("resumed")

    meter = Meter(budget_usd=5, max_usage=0.85, wait_hours=1, on_wait=waiting, on_resume=resumed)
    claude = ClaudeCode(Settings(binary=str(fake_claude)), store, meter, lanes=2)
    setup = Setup(width=64, actions=10, work_dir=tmp_path / "s", parents=2, judge=False)
    nodes, _ = build(setup, store, claude)
    conductor = Conductor(nodes, store, lanes=2, schedule=[("painter", 1), ("instrument", 1)],
                          should_stop=meter.should_stop, wait_out_limits=True)
    conductor.run(1)
    store.close()
    assert conductor.stop_reason is None and len(waits) == 1 and resumes == [True]
    conn = sqlite3.connect(tmp_path / "w.db")
    [(cut,)] = conn.execute("SELECT id FROM sessions WHERE purpose='paint' AND status='error'").fetchall()
    assert conn.execute("SELECT count(*) FROM evaluations WHERE session_id=?", (cut,)).fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM evaluations WHERE reason='seed' AND viable=1").fetchone()[0] == 2

    server, get = serve(tmp_path / "w.db")
    try:
        ov = get(f"/api/runs/{get('/api/runs')[0]['id']}")
        assert ov["waiting"] is None and ov["finished"]
    finally:
        server.shutdown()


def test_two_runs_can_share_one_database(tmp_path):
    """Two offline runs writing the same file at once: both complete, nothing dropped."""
    import sqlite3
    import subprocess
    import sys

    db = tmp_path / "shared.db"
    cmd = [sys.executable, "-m", "conveyor", "run", "--offline", "--no-serve", "--cycles", "2", "--actions", "15",
           "--width", "64", "--parents", "2", "--seeds", "round,pen", "--db", str(db)]
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=300)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    assert not any("[store]" in o for o in outs), outs
    conn = sqlite3.connect(db)
    runs = [r for (r,) in conn.execute("SELECT id FROM runs")]
    assert len(runs) == 2
    for run in runs:
        finished = conn.execute("SELECT count(*) FROM events WHERE run_id=? AND kind='run_finished'", (run,)).fetchone()[0]
        organisms = conn.execute("SELECT count(*) FROM organisms WHERE run_id=? AND node='instrument'", (run,)).fetchone()[0]
        assert finished == 1 and organisms == 2 + 4
