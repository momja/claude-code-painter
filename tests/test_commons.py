"""The shared canvas: tiles and replay, concurrent agents, the tool-call budget, moves, messages, the catalog,
spawning, a whole agent driven by the fake CLI, and the HTTP routes."""

import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from conveyor.__main__ import build_parser
from conveyor.commons import catalog
from conveyor.commons import views
from conveyor.commons.agent import create_agent
from conveyor.commons.agent import run_agent
from conveyor.commons.agent import session_scope
from conveyor.commons.server import AgentSession
from conveyor.commons.server import CommonsServer
from conveyor.commons.tiles import PAPER_RGB
from conveyor.commons.tiles import TILE
from conveyor.commons.tiles import SharedCanvas
from conveyor.commons.tiles import create_canvas
from conveyor.commons.tiles import decode_tile
from conveyor.launch import LaunchError
from conveyor.launch import Launcher
from conveyor.painting.seeds import PEN
from conveyor.painting.seeds import ROUND
from conveyor.server import make_server
from conveyor.store import Store
from conveyor.store import connect
from test_launch import call as http
from test_launch import wait_for

RED = "#c03030"


@pytest.fixture
def canvas(tmp_path):
    db = tmp_path / "c.db"
    return db, create_canvas(db, "Commons", viewport=256, max_calls=20)


def agent(db: Path, canvas_row: dict, tmp: Path, *, x=0, y=0, source=ROUND, max_calls=None, name="a") -> CommonsServer:
    conn = connect(db)
    conn.execute("INSERT INTO canvas_agents (id, canvas_id, created, status, config, x, y, max_calls) "
                 "VALUES (?, ?, ?, 'running', '{}', ?, ?, ?)",
                 (name, canvas_row["id"], time.time(), x, y, max_calls or canvas_row["config"]["max_calls"]))
    conn.commit()
    conn.close()
    d = tmp / f"agent-{name}"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"db": str(db), "canvas_id": canvas_row["id"], "agent_id": name,
                                            "source": source, "x": x, "y": y, "max_calls": max_calls}))
    return CommonsServer(AgentSession(d))


def tool(server: CommonsServer, name: str, args: dict | None = None) -> dict:
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": args or {}}})
    return reply["result"]


def text_of(result: dict) -> str:
    return "\n".join(b.get("text", "") for b in result["content"])


def stroke(x, y, color=RED, **kw):
    return {"x": x, "y": y, "length": 40, "size": 6, "color": color, "pressure": 1.0, **kw}


# ---- tiles and replay ---------------------------------------------------------------------------------------


def test_unpainted_canvas_is_paper_everywhere_including_negative_coordinates(canvas):
    db, row = canvas
    shared = SharedCanvas(db, row["id"])
    region = shared.read(-300, -77, 200, 150)
    assert region.image().shape == (150, 200, 3)
    assert sorted(region.tiles) == [(tx, ty) for tx in (-3, -2, -1) for ty in (-1, 0)]  # x -300..-101, y -77..72
    assert all((t == PAPER_RGB).all() for t in region.tiles.values())
    assert region.changed(region.image()) == {}


def test_a_stroke_writes_versions_that_replay_reads_back(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path, x=-200, y=-60)
    assert not tool(server, "stroke", stroke(180, 30))["isError"]  # canvas x -20..20 at y -30: crosses x = 0
    conn = connect(db, readonly=True)
    history = views.history(conn, row["id"])
    seq = history["ops"][-1][0]
    touched = {(tx, ty) for s, tx, ty in history["versions"] if s == seq}
    assert {(-1, -1), (0, -1)} <= touched
    now = SharedCanvas(db, row["id"]).read(-200, -60, 256, 256).image()
    for tx, ty in touched:
        tile = decode_tile(views.tile(conn, row["id"], tx, ty, seq))
        x0, y0 = tx * TILE + 200, ty * TILE + 60  # where that tile sits in the viewport
        ys, xs = slice(max(0, y0), min(256, y0 + TILE)), slice(max(0, x0), min(256, x0 + TILE))
        part = tile[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        assert np.array_equal(part, (now[ys, xs] * 255).round().astype(np.uint8))
    # A later stroke over the same tile adds a version; the old one stays for replay.
    tool(server, "stroke", stroke(180, 32, color="#2040a0"))
    heads = dict(((tx, ty), s) for tx, ty, s in views.canvas(conn, row["id"])["heads"])
    assert heads[(0, -1)] > seq
    assert views.tile(conn, row["id"], 0, -1, seq) is not None


def test_two_agents_painting_one_tile_at_once_lose_nothing(canvas, tmp_path):
    db, row = canvas
    a, b = agent(db, row, tmp_path, name="a"), agent(db, row, tmp_path, x=64, name="b")
    errors = []

    def paint(server, y0, color):
        for i in range(12):
            r = tool(server, "stroke", {"x": 70, "y": y0 + 4 * i, "angle": 0, "length": 20, "size": 1, "color": color,
                                        "pressure": 1.0})
            errors.extend([text_of(r)] if r["isError"] else [])

    threads = [threading.Thread(target=paint, args=(a, 10, "#ff0000")), threading.Thread(target=paint, args=(b, 12, "#0000ff"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    img = SharedCanvas(db, row["id"]).read(0, 0, 256, 256).image()
    # a's strokes at x 70..90 (canvas), b's at 134..154 (its viewport starts at x 64); every row of each is there.
    for i in range(12):
        assert img[10 + 4 * i, 80, 0] > 0.9 and img[10 + 4 * i, 80, 2] < 0.1
        assert img[12 + 4 * i, 144, 2] > 0.9 and img[12 + 4 * i, 144, 0] < 0.1


# ---- the budget, moves, messages ------------------------------------------------------------------------------


def test_every_call_counts_and_the_last_one_stops_the_session(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path, max_calls=3)
    assert "2 left" in text_of(tool(server, "look"))
    refused = tool(server, "stroke", {"x": 10})  # missing arguments: refused, still counted
    assert refused["isError"] and "1 left" in text_of(refused)
    last = tool(server, "look")
    assert "last tool call" in text_of(last) and last["structuredContent"] == {"stop": True}
    over = tool(server, "stroke", stroke(10, 10))
    assert over["isError"] and "used all 3" in text_of(over)
    agent_row = connect(db).execute("SELECT calls_used FROM canvas_agents WHERE id='a'").fetchone()
    assert agent_row[0] == 3
    assert len(views.history(connect(db, readonly=True), row["id"])["ops"]) == 3  # the refused fourth isn't an op


def test_a_move_goes_at_most_three_quarters_of_the_viewport(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path, x=100, y=100)
    result = tool(server, "move_viewport", {"angle": 90, "distance": 10_000})
    assert not result["isError"] and result["content"][1]["type"] == "image"
    assert (server.s.x, server.s.y) == (100, 100 + 192)  # 0.75 x 256
    tool(server, "move_viewport", {"angle": 180, "distance": 50})
    assert (server.s.x, server.s.y) == (50, 292)
    stored = connect(db).execute("SELECT x, y FROM canvas_agents WHERE id='a'").fetchone()
    assert tuple(stored) == (50, 292)
    assert tool(server, "move_viewport", {"distance": 5})["isError"]


def test_messages_are_pixels_that_can_be_painted_over(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path)
    result = tool(server, "write_message", {"text": "MEET AT 900,0", "x": 10, "y": 10, "scale": 2, "color": "#000000"})
    assert not result["isError"], text_of(result)
    img = SharedCanvas(db, row["id"]).read(0, 0, 256, 256).image()
    ink = img[10:36, 10:200].min(axis=-1) < 0.05
    assert ink.sum() > 100 and not (img[60:, :].min(axis=-1) < 0.05).any()
    tool(server, "paint_batch", {"tool": "stroke", "defaults": {"length": 200, "size": 12, "color": "#ffffff",
                                                               "pressure": 1.0, "angle": 0},
                                 "calls": [{"x": 10, "y": 14}, {"x": 10, "y": 30}]})
    img = SharedCanvas(db, row["id"]).read(0, 0, 256, 256).image()
    assert (img[10:36, 10:200].min(axis=-1) < 0.05).sum() < ink.sum() / 4
    huge = tool(server, "write_message", {"text": "x" * 200, "x": 0, "y": 0, "scale": 4, "background": "#ffffff"})
    assert huge["isError"] and "area" in text_of(huge)
    messages = views.canvas(connect(db, readonly=True), row["id"])["messages"]
    assert [m["text"] for m in messages] == ["MEET AT 900,0"]


def test_the_pen_keeps_its_state_across_calls(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path, source=PEN)
    assert not tool(server, "start", {"x": 20, "y": 20, "color": RED})["isError"]
    assert not tool(server, "move", {"dx": 60, "dy": 0})["isError"]
    img = SharedCanvas(db, row["id"]).read(0, 0, 256, 256).image()
    assert img[20, 50, 0] - img[20, 50, 2] > 0.2


def test_an_instrument_with_a_canvas_tool_name_is_refused(canvas, tmp_path):
    db, row = canvas
    clash = ROUND.replace("stroke", "move_viewport")
    with pytest.raises(ValueError, match="clash"):
        agent(db, row, tmp_path, source=clash)


# ---- the catalog and spawning ---------------------------------------------------------------------------------


def seeded_db(tmp_path) -> Path:
    db = tmp_path / "c.db"
    for name, score in (("run one", 0.6), ("run two", 0.7)):
        store = Store(db, run_name=name, config={})
        store.organism({"id": f"{name}-round", "node": "instrument", "created": time.time(), "genome": {"source": ROUND},
                        "summary": "seed: round", "viable": 1})
        store.organism({"id": f"{name}-prompt", "node": "painter", "created": time.time(), "genome": {"prompt": "Paint big."},
                        "viable": 1})
        store.evaluation({"id": f"{name}-ev", "node": "instrument", "organism_id": f"{name}-round", "score": score,
                          "viable": 1, "ended": time.time()})
        store.close()
    store = Store(db, run_name="clash", config={})
    store.organism({"id": "clash", "node": "instrument", "created": time.time(), "viable": 1,
                    "genome": {"source": 'TOOLS = {"write_message": {}}\n'}, "summary": "talks too much"})
    store.organism({"id": "dead", "node": "instrument", "created": time.time(), "viable": 0,
                    "genome": {"source": "TOOLS = {}\n"}, "summary": "never worked"})
    store.close()
    return db


def test_the_catalog_lists_each_text_once_across_runs(tmp_path):
    db = seeded_db(tmp_path)
    cat = catalog.catalog(connect(db, readonly=True))
    rounds = [e for e in cat["instruments"] if e["text"] == ROUND]
    assert len(rounds) == 1 and rounds[0]["best_score"] == 0.7 and rounds[0]["tools"] == ["stroke"]
    assert {o["run"] for o in rounds[0]["origins"]} == {"run one", "run two"}
    assert rounds[0]["id"] == catalog.key("instrument", ROUND)
    assert not any(e["summary"] == "never worked" or e["label"] == "never worked" for e in cat["instruments"])
    assert [e for e in cat["prompts"] if e["text"] == "Paint big."][0]["origins"]
    assert any(e["label"] == "starting strategy" for e in cat["prompts"])


@pytest.mark.parametrize("change,match", [
    ({"harness": "gpt"}, "harness"), ({"effort": "huge"}, "effort"), ({"instrument_id": "i-nope"}, "instrument"),
    ({"prompt_id": catalog.key("instrument", ROUND)}, "prompt"), ({"x": "left"}, "whole numbers"),
    ({"x": 10**9}, "within"), ({"cap": 0}, "cap"), ({"model": "--dangerous"}, "dash"), ({"color": "red"}, "Expected"),
])
def test_spawning_checks_its_request(tmp_path, change, match):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C")
    body = {"instrument_id": catalog.key("instrument", ROUND), "prompt_id": catalog.key("prompt", "Paint big."),
            "harness": "claude", **change}
    with pytest.raises(LaunchError, match=match):
        create_agent(db, row["id"], body)


def test_spawning_refuses_an_instrument_whose_tools_clash(tmp_path):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C")
    with pytest.raises(LaunchError, match="write_message"):
        create_agent(db, row["id"], {"instrument_id": catalog.key("instrument", 'TOOLS = {"write_message": {}}\n'),
                                     "prompt_id": catalog.key("prompt", "Paint big.")})


def test_an_agent_runs_to_the_end_of_its_budget(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_claude.parent}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("FAKE_CLAUDE_MESSAGE", "east is empty")
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=9)
    spawned = create_agent(db, row["id"], {"instrument_id": catalog.key("instrument", ROUND),
                                           "prompt_id": catalog.key("prompt", "Paint big."), "x": -40, "y": 30,
                                           "model": "fake-model"})
    run_agent(db, spawned["id"])
    conn = connect(db, readonly=True)
    state = views.canvas(conn, row["id"])
    done = state["agents"][0]
    assert done["status"] == "finished" and done["calls_used"] == 9 and done["error"] is None
    assert (done["x"], done["y"]) == (-40 + 192, 30)
    assert state["heads"] and [m["text"] for m in state["messages"]] == ["east is empty"]
    session = conn.execute("SELECT * FROM sessions WHERE id=?", (done["session_id"],)).fetchone()
    assert session["run_id"] == session_scope(row["id"]) and session["purpose"] == "canvas agent"
    request = json.loads(session["request"])
    assert "Paint big." in request["system"] and "shared canvas" in request["system"]
    assert [c["type"] for c in request["content"]].count("image") == 2  # the viewport and the demo sheet
    strokes = conn.execute("SELECT count(*), sum(snapshot IS NOT NULL) FROM strokes WHERE session_id=?",
                           (done["session_id"],)).fetchone()
    assert strokes[0] == 9 and strokes[1] >= 3
    assert not conn.execute("SELECT 1 FROM runs WHERE id=?", (session_scope(row["id"]),)).fetchone()


# ---- HTTP ------------------------------------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path):
    db = seeded_db(tmp_path)
    launcher = Launcher(db, build_parser)
    server = make_server(db, "127.0.0.1", 0, launcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", db, launcher
    launcher.shutdown(grace=5)
    server.shutdown()


def test_the_page_and_its_reads(served, tmp_path):
    base, db, _ = served
    with urllib.request.urlopen(base + "/canvas") as r:
        assert b"Shared canvas" in r.read()
    assert http(base, "GET", "/api/canvases") == (200, [])
    status, made = http(base, "POST", "/api/canvases", {"name": "Commons", "viewport": 256})
    assert status == 201 and made["config"]["viewport"] == 256
    assert http(base, "POST", "/api/canvases", {"viewport": 9000})[0] == 422
    server = agent(db, made, tmp_path)
    tool(server, "stroke", stroke(10, 10))
    status, state = http(base, "GET", f"/api/canvases/{made['id']}")
    assert status == 200 and state["heads"] and state["agents"][0]["calls_used"] == 1
    status, later = http(base, "GET", f"/api/canvases/{made['id']}?since={state['seq']}")
    assert later["heads"] == []
    tx, ty, seq = state["heads"][0]
    with urllib.request.urlopen(f"{base}/api/canvases/{made['id']}/tiles/{tx}/{ty}/{seq}") as r:
        assert r.headers["Content-Type"] == "image/png" and r.read().startswith(b"\x89PNG")
    status, history = http(base, "GET", f"/api/canvases/{made['id']}/history")
    assert history["ops"][0][2] == "stroke" and history["versions"]
    status, op = http(base, "GET", f"/api/canvases/{made['id']}/ops/{history['ops'][0][0]}")
    assert op["args"]["color"] == RED
    status, cat = http(base, "GET", "/api/canvas-catalog")
    assert status == 200 and cat["instruments"] and cat["prompts"]
    assert http(base, "GET", f"/api/canvases/{made['id']}?since=soon")[0] == 400
    assert http(base, "GET", "/api/canvases/nope")[0] == 404


def test_spawn_and_stop_over_http(served, monkeypatch):
    base, db, launcher = served
    _, made = http(base, "POST", "/api/canvases", {"name": "Commons"})
    body = {"instrument_id": catalog.key("instrument", ROUND), "prompt_id": catalog.key("prompt", "Paint big."),
            "harness": "claude"}
    assert http(base, "POST", f"/api/canvases/{made['id']}/agents", {**body, "harness": "x"})[0] == 422
    # A process that sleeps stands in for the agent, so stopping it is what's tested.
    monkeypatch.setattr(launcher, "python", "/bin/sleep")
    real_popen = __import__("subprocess").Popen
    monkeypatch.setattr("conveyor.launch.subprocess.Popen", lambda argv, **kw: real_popen(["/bin/sleep", "30"], **kw))
    status, spawned = http(base, "POST", f"/api/canvases/{made['id']}/agents", body)
    assert status == 201, spawned
    assert http(base, "POST", f"/api/canvases/other/agents/{spawned['id']}/stop", {})[0] == 404
    status, _ = http(base, "POST", f"/api/canvases/{made['id']}/agents/{spawned['id']}/stop", {})
    assert status == 200
    wait_for(lambda: connect(db).execute("SELECT status FROM canvas_agents WHERE id=?",
                                         (spawned["id"],)).fetchone()[0] == "stopped")
    assert http(base, "POST", "/api/canvases", {"name": "x"}, headers={"Origin": "http://evil.example"})[0] == 403
