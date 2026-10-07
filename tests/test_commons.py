"""The shared canvas: tiles and replay, concurrent agents, the tool-call budget, moves, messages, the catalog,
spawning, a whole agent driven by the fake CLI, and the HTTP routes."""

import base64
import copy
import io
import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conveyor.__main__ import build_parser
from conveyor.commons import catalog
from conveyor.commons import lettering
from conveyor.commons import prompts
from conveyor.commons import views
from conveyor.commons.agent import create_agent
from conveyor.commons.agent import run_agent
from conveyor.commons.agent import session_scope
from conveyor.commons.agent import system_prompt
from conveyor.commons.server import AgentSession
from conveyor.commons.server import CommonsServer
from conveyor.commons.tiles import PAPER_RGB
from conveyor.commons.tiles import TILE
from conveyor.commons.tiles import SharedCanvas
from conveyor.commons.tiles import create_canvas
from conveyor.commons.tiles import decode_tile
from conveyor.launch import LaunchError
from conveyor.launch import Launcher
from conveyor.painting.prompts import INITIAL_STRATEGY
from conveyor.painting.seeds import PEN
from conveyor.painting.seeds import ROUND
from conveyor.server import make_server
from conveyor.store import Store
from conveyor.store import connect
from conveyor.store import init_db
from test_launch import call as http
from test_launch import wait_for

RED = "#c03030"


@pytest.fixture
def canvas(tmp_path):
    db = tmp_path / "c.db"
    return db, create_canvas(db, "Commons", viewport=256, max_calls=20)


def agent(db: Path, canvas_row: dict, tmp: Path, *, x=0, y=0, source=ROUND, max_calls=None, name="a",
          successors=False) -> CommonsServer:
    conn = connect(db)
    conn.execute("INSERT INTO canvas_agents (id, canvas_id, created, status, config, x, y, max_calls) "
                 "VALUES (?, ?, ?, 'running', '{}', ?, ?, ?)",
                 (name, canvas_row["id"], time.time(), x, y, max_calls or canvas_row["config"]["max_calls"]))
    conn.commit()
    conn.close()
    d = tmp / f"agent-{name}"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"db": str(db), "canvas_id": canvas_row["id"], "agent_id": name,
                                            "source": source, "x": x, "y": y, "max_calls": max_calls,
                                            "successors": successors}))
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


def test_messages_float_above_the_paint_for_every_agent(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path)
    result = tool(server, "write_message", {"text": "MEET AT 900,0", "x": 10, "y": 10, "scale": 2, "color": "#000000"})
    assert not result["isError"], text_of(result)
    shared = SharedCanvas(db, row["id"])
    assert views.canvas(connect(db, readonly=True), row["id"])["heads"] == []  # no tile changed: tiles hold paint only
    dark = lambda img: img.min(axis=-1) < 0.05  # noqa: E731
    ink = dark(server.s._seen()[10:36, 10:200]).sum()
    assert ink > 100
    tool(server, "paint_batch", {"tool": "stroke", "defaults": {"length": 200, "size": 12, "color": "#ffffff",
                                                               "pressure": 1.0, "angle": 0},
                                 "calls": [{"x": 10, "y": 14}, {"x": 10, "y": 30}]})
    assert shared.read(0, 0, 256, 256).image()[14, 60].min() > 0.95  # the paint went down
    assert dark(server.s._seen()[10:36, 10:200]).sum() == ink  # and the text is still on top of it
    other = agent(db, row, tmp_path, x=5, y=4, name="b")  # another agent sees it where it is on the canvas
    assert not tool(other, "look")["isError"]
    assert dark(other.s._seen()[6:32, 5:195]).sum() == ink
    huge = tool(server, "write_message", {"text": "x" * 200, "x": 0, "y": 0, "scale": 4})
    assert huge["isError"] and "area" in text_of(huge)
    tool(other, "write_message", {"text": "hi", "x": 0, "y": 100})
    messages = views.canvas(connect(db, readonly=True), row["id"])["messages"]
    assert [(m["text"], m["x"], m["y"]) for m in messages] == [("MEET AT 900,0", 10, 10), ("hi", 5, 104)]
    assert messages[0]["lines"] == ["MEET AT 900,0"] and messages[0]["color"] == "#000000"


def test_messages_painted_into_tiles_before_the_text_layer_come_back_on_top(canvas, tmp_path):
    db, row = canvas
    shared = SharedCanvas(db, row["id"])
    shared.record(agent_id="old", tool="write_message", status="painted", x=0, y=0, note="FROM BEFORE",
                  args={"text": "FROM BEFORE", "x": 20, "y": 40, "background": "#ffffff"})
    assert [m["text"] for m in shared.messages(0, 0, 256, 256)] == ["FROM BEFORE"]
    assert shared.messages(300, 0, 400, 100) == []


def test_lettering_is_outlined_so_dark_ink_reads_on_dark_paint():
    black = np.zeros((40, 120, 3), dtype=np.float32)
    m = {"x": 4, "y": 4, "text": "DARK", "scale": 2, "width": 116, "height": 36, "color": (0.0, 0.0, 0.0)}
    out = lettering.draw_messages(black, 0, 0, [m])
    assert (out.min(axis=-1) > 0.85).sum() > 50  # a light outline around black letters
    small = lettering.draw_messages(np.zeros((10, 30, 3), dtype=np.float32), 0, 0, [m], scale=0.25)
    assert small.max() > 0.1  # shrunk for an overview, the text still shows


def test_the_overview_shows_the_region_around_the_viewport_and_how_far_the_paint_reaches(canvas, tmp_path):
    db, row = canvas
    near, far = agent(db, row, tmp_path, name="near"), agent(db, row, tmp_path, x=3000, y=1000, name="far")
    shared = SharedCanvas(db, row["id"])
    assert shared.extent() is None
    first = tool(near, "overview")
    assert "Nothing is painted anywhere yet" in text_of(first)
    tool(near, "stroke", stroke(10, 10))
    tool(far, "stroke", stroke(10, 10, color="#2040a0"))
    assert shared.extent() == (0, 0, 24 * TILE, 8 * TILE)
    # A 256 px viewport at (0, 0): the region is four viewports across, centred on it, at half size.
    img = shared.shrunk((-384, -384, 640, 640), 0.5)
    assert img.shape == (512, 512, 3)
    red = (img[..., 0].astype(int) - img[..., 2] > 60).nonzero()
    # near's stroke starts at canvas (10, 10), which is (197, 197) at half size from the region's corner at -384
    assert 185 <= red[0].min() and red[0].max() <= 215 and 180 <= red[1].min() and red[1].max() <= 230
    assert not (img[..., 2].astype(int) - img[..., 0] > 60).any()  # far's is outside the region
    result = tool(near, "overview")
    said = text_of(result)
    assert not result["isError"] and "x -384 to 640 and y -384 to 640" in said and "4 viewports across" in said
    assert "2 moves out" in said and "reaches from x 0 to 3072 and y 0 to 1024" in said
    with Image.open(io.BytesIO(base64.b64decode(result["content"][1]["data"]))) as pic:
        arr = np.asarray(pic.convert("RGB")).astype(int)
    magenta = ((arr[..., 0] > 200) & (arr[..., 1] < 60) & (arr[..., 2] > 130)).nonzero()
    assert pic.width / 3 < magenta[1].mean() < pic.width * 2 / 3  # its own viewport, in the middle
    assert "17 left" in said  # it costs a call like any other: three made
    assert views.history(connect(db, readonly=True), row["id"])["ops"][-1][2] == "overview"


def test_a_canvas_task_reaches_every_agent_word_for_word(tmp_path):
    config = {"source": ROUND, "prompt": "Paint big."}
    task = "A lighthouse on a cliff at night,\nits beam crossing the whole sky."
    with_task, without = system_prompt(config, 256, 0, 0, 20, task), system_prompt(config, 256, 0, 0, 20)
    assert task in with_task and prompts.NO_TASK not in with_task and "far beyond your viewport" in with_task
    assert prompts.NO_TASK in without and "same task" not in without and "far beyond" not in without
    assert "`overview`" in with_task and "`overview`" in without


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


CLASH = 'TOOLS = {"write_message": {}}\n'
PAIR = catalog.pair_key(ROUND, "Paint big.")


def seeded_db(tmp_path) -> Path:
    """Two runs that painted the round brush with "Paint big.", one painting each; an instrument whose tool name
    clashes, painted with that prompt; an evolved prompt that never painted; and a dead instrument."""
    db = tmp_path / "c.db"
    for name, score in (("run one", 0.6), ("run two", 0.7)):
        store = Store(db, run_name=name, config={})
        store.organism({"id": f"{name}-round", "node": "instrument", "created": time.time(), "genome": {"source": ROUND},
                        "summary": "seed: round", "viable": 1, "sheet": "sheet.png"})
        store.organism({"id": f"{name}-prompt", "node": "painter", "created": time.time(), "genome": {"prompt": "Paint big."},
                        "viable": 1})
        for org, partner in ((f"{name}-round", f"{name}-prompt"), (f"{name}-prompt", f"{name}-round")):
            store.evaluation({"id": f"{org}-ev", "node": "x", "organism_id": org, "partner_id": partner, "score": score,
                              "viable": 1, "ended": time.time(), "session_id": f"{name}-session",
                              "artifacts": {"painting": f"{name}.png"}})
        store.close()
    store = Store(db, run_name="clash", config={})
    store.organism({"id": "clash", "node": "instrument", "created": time.time(), "viable": 1,
                    "genome": {"source": CLASH}, "summary": "talks too much"})
    store.organism({"id": "clash-prompt", "node": "painter", "created": time.time(), "viable": 1,
                    "genome": {"prompt": "Paint big."}})
    store.evaluation({"id": "clash-ev", "node": "instrument", "organism_id": "clash", "partner_id": "clash-prompt",
                      "score": 0.1, "viable": 1, "ended": time.time(), "details": {"instrument_id": "clash",
                                                                                   "prompt_id": "clash-prompt"}})
    store.organism({"id": "lonely", "node": "painter", "created": time.time(), "viable": 1,
                    "genome": {"prompt": "Never painted."}})
    store.organism({"id": "dead", "node": "instrument", "created": time.time(), "viable": 0,
                    "genome": {"source": "TOOLS = {}\n"}, "summary": "never worked"})
    store.close()
    return db


def test_the_catalog_offers_pairs_that_painted_together(tmp_path):
    db = seeded_db(tmp_path)
    pairs = catalog.catalog(connect(db, readonly=True))["pairs"]
    assert [p["id"] for p in pairs] == [PAIR, catalog.pair_key(CLASH, "Paint big.")]  # best score first
    best = pairs[0]
    assert best["best_score"] == 0.7 and best["painting"] == "run two.png"
    assert best["paintings"] == 2  # two sessions, each evaluated for both organisms, counted once
    assert {o["run"] for o in best["origins"]} == {"run one", "run two"}
    assert [(r["run"], r["place"]) for r in best["ranks"]] == [("run two", 1), ("run one", 1)]  # same place: higher score first
    assert best["instrument"]["label"] == "seed: round" and best["instrument"]["tools"] == ["stroke"]
    assert best["instrument"]["sheet"] == "sheet.png" and best["prompt"]["text"] == "Paint big."
    # A prompt that never painted isn't a painter, and the unpainted seed pairs only stand in on an empty database.
    assert all(p["prompt"]["text"] not in ("Never painted.", INITIAL_STRATEGY) for p in pairs)


def test_only_each_runs_top_five_pairs_are_offered(tmp_path):
    db = tmp_path / "c.db"
    store = Store(db, run_name="crowded", config={})
    store.organism({"id": "inst", "node": "instrument", "created": time.time(), "genome": {"source": ROUND}, "viable": 1})
    for i in range(7):
        store.organism({"id": f"p{i}", "node": "painter", "created": time.time(), "genome": {"prompt": f"Prompt {i}."},
                        "viable": 1})
        store.evaluation({"id": f"e{i}", "node": "painter", "organism_id": f"p{i}", "partner_id": "inst",
                          "score": i / 10, "viable": 1, "ended": time.time(), "session_id": f"s{i}"})
    store.evaluation({"id": "again", "node": "painter", "organism_id": "p6", "partner_id": "inst", "score": 0.65,
                      "viable": 1, "ended": time.time(), "session_id": "s6-confirm"})  # a repaint takes no extra place
    store.close()
    pairs = catalog.catalog(connect(db, readonly=True))["pairs"]
    assert [p["prompt"]["text"] for p in pairs] == [f"Prompt {i}." for i in (6, 5, 4, 3, 2)]
    assert [p["ranks"][0]["place"] for p in pairs] == [1, 2, 3, 4, 5]
    assert pairs[0]["paintings"] == 2 and pairs[0]["best_score"] == 0.65


def test_a_database_without_scores_offers_the_seed_pairs(tmp_path):
    db = tmp_path / "empty.db"
    init_db(db)
    pairs = catalog.catalog(connect(db, readonly=True))["pairs"]
    assert {p["instrument"]["label"] for p in pairs} == {"seed: round", "seed: pen"}
    assert {p["prompt"]["label"] for p in pairs} == {"starting strategy"} and not any(p["ranks"] for p in pairs)


@pytest.mark.parametrize("change,match", [
    ({"harness": "gpt"}, "harness"), ({"effort": "huge"}, "effort"), ({"pair_id": "c-nope"}, "painter"),
    ({"instrument_id": "i-x"}, "Expected"), ({"x": "left"}, "whole numbers"),
    ({"x": 10**9}, "within"), ({"cap": 0}, "cap"), ({"model": "--dangerous"}, "dash"), ({"color": "red"}, "Expected"),
    ({"successors": "yes"}, "on or off"),
])
def test_spawning_checks_its_request(tmp_path, change, match):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C")
    body = {"pair_id": PAIR, "harness": "claude", **change}
    with pytest.raises(LaunchError, match=match):
        create_agent(db, row["id"], body)


def test_a_random_painter_is_drawn_from_every_pair_the_canvas_can_run(tmp_path, monkeypatch):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C")
    offered = []

    def choice(pairs):
        offered.extend(p["id"] for p in pairs)
        return pairs[-1]

    monkeypatch.setattr("conveyor.commons.agent.random.choice", choice)
    spawned = create_agent(db, row["id"], {"pair_id": "random"})
    everything = {p["id"] for p in catalog.catalog(connect(db, readonly=True))["pairs"]}
    assert set(offered) == everything - {catalog.pair_key(CLASH, "Paint big.")}  # all but the one that clashes
    assert spawned["painter"]["random"] and spawned["painter"]["id"] == offered[-1]
    config = views.canvas(connect(db, readonly=True), row["id"])["agents"][0]["config"]
    assert config["random"] and config["pair_id"] == offered[-1]
    chosen = create_agent(db, row["id"], {"pair_id": PAIR})
    assert chosen["painter"] == {"id": PAIR, "random": False, "instrument": "seed: round", "prompt": "Paint big.",
                                 "best_score": 0.7}


def test_spawning_refuses_an_instrument_whose_tools_clash(tmp_path):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C")
    with pytest.raises(LaunchError, match="write_message"):
        create_agent(db, row["id"], {"pair_id": catalog.pair_key(CLASH, "Paint big.")})


def test_an_agent_runs_to_the_end_of_its_budget(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_claude.parent}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("FAKE_CLAUDE_MESSAGE", "east is empty")
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=9, task="A harbour at dusk.")
    spawned = create_agent(db, row["id"], {"pair_id": PAIR, "x": -40, "y": 30,
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
    assert "making one image:\n\nA harbour at dusk." in request["system"] and "overview" in request["tools"]
    assert [c["type"] for c in request["content"]].count("image") == 2  # the viewport and the demo sheet
    strokes = conn.execute("SELECT count(*), sum(snapshot IS NOT NULL) FROM strokes WHERE session_id=?",
                           (done["session_id"],)).fetchone()
    assert strokes[0] == 9 and strokes[1] >= 3
    assert not conn.execute("SELECT 1 FROM runs WHERE id=?", (session_scope(row["id"]),)).fetchone()


# ---- successors ------------------------------------------------------------------------------------------------


def test_spawning_a_successor_ends_the_session_where_it_stands(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path, max_calls=6, successors=True)
    assert "spawn_successor" in [t["name"] for t in server.tools()]
    assert "spawn_successor" not in [t["name"] for t in agent(db, row, tmp_path, name="b").tools()]
    tool(server, "look")
    tool(server, "move_viewport", {"angle": 0, "distance": 100})
    assert "spawn_successor" in text_of(tool(server, "look"))  # three calls left: the reminder
    spawn = next(t for t in server.tools() if t["name"] == "spawn_successor")
    assert spawn["inputSchema"]["properties"] == {}  # no note: the canvas is all it hands on
    result = tool(server, "spawn_successor", {})
    assert not result["isError"] and "don't call any more tools" in text_of(result)
    assert result["structuredContent"] == {"stop": True}
    assert json.loads((server.s.dir / "successor.json").read_text())["x"] == 100
    after = tool(server, "look")
    assert after["isError"] and "successor" in text_of(after)
    assert server.s.calls_used == 4  # the refused call after the hand-off doesn't count


def test_a_successor_can_only_be_spawned_in_the_last_ten_calls(canvas, tmp_path):
    db, row = canvas
    server = agent(db, row, tmp_path, max_calls=20, successors=True)
    early = tool(server, "spawn_successor", {})
    assert early["isError"] and "last 10" in text_of(early) and "19 left" in text_of(early)
    assert not server.s.handed_off and not (server.s.dir / "successor.json").exists()
    for _ in range(8):
        tool(server, "look")
    assert tool(server, "spawn_successor", {})["isError"]  # call 10: ten left after it, still too early
    result = tool(server, "spawn_successor", {})  # call 11, the first of the last ten
    assert not result["isError"] and server.s.handed_off and server.s.calls_used == 11


def test_a_handed_off_agent_queues_its_successor_with_only_the_canvas(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_claude.parent}:{__import__('os').environ['PATH']}")
    flag = tmp_path / "handoff"
    flag.touch()
    monkeypatch.setenv("FAKE_CLAUDE_HANDOFF", str(flag))
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=20)
    offered = []

    def choice(pairs):  # the test catalog has one usable painter; hand the successor a different prompt
        offered.extend(p["id"] for p in pairs)
        pair = copy.deepcopy(pairs[0])
        pair["id"], pair["prompt"]["text"] = "c-drawn", "Paint small."
        return pair

    monkeypatch.setattr("conveyor.commons.agent.random.choice", choice)
    first = create_agent(db, row["id"], {"pair_id": PAIR, "name": "Ada", "x": 5, "y": 7})
    run_agent(db, first["id"])
    agents = views.canvas(connect(db, readonly=True), row["id"])["agents"]
    parent, child = agents
    assert parent["status"] == "handed_off" and parent["calls_used"] == 11 and parent["error"] is None
    assert child["status"] == "queued" and child["name"] == "Ada #2"
    assert (child["x"], child["y"]) == (parent["x"], parent["y"]) == (5 + 192, 7)
    c = child["config"]
    assert (c["generation"], c["parent_id"], c["lineage"]) == (2, parent["id"], parent["id"])
    assert "handoff_note" not in c and parent["config"]["pair_id"] == PAIR and offered == [PAIR]
    assert c["pair_id"] == "c-drawn" and c["random"] and c["model"] == parent["config"]["model"]  # a new painter
    run_agent(db, child["id"])  # the flag is gone, so this one paints to the end of its budget
    conn = connect(db, readonly=True)
    done = views.canvas(conn, row["id"])["agents"][1]
    assert done["status"] == "finished" and done["calls_used"] == 20
    request = json.loads(conn.execute("SELECT request FROM sessions WHERE id=?", (done["session_id"],)).fetchone()[0])
    # It starts like an agent spawned by hand at that spot: the same prompt, and nothing saying it's a successor.
    first = json.loads(conn.execute("SELECT request FROM sessions WHERE id=?", (parent["session_id"],)).fetchone()[0])
    assert "Paint small." in request["system"] and "Paint big." not in request["system"]
    assert request["system"].replace("(197, 7)", "(5, 7)").replace("Paint small.", "Paint big.") == first["system"]
    assert request["content"][0]["text"] == "Your viewport as it is now, top-left corner at canvas (197, 7):"
    assert not any("session" in b.get("text", "") for b in request["content"])
    assert "spawn_successor" in request["system"] and "spawn_successor" in request["tools"]


def test_no_successor_starts_once_a_usage_window_is_past_the_limit(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_claude.parent}:{__import__('os').environ['PATH']}")
    flag = tmp_path / "handoff"
    flag.touch()
    monkeypatch.setenv("FAKE_CLAUDE_HANDOFF", str(flag))
    monkeypatch.setattr("conveyor.harness.Meter.should_stop", lambda self: "the claude five-hour usage window is at 91%")
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=20)
    spawned = create_agent(db, row["id"], {"pair_id": PAIR})
    run_agent(db, spawned["id"])
    agents = views.canvas(connect(db, readonly=True), row["id"])["agents"]
    assert len(agents) == 1 and agents[0]["status"] == "finished" and "91%" in agents[0]["error"]


def test_the_launcher_starts_each_queued_agent_once(tmp_path, monkeypatch):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C")
    body = {"pair_id": PAIR}
    waiting, other = create_agent(db, row["id"], body), create_agent(db, row["id"], body)
    launcher = Launcher(db, build_parser)
    started = []
    real_popen = __import__("subprocess").Popen

    def popen(argv, **kw):
        started.append(argv[-1])
        return real_popen(["/bin/sleep", "30"], **kw)

    monkeypatch.setattr("conveyor.launch.subprocess.Popen", popen)
    assert launcher.stop_agent(row["id"], other["id"]) == {"id": other["id"], "state": "stopped"}
    try:
        assert launcher.start_queued() == [waiting["id"]] and started == [waiting["id"]]
        assert launcher.start_queued() == [] and started == [waiting["id"]]
        status = dict(connect(db).execute("SELECT id, status FROM canvas_agents").fetchall())
        assert status == {waiting["id"]: "starting", other["id"]: "stopped"}
    finally:
        launcher.shutdown(grace=5)


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
    assert http(base, "POST", "/api/canvases", {"task": "x" * 2001})[0] == 422
    status, tasked = http(base, "POST", "/api/canvases", {"name": "Harbour", "task": "  A harbour at dusk.\n"})
    assert status == 201 and tasked["config"]["task"] == "A harbour at dusk."
    assert "task" not in made["config"]
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
    assert status == 200 and cat["pairs"][0]["id"] == PAIR
    assert http(base, "GET", f"/api/canvases/{made['id']}?since=soon")[0] == 400
    assert http(base, "GET", "/api/canvases/nope")[0] == 404


def test_spawn_and_stop_over_http(served, monkeypatch):
    base, db, launcher = served
    _, made = http(base, "POST", "/api/canvases", {"name": "Commons"})
    body = {"pair_id": PAIR,
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
