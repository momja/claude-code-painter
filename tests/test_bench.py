"""Benchmarks, the successor cap, the kill switch, PNG export, and reference pictures for Claude Code agents."""

import io
import json
import os
import time
import urllib.request

import numpy as np
import pytest
from PIL import Image

from conveyor.__main__ import build_parser
from conveyor.commons import bench
from conveyor.commons import prompts
from conveyor.commons import views
from conveyor.commons.agent import create_agent
from conveyor.commons.agent import queue_judges
from conveyor.commons.agent import queue_successor
from conveyor.commons.agent import run_agent
from conveyor.commons.agent import system_prompt
from conveyor.commons.export import export_png
from conveyor.commons.tiles import PAPER_RGB
from conveyor.commons.tiles import SharedCanvas
from conveyor.commons.tiles import create_canvas
from conveyor.launch import LaunchError
from conveyor.launch import Launcher
from conveyor.painting.seeds import ROUND
from conveyor.store import connect
from test_commons import PAIR
from test_commons import agent
from test_commons import fake_images  # noqa: F401 - a fixture
from test_commons import seeded_db
from test_commons import served  # noqa: F401 - a fixture
from test_commons import stroke
from test_commons import tool
from test_launch import call as http
from test_launch import wait_for

SPEC = {
    "name": "Harbour", "seed": 11,
    "canvas": {"viewport": 256, "max_calls": 12, "task": "A harbour at dusk.", "frame": {"width": 768, "height": 512}},
    "defaults": {"cap": 2, "effort": "low"},
    "agents": [{"label": "left", "x": 0, "y": 0}, {"label": "right", "x": 512, "y": 256, "task": "The boats."},
               {"label": "critic", "kind": "judge", "x": 100, "y": 0}],
    "models": {"fake": {"harness": "claude", "model": "fake-model"}},
    "variants": {"claude": "fake", "glm": {"harness": "pi", "provider": "opencode-go", "model": "glm-x"},
                 "mix": ["fake", "glm"]},
}


def agents_of(db, canvas_id) -> list[dict]:
    return views.canvas(connect(db, readonly=True), canvas_id)["agents"]


def set_status(db, ids, status):
    conn = connect(db)
    conn.executemany("UPDATE canvas_agents SET status=?, ended=? WHERE id=?", [(status, time.time(), i) for i in ids])
    conn.commit()
    conn.close()


def png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---- the successor cap -----------------------------------------------------------------------------------------


def test_a_chain_hands_off_at_most_max_successors_times(tmp_path):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=20)
    cid = row["id"]
    none = create_agent(db, cid, {"pair_id": PAIR, "max_successors": 0})
    free = create_agent(db, cid, {"pair_id": PAIR})
    two = create_agent(db, cid, {"pair_id": PAIR, "max_successors": 2})
    for bad in (-1, 101, "3", True, 1.5):
        with pytest.raises(LaunchError, match="max_successors"):
            create_agent(db, cid, {"pair_id": PAIR, "max_successors": bad})
    configs = {a["id"]: a["config"] for a in agents_of(db, cid)}
    assert configs[none["id"]]["successors"] is False and configs[none["id"]]["max_successors"] is None
    assert configs[free["id"]]["successors"] is True and configs[free["id"]]["max_successors"] is None
    assert configs[two["id"]]["successors"] is True and configs[two["id"]]["max_successors"] == 2

    shared = SharedCanvas(db, cid)
    raw = lambda i: dict(shared.conn.execute("SELECT * FROM canvas_agents WHERE id=?", (i,)).fetchone())  # noqa: E731
    stored = lambda i: json.loads(raw(i)["config"])  # noqa: E731
    child = queue_successor(shared, raw(two["id"]), stored(two["id"]), 0, 0)
    grandchild = queue_successor(shared, raw(child["id"]), stored(child["id"]), 0, 0)
    assert stored(child["id"])["successors"] is True and stored(grandchild["id"])["successors"] is False
    assert stored(grandchild["id"])["generation"] == 3

    prompt = lambda config: system_prompt({**config, "source": ROUND, "prompt": "Paint big."}, 256, 0, 0, 20)  # noqa: E731
    assert prompts.SUCCESSOR_LEFT.format(left=2) in prompt(stored(two["id"]))
    assert prompts.SUCCESSOR_LAST in prompt(stored(child["id"]))
    assert "spawn_successor" not in prompt(stored(grandchild["id"]))
    assert prompts.SUCCESSOR_OPEN in prompt(stored(free["id"]))  # no cap: the prompt reads as it always did
    shared.close()


# ---- the kill switch -------------------------------------------------------------------------------------------


def test_stopping_a_canvas_stops_its_agents_and_lets_nothing_new_start_there(tmp_path, monkeypatch):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=20)
    cid = row["id"]
    launcher = Launcher(db, build_parser)
    real_popen = __import__("subprocess").Popen
    monkeypatch.setattr("conveyor.launch.subprocess.Popen", lambda argv, **kw: real_popen(["/bin/sleep", "30"], **kw))
    try:
        running = launcher.start_agent(cid, {"pair_id": PAIR})
        create_agent(db, cid, {"pair_id": PAIR})  # waiting for the watcher
        assert launcher.stop_canvas(cid) == {"id": cid, "interrupted": 1, "dropped": 1}
        wait_for(lambda: {a["status"] for a in agents_of(db, cid)} == {"stopped"})
        assert views.canvas(connect(db, readonly=True), cid)["canvas"]["config"]["halted"]

        late = create_agent(db, cid, {"pair_id": PAIR})  # a successor queued just as the canvas stopped
        assert launcher.start_queued() == []
        assert next(a for a in agents_of(db, cid) if a["id"] == late["id"])["status"] == "stopped"
        create_agent(db, cid, {"pair_id": PAIR})
        conn = connect(db)
        conn.executemany("INSERT INTO canvas_ops (canvas_id, agent_id, ts, tool, status) VALUES (?, ?, ?, 'look', 'view')",
                         [(cid, running["id"], time.time())] * 1000)
        conn.commit()
        conn.close()
        assert queue_judges(db) == []  # due, but the canvas is stopped

        launcher.start_agent(cid, {"pair_id": PAIR})  # by hand: carry on
        assert "halted" not in views.canvas(connect(db, readonly=True), cid)["canvas"]["config"]
        assert launcher.stop_canvas("nope") is None
    finally:
        launcher.shutdown(grace=5)


def test_an_agent_that_hands_off_on_a_stopped_canvas_queues_no_successor(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_claude.parent}:{os.environ['PATH']}")
    flag = tmp_path / "handoff"
    flag.touch()
    monkeypatch.setenv("FAKE_CLAUDE_HANDOFF", str(flag))
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=20)
    spawned = create_agent(db, row["id"], {"pair_id": PAIR})
    Launcher(db, build_parser)._set_halted(row["id"], True)
    run_agent(db, spawned["id"])
    [only] = agents_of(db, row["id"])
    assert only["status"] == "finished" and "canvas was stopped" in only["error"]


# ---- export ----------------------------------------------------------------------------------------------------


def test_a_canvas_exports_as_one_png_now_or_as_of_any_op(tmp_path):
    db = tmp_path / "c.db"
    framed = create_canvas(db, "F", viewport=256, max_calls=20, frame=(600, 400))
    server = agent(db, framed, tmp_path)
    tool(server, "stroke", stroke(50, 50))
    conn = connect(db, readonly=True)
    first = conn.execute("SELECT max(seq) FROM canvas_ops").fetchone()[0]
    tool(server, "stroke", stroke(150, 150, color="#3050c0"))
    now = png(export_png(conn, framed["id"]))
    assert now.size == (600, 400)
    paper = tuple(int(c) for c in PAPER_RGB)
    assert now.getpixel((70, 50)) != paper and now.getpixel((170, 150)) != paper and now.getpixel((590, 390)) == paper
    then = png(export_png(conn, framed["id"], seq=first))
    assert then.getpixel((70, 50)) == now.getpixel((70, 50)) and then.getpixel((170, 150)) == paper
    assert png(export_png(conn, framed["id"], side=300)).size == (300, 200)
    with pytest.raises(ValueError):
        export_png(conn, framed["id"], side=8)
    with pytest.raises(KeyError):
        export_png(conn, "nope")

    open_canvas = create_canvas(db, "Open", viewport=256, max_calls=20)
    assert png(export_png(conn, open_canvas["id"])).size == (256, 256)  # blank paper: one viewport
    tool(agent(db, open_canvas, tmp_path, x=-300, y=40, name="b"), "stroke", stroke(10, 10))
    reach = png(export_png(conn, open_canvas["id"]))  # the paint's reach, tile-aligned, wherever it lies
    assert reach.size[0] % 128 == 0 and reach.size[1] % 128 == 0 and reach.size[0] <= 384
    assert (np.asarray(reach) != PAPER_RGB).any()


# ---- planning ----------------------------------------------------------------------------------------------------


def test_a_spec_lays_out_one_canvas_per_variant_with_the_same_agents_and_painters(tmp_path):
    db = seeded_db(tmp_path)
    conn = connect(db)
    p = bench.plan(conn, SPEC)
    assert [r["label"] for r in p["runs"]] == ["claude", "glm", "mix"]
    for r in p["runs"]:
        assert r["canvas"]["config"]["task"] == "A harbour at dusk." and r["canvas"]["config"]["draw_seed"] == 11
        assert r["canvas"]["config"]["frame"] == [0, 0, 768, 512]
        assert [m["config"]["slot_label"] for m in r["agents"]] == ["left", "right", "critic"]
        assert [m["config"]["cap"] for m in r["agents"]] == [2, 2, 2]
        assert r["agents"][1]["config"]["agent_task"] == "The boats."
        assert r["agents"][0]["config"]["max_successors"] == bench.DEFAULT_MAX_SUCCESSORS
        assert r["agents"][2]["config"]["kind"] == "judge"
    painters = {tuple(m["config"].get("pair_id") for m in r["agents"]) for r in p["runs"]}
    assert len(painters) == 1  # every variant draws the same painter for each slot
    models = {r["label"]: [(m["config"]["harness"], m["config"]["model"]) for m in r["agents"]] for r in p["runs"]}
    assert models["claude"] == [("claude", "fake-model")] * 3
    assert models["glm"] == [("pi", "glm-x")] * 3
    assert models["mix"] == [("claude", "fake-model"), ("pi", bench.PRESETS["glm"]["model"]), ("claude", "fake-model")]
    assert [m["config"]["provider"] for m in p["runs"][1]["agents"]] == ["opencode-go"] * 3
    assert conn.execute("SELECT count(*) FROM canvases").fetchone()[0] == 0  # a plan saves nothing
    # The most it can spend: two painters at $2 for up to 4 sessions each, a judge at $2 once, on 3 canvases.
    assert bench.preview(p)["ceiling"] == 3 * (2 * 2 * 4 + 2)
    open_ended = bench.plan(conn, {**SPEC, "defaults": {**SPEC["defaults"], "max_successors": None}})
    assert bench.preview(open_ended)["ceiling"] is None
    again = bench.plan(conn, {**SPEC, "variants": {"x": "fake"}, "repeats": 2})
    assert [r["label"] for r in again["runs"]] == ["x #1", "x #2"]
    assert again["runs"][0]["agents"][0]["config"]["pair_id"] == p["runs"][0]["agents"][0]["config"]["pair_id"]


@pytest.mark.parametrize("change, match", [
    ({"variants": {"a": "gpt5"}}, "no model called 'gpt5'"),
    ({"agents": [{"x": 0, "harness": "pi"}]}, r"agents\[0\] takes"),
    ({"variants": {f"v{i}": "fake" for i in range(5)}, "repeats": 5}, "at most 24"),
    ({"repeats": 0}, "repeats"),
    ({"order": "random"}, "order"),
    ({"canvas": {"viewport": 9000}}, "viewport"),
    ({"agents": [{"x": 0, "viewport": 1024}]}, r"agents\[0\] \(#1\) in claude: A 1024 px viewport"),
    ({"agents": []}, "agents must be"),
    ({"defaults": {"x": 3}}, "defaults takes"),
    ({"canvas": {"painters": ["nope"]}}, "not in the catalog"),
    ({"variants": {"a": {"harness": "pi", "provider": "nowhere"}}}, "provider"),
    ({"surprise": 1}, "takes"),
])
def test_a_spec_is_checked_before_anything_starts(tmp_path, change, match):
    db = seeded_db(tmp_path)
    with pytest.raises(LaunchError, match=match):
        bench.create(db, {**SPEC, **change})
    conn = connect(db)
    assert conn.execute("SELECT count(*) FROM canvases").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM benchmarks").fetchone()[0] == 0


def test_successors_on_benchmark_canvases_draw_the_same_painters(tmp_path):
    db = seeded_db(tmp_path)
    made = bench.create(db, {**SPEC, "variants": {"a": "fake", "b": "fake"}})
    conn = connect(db, readonly=True)
    runs = bench.results(conn, made["id"])["runs"]
    drawn = []
    for r in runs:
        shared = SharedCanvas(db, r["canvas_id"])
        first = dict(shared.conn.execute("SELECT * FROM canvas_agents WHERE canvas_id=? ORDER BY created",
                                         (r["canvas_id"],)).fetchone())
        line = []
        for _ in range(3):
            child = queue_successor(shared, first, json.loads(first["config"]), 0, 0)
            first = dict(shared.conn.execute("SELECT * FROM canvas_agents WHERE id=?", (child["id"],)).fetchone())
            line.append(json.loads(first["config"])["pair_id"])
        drawn.append(line)
        shared.close()
    assert drawn[0] == drawn[1]


# ---- running -----------------------------------------------------------------------------------------------------


def test_a_sequential_benchmark_runs_one_canvas_at_a_time_and_then_reports(tmp_path):
    db = seeded_db(tmp_path)
    made = bench.create(db, SPEC)
    assert made["canvases"] == 3 and made["agents"] == 9
    conn = connect(db, readonly=True)
    runs = bench.results(conn, made["id"])["runs"]
    assert [r["state"] for r in runs] == ["running", "waiting", "waiting"]
    assert {a["status"] for a in agents_of(db, runs[1]["canvas_id"])} == {"held"}
    assert bench.advance(db) == []  # the first canvas is still at work
    for k, r in enumerate(runs):
        ids = [a["id"] for a in agents_of(db, r["canvas_id"])]
        set_status(db, ids, "finished")
        released = bench.advance(db)
        if k + 1 < len(runs):
            assert sorted(released) == sorted(a["id"] for a in agents_of(db, runs[k + 1]["canvas_id"]))
            assert {a["status"] for a in agents_of(db, runs[k + 1]["canvas_id"])} == {"queued"}
        else:
            assert released == []
    res = bench.results(conn, made["id"])
    assert res["status"] == "done" and [r["state"] for r in res["runs"]] == ["done"] * 3
    assert res["runs"][2]["models"] == ["fake-model low", "glm-5.3-flash (opencode-go) low", "fake-model low"]
    assert {m["model"] for m in res["runs"][2]["by_model"]} == {"fake-model low", "glm-5.3-flash (opencode-go) low"}
    assert res["runs"][0]["coverage"] == 0 and res["runs"][0]["agents"] == 3
    sheet = png(bench.sheet_png(conn, made["id"], cell=200))
    assert sheet.size[0] == 3 * (200 + 16) + 16
    written = bench.write_results(conn, made["id"], tmp_path / "out", side=256)
    assert {p.name for p in written} == {"claude.png", "glm.png", "mix.png", "comparison.png", "results.json"}
    assert png((tmp_path / "out" / "claude.png").read_bytes()).size == (256, 171)


def test_a_parallel_benchmark_starts_every_canvas_and_a_stopped_one_drops_the_rest(tmp_path):
    db = seeded_db(tmp_path)
    parallel = bench.create(db, {**SPEC, "order": "parallel"})
    conn = connect(db, readonly=True)
    assert [r["state"] for r in bench.results(conn, parallel["id"])["runs"]] == ["running"] * 3
    sequential = bench.create(db, SPEC)
    canvases = bench.stop(db, sequential["id"])
    assert len(canvases) == 3
    res = bench.results(conn, sequential["id"])
    assert res["status"] == "stopped"
    assert {a["status"] for a in agents_of(db, canvases[1])} == {"stopped"}
    assert bench.stop(db, "nope") is None


def test_judges_on_a_benchmark_canvas_follow_its_settings(tmp_path):
    db = seeded_db(tmp_path)
    fixed = bench.create(db, {**SPEC, "order": "parallel", "judge": {"harness": "claude", "model": "judge-model"},
                              "variants": {"glm": SPEC["variants"]["glm"]}})
    off = bench.create(db, {**SPEC, "order": "parallel", "canvas": {**SPEC["canvas"], "auto_judges": False},
                            "variants": {"glm": SPEC["variants"]["glm"]}})
    conn = connect(db)
    for b in (fixed, off):
        cid = bench.results(conn, b["id"])["runs"][0]["canvas_id"]
        conn.executemany("INSERT INTO canvas_ops (canvas_id, agent_id, ts, tool, status) VALUES (?, NULL, ?, 'look', 'view')",
                         [(cid, time.time())] * 1000)
    conn.commit()
    [judge] = queue_judges(db)
    config = json.loads(conn.execute("SELECT config FROM canvas_agents WHERE id=?", (judge,)).fetchone()[0])
    assert (config["harness"], config["model"], config["effort"]) == ("claude", "judge-model", "high")


def test_a_canvas_set_up_by_hand_becomes_a_spec(tmp_path):
    db = seeded_db(tmp_path)
    row = create_canvas(db, "By hand", viewport=256, max_calls=20, task="A harbour.", frame=(1024, 768))
    first = create_agent(db, row["id"], {"pair_id": "random", "x": 10, "y": 20, "max_successors": 1, "task": "Sky."})
    create_agent(db, row["id"], {"pair_id": PAIR, "harness": "pi", "model": "glm-x", "x": 500, "y": 300, "cap": 4})
    create_agent(db, row["id"], {"kind": "judge"}, extra={"auto": True})  # started by itself: not part of the setup
    shared = SharedCanvas(db, row["id"])
    raw = dict(shared.conn.execute("SELECT * FROM canvas_agents WHERE id=?", (first["id"],)).fetchone())
    queue_successor(shared, raw, json.loads(raw["config"]), 0, 0)  # a successor: comes again by itself
    shared.close()
    spec = bench.spec_from_canvas(connect(db, readonly=True), row["id"])
    assert spec["canvas"] == {"viewport": 256, "max_calls": 20, "task": "A harbour.", "frame": {"width": 1024, "height": 768}}
    assert [(a["x"], a["y"], a["pair_id"]) for a in spec["agents"]] == [(10, 20, "random"), (500, 300, PAIR)]
    assert spec["agents"][0]["max_successors"] == 1 and spec["agents"][0]["task"] == "Sky."
    assert spec["variants"]["as spawned"][1] == {"harness": "pi", "model": "glm-x", "effort": "high", "provider": "opencode-go"}
    assert len(bench.plan(connect(db), spec)["runs"]) == 4


# ---- HTTP and the command line -------------------------------------------------------------------------------------


def test_benchmarks_and_the_kill_switch_over_http(served, monkeypatch):  # noqa: F811
    base, db, launcher = served
    real_popen = __import__("subprocess").Popen
    monkeypatch.setattr("conveyor.launch.subprocess.Popen", lambda argv, **kw: real_popen(["/bin/sleep", "30"], **kw))
    with urllib.request.urlopen(base + "/bench") as r:
        assert b"Benchmarks" in r.read()
    status, listed = http(base, "GET", "/api/benchmarks")
    assert status == 200 and listed["benchmarks"] == [] and {"claude", "gpt", "muse"} <= set(listed["presets"])
    assert bench.plan(connect(db), {**listed["example"], "seed": 1})  # the example checks out
    status, planned = http(base, "POST", "/api/benchmarks/plan", SPEC)
    assert status == 200 and planned["canvases"] == 3 and planned["runs"][2]["agents"][1]["model"].startswith("glm")
    assert http(base, "GET", "/api/canvases") == (200, [])
    assert http(base, "POST", "/api/benchmarks", {**SPEC, "order": "sideways"})[0] == 422
    status, made = http(base, "POST", "/api/benchmarks", SPEC)
    assert status == 201, made
    status, res = http(base, "GET", f"/api/benchmarks/{made['id']}")
    first = res["runs"][0]["canvas_id"]
    assert {a["status"] for a in agents_of(db, first)} == {"starting"}  # started at once, by the watcher's pass
    with urllib.request.urlopen(f"{base}/api/benchmarks/{made['id']}/sheet.png?cell=160") as r:
        assert r.headers["Content-Type"] == "image/png" and r.read().startswith(b"\x89PNG")
    with urllib.request.urlopen(f"{base}/api/canvases/{first}/export.png?side=200") as r:
        assert r.headers["Content-Disposition"].endswith('.png"') and png(r.read()).size == (200, 133)
    assert http(base, "GET", f"/api/canvases/{first}/export.png?side=5")[0] == 400
    status, spec = http(base, "GET", f"/api/canvases/{first}/spec")
    assert status == 200 and [a["label"] for a in spec["agents"]] == ["left", "right", "critic"]
    status, stopped = http(base, "POST", "/api/stop-all", {})
    assert status == 200 and stopped["benchmarks"] == 1 and stopped["agents"] >= 3
    wait_for(lambda: {a["status"] for a in agents_of(db, first)} == {"stopped"})
    assert http(base, "GET", f"/api/benchmarks/{made['id']}")[1]["status"] == "stopped"
    assert http(base, "POST", "/api/canvases/nope/stop", {})[0] == 404
    assert http(base, "POST", "/api/benchmarks/nope/stop", {})[0] == 404
    assert http(base, "POST", "/api/stop-all", {}, headers={"Origin": "http://evil.example"})[0] == 403


def test_the_command_line_checks_a_spec_and_exports_a_canvas(tmp_path, capsys):
    db = seeded_db(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    args = build_parser().parse_args(["bench", "run", str(spec), "--dry-run", "--db", str(db)])
    args.func(args)
    assert json.loads(capsys.readouterr().out)["canvases"] == 3
    assert connect(db).execute("SELECT count(*) FROM canvases").fetchone()[0] == 0
    row = create_canvas(db, "Shore line", viewport=256, max_calls=20, frame=(512, 256))
    out = tmp_path / "shore.png"
    args = build_parser().parse_args(["export", row["id"], "-o", str(out), "--db", str(db)])
    args.func(args)
    assert png(out.read_bytes()).size == (512, 256)


# ---- reference pictures on Claude Code -----------------------------------------------------------------------------


def test_a_claude_agent_studies_reference_pictures_like_a_pi_one(tmp_path, fake_claude, fake_images, monkeypatch):  # noqa: F811
    monkeypatch.setenv("PATH", f"{fake_claude.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLAUDE_REFERENCE", "A lighthouse from three angles")
    db = seeded_db(tmp_path)
    row = create_canvas(db, "C", viewport=256, max_calls=12)
    spawned = create_agent(db, row["id"], {"pair_id": PAIR, "harness": "claude", "model": "fake-model"})
    run_agent(db, spawned["id"])
    conn = connect(db, readonly=True)
    state = views.canvas(conn, row["id"])
    done = state["agents"][0]
    assert done["status"] == "finished", done["error"]
    assert [r["prompt"] for r in state["references"]] == ["A lighthouse from three angles"]
    request = json.loads(conn.execute("SELECT request FROM sessions WHERE id=?", (done["session_id"],)).fetchone()[0])
    assert "generate_reference" in request["tools"] and "`generate_reference` makes a picture" in request["system"]
    assert len(fake_images.read_text().splitlines()) == 1
