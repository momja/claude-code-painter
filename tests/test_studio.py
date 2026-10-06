"""Graph completeness, request guards, inherited settings, and unscored text-only paintings."""
import base64
import io
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from conveyor.__main__ import build_parser, _config
from conveyor.evolve import Organism
from conveyor.harness import Outcome
from conveyor.launch import LaunchError, Launcher
from conveyor.painting.paintserver import PaintSession
from conveyor.painting.problem import Painter, Setup
from conveyor.painting.seeds import PEN
from conveyor.server import Views, make_server
from conveyor.store import Store, connect
from conveyor.studio import create_request, inherited_config, run_request
from test_launch import call, wait_for


def png_data():
    output = io.BytesIO()
    Image.new("RGB", (64, 80), "#556677").save(output, "PNG")
    return base64.b64encode(output.getvalue()).decode()


@pytest.fixture
def saved(tmp_path, fake_claude):
    db = tmp_path / "studio.db"
    args = build_parser().parse_args(["run", "--width=64", "--actions=10", "--no-judge",
                                     f"--claude={fake_claude}", "--model=global-model", "--paint-model=paint-model",
                                     "--paint-effort=medium", "--paint-context-turns=0", "--no-paint-batch", "--scope"])
    store = Store(db, config=_config(args))
    inst = Organism(id="instrument", node="instrument", genome={"source": PEN})
    prompt = Organism(id="prompt", node="painter", genome={"prompt": "Saved painter strategy"})
    for org in (inst, prompt):
        store.organism({"id": org.id, "node": org.node, "created": org.created, "genome": org.genome, "viable": 1})
    store.emit("run_finished")
    store.flush()
    yield db, store, inst, prompt
    store.close()


def body(**kwargs):
    return {"instrument_id": "instrument", "prompt_id": "prompt", "text": "A blue mountain at night", **kwargs}


class LocalPainter:
    def __init__(self, store):
        self.store = store
        self.job = None

    def run(self, job):
        self.job = job
        sid = "text-session"
        self.store.start_session(sid, node="painting", organism_id=job.organism_id, purpose="paint", model="test",
                                 effort="medium", request={}, dir=str(job.cwd))
        job.on_start(sid)
        session = PaintSession(job.cwd)
        session.apply("start", {"x": 10, "y": 10, "color": "#223344"})
        session.apply("move", {"dx": 20, "dy": 0})
        session.look()
        session.view("detail", {"x": 32, "y": 32, "span": 16})
        session.scope({"x": 32, "y": 32, "span": 16})
        session.scope({"clear": True})
        session.finish("Done")
        session._log.close()
        self.store.update_session(sid, status="ok", ended=time.time())
        return Outcome(sid, ok=True)


def test_text_only_on_pi_uses_real_tools_and_bounded_context(saved):
    from conveyor.pi import PiAgent, available
    from conveyor.painting.seeds import ROUND
    if available() is not None:
        pytest.skip(available())
    db, store, _, prompt = saved
    faux = [{"blocks": [{"tool": "stroke", "args": {"x": 20, "y": 20, "angle": 0, "length": 30,
                                                   "size": 6, "color": "#223344"}}, {"tool": "look", "args": {}}]},
            {"blocks": [{"tool": "finish", "args": {"note": "Painted from text"}}]}]
    harness = PiAgent("faux-model", store=store, faux=faux, paint_context_turns=2)
    painter = Painter(Setup(target=None, width=64, height=80, brief="Blue mountains", work_dir=db.parent), store, harness)
    painting = painter.paint(Organism(node="instrument", genome={"source": ROUND}), prompt)
    assert painting.viable and painting.score is None and painting.critic == {}
    assert painting.details["note"] == "Painted from text"
    store.flush()
    session = Views(db).session(painting.session_id)
    assert session["request"]["harness"] == "pi"
    assert session["strokes"] and all("score_after" not in row["data"] for row in session["strokes"])


def test_text_only_reuses_strategy_tools_and_has_no_metrics(saved, monkeypatch):
    db, store, inst, prompt = saved
    harness = LocalPainter(store)
    monkeypatch.setattr("conveyor.painting.critic.Critic.score", lambda *a: pytest.fail("No target means no critic"))
    painter = Painter(Setup(target=None, width=64, height=80, brief="Blue mountains", work_dir=db.parent,
                            actions=10, scope_views=True), store, harness)
    painting = painter.paint(inst, prompt)
    assert painting.viable and painting.score is None and painting.critic == {}
    assert set(painting.artifacts) == {"painting"}
    assert "Saved painter strategy" in harness.job.system_prompt
    assert "Blue mountains" in json.dumps([b for b in harness.job.content if b["type"] == "text"])
    assert {"start", "move", "detail", "paint_batch", "scope", "look", "finish"} <= set(harness.job.tools)
    assert painting.details["canvas"] == {"width": 64, "height": 80}
    assert not painter.judge
    calls = [json.loads(line) for line in (harness.job.cwd / "calls.jsonl").read_text().splitlines()]
    assert not any(k.startswith(("score_", "error_")) for c in calls for k in c)
    finish = json.loads((harness.job.cwd / "finish.json").read_text())
    assert "score" not in finish and "error" not in finish


@pytest.mark.parametrize("values,match", [
    (body(model="override"), "cannot be changed"),
    (body(text="", image=None), "text or upload"),
    (body(text="a" * 12001), "12000"),
    (body(prompt_id="missing"), "viable painter"),
    (body(instrument_id="prompt"), "viable instrument"),
    (body(image="not-an-image"), "valid PNG"),
    (body(image={"path": "/etc/passwd"}), "6 MB"),
])
def test_bad_requests_never_launch(saved, values, match):
    db, store, *_ = saved
    with pytest.raises(LaunchError, match=match):
        create_request(db, store.run_id, values)


def test_other_runs_organisms_cannot_be_selected(saved):
    db, _store, *_ = saved
    other = Store(db)
    try:
        with pytest.raises(LaunchError, match="viable instrument"):
            create_request(db, other.run_id, body())
    finally:
        other.close()


def test_inherits_resolved_roles_and_does_not_use_current_defaults(saved):
    db, store, *_ = saved
    store.emit("role_settings", node="paint", harness="pi", model="resolved-pi-model", effort="xhigh",
               provider="openrouter", definition={"id": "resolved-pi-model"}, paint_context_turns=0)
    store.emit("role_settings", node="judge", harness="claude", model="resolved-judge-model", effort="low")
    store.flush()
    conn = connect(db)
    try:
        config = inherited_config(conn, store.run_id)
    finally:
        conn.close()
    assert config["paint_harness"] == "pi" and config["paint_model"] == "resolved-pi-model"
    assert config["paint_effort"] == "xhigh" and config["provider"] == "openrouter"
    assert config["judge_model"] == "resolved-judge-model" and config["judge_effort"] == "low"
    assert config["paint_batch"] is False and config["scope"] is True and config["paint_context_turns"] == 0
    assert config["saved_roles"]["paint"]["definition"] == {"id": "resolved-pi-model"}


def test_legacy_run_uses_recorded_session_model(saved):
    db, store, *_ = saved
    store.start_session("old", node="painting", organism_id="instrument", purpose="paint", model="actual-model",
                        effort="low", request={"harness": "pi"}, dir=None)
    store.flush()
    conn = connect(db)
    try:
        config = inherited_config(conn, store.run_id)
    finally:
        conn.close()
    assert config["paint_model"] == "actual-model" and config["paint_effort"] == "low"
    assert config["paint_harness"] == "pi"


@pytest.mark.parametrize("image", [None, True])
def test_real_worker_records_request_without_affecting_evaluations(saved, image):
    db, store, *_ = saved
    request = create_request(db, store.run_id, body(image=png_data() if image else None))
    run_request(db, request["id"])
    graph = Views(db).studio(store.run_id)
    assert len(graph["organisms"]) == 2 and len(graph["paintings"]) == 1
    painting = graph["paintings"][0]
    assert painting["status"] == "finished" and painting["request"]
    assert painting["instrument_id"] == "instrument" and painting["prompt_id"] == "prompt"
    assert bool(painting["details"]["critic"]) == bool(image)
    assert (painting["score"] is not None) == bool(image)
    session = Views(db).session(painting["session_id"])
    assert session["model"] == "paint-model" and session["effort"] == "medium"
    assert "Saved painter strategy" in session["request"]["system"]
    assert "A blue mountain" in json.dumps(session["request"]["content"])
    assert session["strokes"] and painting["artifacts"]["painting"]
    conn = connect(db)
    try:
        assert conn.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    finally:
        conn.close()


def test_graph_contains_all_instruments_prompts_and_unique_paintings(saved):
    db, store, *_ = saved
    for sid in ("one", "two"):
        store.start_session(sid, node="painting", organism_id="instrument", purpose="paint", model="x", effort="high", request={}, dir=None)
        store.update_session(sid, status="ok")
        for org, partner, node in (("instrument", "prompt", "instrument"), ("prompt", "instrument", "painter")):
            store.evaluation({"id": sid+org, "node": node, "organism_id": org, "partner_id": partner,
                              "viable": 1, "score": .7, "started": 1, "ended": 2,
                              "session_id": sid, "artifacts": {"painting": sid+".png"},
                              "details": {"instrument_id": "instrument", "prompt_id": "prompt"}})
    store.organism({"id": "unused", "node": "instrument", "created": time.time(), "genome": {"source": PEN}, "viable": 0})
    store.flush()
    graph = Views(db).studio(store.run_id)
    assert {p["id"] for p in graph["paintings"]} == {"one", "two"}
    assert len(graph["organisms"]) == 3
    assert {"start", "move"} <= set(graph["organisms"][0]["tools"])
    assert "detail" in graph["organisms"][0]["views"]


def test_live_request_on_finished_run_is_not_interrupted(saved):
    db, store, *_ = saved
    request = create_request(db, store.run_id, body())
    store.start_session("live", node="painting", organism_id="instrument", purpose="paint", model="test", effort="high", request={}, dir=None)
    store.update_painting_request(request["id"], session_id="live", status="running")
    store.flush()
    assert Views(db).studio(store.run_id)["paintings"][0]["status"] == "running"
    assert Views(db).session("live")["status"] == "running"


def test_request_post_uses_launch_guards_and_runs_child(saved):
    db, store, *_ = saved
    launcher = Launcher(db, build_parser, token="a-long-enough-token")
    server = make_server(db, "127.0.0.1", 0, launcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    path = f"/api/runs/{store.run_id}/paintings"
    try:
        assert call(base, "POST", path, body())[0] == 401
        headers = {"X-Conveyor-Token": launcher.token}
        assert call(base, "POST", path, body(), {**headers, "Origin": "http://evil.example"})[0] == 403
        assert call(base, "POST", path, body(), {**headers, "Content-Type": "text/plain"})[0] == 403
        status, launch = call(base, "POST", path, body(), headers)
        assert status == 201 and launch["request_id"]
        finished = wait_for(lambda: (value := launcher.get(launch["id"]))["state"] != "running" and value)
        assert finished["state"] == "finished", finished["log"]
        painting = call(base, "GET", f"/api/runs/{store.run_id}/studio")[1]["paintings"][0]
        assert painting["status"] == "finished" and painting["score"] is None
    finally:
        launcher.shutdown(grace=5)
        server.shutdown()
        server.server_close()


def test_offline_rejects_text_but_supports_image(saved):
    db, store, *_ = saved
    conn = connect(db)
    config = json.loads(conn.execute("SELECT config FROM runs WHERE id=?", (store.run_id,)).fetchone()[0])
    config["offline"] = True
    config["actions"] = 50
    from conveyor.painting.seeds import SEEDS
    conn.execute("UPDATE organisms SET genome=? WHERE id='instrument'", (json.dumps({"source": SEEDS["round"]}),))
    conn.execute("UPDATE runs SET config=? WHERE id=?", (json.dumps(config), store.run_id))
    conn.commit()
    conn.close()
    with pytest.raises(LaunchError, match="needs a model"):
        create_request(db, store.run_id, body())
    from conveyor.painting.canvas import load_target, to_png
    image = base64.b64encode(to_png(load_target("self_portrait", width=64).image)).decode()
    request = create_request(db, store.run_id, body(image=image))
    run_request(db, request["id"])
    painting = Views(db).studio(store.run_id)["paintings"][0]
    assert painting["status"] == "finished" and painting["score"] is not None


def test_old_database_can_browse_without_request_table(saved):
    db, store, *_ = saved
    conn = connect(db)
    conn.execute("DROP TABLE painting_requests")
    conn.commit()
    conn.close()
    assert len(Views(db).studio(store.run_id)["organisms"]) == 2


def test_stopping_request_records_status_without_evaluations(saved, monkeypatch, tmp_path):
    db, store, *_ = saved
    flag = tmp_path / "stall"
    flag.touch()
    monkeypatch.setenv("FAKE_CLAUDE_STALL", str(flag))
    launcher = Launcher(db, build_parser)
    try:
        launch = launcher.start_painting(store.run_id, body())
        wait_for(lambda: Views(db).studio(store.run_id)["paintings"][0].get("session_id"))
        launcher.stop(launch["id"])
        wait_for(lambda: launcher.get(launch["id"])["state"] == "stopped")
        painting = wait_for(lambda: (p := Views(db).studio(store.run_id)["paintings"][0])["status"] == "stopped" and p)
        assert painting["request"]
        conn = connect(db)
        assert conn.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 0
        conn.close()
    finally:
        launcher.shutdown(grace=5)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_studio_script_syntax():
    page = Path(__file__).parents[1] / "src/conveyor/studio.html"
    script = re.findall(r"<script[^>]*>(.*?)</script>", page.read_text(), re.DOTALL)[0]
    subprocess.run(["node", "--check", "--input-type=module"], input=script, text=True, capture_output=True,
                   check=True, timeout=10)
