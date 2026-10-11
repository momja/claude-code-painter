"""Brush lab is a separate, human-selected archive. Model tests use the real worker and a fake CLI."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_launch import call, wait_for

from conveyor import brushlab as lab
from conveyor.__main__ import build_parser
from conveyor.commons.catalog import catalog, pair_key
from conveyor.launch import Launcher, LaunchError
from conveyor.mcp import ToolFailure
from conveyor.painting.seeds import PEN, ROUND
from conveyor.painting.workbench import Workbench
from conveyor.server import make_server
from conveyor.store import connect, init_db


@pytest.fixture
def saved(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("PATH", str(fake_claude.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", ROUND)
    db = tmp_path / "lab.db"
    row = lab.create_lab(db, {"name": "Strange marks", "brief": "Remember earlier gestures."})
    return db, row["id"]


def view(saved):
    db, lid = saved
    conn = connect(db, readonly=True)
    try:
        return lab.detail(conn, lid)
    finally:
        conn.close()


def design(saved, **body):
    return lab.create_job(*saved, "design", {"count": 1, **body})


def ready(saved, **body):
    job = design(saved, **body)
    lab.run_job(saved[0], job["id"])
    return next(c for c in view(saved)["candidates"] if c["job_id"] == job["id"])


def test_empty_lab_needs_no_run_and_no_model(saved):
    data = view(saved)
    assert data["name"] == "Strange marks" and data["jobs"] == [] and data["candidates"] == []
    conn = connect(saved[0])
    try:
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
        assert {s["id"] for s in lab.sources(conn)} == {"seed:round", "seed:pen"}
    finally:
        conn.close()


@pytest.mark.parametrize("body", [
    [], {"count": True}, {"count": 0}, {"count": 5}, {"count": 1.5}, {"count": "2"},
    {"locks": "keep it"}, {"locks": [""]}, {"locks": ["x"] * 13}, {"locks": [1]},
    {"lock_interface": "yes"}, {"parent_id": "missing"}, {"parent2_id": "missing"},
    {"source_id": "../../etc/passwd"}, {"source_id": []}, {"source": ROUND},
    {"config": {"cap": True}}, {"config": {"cap": float("nan")}}, {"config": {"cap": float("inf")}},
    {"config": {"cap": 101}}, {"config": {"cap": 0}}, {"config": {"claude": "/bin/sh"}},
    {"config": {"model": "--command"}}, {"config": {"harness": []}}, {"config": None},
    {"config": {"provider": "missing"}}, {"brief": "x" * 12001},
])
def test_invalid_jobs_never_create_work(saved, body):
    with pytest.raises(LaunchError):
        lab.create_job(*saved, "design", body)
    assert view(saved)["jobs"] == []


def test_one_batch_at_a_time_even_for_racing_requests(saved):
    def attempt():
        try:
            return design(saved)
        except LaunchError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sum(r is not None for r in results) == 1
    assert len(view(saved)["jobs"]) == 1


def test_worker_saves_candidates_replays_and_sessions_without_evaluations(saved):
    job = design(saved, count=2, lock_interface=True, locks=["Keep the broken edges"])
    lab.run_job(saved[0], job["id"])
    data = view(saved)
    assert data["jobs"][0]["status"] == "finished"
    assert data["jobs"][0]["cost"] == pytest.approx(.02)
    assert len(data["candidates"]) == 2
    conn = connect(saved[0])
    try:
        for c in data["candidates"]:
            assert c["status"] == "ready" and c["source"] == ROUND and not c["published"]
            assert c["session_id"] and c["report"]["ok"]
            assert conn.execute("SELECT data FROM artifacts WHERE name=?", (c["sheet"],)).fetchone()
            sequence = c["report"]["replays"][0]
            assert sequence[0]["tool"] == "Blank canvas" and sequence[1]["tool"] == "stroke"
            for frame in sequence:
                assert conn.execute("SELECT data FROM artifacts WHERE name=?", (frame["image"],)).fetchone()
        for table in ("runs", "organisms", "evaluations"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        sessions = conn.execute("SELECT request FROM sessions").fetchall()
        assert len(sessions) == 2
        assert "Human's request for this round" in sessions[0]["request"]
    finally:
        conn.close()
    # Re-running the same worker or its process-exit callback cannot erase a successful result.
    lab.run_job(saved[0], job["id"])
    lab.close_job(saved[0], job["id"], "failed", "late callback")
    assert view(saved)["jobs"][0]["status"] == "finished"


def test_branch_and_recombine_preserve_parents(saved):
    a, b = ready(saved), ready(saved)
    j = design(saved, parent_id=a["id"], parent2_id=b["id"], brief="Combine their tearing and color pickup.",
               locks=["Keep the gaps"], lock_interface=True)
    data = view(saved)
    child = data["candidates"][-1]
    assert (child["parent_id"], child["parent2_id"]) == (a["id"], b["id"])
    req = data["jobs"][-1]["request"]
    assert req["source"] == a["source"] and req["other_source"] == b["source"]
    text = lab.design_prompt(req, 0, [])
    assert "Keep the gaps" in text and "Second parent" in text and "Interface lock: ON" in text
    lab.run_job(saved[0], j["id"])
    assert view(saved)["candidates"][0] == a


def test_cannot_use_candidate_from_another_lab(saved):
    c = ready(saved)
    other = lab.create_lab(saved[0], {"name": "Other", "brief": "Other design"})
    with pytest.raises(LaunchError, match="from this lab"):
        lab.create_job(saved[0], other["id"], "design", {"parent_id": c["id"]})


def test_interface_lock_checks_tool_params_and_allows_doc_changes(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({"width": 512, "height": 512, "max_tries": 4,
                                                   "locked_interface": lab.interface_contract(ROUND)}))
    bench = Workbench(tmp_path)
    with pytest.raises(ToolFailure, match="Interface lock"):
        bench.call("submit_instrument", {"source": PEN, "summary": "wrong interface"}, {})
    assert not (tmp_path / "submitted.json").exists()
    changed = ROUND.replace('"type": "color"', '"type": "color", "default": "#ff0000"', 1)
    assert lab.interface_contract(changed) != lab.interface_contract(ROUND)
    with pytest.raises(ToolFailure, match="Interface lock"):
        bench.call("submit_instrument", {"source": changed, "summary": "wrong default"}, {})
    documented = ROUND.replace("Start x.", "Horizontal origin.")
    assert lab.interface_contract(documented) == lab.interface_contract(ROUND)
    bench.call("submit_instrument", {"source": documented, "summary": "same interface"}, {})
    assert (tmp_path / "submitted.json").exists()


def test_invalid_designer_output_does_not_become_ready(saved, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", "not valid python!")
    j = design(saved)
    lab.run_job(saved[0], j["id"])
    data = view(saved)
    assert data["jobs"][0]["status"] == "failed"
    assert data["candidates"][0]["status"] == "failed" and data["candidates"][0]["error"]
    with pytest.raises(LaunchError, match="ready"):
        lab.publish(*saved, data["candidates"][0]["id"], {})


def test_publication_is_explicit_confirmed_and_immutable(saved):
    c = ready(saved, locks=["Keep the gaps"])
    prompt = "Deliberately explore the gaps between marks."
    conn = connect(saved[0])
    try:
        before = {p["id"] for p in catalog(conn)["pairs"]}
        assert pair_key(c["source"], c["prompt"]) not in before
        with pytest.raises(LaunchError, match="confirm"):
            lab.publish(*saved, c["id"], {"prompt": prompt})
        published = lab.publish(*saved, c["id"], {"prompt": prompt, "confirm_traits": True})
        assert published["pair_id"] == pair_key(c["source"], prompt)
        found = next(p for p in catalog(conn)["pairs"] if p["id"] == published["pair_id"])
        assert found["published"] and found["lab"]["id"] == saved[1]
        assert found["best_score"] is None and found["instrument"]["sheet"]
        assert lab.publish(*saved, c["id"], {"prompt": prompt, "confirm_traits": True}) == published
        with pytest.raises(LaunchError, match="immutable"):
            lab.publish(*saved, c["id"], {"prompt": "different", "confirm_traits": True})
        # A high-scoring run removes the seed fallback, but never the published Brush lab pair.
        conn.execute("INSERT INTO runs VALUES ('ranked','Ranked run',?, '{}')", (time.time(),))
        for oid, node, genome in [("i", "instrument", {"source": PEN}), ("p", "painter", {"prompt": "copy"})]:
            conn.execute("INSERT INTO organisms (id,run_id,node,created,genome,viable) VALUES (?,'ranked',?,?,?,1)",
                         (oid, node, time.time(), json.dumps(genome)))
        conn.execute("INSERT INTO evaluations (id,run_id,node,organism_id,partner_id,score,viable) "
                     "VALUES ('ev','ranked','instrument','i','p',.9,1)")
        conn.commit()
        offered = {p["id"] for p in catalog(conn)["pairs"]}
        assert offered == {published["pair_id"], pair_key(PEN, "copy")}
    finally:
        conn.close()


def test_audition_uses_saved_pair_and_has_no_scores(saved):
    c = ready(saved)
    j = lab.create_job(*saved, "audition", {"candidate_id": c["id"], "brief": "Blue echoes", "actions": 12})
    lab.run_job(saved[0], j["id"])
    audition = view(saved)["jobs"][-1]
    assert audition["status"] == "finished", audition["error"]
    assert audition["request"]["source"] == c["source"]
    assert audition["request"]["prompt"] == c["prompt"]
    result = audition["result"]
    assert result["artifacts"]["painting"] and result["session_id"]
    assert result["details"].get("score") is None
    conn = connect(saved[0])
    try:
        assert not conn.execute("SELECT 1 FROM evaluations").fetchone()
        assert not conn.execute("SELECT 1 FROM painting_requests").fetchone()
    finally:
        conn.close()


@pytest.fixture
def served(saved):
    launcher = Launcher(saved[0], build_parser, token="secret")
    server = make_server(saved[0], "127.0.0.1", 0, launcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", launcher, saved[1]
    launcher.shutdown(grace=5)
    server.shutdown()
    server.server_close()


def test_http_guard_and_real_launch(served):
    base, launcher, lid = served
    path = f"/api/brush-labs/{lid}/design"
    headers = {"X-Conveyor-Token": "secret"}
    assert call(base, "POST", path, {})[0] == 401
    assert call(base, "POST", path, {}, {**headers, "Origin": "https://evil.example"})[0] == 403
    assert call(base, "POST", path, {}, {**headers, "Content-Type": "text/plain"})[0] == 403
    assert call(base, "POST", path, [], headers)[0] == 422
    status, job = call(base, "POST", path, {"count": 1}, headers)
    assert status == 201
    wait_for(lambda: launcher.get(job["launch_id"])["state"] in ("finished", "failed"))
    status, data = call(base, "GET", f"/api/brush-labs/{lid}")
    assert status == 200 and data["candidates"][0]["status"] == "ready"
    cid = data["candidates"][0]["id"]
    assert call(base, "POST", f"/api/brush-labs/{lid}/candidates/{cid}/publish", {})[0] == 401
    assert call(base, "POST", f"/api/brush-labs/{lid}/candidates/{cid}/publish", {}, headers)[0] == 200
    assert call(base, "GET", "/api/brush-labs/missing")[0] == 404


def test_stop_batch_preserves_terminal_candidates_and_stops_queued_work(served, monkeypatch, tmp_path):
    base, launcher, lid = served
    flag = tmp_path / "stall"
    flag.touch()
    monkeypatch.setenv("FAKE_CLAUDE_STALL", str(flag))
    j = launcher.start_brush_job(lid, "design", {"count": 3})
    wait_for(lambda: not flag.exists())
    assert launcher.stop_brush_job("different-lab", j["id"]) is None
    launcher.stop_brush_job(lid, j["id"])
    wait_for(lambda: launcher.get(j["launch_id"])["state"] == "stopped")
    data = wait_for(lambda: (d if not any(lab.ACTIVE.count(c["status"]) for c in d["candidates"]) else None)
                    if (d := call(base, "GET", f"/api/brush-labs/{lid}")[1]) else None)
    assert data["jobs"][0]["status"] == "stopped"
    assert all(c["status"] == "stopped" for c in data["candidates"])


def test_spawn_failure_reconciles_job(saved, monkeypatch):
    launcher = Launcher(saved[0], build_parser)
    def fail(*a, **kw):
        raise OSError("test launch failure")
    monkeypatch.setattr("conveyor.launch.subprocess.Popen", fail)
    with pytest.raises(LaunchError, match="test launch failure"):
        launcher.start_brush_job(saved[1], "design", {"count": 2})
    data = view(saved)
    assert data["jobs"][0]["status"] == "failed"
    assert all(c["status"] == "failed" for c in data["candidates"])


def test_locks_and_human_history_carry_forward_unless_explicitly_changed(saved):
    parent = ready(saved, brief="Make broken contact", locks=["Keep the gaps"], lock_interface=True)
    design(saved, parent_id=parent["id"], brief="Now add memory")
    request = view(saved)["jobs"][-1]["request"]
    assert request["locks"] == ["Keep the gaps"] and request["lock_interface"]
    assert request["history"] == ["Make broken contact"]
    assert request["lab_brief"] == "Remember earlier gestures."


def test_recovery_releases_dead_workers_without_discarding_ready_candidates(saved, monkeypatch):
    j = design(saved, count=2)
    conn = connect(saved[0])
    conn.execute("UPDATE brush_jobs SET worker_pid=12345,status='running' WHERE id=?", (j["id"],))
    conn.execute("UPDATE brush_candidates SET status='ready' WHERE job_id=? AND ordinal=0", (j["id"],))
    conn.commit()
    conn.close()
    def dead(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(lab.os, "kill", dead)
    lab.recover_jobs(saved[0])
    data = view(saved)
    assert data["jobs"][0]["status"] == "failed"
    assert [c["status"] for c in data["candidates"]] == ["ready", "failed"]
    design(saved)  # The recovered lab isn't permanently blocked.


def test_recovery_leaves_live_and_recently_queued_jobs_alone(saved):
    j = design(saved)
    lab.recover_jobs(saved[0])
    assert view(saved)["jobs"][0]["status"] == "queued"
    conn = connect(saved[0])
    conn.execute("UPDATE brush_jobs SET worker_pid=?,created=0,status='running' WHERE id=?", (os.getpid(), j["id"]))
    conn.commit()
    conn.close()
    lab.recover_jobs(saved[0])
    assert view(saved)["jobs"][0]["status"] == "running"


def test_failed_audition_keeps_its_transcript_link(saved):
    c = ready(saved)
    j = lab.create_job(*saved, "audition", {"candidate_id": c["id"], "prompt": "Try these instructions"})
    conn = connect(saved[0])
    conn.execute("UPDATE brush_jobs SET result=? WHERE id=?", (json.dumps({"session_id": "partial"}), j["id"]))
    conn.commit()
    conn.close()
    lab.close_job(saved[0], j["id"], "failed", "connection dropped")
    job = view(saved)["jobs"][-1]
    assert job["result"]["session_id"] == "partial"
    assert job["request"]["prompt"] == "Try these instructions"


def test_designer_sees_parent_sheet_and_latest_audition(saved):
    c = ready(saved)
    j = lab.create_job(*saved, "audition", {"candidate_id": c["id"], "brief": "Blue echoes", "actions": 10})
    lab.run_job(saved[0], j["id"])
    design(saved, parent_id=c["id"])
    request = view(saved)["jobs"][-1]["request"]
    content = lab._design_content(saved[0], request, 0, [])
    assert sum(block["type"] == "png" for block in content) == 2
    assert any("Blue echoes" in b.get("text", "") and "thinner lines" in b["text"] for b in content)


def test_pi_designer_uses_real_sidecar_and_workbench(saved, monkeypatch):
    from conveyor.pi import PiAgent, available
    if available() is not None:
        pytest.skip(available())
    faux = [{"blocks": [{"tool": "try_instrument", "args": {"source": PEN}}]},
            {"blocks": [{"tool": "submit_instrument", "args": {"source": PEN, "summary": "A loaded pen"}}]}]
    def harness(kind, model, effort, args, store, meter, cache):
        return PiAgent("faux", store=store, meter=meter, api_key="test", faux=faux)
    monkeypatch.setattr("conveyor.__main__._harness", harness)
    c = ready(saved, config={"harness": "pi"})
    assert c["status"] == "ready" and c["source"] == PEN


def test_schema_upgrade_is_additive(tmp_path):
    db = tmp_path / "old.db"
    conn = connect(db)
    conn.execute("CREATE TABLE artifacts (name TEXT PRIMARY KEY, data BLOB NOT NULL)")
    conn.execute("INSERT INTO artifacts VALUES ('original', ?)", (b"unchanged",))
    conn.commit()
    conn.close()
    init_db(db)
    init_db(db)
    conn = connect(db)
    try:
        assert conn.execute("SELECT data FROM artifacts").fetchone()[0] == b"unchanged"
        assert lab.listing(conn) == []
    finally:
        conn.close()
