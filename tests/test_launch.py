"""Starting runs from the dashboard: the form's values to argv, the POST guards, and a real offline child process."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from conveyor.__main__ import build_parser
from conveyor.launch import LaunchError
from conveyor.launch import Launcher
from conveyor.server import make_server
from conveyor.store import init_db


@pytest.fixture
def launcher(tmp_path):
    return Launcher(tmp_path / "runs" / "t.db", build_parser)


@pytest.fixture
def served(tmp_path, launcher):
    init_db(launcher.db)
    server = make_server(launcher.db, "127.0.0.1", 0, launcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", launcher
    launcher.shutdown(grace=5)
    server.shutdown()


def call(base, method, path, body=None, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(base + path, method=method, headers=h,
                                 data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_for(fn, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if (v := fn()):
            return v
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_argv_sends_only_what_was_set(launcher):
    argv = launcher.argv({"cycles": 3, "offline": True, "judge": False, "seeds": ["round", "pen"],
                          "name": "a b", "wait": 4, "model": "", "looks": None})
    assert argv[0] == "run"
    assert {"--cycles=3", "--offline", "--no-judge", "--seeds=round,pen", "--name=a b", "--wait=4.0"} <= set(argv)
    assert argv[-2:] == ["--no-serve", f"--db={launcher.db}"]
    assert not any(a.startswith("--model") or a.startswith("--looks") for a in argv)


@pytest.mark.parametrize("values, message", [
    ({"db": "/etc/passwd"}, "Unknown option"),
    ({"claude": "/bin/sh"}, "Unknown option"),
    ({"no_serve": False}, "Unknown option"),
    ({"effort": "turbo"}, "invalid choice"),
    ({"model": "--budget=9999"}, "dash"),
    ({"cycles": 0}, "between"),
    ({"cycles": "many"}, "whole number"),
    ({"lanes": True}, "number"),
    ({"seeds": ["../x"]}, "Unknown seed"),
    ({"target": "/etc/passwd"}, "Unknown target"),
    ({"refine": 0, "invent": 0, "recombine": 0}, "weights"),
    ({"judge": "yes"}, "on or off"),
])
def test_argv_refuses(launcher, values, message):
    with pytest.raises(LaunchError, match=message):
        launcher.argv(values)


def test_options_come_from_the_parser(launcher):
    o = launcher.options()
    assert o["launch"] and o["defaults"]["cycles"] == 6 and o["defaults"]["harness"] == "claude"
    assert "round" in o["seeds"] and "self_portrait" in o["targets"]
    assert o["choices"]["effort"][0] == "low" and "db" not in o["defaults"]
    assert o["defaults"]["paint_batch"] is True and o["defaults"]["paint_context_turns"] == 2


def test_painter_cost_controls_round_trip_through_launcher(launcher):
    argv = launcher.argv({"paint_batch": False, "paint_context_turns": 0})
    assert "--no-paint-batch" in argv and "--paint-context-turns=0" in argv
    args = build_parser().parse_args(argv)
    from conveyor.__main__ import _setup
    assert _setup(args).paint_batch is False and args.paint_context_turns == 0
    for value in (-1, 21):
        with pytest.raises(LaunchError, match="between"):
            launcher.argv({"paint_context_turns": value})


def test_post_guards(served):
    base, _ = served
    assert call(base, "POST", "/api/launches", {"nope": 1})[0] == 422
    assert call(base, "POST", "/api/launches", {}, {"Origin": "http://evil.example"})[0] == 403
    assert call(base, "POST", "/api/launches", {}, {"Host": "evil.example"})[0] == 403
    assert call(base, "POST", "/api/launches", {}, {"Content-Type": "text/plain"})[0] == 403
    assert call(base, "POST", "/api/launches/nope/stop", {})[0] == 404
    assert call(base, "GET", "/api/launches")[1] == []


def test_no_launching_without_a_launcher(tmp_path):
    init_db(tmp_path / "x.db")
    server = make_server(tmp_path / "x.db", "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    assert call(base, "GET", "/api/options")[1] == {"launch": False}
    assert call(base, "POST", "/api/launches", {"offline": True})[0] == 404
    server.shutdown()


def test_start_an_offline_run_and_read_it_back(served):
    base, _ = served
    status, launch = call(base, "POST", "/api/launches", {"offline": True, "cycles": 1, "width": 64, "actions": 10,
                                                          "name": "from the form"})
    assert status == 201
    done = wait_for(lambda: (d := call(base, "GET", f"/api/launches/{launch['id']}")[1])["state"] != "running" and d)
    assert done["state"] == "finished" and done["exit_code"] == 0 and done["run_id"]
    runs = call(base, "GET", "/api/runs")[1]
    assert [(r["id"], r["name"], r["finished"]) for r in runs] == [(done["run_id"], "from the form", True)]


def test_stop_a_running_run(served):
    base, _ = served
    _, launch = call(base, "POST", "/api/launches", {"offline": True, "cycles": 1000, "width": 64, "actions": 10})
    run_id = wait_for(lambda: call(base, "GET", f"/api/launches/{launch['id']}")[1]["run_id"])
    wait_for(lambda: call(base, "GET", f"/api/runs/{run_id}")[1]["cycle"])  # the conductor is going
    assert call(base, "POST", f"/api/launches/{launch['id']}/stop", {})[1]["state"] in ("stopping", "stopped")
    done = wait_for(lambda: (d := call(base, "GET", f"/api/launches/{launch['id']}")[1])["state"] == "stopped" and d)
    run = call(base, "GET", f"/api/runs/{done['run_id']}")[1]
    assert run["finished"]["stopped_early"] is True


def test_a_run_that_cannot_start_reports_why(served, monkeypatch, tmp_path):
    base, _ = served
    # `claude` isn't on this PATH, so the real harness refuses at once; the output says so.
    monkeypatch.setenv("PATH", str(tmp_path))
    _, launch = call(base, "POST", "/api/launches", {"cycles": 1})
    done = wait_for(lambda: (d := call(base, "GET", f"/api/launches/{launch['id']}")[1])["state"] != "running" and d)
    assert done["state"] == "failed" and done["run_id"] is None
    assert "claude --version" in " ".join(done["log"])


TOKEN = "a-long-enough-launch-token"


@pytest.fixture
def guarded(tmp_path):
    launcher = Launcher(tmp_path / "t.db", build_parser, token=TOKEN)
    init_db(launcher.db)
    server = make_server(launcher.db, "127.0.0.1", 0, launcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", launcher
    launcher.shutdown(grace=5)
    server.shutdown()


def test_a_token_locks_every_launch_endpoint(guarded):
    base, _ = guarded
    assert call(base, "GET", "/api/options")[1] == {"launch": True, "locked": True}
    assert call(base, "GET", "/api/options", headers={"X-Conveyor-Token": "wrong"})[1]["locked"] is True
    assert call(base, "GET", "/api/launches")[0] == 401
    assert call(base, "POST", "/api/launches", {"offline": True})[0] == 401
    assert call(base, "POST", "/api/launches", {"offline": True}, {"X-Conveyor-Token": "wrong"})[0] == 401
    assert call(base, "POST", "/api/launches/x/stop", {})[0] == 401
    assert call(base, "GET", "/api/runs")[0] == 200  # the dashboard's reads stay open


def test_with_the_token_any_host_may_launch(guarded):
    base, _ = guarded
    ok = {"X-Conveyor-Token": TOKEN}
    assert call(base, "GET", "/api/options", headers=ok)[1]["launch"] is True
    # Behind a proxy the Host is the public name; the token is what proves the caller.
    h = {**ok, "Host": "conveyor.example.com"}
    status, launch = call(base, "POST", "/api/launches", {"offline": True, "cycles": 1, "width": 64, "actions": 8}, h)
    assert status == 201
    wait_for(lambda: call(base, "GET", f"/api/launches/{launch['id']}", headers=ok)[1]["state"] == "finished")
    # A foreign Origin is still refused, token or not.
    assert call(base, "POST", "/api/launches", {}, {**h, "Origin": "http://evil.example"})[0] == 403


def test_the_form_defaults_to_pi_when_claude_is_missing(launcher, monkeypatch):
    monkeypatch.setattr(Launcher, "harnesses", lambda self: {
        "claude": {"ok": False, "detail": "missing"},
        "pi": {"ok": True, "detail": "ready", "keys": {}, "env_vars": {}}})
    assert launcher.options()["defaults"]["harness"] == "pi"


def test_a_claude_that_is_not_logged_in_blocks_the_claude_harness(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text('''#!/bin/sh
case "$1" in
  --version) echo "9.9.9 (Claude Code)" ;;
  auth) echo '{"loggedIn": false}'; exit 1 ;;
esac
''')
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    from conveyor.claude import logged_in
    assert logged_in("claude") is False
    claude = Launcher(tmp_path / "t.db", build_parser).harnesses()["claude"]
    assert claude["ok"] is False and "not logged in" in claude["detail"] and "auth login" in claude["detail"]
    assert logged_in("/nonexistent/claude") is None


def _png(size, mode="RGB", color=(200, 30, 30)):
    import base64
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new(mode, size, color).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def test_import_target_stores_a_shrunk_png_and_argv_uses_its_path(launcher):
    done = launcher.import_target({"name": "My Photo!.png", "image": _png((4000, 3000))})
    assert (done["width"], done["height"]) == (2048, 1536)
    assert done["target"].startswith("My-Photo-") and done["target"] in launcher.targets()
    assert launcher.import_target({"name": "My Photo!.png", "image": _png((4000, 3000))})["target"] == done["target"]
    argv = launcher.argv({"target": done["target"]})
    path = next(a for a in argv if a.startswith("--target=")).split("=", 1)[1]
    assert path.endswith(done["target"] + ".png") and build_parser().parse_args(argv).target == path


@pytest.mark.parametrize("body, message", [
    ({"image": "not base64!"}, "valid image"),
    ({"image": _png((8, 8))}, "at least 16"),
    ({"image": "A" * (9 * 1024 * 1024)}, "6 MB"),
    ({}, "6 MB"),
])
def test_import_target_refuses(launcher, body, message):
    with pytest.raises(LaunchError, match=message):
        launcher.import_target(body)


def test_import_target_over_http(served):
    base, launcher = served
    status, done = call(base, "POST", "/api/targets", {"name": "x.png", "image": _png((64, 64))})
    assert status == 201 and done["target"] in call(base, "GET", "/api/options")[1]["targets"]
    assert call(base, "POST", "/api/targets", {"image": "zzz"})[0] == 422
