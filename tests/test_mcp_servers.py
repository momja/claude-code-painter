"""The paint server and the workbench, driven over stdio the way Claude Code drives them."""

import json
import subprocess
import sys
from pathlib import Path

from conveyor.painting.seeds import PEN
from conveyor.painting.seeds import ROUND


class Client:
    def __init__(self, module: str, session_dir: Path) -> None:
        self.proc = subprocess.Popen([sys.executable, "-m", module, str(session_dir)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        self.n = 0
        init = self.rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})
        assert init["result"]["capabilities"]["tools"] is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")

    def rpc(self, method: str, params: dict | None = None) -> dict:
        self.n += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def call(self, name: str, arguments: dict, tool_use_id: str = "t") -> dict:
        return self.rpc("tools/call", {"name": name, "arguments": arguments, "_meta": {"claudecode/toolUseId": tool_use_id}})["result"]

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def test_paint_server(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({"source": PEN, "target": "self_portrait", "width": 64, "actions": 3,
                                                   "looks": 1, "snapshot_every": 1}))
    c = Client("conveyor.painting.paintserver", tmp_path)
    names = [t["name"] for t in c.rpc("tools/list")["result"]["tools"]]
    assert names == ["start", "move", "stop", "detail", "paint_batch", "look", "finish"]

    bad = c.call("move", {"dx": "far"})
    assert bad["isError"] and "no action was used" in bad["content"][0]["text"]
    r = c.call("start", {"x": 10, "y": 10, "color": "#223344"}, "a")
    assert not r["isError"] and "2 actions left" in r["content"][0]["text"]
    r = c.call("move", {"dx": 20, "dy": 0}, "b")
    text = r["content"][0]["text"]
    assert text.startswith("score 0.") and "pixel error" in text and "load" in text
    # The instrument's viewing tools are free: the window comes back gridded, and no action or look is spent.
    v = c.call("detail", {"x": 32, "y": 32, "span": 16}, "v")
    assert not v["isError"] and [b["type"] for b in v["content"]] == ["text", "image"]
    vtext = v["content"][0]["text"]
    assert "Free view: no action and no look used" in vtext and "1 actions left" in vtext
    assert "x 24-40, y 24-40" in vtext  # the window is named in canvas pixels
    bad = c.call("detail", {"span": "wide"})
    assert bad["isError"] and "No action was used" in bad["content"][0]["text"]
    look = c.call("look", {})
    assert [b["type"] for b in look["content"]] == ["text", "image"]
    assert "texture and palette match" in look["content"][0]["text"]
    assert "No looks left" in c.call("look", {})["content"][0]["text"]
    r = c.call("stop", {}, "c")
    assert "last action" in r["content"][0]["text"]
    assert c.call("move", {"dx": 1, "dy": 1})["isError"]
    assert "Finished" in c.call("finish", {"note": "needed a fill"})["content"][0]["text"]
    c.close()

    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [(x["tool"], x["status"]) for x in calls][:3] == [("move", "rejected"), ("start", "applied"), ("move", "applied")]
    assert [(x["tool"], x["status"]) for x in calls][3] == ("detail", "view")
    assert calls[3]["views"] == [[24, 24, 40, 40]] and "score_before" not in calls[3]  # a view changes nothing
    assert calls[1]["tool_use_id"] == "a" and '"down": true' in calls[1]["pen"]
    assert calls[2]["score_after"] != calls[2]["score_before"]
    assert json.loads((tmp_path / "finish.json").read_text())["note"] == "needed a fill"
    assert (tmp_path / "canvas.npy").exists() and (tmp_path / calls[1]["snapshot"]).exists()


def test_greedy_painter_writes_the_same_outputs(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({"source": ROUND, "target": "self_portrait", "width": 64, "actions": 20}))
    subprocess.run([sys.executable, "-m", "conveyor.painting.paintserver", str(tmp_path), "--greedy"], check=True)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert sum(1 for x in calls if x["tool"] == "stroke") == 20
    assert json.loads((tmp_path / "finish.json").read_text())["actions_used"] == 20


def test_workbench(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({"width": 64, "height": 80, "max_tries": 1,
                                                   "wanted_niche": "stateful/scalar/medium"}))
    c = Client("conveyor.painting.workbench", tmp_path)
    tried = c.call("try_instrument", {"source": PEN})
    assert "Valid. Niche: stateful/scalar" in tried["content"][0]["text"]
    assert tried["content"][1]["type"] == "image"
    assert "No tries left" in c.call("try_instrument", {"source": PEN})["content"][0]["text"]
    refused = c.call("submit_instrument", {"source": "import os", "summary": "x"})
    assert refused["isError"] and "Not valid" in refused["content"][0]["text"]
    assert not (tmp_path / "submitted.json").exists()
    ok = c.call("submit_instrument", {"source": PEN, "summary": "a pen"})
    assert "Accepted" in ok["content"][0]["text"]
    c.close()
    record = json.loads((tmp_path / "submitted.json").read_text())
    assert record["summary"] == "a pen" and record["traits"]["stateful"]
