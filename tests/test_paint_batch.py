"""Batch calls preserve the paint physics, ordered state, failures and per-action recording."""

import json

import numpy as np
import pytest

from conveyor.mcp import ToolFailure
from conveyor.painting.instrument import Instrument, InstrumentError
from conveyor.painting.paintserver import (
    MAX_BATCH_CALLS,
    MAX_REJECTS,
    PaintServer,
    PaintSession,
)
from conveyor.painting.seeds import PEN, ROUND


def session(path, **job):
    path.mkdir(exist_ok=True)
    base = {"source": ROUND, "target": "self_portrait", "width": 64, "actions": 10,
            "looks": 2, "snapshot_every": 1, "seed": 42}
    (path / "job.json").write_text(json.dumps({**base, **job}))
    return PaintSession(path)


def entries(s):
    return [json.loads(line) for line in (s.dir / "calls.jsonl").read_text().splitlines()]


def batch_defaults():
    return {"angle": 0, "length": 30, "size": 6, "color": "#6f8fb5"}


def test_shared_arguments_and_overrides_match_individual_calls(tmp_path):
    batched = session(tmp_path / "batch")
    direct = session(tmp_path / "direct")
    defaults = batch_defaults()
    calls = [{"x": 10, "y": 10}, {"x": 20, "y": 20, "color": "#223344"}, {"x": 30, "y": 30}]
    result = batched.batch({"tool": "stroke", "defaults": defaults, "calls": calls,
                            "plan": "Background done; face next."}, "batch-1")
    for args in calls:
        direct.apply("stroke", {**defaults, **args})
    assert np.array_equal(batched.canvas.img, direct.canvas.img)
    assert batched.pen == direct.pen and batched.actions_left == direct.actions_left == 7
    assert result.startswith("Applied 3/3 calls") and "State:" in result
    state = json.loads(result.split("State: ")[1])
    assert state["plan"] == "Background done; face next." and state["actions_used"] == 3
    log = entries(batched)
    assert [e["batch_call"] for e in log] == [1, 2, 3]
    assert all(e["tool_use_id"] == "batch-1" for e in log)
    assert all(e["area"] <= batched.canvas.area_cap for e in log)
    assert all((batched.dir / e["snapshot"]).exists() for e in log)
    assert log[1]["args"]["color"] == "#223344"
    assert [e["score_after"] for e in log] == [e["score_after"] for e in entries(direct)]


def test_mixed_stateful_tools_scope_and_defaults(tmp_path):
    s = session(tmp_path, source=PEN, scope=True)
    s.scope({"x": 32, "y": 32, "span": 16})
    result = s.batch({"defaults": {"color": "#223344"}, "calls": [
        ["start", {"x": 8, "y": 8, "size": 4}], ["move", {"dx": 4, "dy": -4}], ["stop", {}],
    ]}, "b")
    assert "Applied 3/3" in result and s.actions_left == 7
    assert (s.pen["x"], s.pen["y"]) == (36, 28) and not s.pen["down"]
    assert s.working_state()["scope"] == [24, 24, 40, 40]
    assert [e["tool"] for e in entries(s)] == ["scope", "start", "move", "stop"]
    assert entries(s)[2]["args"] == {"dx": 4, "dy": -4}  # colour is not a move parameter


@pytest.mark.parametrize("args", [
    {"tool": "stroke", "calls": []},
    {"tool": "stroke", "calls": [{}] * (MAX_BATCH_CALLS + 1)},
    {"tool": "stroke", "calls": [{}, ["look", {}]]},
    {"calls": [["finish", {}]]},
    {"calls": [["paint_batch", {}]]},
    {"calls": [["scope", {}]]},
    {"calls": [["detail", {}]]},
    {"calls": [{"x": 1}]},
    {"tool": "stroke", "calls": [None]},
    {"tool": "stroke", "calls": [{}], "defaults": {"typo": 2}},
    {"tool": "stroke", "calls": [{}], "defaults": []},
    {"tool": "stroke", "calls": [{}], "plan": "x" * 1201},
])
def test_invalid_envelope_cannot_partially_paint(tmp_path, args):
    s = session(tmp_path)
    before = s.canvas.img.copy()
    response = PaintServer(s).handle({"id": 1, "method": "tools/call",
                                    "params": {"name": "paint_batch", "arguments": args,
                                               "_meta": {"conveyor/paintingState": True}}})["result"]
    assert response["isError"] and "Nothing was painted" in response["content"][0]["text"]
    assert s.actions_left == 10 and np.array_equal(s.canvas.img, before)
    assert [(e["tool"], e["status"]) for e in entries(s)] == [("paint_batch", "rejected")]
    assert response["structuredContent"]["painting_state"]["actions_used"] == 0


def test_first_failure_keeps_earlier_calls_and_skips_the_rest(tmp_path):
    s = session(tmp_path)
    server = PaintServer(s)
    response = server.handle({"id": 1, "method": "tools/call", "params": {
        "name": "paint_batch", "_meta": {"claudecode/toolUseId": "b", "conveyor/paintingState": True}, "arguments": {
            "tool": "stroke", "defaults": batch_defaults(),
            "calls": [{"x": 10, "y": 10}, {"x": "bad", "y": 20}, {"x": 30, "y": 30}],
            "plan": "One background stroke done.",
        }}})["result"]
    assert response["isError"]
    assert "Call 2 (stroke) failed" in response["content"][0]["text"]
    assert "Applied 1/3" in response["content"][0]["text"] and "Skipped 1 calls" in response["content"][0]["text"]
    assert [(e["status"], e["batch_call"]) for e in entries(s)] == [("applied", 1), ("rejected", 2)]
    state = response["structuredContent"]["painting_state"]
    assert state["actions_left"] == 9 and state["plan"] == "One background stroke done."
    # A later successful batch can recover. Shared arguments do not leak between batches.
    s.batch({"calls": [["stroke", {**batch_defaults(), "x": 30, "y": 30}]]})
    assert s.actions_left == 8 and s.rejected == 1
    s.apply("stroke", {**batch_defaults(), "x": 40, "y": 40})
    assert "batch_call" not in entries(s)[-1]


def test_tool_exception_rolls_back_only_the_failed_action(tmp_path):
    source = '''
TOOLS = {"good": {"params": {}}, "bad": {"params": {}}}
EXAMPLES = [[["good", {}]]]
def good(args, pen, canvas, rng):
    pen["count"] = pen.get("count", 0) + 1
    canvas.dab(10, 10, 2, "#223344")
def bad(args, pen, canvas, rng):
    pen["count"] = 999
    canvas.dab(40, 40, 2, "#223344")
    return 1 / 0
'''
    s = session(tmp_path / "batch", source=source)
    direct = session(tmp_path / "direct", source=source)
    direct.apply("good", {})
    response = PaintServer(s).handle({"id": 1, "method": "tools/call", "params": {
        "name": "paint_batch", "arguments": {"calls": [["good", {}], ["bad", {}], ["good", {}]]}}})["result"]
    assert response["isError"] and s.actions_left == 9 and s.pen == {"count": 1}
    assert np.array_equal(s.canvas.img, direct.canvas.img)


def test_batch_stops_at_action_budget_and_can_finish(tmp_path):
    s = session(tmp_path, actions=2)
    result = s.batch({"tool": "stroke", "defaults": batch_defaults(),
                      "calls": [{"x": x, "y": 10} for x in [10, 20, 30, 40]]})
    assert "Applied 2/4" in result and "Skipped 2 calls" in result and "Call finish" in result
    assert s.applied == 2 and len(entries(s)) == 2
    s.finish("ok")
    assert json.loads((tmp_path / "finish.json").read_text())["actions_used"] == 2


def test_invalid_batches_share_the_refusal_limit(tmp_path):
    s = session(tmp_path)
    for _ in range(MAX_REJECTS - 1):
        with pytest.raises(ToolFailure):
            s.batch({"calls": []})
    with pytest.raises(ToolFailure, match="Too many refused calls"):
        s.batch({"calls": []})
    assert s.finished and s.rejected == MAX_REJECTS and s.actions_left == 10


def test_batch_can_be_disabled_and_name_is_reserved(tmp_path):
    server = PaintServer(session(tmp_path, paint_batch=False))
    assert "paint_batch" not in {t["name"] for t in server.tools()}
    result = server.handle({"id": 1, "method": "tools/call", "params": {"name": "paint_batch"}})["result"]
    assert result["isError"]
    with pytest.raises(InstrumentError, match="taken by the harness"):
        Instrument(ROUND.replace("stroke", "paint_batch"))
