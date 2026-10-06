"""Painting in a scope: local coordinates map into the window, everything else is untouched."""

import json

import numpy as np

from conveyor.painting.canvas import Canvas
from conveyor.painting.canvas import PAPER
from conveyor.painting.paintserver import PaintServer
from conveyor.painting.paintserver import PaintSession
from conveyor.painting.instrument import Instrument
from conveyor.painting.seeds import PEN
from conveyor.painting.seeds import ROUND


def stroke_at(x, y, size=4):
    return {"x": x, "y": y, "angle": 0, "length": 0, "size": size,
            "color": "#223344", "pressure": 1.0}


def changed(canvas):
    return np.abs(canvas.img - PAPER).sum(axis=-1) > 1e-6


def test_positions_shift_and_sizes_do_not():
    inst = Instrument(ROUND)
    canvas = Canvas(64, 64)
    rng = np.random.default_rng(0)
    inst.call("stroke", stroke_at(8, 8), inst.new_state(), canvas, rng, scope=(24, 24, 40, 40))
    marks = changed(canvas)
    assert marks[32, 32] and not marks[8, 8]  # local (8, 8) landed on canvas (32, 32)


def test_deltas_do_not_shift():
    inst = Instrument(PEN)
    canvas = Canvas(64, 64)
    rng = np.random.default_rng(0)
    pen = inst.new_state()
    # start.x/y are positions and shift; move.dx/dy are deltas and do not.
    inst.call("start", {"x": 8, "y": 8, "color": "#223344", "size": 4},
              pen, canvas, rng, scope=(24, 24, 40, 40))
    assert (pen["x"], pen["y"]) == (32, 32)
    inst.call("move", {"dx": 4, "dy": -4}, pen, canvas, rng, scope=(24, 24, 40, 40))
    assert (pen["x"], pen["y"]) == (36, 28)


def test_shift_clamps_into_canvas():
    inst = Instrument(ROUND)
    canvas = Canvas(64, 64)
    rng = np.random.default_rng(0)
    # Local (60, 60) in a window ending at 40 would leave the canvas; coercion clamps it back in.
    inst.call("stroke", stroke_at(60, 60), inst.new_state(), canvas, rng, scope=(24, 24, 40, 40))
    marks = changed(canvas)
    assert marks[63, 63] and not marks[0, 0]  # clamped to the canvas corner, painted anyway


def session(tmp_path, **job):
    base = {"source": ROUND, "target": "self_portrait", "width": 64, "actions": 10,
            "looks": 2, "snapshot_every": 5}
    (tmp_path / "job.json").write_text(json.dumps({**base, **job}))
    return PaintSession(tmp_path)


def test_scope_tool_sets_clears_and_records(tmp_path):
    s = session(tmp_path, scope=True)
    assert s.scope_rect is None
    text, pngs = s.scope({"x": 32, "y": 32, "span": 16}, "s1")
    assert s.scope_rect == (24, 24, 40, 40)
    assert "local (0, 0) is canvas (24, 24)" in text and len(pngs) == 1
    # A scoped dab lands in the window and says which coordinates it used.
    out = s.apply("stroke", stroke_at(8, 8), "t1")
    assert "Scope is x 24-40, y 24-40" in out
    entries = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [e["status"] for e in entries] == ["scope", "applied"]
    assert entries[0]["scope"] == [24, 24, 40, 40] and entries[1]["scope"] == [24, 24, 40, 40]
    marks = changed(s.canvas)
    assert marks[32, 32] and not marks[8, 8]
    # Clearing returns to canvas pixels.
    text, pngs = s.scope({"clear": True}, "s2")
    assert s.scope_rect is None and pngs == [] and "cleared" in text
    s.apply("stroke", stroke_at(8, 8), "t2")
    assert changed(s.canvas)[8, 8]


def test_scope_disabled_by_default(tmp_path):
    s = session(tmp_path)
    assert s.scope_enabled is False
    assert "scope" not in [t["name"] for t in PaintServer(s).tools()]
    s2 = session(tmp_path, scope=True)
    assert "scope" in [t["name"] for t in PaintServer(s2).tools()]
