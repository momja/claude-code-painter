import numpy as np
import pytest

from conveyor.painting.canvas import Canvas
from conveyor.painting.instrument import Instrument
from conveyor.painting.instrument import InstrumentError
from conveyor.painting.instrument import ToolError
from conveyor.painting.instrument import all_niches
from conveyor.painting.instrument import probe
from conveyor.painting.instrument import static_traits
from conveyor.painting.seeds import PEN
from conveyor.painting.seeds import ROUND

W, H = 128, 160

MINIMAL = '''
"""One dab."""
TOOLS = {"dot": {"doc": "A dot.", "params": {"x": {"type": "number", "min": 0, "max": "width"},
                                            "y": {"type": "number", "min": 0, "max": "height"},
                                            "color": {"type": "color", "default": "#000000"}}}}
EXAMPLES = [[["dot", {"x": 10, "y": 10}]]]
def dot(args, pen, canvas, rng):
    canvas.dab(args["x"], args["y"], 3, args["color"])
'''


def with_body(body: str) -> str:
    return MINIMAL.replace('    canvas.dab(args["x"], args["y"], 3, args["color"])\n', body)


def test_seeds_probe_into_different_niches():
    round_report, pen_report = probe(ROUND, W, H), probe(PEN, W, H)
    assert round_report["ok"] and pen_report["ok"], (round_report["errors"], pen_report["errors"])
    assert round_report["niche"] == "stateless/scalar/medium"
    assert pen_report["niche"] == "stateful/scalar/medium"
    assert round_report["sheet"].startswith(b"\x89PNG")
    assert len(all_niches()) == 12


@pytest.mark.parametrize("snippet, words", [
    ("import os\n", "top level"),
    ("def f(a, pen, canvas, rng):\n    import os\n", "Import"),
    ("x = open('f')\n", "must be a plain literal"),
    ("def f(a, pen, canvas, rng):\n    return a.__class__\n", "underscore"),
    ("def f(a, pen, canvas, rng):\n    print(1)\n", "print"),
    ("def f(a, pen, canvas, rng):\n    return np.load('x')\n", "np.load"),
    ("def f(a, pen, canvas, rng):\n    return '{0}'.format(a)\n", "format"),
    ("def f(a, pen, canvas, rng):\n    try:\n        pass\n    except Exception:\n        pass\n", "Try"),
    ("def f(a, pen, canvas, rng):\n    return canvas.img\n", "canvas.img"),
    ("class K:\n    pass\n", "top level"),
])
def test_sandbox_refuses(snippet, words):
    with pytest.raises(InstrumentError, match=words):
        Instrument(MINIMAL + snippet)


def test_spec_errors_are_specific():
    with pytest.raises(InstrumentError, match="has no function"):
        Instrument(MINIMAL.replace("def dot(", "def dab("))
    with pytest.raises(InstrumentError, match="taken by the harness"):
        Instrument(MINIMAL.replace('"dot": {', '"look": {').replace('["dot"', '["look"').replace("def dot(", "def look("))
    with pytest.raises(InstrumentError, match="type must be one of"):
        Instrument(MINIMAL.replace('"type": "color"', '"type": "colour"'))
    with pytest.raises(InstrumentError, match="EXAMPLES"):
        Instrument(MINIMAL.replace("EXAMPLES = [[[\"dot\", {\"x\": 10, \"y\": 10}]]]", ""))


def test_arguments_are_coerced_clamped_and_defaulted():
    inst = Instrument(MINIMAL)
    args = inst.coerce("dot", {"x": 999, "y": "12.5"}, W, H)
    assert args == {"x": W, "y": 12.5, "color": (0.0, 0.0, 0.0)}
    with pytest.raises(ToolError, match="needs `x`"):
        inst.coerce("dot", {"y": 1}, W, H)
    with pytest.raises(ToolError, match="takes no parameter"):
        inst.coerce("dot", {"x": 1, "y": 1, "angle": 3}, W, H)
    with pytest.raises(ToolError, match="as a color"):
        inst.coerce("dot", {"x": 1, "y": 1, "color": "blue-ish"}, W, H)


def test_list_parameters_and_the_mcp_schema():
    source = MINIMAL.replace('"color": {"type": "color", "default": "#000000"}',
                             '"color": {"type": "color", "default": "#000000"}, "path": {"type": "points", "max_items": 4}')
    source = source.replace('[["dot", {"x": 10, "y": 10}]]', '[["dot", {"x": 10, "y": 10, "path": [[1, 2], [3, 4]]}]]')
    inst = Instrument(source)
    schema = inst.mcp_tools(W, H)[0]["inputSchema"]
    assert schema["properties"]["path"]["maxItems"] == 4
    assert schema["required"] == ["x", "y", "path"]
    assert schema["properties"]["x"]["maximum"] == W
    with pytest.raises(ToolError, match="takes 1 to 4 items"):
        inst.coerce("dot", {"x": 1, "y": 1, "path": [[0, 0]] * 5}, W, H)
    assert static_traits(inst)["list_params"]


def test_state_persists_across_calls_and_is_detected():
    inst = Instrument(PEN)
    canvas, pen, rng = Canvas(H, W), inst.new_state(), np.random.default_rng(0)
    assert "brush is up" in inst.call("move", {"dx": 10, "dy": 0}, pen, canvas, rng)
    inst.call("start", {"x": 10, "y": 10, "color": "#ff0000"}, pen, canvas, rng)
    inst.call("move", {"dx": 20, "dy": 5}, pen, canvas, rng)
    assert (pen["x"], pen["y"]) == (30, 15) and pen["load"] < 1.0
    assert static_traits(inst)["stateful"]
    # A write through a helper is invisible to the static check, but the probe sees the pen change.
    hidden = with_body('    bump(pen)\n    canvas.dab(args["x"], args["y"], 3, args["color"])\n'
                       'def bump(p):\n    p["n"] = p.get("n", 0) + 1\n')
    assert not static_traits(Instrument(hidden))["stateful"]
    assert probe(hidden, W, H)["traits"]["stateful"]


def test_a_runaway_call_is_stopped():
    inst = Instrument(with_body("    while True:\n        pass\n"))
    with pytest.raises(ToolError, match="limit"):
        inst.call("dot", {"x": 1, "y": 1}, {}, Canvas(H, W), np.random.default_rng(0), timeout=0.2)


def test_probe_reports_examples_that_fail_or_lay_nothing():
    broken = probe(with_body("    return 1 / 0\n"), W, H)
    assert not broken["ok"] and "ZeroDivisionError" in broken["errors"][0]
    empty = probe(with_body("    return 'nothing'\n"), W, H)
    assert not empty["ok"] and "no example laid any paint" in empty["errors"][0]


def test_numpy_lazy_imports_work_inside_the_sandbox():
    """A fresh process whose first ndarray.max() runs inside an instrument. It used to raise KeyError: '__import__'."""
    import subprocess
    import sys

    source = with_body("    if np.zeros(3).max() >= 0:\n        canvas.dab(args['x'], args['y'], 3, args['color'])\n")
    code = ("import numpy as np, sys\n"
            "from conveyor.painting.instrument import Instrument\n"
            "from conveyor.painting.canvas import Canvas\n"
            "inst = Instrument(sys.stdin.read())\n"
            "inst.call('dot', {'x': 5, 'y': 5}, {}, Canvas(32, 32), np.random.default_rng(0))\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], input=source, capture_output=True, text=True)
    assert r.stdout.strip() == "ok", r.stderr[-500:]
