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


VIEW_SRC = MINIMAL + '''

VIEWS = {"detail": {"doc": "A window on the canvas.",
                   "params": {"x": {"type": "number", "min": 0, "max": "width"},
                              "y": {"type": "number", "min": 0, "max": "height"},
                              "span": {"type": "number", "min": 8, "max": "width", "default": 32}}}}


def detail(args, pen, canvas, rng):
    return canvas.view(args["x"], args["y"], args["span"]), "look here"
'''


def view_module(name: str = "detail", body: str = "    return canvas.view(1, 1, 32)",
                def_line: str | None = None) -> str:
    sig = def_line or f"def {name}(args, pen, canvas, rng):"
    return MINIMAL + f'''\n\nVIEWS = {{"{name}": {{"doc": "A window.", "params": {{}}}}}}\n\n\n{sig}\n{body}\n'''


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


def test_view_spec_errors_are_specific():
    with pytest.raises(InstrumentError, match="already a tool name"):
        Instrument(view_module("dot"))
    with pytest.raises(InstrumentError, match="taken by the harness"):
        Instrument(view_module("look"))
    with pytest.raises(InstrumentError, match="has no function"):
        Instrument(view_module("peek", def_line="def detail(args, pen, canvas, rng):"))
    with pytest.raises(InstrumentError, match=r"must take \(args, pen, canvas, rng\)"):
        Instrument(view_module("detail", def_line="def detail(a, pen, canvas, rng):"))
    with pytest.raises(InstrumentError, match="example"):
        Instrument(VIEW_SRC.replace('[["dot", {"x": 10, "y": 10}]]', '[["detail", {"x": 1, "y": 1}]]'))
    five = "\n".join(f'def v{i}(args, pen, canvas, rng):\n    return canvas.view(1, 1, 32)' for i in range(5))
    tools = " ".join('"v%d": {"doc": "w", "params": {}},' % i for i in range(5))
    with pytest.raises(InstrumentError, match="viewing tools, limit"):
        Instrument(MINIMAL + f'\n\nVIEWS = {{{tools}}}\n\n{five}\n')


def test_views_return_views_and_never_change_anything():
    inst = Instrument(VIEW_SRC)
    assert [v.name for v in inst.spec.views] == ["detail"] and static_traits(inst)["n_views"] == 1
    canvas, pen = Canvas(H, W), inst.new_state()
    before = canvas.snapshot()
    views, note = inst.view("detail", {"x": 900, "y": 10, "span": 40}, pen, canvas, np.random.default_rng(0))
    assert note == "look here" and [v.rect for v in views] == [(W - 40, 0, W, 40)]  # clamped into the canvas
    assert np.array_equal(canvas.img, before) and pen == inst.new_state()
    assert inst.coerce("detail", {"x": 5, "y": 5}, W, H)["span"] == 32  # views coerce like any other tool


def test_a_view_that_draws_is_put_back_and_one_that_returns_nothing_is_refused():
    body = '    pen["seen"] = True\n    canvas.dab(args["x"], args["y"], 5, "#000000")\n    return canvas.view(args["x"], args["y"], args["span"])\n'
    src = VIEW_SRC.replace('    return canvas.view(args["x"], args["y"], args["span"]), "look here"\n', body)
    inst = Instrument(src)
    canvas, pen = Canvas(H, W), inst.new_state()
    before = canvas.snapshot()
    views, _ = inst.view("detail", {"x": 10, "y": 10}, pen, canvas, np.random.default_rng(0))
    assert views and np.array_equal(canvas.img, before) and pen == inst.new_state()
    bad = Instrument(VIEW_SRC.replace('    return canvas.view(args["x"], args["y"], args["span"]), "look here"',
                                      '    return "no view here"'))
    with pytest.raises(ToolError, match="must return canvas.view"):
        bad.view("detail", {"x": 1, "y": 1}, {}, Canvas(H, W), np.random.default_rng(0))
    many = Instrument(VIEW_SRC.replace('    return canvas.view(args["x"], args["y"], args["span"]), "look here"',
                                       '    return [canvas.view(1, 1, 8) for i in range(5)]'))
    with pytest.raises(ToolError, match="limit 4"):
        many.view("detail", {"x": 1, "y": 1}, {}, Canvas(H, W), np.random.default_rng(0))


def test_the_probe_calls_viewing_tools_and_can_refuse_them():
    report = probe(VIEW_SRC, W, H)
    assert report["ok"] and report["traits"]["n_views"] == 1
    assert report["views"] == [{"name": "detail", "params": ["x", "y", "span"]}]
    broken = probe(view_module("detail", body="    return 1 / 0"), W, H)
    assert not broken["ok"] and "view detail failed its probe call" in broken["errors"][0]


def test_the_reference_shows_viewing_tools_apart():
    ref = Instrument(VIEW_SRC).reference(W, H)
    assert "Viewing tools (free" in ref and "- detail: A window on the canvas." in ref
    assert "span: number 8 to 128 (default 32)" in ref


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


def test_bounds_can_follow_the_canvas():
    from conveyor.painting.canvas import Canvas

    inst = Instrument(PEN)  # `size` is bounded by "radius", `dx` by "-width".."width"
    size = inst.mcp_tools(512, 640)[0]["inputSchema"]["properties"]["size"]
    assert size["maximum"] == Canvas(640, 512).max_radius
    assert inst.coerce("move", {"dx": -9999, "dy": 0}, 512, 640)["dx"] == -512
    with pytest.raises(InstrumentError, match="must be a number"):
        Instrument(MINIMAL.replace('"max": "width"}', '"max": "wide"}'))


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
