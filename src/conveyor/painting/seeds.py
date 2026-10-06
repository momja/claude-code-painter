"""
Seed instruments.

`ROUND` is the default seed and deliberately plain: one stateless tool that drags a round brush in a straight
line, the same mark and nearly the same call the old toolkit offered. A run that starts here and ends somewhere
else got there by evolution. `PEN` is a stateful pen plotter (start, move, stop), kept for tests and for runs
that want a second, structurally different starting point in the archive. Both carry one plain viewing tool, a
centred window at a chosen zoom, because a painter cannot place fine marks on a 512 px canvas from the
whole-canvas view alone; how the painter sees is part of the interface and evolves with the rest.
"""

ROUND = '''\
"""
Round brush. `stroke` drags a soft round brush in a straight line and lays translucent paint that layers over
what is already there. A stroke with length 0 is a single dab. `detail` shows a window on the canvas, for
placing marks finer than the whole-canvas view can show.
"""

TOOLS = {
    "stroke": {
        "doc": "Drag a round brush in a straight line from (x, y), heading `angle` degrees (0 points right, 90 down), for `length` px.",
        "params": {
            "x": {"type": "number", "min": 0, "max": "width", "doc": "Start x."},
            "y": {"type": "number", "min": 0, "max": "height", "doc": "Start y."},
            "angle": {"type": "number", "min": -360, "max": 360, "default": 0, "doc": "Direction in degrees."},
            "length": {"type": "number", "min": 0, "max": "width", "default": 48, "doc": "Path length in px."},
            "size": {"type": "number", "min": 0.5, "max": "radius", "default": 12, "doc": "Brush radius in px."},
            "color": {"type": "color"},
            "pressure": {"type": "number", "min": 0.05, "max": 1, "default": 0.6, "doc": "How opaque the stroke is."},
        },
    },
}

VIEWS = {
    "detail": {
        "doc": "Look closely at (x, y): a window `span` canvas px across, shown at `zoom` image px per canvas px.",
        "params": {
            "x": {"type": "number", "min": 0, "max": "width", "doc": "Centre x."},
            "y": {"type": "number", "min": 0, "max": "height", "doc": "Centre y."},
            "span": {"type": "number", "min": 8, "max": "width", "default": 64, "doc": "Window width in canvas px."},
            "zoom": {"type": "integer", "min": 1, "max": 8, "default": 4, "doc": "Image px per canvas px."},
        },
    },
}

EXAMPLES = [
    [["stroke", {"x": 48, "y": 80, "angle": 0, "length": 176, "size": 36, "color": "#4a6fa5", "pressure": 0.5}]],
    [["stroke", {"x": 56, "y": 240, "angle": 25, "length": 160, "size": 6, "color": "#1d2a3a", "pressure": 0.9}]],
    [["stroke", {"x": 160, "y": 440, "length": 0, "size": 24, "color": "#c8553d"}]],
]


def stroke(args, pen, canvas, rng):
    a = math.radians(args["angle"])
    r = args["size"]
    step = max(0.5, r * 0.35)
    n = int(args["length"] / step) + 1
    # Dabs overlap about 2r/step deep, so each lays less pigment and the stroke ends up near `pressure`.
    depth = max(1, min(n, int(2 * r / step) + 1))
    flow = 1.0 - (1.0 - args["pressure"]) ** (1.0 / depth)
    for i in range(n):
        d = i * step
        if not canvas.dab(args["x"] + math.cos(a) * d, args["y"] + math.sin(a) * d, r, args["color"], flow, 0.5):
            return "ran dry: the stroke was longer than one call can cover"


def detail(args, pen, canvas, rng):
    return canvas.view(args["x"], args["y"], args["span"], args["zoom"])
'''

PEN = '''\
"""
Pen plotter with a loaded brush. `start` puts the brush down and loads it with paint, `move` drags it in a
straight segment from wherever it is, and `stop` lifts it. Several moves in a row make a bent or curved stroke.
The brush runs out as it travels: each move lays a little less paint than the one before, until `start` reloads.
`detail` shows a window on the canvas, for placing marks finer than the whole-canvas view can show.
"""

STATE = {"down": False, "x": 0.0, "y": 0.0, "size": 3.0, "load": 0.0, "color": [0.0, 0.0, 0.0]}

TOOLS = {
    "start": {
        "doc": "Put the brush down at (x, y), loaded with `color`.",
        "params": {
            "x": {"type": "number", "min": 0, "max": "width"},
            "y": {"type": "number", "min": 0, "max": "height"},
            "color": {"type": "color"},
            "size": {"type": "number", "min": 0.5, "max": "radius", "default": 12, "doc": "Brush radius in px."},
        },
    },
    "move": {
        "doc": "Drag the brush by (dx, dy) px from where it is. Does nothing while the brush is up.",
        "params": {
            "dx": {"type": "number", "min": "-width", "max": "width"},
            "dy": {"type": "number", "min": "-height", "max": "height"},
            "pressure": {"type": "number", "min": 0.05, "max": 1, "default": 0.7},
        },
    },
    "stop": {"doc": "Lift the brush.", "params": {}},
}

VIEWS = {
    "detail": {
        "doc": "Look closely at (x, y): a window `span` canvas px across, shown at `zoom` image px per canvas px.",
        "params": {
            "x": {"type": "number", "min": 0, "max": "width", "doc": "Centre x."},
            "y": {"type": "number", "min": 0, "max": "height", "doc": "Centre y."},
            "span": {"type": "number", "min": 8, "max": "width", "default": 64, "doc": "Window width in canvas px."},
            "zoom": {"type": "integer", "min": 1, "max": 8, "default": 4, "doc": "Image px per canvas px."},
        },
    },
}

EXAMPLES = [
    [["start", {"x": 48, "y": 120, "color": "#2b4c7e", "size": 16}], ["move", {"dx": 80, "dy": -24}],
     ["move", {"dx": 80, "dy": 24}], ["move", {"dx": 80, "dy": 56}], ["stop", {}]],
    [["start", {"x": 80, "y": 360, "color": "#8c2f39", "size": 8}], ["move", {"dx": 120, "dy": 0}],
     ["move", {"dx": 0, "dy": 120}], ["move", {"dx": -120, "dy": 0}], ["stop", {}]],
]


def start(args, pen, canvas, rng):
    pen.update(down=True, x=args["x"], y=args["y"], size=args["size"], load=1.0, color=list(args["color"]))
    return f"brush down at ({args['x']:.0f}, {args['y']:.0f}), fully loaded"


def move(args, pen, canvas, rng):
    if not pen["down"]:
        return "the brush is up; call start first"
    x0, y0 = pen["x"], pen["y"]
    length = math.hypot(args["dx"], args["dy"])
    r = pen["size"]
    step = max(0.5, r * 0.4)
    n = int(length / step) + 1
    for i in range(n):
        t = i / max(n - 1, 1)
        flow = 0.35 * args["pressure"] * pen["load"]
        if not canvas.dab(x0 + args["dx"] * t, y0 + args["dy"] * t, r, pen["color"], flow, 0.6):
            break
    pen["x"], pen["y"] = x0 + args["dx"], y0 + args["dy"]
    # The brush runs out over a distance proportional to the canvas, so it empties at the same rate at any size.
    pen["load"] = max(0.15, pen["load"] * (1.0 - length / (1.25 * canvas.width)))
    return f"at ({pen['x']:.0f}, {pen['y']:.0f}), load {pen['load']:.0%}"


def stop(args, pen, canvas, rng):
    pen["down"] = False
    return "brush up"


def detail(args, pen, canvas, rng):
    return canvas.view(args["x"], args["y"], args["span"], args["zoom"])
'''

SEEDS = {"round": ROUND, "pen": PEN}
