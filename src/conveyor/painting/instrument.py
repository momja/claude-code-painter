"""
Instruments: the toolkit organism.

An instrument is a small Python module that declares its own tools. The painter gets exactly these tools, so the
instrument decides the whole calling interface: how many tools there are, what each call takes, and whether
calls share state. That is the point of the design. When every brush took the same `x, y, angle, color,
pressure`, no mutation could change what a call looks like, only what one straight mark looks like.

The module contract:

    '''What the painter should know about this instrument. Shown to it in full.'''

    STATE = {"down": False}            # the pen's starting state, copied fresh for every painting

    TOOLS = {
        "start": {"doc": "Put the pen down at (x, y).",
                  "params": {"x": {"type": "number", "min": 0, "max": "width"},
                             "y": {"type": "number", "min": 0, "max": "height"},
                             "color": {"type": "color"}}},
        ...
    }

    EXAMPLES = [                        # call sequences, run on a blank canvas for the demo sheet
        [["start", {"x": 20, "y": 30, "color": "#223344"}], ["move", {"dx": 30, "dy": 4}], ["stop", {}]],
    ]

    def start(args, pen, canvas, rng):  # one function per tool, same name
        pen.update(x=args["x"], y=args["y"], color=args["color"], down=True)
        return "pen down"                # optional: a short note the painter sees after the call

    VIEWS = {                          # optional: 0 to 4 viewing tools, the painter's eyes on its own canvas
        "detail": {"doc": "Look closely at (x, y): a window `span` px across.",
                   "params": {"x": {"type": "number", "min": 0, "max": "width"},
                              "y": {"type": "number", "min": 0, "max": "height"},
                              "span": {"type": "number", "min": 16, "max": 256, "default": 64}}},
    }

    def detail(args, pen, canvas, rng):  # a view function returns canvas.view(...) and never paints
        return canvas.view(args["x"], args["y"], args["span"])

Parameter types: number, integer (with `min`/`max`, which may be "width", "height" or "radius" — the canvas's
brush limit — or their negatives), boolean, color (the tool
receives an (r, g, b) tuple in 0..1), choice (`options`), points (a list of [x, y], up to 64), numbers (a list of
numbers, up to 64). A parameter with a `default` is optional.

`pen` persists across every call of one painting. `canvas` is the only way to put paint down (see canvas.py),
and `canvas.view(x, y, span, scale=4)` is the only way to show the painter a piece of it. Tools never see the
target. A viewing call costs the painter no action and no look, and cannot change the painting: the canvas and
the pen are put back after every view call.

The painter may set a scope, a square window it then paints in with local coordinates. Scoped calls arrive
with their canvas positions already shifted into place, so tool code never sees the scope. For the shift to
treat a parameter as a position, bound it 0-based against the canvas (`max` of 'width'/'height' with no
negative `min`); a symmetric bound ('-width' to 'width') marks a delta and is left alone, as are sizes
('radius'), `points` are always positions. Keep positions 0-based and the tools paint scoped for free.

The sandbox is not a security boundary. It stops a mutator from wandering into the filesystem or the MCP
channel by accident: no imports, no classes, no try/raise, no underscored names, no print, and attributes only
from allowlists. Every tool call also runs under a wall-clock limit. Instrument code only ever runs in a
subprocess (the paint server, the workbench, the probe), never in the process that runs the evolution.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np

from conveyor.painting.canvas import Canvas
from conveyor.painting.canvas import CanvasError
from conveyor.painting.canvas import PAPER
from conveyor.painting.canvas import View
from conveyor.painting.canvas import brush_limit
from conveyor.painting.canvas import labelled_sheet
from conveyor.painting.canvas import parse_color

MAX_SOURCE = 12_000
MAX_TOOLS = 8
MAX_VIEWS = 4
MAX_PARAMS = 8
MAX_LIST_ITEMS = 64
MAX_EXAMPLES = 6
MAX_EXAMPLE_CALLS = 16
MAX_DOC = 1500
MAX_NOTE = 300
MAX_VIEWS_PER_CALL = 4  # pictures one viewing call may return; each costs the painter tokens
CALL_TIMEOUT = 1.0  # seconds of wall clock per tool call
RESERVED = {"look", "finish", "scope", "paint_batch"}
PARAM_TYPES = {"number", "integer", "boolean", "color", "choice", "points", "numbers"}
TOOL_ARGS = ("args", "pen", "canvas", "rng")

ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Expr, ast.Pass, ast.If, ast.IfExp, ast.For, ast.While, ast.Break, ast.Continue, ast.Delete,
    ast.comprehension, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.keyword, ast.Attribute, ast.Name,
    ast.Load, ast.Store, ast.Del, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Subscript,
    ast.Slice, ast.Starred, ast.NamedExpr, ast.JoinedStr, ast.FormattedValue,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
    ast.Invert, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.MatMult,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
)
# What a program may reach through a dot. The module roots get allowlists because numpy has plain-named doors
# out of any sandbox (`np.ctypeslib`, `np.load`). Anything else on the left of a dot is a local value (a dict,
# a list, an array, a number) and only gets the methods those need. `format` is missing on purpose:
# "{0.__class__}".format(x) walks attributes without an Attribute node the checker could see.
NP_ATTRS = {
    "abs", "absolute", "all", "any", "arange", "arccos", "arcsin", "arctan", "arctan2", "argmax", "argmin",
    "array", "asarray", "cbrt", "ceil", "clip", "concatenate", "convolve", "cos", "cosh", "cumsum", "deg2rad",
    "degrees", "diff", "dot", "e", "exp", "expm1", "float32", "float64", "fliplr", "flipud", "floor", "full",
    "full_like", "hypot", "int32", "int64", "interp", "isfinite", "linspace", "log", "log1p", "maximum", "mean",
    "meshgrid", "mgrid", "minimum", "mod", "nan_to_num", "ogrid", "ones", "ones_like", "outer", "pad", "pi",
    "polyval", "power", "rad2deg", "radians", "rot90", "round", "sign", "sin", "sinh", "sort", "sqrt", "square",
    "stack", "std", "sum", "tan", "tanh", "where", "zeros", "zeros_like", "max", "min", "median",
}
RNG_ATTRS = {"choice", "integers", "normal", "permutation", "random", "standard_normal", "uniform"}
CANVAS_ATTRS = {"dab", "stamp", "smudge", "pick", "view", "width", "height", "area_left", "max_radius"}
VALUE_ATTRS = {
    # dict
    "get", "keys", "values", "items", "update", "pop", "setdefault", "copy", "clear",
    # list
    "append", "extend", "insert", "index", "count", "reverse", "sort", "remove",
    # str
    "lower", "upper", "strip", "startswith", "endswith", "split", "join", "replace",
    # array and number
    "shape", "size", "ndim", "dtype", "astype", "reshape", "ravel", "flatten", "sum", "mean", "max", "min",
    "std", "clip", "round", "any", "all", "item", "tolist", "cumsum", "argmax", "argmin", "T", "real", "imag",
    "is_integer",
}
ROOT_ATTRS: dict[str, set[str] | None] = {"np": NP_ATTRS, "math": None, "rng": RNG_ATTRS, "canvas": CANVAS_ATTRS}
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map, "max": max,
    "min": min, "pow": pow, "range": range, "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip, "True": True, "False": False, "None": None,
}


def _numpy_internal_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    numpy's C code imports some of its own modules lazily, on first use, through the builtins of whatever frame
    is running, which inside an instrument is the sandbox's. `ndarray.max()` does this in numpy 2: the first
    `.max()` in a process imports `numpy._core._methods`. Without an `__import__` here that raised KeyError on
    every call, and a painter lost a whole tool to it. Instrument code itself still can't import anything:
    the checker refuses import statements and every name that starts with an underscore.
    """
    if level == 0 and name.split(".")[0] == "numpy":
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"instrument code can't import {name}")


WRITES_STATE = {"update", "pop", "setdefault", "clear"}
READS_CANVAS = {"pick", "smudge"}


class InstrumentError(ValueError):
    """The source is not a valid instrument."""


class ToolError(RuntimeError):
    """A tool call failed: bad arguments, an exception in the tool, or the time limit."""


class ToolTimeout(ToolError):
    pass


# ---- the spec -----------------------------------------------------------------------------------------------


@dataclass
class ParamSpec:
    name: str
    type: str
    doc: str = ""
    min: float | str | None = None
    max: float | str | None = None
    default: Any = None
    has_default: bool = False
    options: list[str] = field(default_factory=list)
    min_items: int = 1
    max_items: int = MAX_LIST_ITEMS

    def bound(self, value, width: int, height: int) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            # Bounds may be written against the canvas, so one instrument reads right at any size: "width",
            # "height", and "radius", the canvas's brush limit. "-width" and friends give the mirror bound,
            # for deltas.
            neg = value.startswith("-")
            size = float({"width": width, "height": height, "radius": brush_limit(width, height)}[value.lstrip("-")])
            return -size if neg else size
        return float(value)

    def json_schema(self, width: int, height: int) -> dict:
        lo, hi = self.bound(self.min, width, height), self.bound(self.max, width, height)
        doc = self.doc
        if self.type in ("number", "integer"):
            s: dict = {"type": self.type}
            if lo is not None:
                s["minimum"] = lo if self.type == "number" else int(math.ceil(lo))
            if hi is not None:
                s["maximum"] = hi if self.type == "number" else int(math.floor(hi))
        elif self.type == "boolean":
            s = {"type": "boolean"}
        elif self.type == "color":
            s = {"type": "string"}
            doc = (doc + " " if doc else "") + "Hex colour, for example #8a6d4f."
        elif self.type == "choice":
            s = {"type": "string", "enum": list(self.options)}
        elif self.type == "points":
            s = {"type": "array", "minItems": self.min_items, "maxItems": self.max_items,
                 "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}
            doc = (doc + " " if doc else "") + "A list of [x, y] pairs in canvas pixels."
        else:  # numbers
            s = {"type": "array", "minItems": self.min_items, "maxItems": self.max_items, "items": {"type": "number"}}
        if self.has_default:
            doc = (doc + " " if doc else "") + f"Default {json.dumps(self.default)}."
        if doc:
            s["description"] = doc.strip()
        return s


@dataclass
class ToolSpec:
    name: str
    doc: str
    params: list[ParamSpec]

    def json_schema(self, width: int, height: int) -> dict:
        return {
            "type": "object",
            "properties": {p.name: p.json_schema(width, height) for p in self.params},
            "required": [p.name for p in self.params if not p.has_default],
            "additionalProperties": False,
        }


@dataclass
class Spec:
    doc: str
    tools: list[ToolSpec]
    views: list[ToolSpec]
    state: dict
    examples: list[list[tuple[str, dict]]]

    def tool(self, name: str) -> ToolSpec:
        for t in [*self.tools, *self.views]:
            if t.name == name:
                return t
        raise ToolError(f"no tool named {name!r}")


# ---- compiling ----------------------------------------------------------------------------------------------


def _attr_root(node: ast.AST) -> str:
    while True:
        if isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return node.id if isinstance(node, ast.Name) else ""


def _check_code(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            line = getattr(node, "lineno", "?")
            raise InstrumentError(f"line {line}: {type(node).__name__} is not allowed in instrument code")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise InstrumentError(f"line {node.lineno}: decorators are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise InstrumentError(f"line {node.lineno}: attribute {node.attr!r} starts with an underscore")
            root = _attr_root(node.value)
            allowed = ROOT_ATTRS.get(root, VALUE_ATTRS)
            if allowed is not None and node.attr not in allowed:
                where = root or "an expression"
                raise InstrumentError(f"line {node.lineno}: {where}.{node.attr} is not allowed in instrument code")
        name = getattr(node, "id", None) or getattr(node, "arg", None)
        if isinstance(name, str) and name.startswith("_"):
            raise InstrumentError(f"line {getattr(node, 'lineno', '?')}: name {name!r} starts with an underscore")
        if isinstance(node, ast.Name) and node.id in ("print", "eval", "exec", "open", "getattr", "globals", "vars"):
            raise InstrumentError(f"line {node.lineno}: {node.id} is not allowed in instrument code")


def _check_top_level(tree: ast.Module) -> None:
    for i, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.FunctionDef):
            continue
        if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue  # the module docstring
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                ast.literal_eval(stmt.value)
            except (ValueError, SyntaxError, TypeError) as e:
                raise InstrumentError(
                    f"line {stmt.lineno}: top-level {stmt.targets[0].id} must be a plain literal (numbers, strings, "
                    "lists, dicts), not computed") from e
            continue
        raise InstrumentError(f"line {stmt.lineno}: only a docstring, literal constants and functions may sit at "
                              "the top level of an instrument")


def _literal(tree: ast.Module, name: str, default: Any) -> Any:
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == name:
            return ast.literal_eval(stmt.value)
    return default


def _param(tool: str, name: str, raw: Any) -> ParamSpec:
    where = f"{tool}.{name}"
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
        raise InstrumentError(f"parameter {where}: names are lowercase letters, digits and underscores")
    if not isinstance(raw, dict):
        raise InstrumentError(f"parameter {where} must be a dict like {{'type': 'number'}}")
    kind = raw.get("type")
    if kind not in PARAM_TYPES:
        raise InstrumentError(f"parameter {where}: type must be one of {sorted(PARAM_TYPES)}, got {kind!r}")
    p = ParamSpec(name=name, type=kind, doc=str(raw.get("doc", ""))[:300])
    for k in ("min", "max"):
        v = raw.get(k)
        ok = isinstance(v, int | float) or (isinstance(v, str)
                                           and v.lstrip("-") in ("width", "height", "radius")
                                           and v.count("-") == v.startswith("-"))
        if v is not None and not ok:
            raise InstrumentError(f"parameter {where}: {k} must be a number, 'width', 'height' or 'radius' "
                                  f"(or their negatives)")
        setattr(p, k, v)
    if kind == "choice":
        opts = raw.get("options")
        if not isinstance(opts, list) or not opts or not all(isinstance(o, str) for o in opts):
            raise InstrumentError(f"parameter {where}: a choice needs `options`, a list of strings")
        p.options = opts[:32]
    if kind in ("points", "numbers"):
        p.min_items = int(raw.get("min_items", 1))
        p.max_items = int(min(MAX_LIST_ITEMS, raw.get("max_items", MAX_LIST_ITEMS)))
        if not 0 <= p.min_items <= p.max_items:
            raise InstrumentError(f"parameter {where}: need 0 <= min_items <= max_items <= {MAX_LIST_ITEMS}")
    if "default" in raw:
        p.default, p.has_default = raw["default"], True
    return p


def _tool_spec(kind: str, name: str, raw: Any, fns: dict[str, Callable]) -> ToolSpec:
    """One entry of TOOLS or VIEWS: name, docs, parameters, and the function behind it."""
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
        raise InstrumentError(f"tool name {name!r}: use lowercase letters, digits and underscores")
    if name in RESERVED:
        raise InstrumentError(f"tool name {name!r} is taken by the harness")
    if not isinstance(raw, dict):
        raise InstrumentError(f"{kind}[{name!r}] must be a dict with 'doc' and 'params'")
    params_raw = raw.get("params", {})
    if not isinstance(params_raw, dict):
        raise InstrumentError(f"{kind}[{name!r}]['params'] must be a dict")
    if len(params_raw) > MAX_PARAMS:
        raise InstrumentError(f"tool {name} has {len(params_raw)} parameters, limit {MAX_PARAMS}")
    fn = fns.get(name)
    if fn is None:
        raise InstrumentError(f"tool {name} has no function `def {name}(args, pen, canvas, rng)`")
    got = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    if tuple(got) != TOOL_ARGS:
        raise InstrumentError(f"`def {name}` must take (args, pen, canvas, rng), got ({', '.join(got)})")
    return ToolSpec(name=name, doc=str(raw.get("doc", ""))[:600],
                    params=[_param(name, k, v) for k, v in params_raw.items()])


def _spec(tree: ast.Module, fns: dict[str, Callable]) -> Spec:
    doc = ast.get_docstring(tree) or ""
    if len(doc) > MAX_DOC:
        raise InstrumentError(f"the docstring is {len(doc)} characters, limit {MAX_DOC}")
    tools_raw = _literal(tree, "TOOLS", None)
    if not isinstance(tools_raw, dict) or not tools_raw:
        raise InstrumentError("define TOOLS = {name: {'doc': ..., 'params': {...}}} with at least one tool")
    if len(tools_raw) > MAX_TOOLS:
        raise InstrumentError(f"{len(tools_raw)} tools, limit {MAX_TOOLS}")
    tools = [_tool_spec("TOOLS", name, raw, fns) for name, raw in tools_raw.items()]
    views_raw = _literal(tree, "VIEWS", {})
    if not isinstance(views_raw, dict):
        raise InstrumentError("VIEWS must be a dict of viewing tools, or left out")
    if len(views_raw) > MAX_VIEWS:
        raise InstrumentError(f"{len(views_raw)} viewing tools, limit {MAX_VIEWS}")
    for name in views_raw:
        if name in tools_raw:
            raise InstrumentError(f"viewing tool {name!r} is already a tool name")
    views = [_tool_spec("VIEWS", name, raw, fns) for name, raw in views_raw.items()]
    state = _literal(tree, "STATE", {})
    if not isinstance(state, dict):
        raise InstrumentError("STATE must be a dict")
    try:
        json.dumps(state)
    except TypeError as e:
        raise InstrumentError(f"STATE must be plain JSON data: {e}") from e
    examples_raw = _literal(tree, "EXAMPLES", None)
    if not isinstance(examples_raw, list) or not examples_raw:
        raise InstrumentError("define EXAMPLES: a list of call sequences like [[['tool', {...}], ...], ...]")
    examples = []
    for i, seq in enumerate(examples_raw[:MAX_EXAMPLES]):
        if not isinstance(seq, list | tuple) or not seq:
            raise InstrumentError(f"EXAMPLES[{i}] must be a non-empty list of [tool, args] calls")
        calls = []
        for j, call in enumerate(seq[:MAX_EXAMPLE_CALLS]):
            if not (isinstance(call, list | tuple) and len(call) == 2 and isinstance(call[0], str)
                    and isinstance(call[1], dict)):
                raise InstrumentError(f"EXAMPLES[{i}][{j}] must be [tool_name, {{args}}]")
            if call[0] in views_raw:
                raise InstrumentError(f"EXAMPLES[{i}][{j}] calls the viewing tool {call[0]!r}; examples show what "
                                      "the drawing tools draw")
            if call[0] not in tools_raw:
                raise InstrumentError(f"EXAMPLES[{i}][{j}] calls unknown tool {call[0]!r}")
            calls.append((call[0], dict(call[1])))
        examples.append(calls)
    return Spec(doc=doc, tools=tools, views=views, state=state, examples=examples)


def _tool_lines(t: ToolSpec, width: int, height: int) -> list[str]:
    lines = [f"- {t.name}: {t.doc}"]
    for p in t.params:
        lo, hi = p.bound(p.min, width, height), p.bound(p.max, width, height)
        rng_text = f" {lo:g} to {hi:g}" if lo is not None and hi is not None else ""
        opt = f" (default {json.dumps(p.default)})" if p.has_default else ""
        choices = f" one of {p.options}" if p.type == "choice" else ""
        lines.append(f"    {p.name}: {p.type}{rng_text}{choices}{opt}. {p.doc}".rstrip(". ") + ".")
    return lines


def _views_from(out: Any, tool: str) -> tuple[list[View], str]:
    """What a view function may return: a view, a list of views, or either of those with a note."""
    note = ""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], str):
        out, note = out
    if isinstance(out, View):
        views = [out]
    elif isinstance(out, list | tuple) and out and all(isinstance(v, View) for v in out):
        views = list(out)
    else:
        raise ToolError(f"{tool}: a view function must return canvas.view(...) or a list of views, got "
                        f"{type(out).__name__}")
    if len(views) > MAX_VIEWS_PER_CALL:
        raise ToolError(f"{tool}: {len(views)} views in one call, limit {MAX_VIEWS_PER_CALL}")
    return views, note[:MAX_NOTE]


def _scoped_args(spec: ToolSpec, raw_args: Any, scope: tuple[int, int, int, int]) -> Any:
    """
    Shift a scoped call's local coordinates into canvas coordinates. `scope` is the active window's
    (x0, y0, x1, y1); local (0, 0) is its top-left corner. Only canvas positions move: a number/integer
    whose range is 0-based against the canvas (`max` of 'width'/'height' with no negative `min`) is an
    absolute position and gains the window's origin, and `points` are positions by definition. Deltas
    (`min` of '-width'/'-height'), sizes ('radius'), angles, colours and everything else pass through
    unchanged, so instruments must keep positions 0-based for their tools to paint scoped. Anything the
    shift pushes past the canvas edge is clamped into range by coercion, like any other call.
    """
    if not isinstance(raw_args, dict):
        return raw_args
    x0, y0, _, _ = scope
    out = dict(raw_args)
    for p in spec.params:
        if p.name not in out or out[p.name] is None:
            continue
        value = out[p.name]
        if p.type in ("number", "integer") and p.max in ("width", "height") and p.min in (None, 0):
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            out[p.name] = value + (x0 if p.max == "width" else y0)
        elif p.type == "points" and isinstance(value, list | tuple):
            shifted = []
            for item in value:
                try:
                    x, y = (float(c) for c in item)
                except (TypeError, ValueError):
                    shifted.append(item)
                    continue
                shifted.append([x + x0, y + y0])
            out[p.name] = shifted
    return out


def synth_args(params: list[ParamSpec], width: int, height: int) -> dict:
    """Arguments no one chose: what the probe calls a viewing tool with, one in-range value per parameter."""
    out: dict[str, Any] = {}
    for p in params:
        if p.has_default:
            out[p.name] = p.default
        elif p.type in ("number", "integer"):
            lo, hi = p.bound(p.min, width, height), p.bound(p.max, width, height)
            out[p.name] = (0.0 if lo is None or hi is None else (lo + hi) / 2)
        elif p.type == "boolean":
            out[p.name] = False
        elif p.type == "color":
            out[p.name] = "#000000"
        elif p.type == "choice":
            out[p.name] = p.options[0]
        elif p.type == "points":
            out[p.name] = [[width / 2, height / 2]]
        else:
            out[p.name] = [0.0]
    return out


class Instrument:
    """A compiled instrument. Raises InstrumentError when the source isn't one."""

    def __init__(self, source: str) -> None:
        if len(source) > MAX_SOURCE:
            raise InstrumentError(f"the source is {len(source)} characters, limit {MAX_SOURCE}")
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise InstrumentError(f"syntax error on line {e.lineno}: {e.msg}") from e
        _check_top_level(tree)
        _check_code(tree)
        namespace: dict[str, Any] = {"__builtins__": {**SAFE_BUILTINS, "__import__": _numpy_internal_import},
                                     "np": np, "math": math}
        try:
            exec(compile(tree, filename="<instrument>", mode="exec"), namespace)  # noqa: S102 - checked above
        except Exception as e:  # noqa: BLE001 - any failure here is the instrument's
            raise InstrumentError(f"the module failed to load: {type(e).__name__}: {e}") from e
        fns = {k: v for k, v in namespace.items() if callable(v) and hasattr(v, "__code__")}
        self.source = source
        self.tree = tree
        self.spec = _spec(tree, fns)
        self.fns = fns

    def new_state(self) -> dict:
        return copy.deepcopy(self.spec.state)

    # ---- arguments ----------------------------------------------------------------------------------------

    def coerce(self, tool: str, raw: Any, width: int, height: int) -> dict:
        """Validate and fill in a call's arguments. Numbers are clamped into range rather than refused."""
        spec = self.spec.tool(tool)
        if not isinstance(raw, dict):
            raise ToolError("arguments must be an object")
        known = {p.name for p in spec.params}
        extra = sorted(set(raw) - known)
        if extra:
            raise ToolError(f"{tool} takes no parameter {', '.join(extra)}; it takes {', '.join(sorted(known)) or 'none'}")
        out: dict[str, Any] = {}
        for p in spec.params:
            if p.name in raw and raw[p.name] is not None:
                value = raw[p.name]
            elif p.has_default:
                value = p.default
            else:
                raise ToolError(f"{tool} needs `{p.name}`")
            out[p.name] = self._coerce_one(tool, p, value, width, height)
        return out

    def _coerce_one(self, tool: str, p: ParamSpec, value: Any, width: int, height: int) -> Any:
        where = f"{tool}.{p.name}"
        lo, hi = p.bound(p.min, width, height), p.bound(p.max, width, height)
        try:
            if p.type in ("number", "integer"):
                v = float(value)
                if not math.isfinite(v):
                    raise ValueError
                if lo is not None:
                    v = max(lo, v)
                if hi is not None:
                    v = min(hi, v)
                return int(round(v)) if p.type == "integer" else v
            if p.type == "boolean":
                if isinstance(value, str):
                    return value.strip().lower() in ("true", "1", "yes")
                return bool(value)
            if p.type == "color":
                return parse_color(value)
            if p.type == "choice":
                if value not in p.options:
                    raise ToolError(f"{where} must be one of {p.options}")
                return value
            items = list(value)
            if not p.min_items <= len(items) <= p.max_items:
                raise ToolError(f"{where} takes {p.min_items} to {p.max_items} items, got {len(items)}")
            if p.type == "points":
                pts = []
                for item in items:
                    x, y = (float(c) for c in item)
                    if not (math.isfinite(x) and math.isfinite(y)):
                        raise ValueError
                    pts.append((min(max(x, -width), 2 * width), min(max(y, -height), 2 * height)))
                return pts
            nums = []
            for item in items:
                v = float(item)
                if not math.isfinite(v):
                    raise ValueError
                if lo is not None:
                    v = max(lo, v)
                if hi is not None:
                    v = min(hi, v)
                nums.append(v)
            return nums
        except ToolError:
            raise
        except (TypeError, ValueError, CanvasError) as e:
            raise ToolError(f"{where}: can't read {json.dumps(value)[:60]} as a {p.type}") from e

    # ---- calling ------------------------------------------------------------------------------------------

    def call(self, tool: str, raw_args: Any, pen: dict, canvas: Canvas, rng: np.random.Generator,
             timeout: float = CALL_TIMEOUT, scope: tuple[int, int, int, int] | None = None) -> str:
        """Run one tool call. Returns the tool's note ('' if none). Raises ToolError."""
        if scope is not None:
            raw_args = _scoped_args(self.spec.tool(tool), raw_args, scope)
        args = self.coerce(tool, raw_args, canvas.width, canvas.height)
        canvas.begin_call()
        try:
            with deadline(timeout), np.errstate(all="ignore"):
                note = self.fns[tool](args, pen, canvas, rng)
        except ToolTimeout:
            raise
        except (CanvasError, ToolError) as e:
            raise ToolError(f"{tool}: {e}") from e
        except Exception as e:  # noqa: BLE001 - the tool is the instrument's code, not ours
            raise ToolError(f"{tool} raised {type(e).__name__}: {e}") from e
        return "" if note is None else str(note)[:MAX_NOTE]

    def view(self, tool: str, raw_args: Any, pen: dict, canvas: Canvas, rng: np.random.Generator,
             timeout: float = CALL_TIMEOUT) -> tuple[list[View], str]:
        """
        Run one viewing call and return the views it asked for, plus its note. A view call is read-only by
        construction: whatever it does, the canvas and the pen are put back before anyone sees the result.
        """
        args = self.coerce(tool, raw_args, canvas.width, canvas.height)
        before, pen_before = canvas.snapshot(), copy.deepcopy(pen)
        canvas.begin_call()
        try:
            with deadline(timeout), np.errstate(all="ignore"):
                out = self.fns[tool](args, pen, canvas, rng)
        except ToolTimeout:
            raise
        except (CanvasError, ToolError) as e:
            raise ToolError(f"{tool}: {e}") from e
        except Exception as e:  # noqa: BLE001 - the tool is the instrument's code, not ours
            raise ToolError(f"{tool} raised {type(e).__name__}: {e}") from e
        finally:
            canvas.restore(before)
            pen.clear()
            pen.update(pen_before)
        return _views_from(out, tool)

    def mcp_tools(self, width: int, height: int) -> list[dict]:
        return [{"name": t.name, "description": t.doc, "inputSchema": t.json_schema(width, height)}
                for t in self.spec.tools]

    def mcp_views(self, width: int, height: int) -> list[dict]:
        return [{"name": t.name, "description": t.doc, "inputSchema": t.json_schema(width, height)}
                for t in self.spec.views]

    def reference(self, width: int, height: int) -> str:
        """The instrument as the painter reads it: its docstring, then every tool and parameter."""
        lines = [self.spec.doc.strip() or "(no description)", ""]
        for t in self.spec.tools:
            lines += _tool_lines(t, width, height)
        if self.spec.views:
            lines += ["", "Viewing tools (free: no action, no look, and they never change the painting):"]
            for t in self.spec.views:
                lines += _tool_lines(t, width, height)
        return "\n".join(lines)


@contextmanager
def deadline(seconds: float):
    """Raise ToolTimeout if the block runs past `seconds`. Only enforceable on the main thread of a process."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    def fire(signum, frame):
        raise ToolTimeout(f"the call ran past its {seconds:g}s limit")

    old = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---- descriptors: where an instrument sits in the archive -----------------------------------------------------

REACH_BUCKETS = ((0.12, "short"), (0.30, "medium"), (math.inf, "long"))


def static_traits(inst: Instrument) -> dict:
    """What reading the code says: does any tool write the pen, read the canvas, or take a list?"""
    tool_names = {t.name for t in inst.spec.tools}
    writes_state = reads_canvas = False
    for fn in inst.tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Del) \
                    and isinstance(node.value, ast.Name) and node.value.id == "pen":
                writes_state = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "pen" and node.func.attr in WRITES_STATE:
                    writes_state = True
                if node.func.value.id == "canvas" and node.func.attr in READS_CANVAS:
                    reads_canvas = True
    list_params = any(p.type in ("points", "numbers") for t in inst.spec.tools for p in t.params)
    return {
        "n_tools": len(tool_names),
        "n_views": len(inst.spec.views),
        "max_params": max((len(t.params) for t in inst.spec.tools), default=0),
        "stateful": writes_state,
        "list_params": list_params,
        "reads_canvas": reads_canvas,
    }


def niche_of(traits: dict) -> str | None:
    if traits.get("reach") is None:
        return None
    reach = next(label for cut, label in REACH_BUCKETS if traits["reach"] < cut)
    return "/".join([
        "stateful" if traits["stateful"] else "stateless",
        "list" if traits["list_params"] else "scalar",
        reach,
    ])


def niche_distance(a: str | None, b: str) -> float:
    """
    How different two niches are, for aiming an invention. A change in how the tools are called (state, list
    arguments) counts double a change in how far a call reaches: the first is a change in kind, the second one of
    degree, and degree is what mutation finds without being asked.
    """
    if not a:
        return 1.0
    (sa, la, ra), (sb, lb, rb) = a.split("/"), b.split("/")
    return 2.0 * (sa != sb) + 2.0 * (la != lb) + 1.0 * (ra != rb)


def all_niches() -> list[str]:
    return [f"{s}/{a}/{r}" for s in ("stateless", "stateful") for a in ("scalar", "list") for _, r in REACH_BUCKETS]


# ---- probing: compile, run the examples, draw the demo sheet ---------------------------------------------------


def _footprint_reach(before: np.ndarray, after: np.ndarray) -> tuple[int, float]:
    changed = np.abs(after - before).max(axis=-1) > 0.01
    n = int(changed.sum())
    if not n:
        return 0, 0.0
    ys, xs = np.nonzero(changed)
    return n, float(math.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def _caption(i: int, seq: list[tuple[str, dict]]) -> str:
    names = [name for name, _ in seq]
    parts, prev, n = [], None, 0
    for name in names + [None]:
        if name == prev:
            n += 1
            continue
        if prev is not None:
            parts.append(prev if n == 1 else f"{prev} x{n}")
        prev, n = name, 1
    return f"{i + 1}: " + " > ".join(parts)


def probe(source: str, width: int, height: int, seed: int = 0) -> dict:
    """
    Everything the evolution needs to know about an instrument before anyone paints with it: whether it
    compiles, whether its examples run and lay paint, what each example draws (the demo sheet), and the
    traits that place it in a niche. Must run in a subprocess's main thread for the time limit to hold.
    """
    report: dict = {"ok": False, "errors": [], "warnings": [], "traits": {}, "niche": None, "sheet": None}
    try:
        inst = Instrument(source)
    except InstrumentError as e:
        report["errors"].append(str(e))
        return report
    traits = static_traits(inst)
    cells: list[tuple[str, np.ndarray]] = []
    reaches: list[float] = []
    painted_any = False
    for i, seq in enumerate(inst.spec.examples):
        canvas = Canvas(height, width)
        pen = inst.new_state()
        rng = np.random.default_rng(seed + i)
        failed = False
        for j, (tool, args) in enumerate(seq):
            before = canvas.snapshot()
            pen_before = json.dumps(pen, sort_keys=True, default=str)
            try:
                inst.call(tool, args, pen, canvas, rng)
            except ToolError as e:
                report["errors"].append(f"example {i + 1}, call {j + 1} ({tool}): {e}")
                failed = True
                break
            if json.dumps(pen, sort_keys=True, default=str) != pen_before:
                traits["stateful"] = True  # catches writes through a helper the static check can't follow
            area, reach = _footprint_reach(before, canvas.img)
            if area:
                painted_any = True
                reaches.append(reach / math.hypot(width, height))
            if canvas.dry:
                report["warnings"].append(f"example {i + 1}, call {j + 1} ({tool}) hit the per-call area limit "
                                          f"({canvas.area_cap} px) and ran dry")
        cells.append((_caption(i, seq) + ("  [failed]" if failed else ""), canvas.img))
    if not painted_any and not report["errors"]:
        report["errors"].append("no example laid any paint, so there is nothing to show the painter")
    traits["reach"] = float(np.median(reaches)) if reaches else None
    # Viewing tools get the same chance: called once with arbitrary in-range arguments, on a blank canvas, and
    # refused if they raise or return no view. A painter meets them cold, so a view that only works on some
    # arguments is a broken tool.
    for t in inst.spec.views:
        canvas = Canvas(height, width)
        pen = inst.new_state()
        try:
            views, _ = inst.view(t.name, synth_args(t.params, width, height), pen, canvas, np.random.default_rng(seed))
        except ToolError as e:
            report["errors"].append(f"view {t.name} failed its probe call: {e}")
            continue
        if not views:
            report["errors"].append(f"view {t.name} returned no view")
    report["traits"] = traits
    report["niche"] = niche_of(traits)
    report["sheet"] = labelled_sheet(cells)
    report["ok"] = not report["errors"]
    report["tools"] = [{"name": t.name, "params": [p.name for p in t.params]} for t in inst.spec.tools]
    report["views"] = [{"name": t.name, "params": [p.name for p in t.params]} for t in inst.spec.views]
    return report


def probe_cli(argv: list[str]) -> int:
    """`python -m conveyor.painting.instrument <source.py> <out_dir> <width> <height>`: probe in a subprocess."""
    src, out_dir, width, height = Path(argv[0]), Path(argv[1]), int(argv[2]), int(argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = probe(src.read_text(), width, height)
    sheet = report.pop("sheet")
    if sheet:
        (out_dir / "sheet.png").write_bytes(sheet)
    report["seconds"] = round(time.time() - started, 3)
    (out_dir / "probe.json").write_text(json.dumps(report))
    return 0


def blank_like(height: int, width: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(PAPER, dtype=np.float32), (height, width, 3)).copy()


if __name__ == "__main__":
    sys.exit(probe_cli(sys.argv[1:]))
