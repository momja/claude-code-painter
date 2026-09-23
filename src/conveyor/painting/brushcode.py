"""
Brushes as code. Each brush carries a small Python program that draws its own alpha mask, so a mutator can
invent a mark the parameter set never had: a dotted line, a rake of parallel bristles, a stroke that tapers.

The program defines one function:

    def alpha(u, v, rng, radius, length):
        # u: distance along the stroke path, 0 at the start point, `length` at the end
        # v: signed distance across the path
        # both are float32 arrays of the same shape, in canvas pixels
        return <array of the same shape, 0 to 1>

The caller owns everything outside the mark: where the stroke lands, its angle, the bounding box, clipping
to the canvas, and compositing. The program only says how much pigment falls at each offset from the path.

`compile_brush` is a sandbox, not a security boundary. It runs untrusted text in this process with no
imports, no attribute access to private names, no `while`, and nothing in scope but numpy, math and a
handful of builtins. That stops a mutator from wandering into the filesystem by accident. It would not stop
someone who set out to break it, so don't point this at brush code from a source you don't trust.
"""

from __future__ import annotations

import ast
import math
import random
import threading
from typing import Callable

import numpy as np

ENTRY = "alpha"
ENTRY_ARGS = ("u", "v", "rng", "radius", "length")
SIGNATURE = f"def {ENTRY}({', '.join(ENTRY_ARGS)})"
MAX_SOURCE = 4000  # characters; a brush is a few lines of numpy, not a library

# Nodes a brush program may use. Everything else is refused, which is why there is no `while` here: a brush
# is a pure function of its coordinate grids, and a loop that can't be bounded by reading it can hang a run.
ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign,
    ast.AnnAssign, ast.Expr, ast.Pass, ast.If, ast.IfExp, ast.For, ast.Break, ast.Continue,
    ast.comprehension, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.keyword, ast.Attribute,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Subscript, ast.Slice, ast.Starred, ast.NamedExpr,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Not, ast.Invert, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)
# What a program may reach through a dot. Underscored names are refused everywhere, which closes the usual
# `__class__` walk, but numpy also has plain-named doors out of the sandbox (`np.ctypeslib.ctypes`), so the
# module roots get an allowlist rather than a denylist. Anything else on the left of a dot is treated as an
# array, and only gets the handful of attributes array maths actually uses.
NP_ATTRS = {
    "abs", "arange", "arctan2", "asarray", "array", "ceil", "clip", "concatenate", "cos", "cosh", "exp",
    "expand_dims", "float32", "float64", "floor", "full", "full_like", "hypot", "isfinite", "linspace",
    "log", "log1p", "maximum", "meshgrid", "minimum", "mod", "nan_to_num", "ones", "ones_like", "pi",
    "power", "round", "sign", "sin", "sinh", "sqrt", "square", "stack", "tan", "tanh", "where", "zeros",
    "zeros_like",
}
MATH_ATTRS = {"atan2", "ceil", "cos", "e", "exp", "floor", "hypot", "log", "pi", "sin", "sqrt", "tan", "tau"}
RNG_ATTRS = {"integers", "normal", "random", "standard_normal", "uniform"}
ARRAY_ATTRS = {
    "all", "any", "astype", "clip", "copy", "dtype", "flatten", "item", "max", "mean", "min", "ndim",
    "ravel", "reshape", "round", "shape", "size", "std", "sum", "var",
}
ROOT_ATTRS = {"np": NP_ATTRS, "math": MATH_ATTRS, "rng": RNG_ATTRS}
SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round, "sum": sum, "len": len, "range": range,
    "float": float, "int": int, "bool": bool, "pow": pow, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "tuple": tuple, "list": list, "True": True, "False": False, "None": None,
}


class BrushCodeError(ValueError):
    """The source did not compile, or did not behave like a brush when probed."""


def _check(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise BrushCodeError(f"{type(node).__name__} is not allowed in brush code")
        if isinstance(node, ast.Attribute):
            root = node.value
            while True:
                if isinstance(root, ast.Attribute):
                    root = root.value
                elif isinstance(root, ast.Subscript):
                    root = root.value
                elif isinstance(root, ast.Call):
                    root = root.func
                else:
                    break
            base = root.id if isinstance(root, ast.Name) else ""
            allowed = ROOT_ATTRS.get(base, ARRAY_ATTRS)
            if node.attr not in allowed:
                where = base or "an expression"
                raise BrushCodeError(f"{where}.{node.attr} is not allowed in brush code")
        name = getattr(node, "id", None) or getattr(node, "attr", None) or getattr(node, "arg", None)
        if isinstance(name, str) and name.startswith("_"):
            raise BrushCodeError(f"name {name!r} starts with an underscore")


def _entry(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == ENTRY:
            args = node.args
            if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
                raise BrushCodeError(f"`{SIGNATURE}` takes no *args or **kwargs")
            got = ", ".join(a.arg for a in args.args)
            if tuple(a.arg for a in args.args) != ENTRY_ARGS:
                raise BrushCodeError(f"{ENTRY} must take ({', '.join(ENTRY_ARGS)}), got ({got})")
            return node
    raise BrushCodeError(f"brush code must define `{SIGNATURE}`")


_cache: dict[str, Callable] = {}
_lock = threading.Lock()


def compile_brush(source: str) -> Callable:
    """
    Compile brush source to its alpha function, or raise BrushCodeError.

    Cached on the source text itself, because a painting calls this once per stroke and a run re-renders the
    same handful of brushes hundreds of thousands of times. The key is the string rather than a digest of
    it: Python remembers a string's hash after the first lookup, so the cache hit costs nothing, while
    hashing the source each time showed up in a profile.
    """
    with _lock:
        hit = _cache.get(source)
    if hit is not None:
        return hit

    if len(source) > MAX_SOURCE:
        raise BrushCodeError(f"brush code is {len(source)} characters, limit is {MAX_SOURCE}")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise BrushCodeError(f"syntax error on line {e.lineno}: {e.msg}") from e
    _check(tree)
    _entry(tree)

    namespace = {"__builtins__": dict(SAFE_BUILTINS), "np": np, "math": math}
    try:
        exec(compile(tree, filename="<brush>", mode="exec"), namespace)  # noqa: S102 - sandboxed above
    except Exception as e:  # noqa: BLE001 - any failure here is the mutator's fault, not ours
        raise BrushCodeError(f"brush code failed to load: {type(e).__name__}: {e}") from e
    fn = namespace[ENTRY]
    probe(fn)
    with _lock:
        _cache[source] = fn
    return fn


def probe(fn: Callable, radius: float = 3.0, length: float = 4.0) -> np.ndarray:
    """
    Run the function once on a small grid and check it behaves like a brush. Catches the mistakes that would
    otherwise surface as a stack trace in the middle of a painting: wrong shape, NaN, negative alpha.
    """
    n = 9
    u, v = np.meshgrid(
        np.linspace(-radius, length + radius, n, dtype=np.float32),
        np.linspace(-radius, radius, n, dtype=np.float32),
    )
    rng = np.random.default_rng(0)
    try:
        with np.errstate(all="ignore"):
            out = fn(u, v, rng, float(radius), float(length))
    except Exception as e:  # noqa: BLE001 - the program is the thing under test
        raise BrushCodeError(f"{ENTRY} raised {type(e).__name__}: {e}") from e
    arr = np.asarray(out, dtype=np.float32)
    if arr.shape != u.shape:
        raise BrushCodeError(f"{ENTRY} returned shape {arr.shape}, expected {u.shape} (same shape as u)")
    if not np.isfinite(arr).all():
        raise BrushCodeError(f"{ENTRY} returned NaN or infinity")
    if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
        raise BrushCodeError(f"{ENTRY} returned alpha outside 0 to 1 (min {arr.min():.3f}, max {arr.max():.3f})")
    if arr.max() <= 0.01:
        raise BrushCodeError(f"{ENTRY} lays no pigment anywhere (max alpha {arr.max():.4f})")
    return arr


# --- Templates -------------------------------------------------------------------------------------------
#
# Source the scripted mutators write, so the no-API-key demo evolves brush code too. Each one is a readable
# function with its numbers inline, which is what `RandomToolkitMutator` jitters and what an LLM mutator
# reads as an example of the contract.

_CAPSULE = '''\
def alpha(u, v, rng, radius, length):
    """{doc}"""
    d = np.hypot(u - np.clip(u, 0.0, length), v)
    ramp = max(radius * {softness:.2f}, 0.5)
    core = np.clip((radius - d) / ramp + 0.5, 0.0, 1.0)
    wet = 4.0 * core * (1.0 - core)          # pigment pools at the wet edge of a stroke
    a = core * (1.0 - 0.6 * {bleed:.2f}) + 0.9 * {bleed:.2f} * wet
{grain}    return np.clip(a * {opacity:.2f}, 0.0, 1.0)
'''

_DOTTED = '''\
def alpha(u, v, rng, radius, length):
    """{doc}"""
    period = max(radius * {spacing:.2f}, 0.5)
    phase = np.abs(np.mod(u, period) - period * 0.5)   # distance to the nearest dot centre
    d = np.hypot(phase, v)
    ramp = max(radius * {softness:.2f}, 0.5)
    a = np.clip((max(radius * {dot_scale:.2f}, 0.5) - d) / ramp + 0.5, 0.0, 1.0)
    a = a * ((u > -radius) & (u < length + radius))     # dots only along the path, not past its ends
    return np.clip(a * {opacity:.2f}, 0.0, 1.0)
'''

_RAKE = '''\
def alpha(u, v, rng, radius, length):
    """{doc}"""
    gap = max({gap:.2f}, 2.0)                               # bristle spacing, in pixels
    half = max({width:.2f} * 0.5, 0.4)                      # half a bristle's width
    ridge = np.abs(np.mod(v + gap * 0.5, gap) - gap * 0.5)  # distance to the nearest bristle
    ramp = max(half, 0.5)                                   # a bristle thinner than a pixel still has to catch one
    a = np.clip((half - ridge) / ramp + 0.5, 0.0, 1.0)
    along = np.clip((radius - np.hypot(u - np.clip(u, 0.0, length), 0.0)) / max(radius, 0.5) + 0.5, 0.0, 1.0)
    a = a * along * (np.abs(v) < radius)                     # the rake is only as wide as the brush
{grain}    return np.clip(a * {opacity:.2f}, 0.0, 1.0)
'''

_TAPER = '''\
def alpha(u, v, rng, radius, length):
    """{doc}"""
    t = np.clip(u / max(length, 0.5), 0.0, 1.0)
    r = radius * (1.0 - {taper:.2f} * t)                 # the mark thins as the stroke is lifted
    d = np.hypot(u - np.clip(u, 0.0, length), v)
    ramp = np.maximum(r * {softness:.2f}, 0.5)
    a = np.clip((r - d) / ramp + 0.5, 0.0, 1.0)
    return np.clip(a * {opacity:.2f} * (1.0 - {fade:.2f} * t), 0.0, 1.0)
'''

TEMPLATES = {"capsule": _CAPSULE, "dotted": _DOTTED, "rake": _RAKE, "taper": _TAPER}

# Defaults for every knob each template takes, so a mutator can build one without naming them all and a
# param edit against an unfamiliar template still produces valid source.
DEFAULTS: dict[str, dict[str, float]] = {
    "capsule": {"softness": 0.5, "opacity": 0.6, "bleed": 0.0, "granulation": 0.0},
    "dotted": {"spacing": 2.2, "dot_scale": 0.55, "softness": 0.35, "opacity": 0.8},
    "rake": {"gap": 3.0, "width": 1.0, "granulation": 0.1, "opacity": 0.7},
    "taper": {"taper": 0.7, "fade": 0.4, "softness": 0.4, "opacity": 0.75},
}
DOCS = {
    "capsule": "A round head dragged along the path.",
    "dotted": "A broken line of separate dots.",
    "rake": "Parallel bristles, like a dry flat brush.",
    "taper": "Thins and fades from the start of the stroke to its end.",
}


def render_source(template: str, params: dict[str, float], doc: str = "") -> str:
    """Fill a template. Unknown knobs fall back to the template's defaults, so a partial edit still compiles."""
    if template not in TEMPLATES:
        raise BrushCodeError(f"unknown brush template {template!r}")
    filled = {**DEFAULTS[template], **{k: float(v) for k, v in params.items() if k in DEFAULTS[template]}}
    if "granulation" in filled:
        # Drawing the grain costs a random number per pixel, so a brush with no grain should not ask for one.
        grain = filled["granulation"]
        filled["grain"] = (
            f"    a = a * (1.0 - {grain:.2f} * rng.random(a.shape, dtype=np.float32))   # grainy pigment\n"
            if grain > 0.005 else ""
        )
    return TEMPLATES[template].format(doc=doc or DOCS[template], **filled)


def jitter_source(source: str, rng: random.Random, lo: float = 0.6, hi: float = 1.5) -> tuple[str, str] | None:
    """
    Scale one numeric literal in the source. Returns the new source and a note naming what moved, or None if
    the program has no number worth touching. This is how the random mutator gets a grip on code: the
    literals are the brush's knobs, whoever wrote them, so they stay mutable even for source an LLM sent.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    spots = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
        and abs(n.value) > 1e-9
        and n.end_lineno == n.lineno  # a literal split across lines has no simple column span to patch
    ]
    if not spots:
        return None
    pick = rng.choice(spots)
    old = float(pick.value)
    new = round(old * rng.uniform(lo, hi), 3)
    lines = source.splitlines()
    line = lines[pick.lineno - 1]
    lines[pick.lineno - 1] = line[: pick.col_offset] + f"{new:g}" + line[pick.end_col_offset :]
    return "\n".join(lines) + "\n", f"line {pick.lineno}, {old:g} to {new:g}"
