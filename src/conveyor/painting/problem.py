"""
The painting problem as two co-evolving nodes.

  instrument  the toolkit: an instrument module (see instrument.py). Archive node: parents are drawn across
              niches, and the Claude mutator has three operators, refine, invent (aimed at an empty niche) and
              recombine (across niches).
  painter     the painter's strategy prompt. Plain node: parents by rank, one Claude mutator.

A painting pairs one instrument with one prompt, and that one painting is both organisms' evaluation. Paintings
are cached by the pair, so the rescore that follows a new champion is usually a painting that already exists.
"""

from __future__ import annotations

import ast
import difflib
import json
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np

from conveyor.harness import Job
from conveyor.harness import ProcessHarness
from conveyor.harness import SessionFailed
from conveyor.evolve import Context
from conveyor.evolve import Evaluation
from conveyor.evolve import Mutator
from conveyor.evolve import Node
from conveyor.evolve import Organism
from conveyor.painting import prompts
from conveyor.painting.canvas import CALL_AREA_SHARE
from conveyor.painting.canvas import Canvas
from conveyor.painting.canvas import gridded_png
from conveyor.painting.canvas import heatmap_png
from conveyor.painting.canvas import load_target
from conveyor.painting.canvas import side_by_side
from conveyor.painting.canvas import to_png
from conveyor.painting.canvas import worst_regions
from conveyor.painting.critic import Critic
from conveyor.painting.instrument import CALL_TIMEOUT
from conveyor.painting.instrument import MAX_VIEWS
from conveyor.painting.instrument import Instrument
from conveyor.painting.instrument import InstrumentError
from conveyor.painting.instrument import all_niches
from conveyor.painting.instrument import niche_distance
from conveyor.painting.judge import ClaudeJudge
from conveyor.painting.judge import JudgeError
from conveyor.painting.seeds import SEEDS
from conveyor.store import Store
from conveyor.store import new_id

# The strategist is asked to stay under the soft limit. Past the hard one a prompt is cut, at a paragraph break,
# and the cut is noted in the summary. A plain slice once cut a prompt off mid-section without anyone knowing.
MAX_PROMPT_CHARS = 3000
HARD_PROMPT_CHARS = 4500
_SHEETS: dict[str, bytes] = {}  # demo sheet PNGs by artifact name, so a painting doesn't re-probe to show one


@dataclass
class Setup:
    target: str = "self_portrait"
    width: int = 512
    actions: int = 200
    looks: int | None = None  # default: one per 25 actions, at least 4; negative means unlimited
    patch: int | None = None  # region grid for the error tables; default scales with width
    mode: str = "model"  # "model" (a harness runs each role), or "offline" for the greedy painter and scripted mutators
    seeds: list[str] = field(default_factory=lambda: ["round"])
    work_dir: Path = Path("runs/sessions")
    paint_budget_usd: float | None = 8.0  # per painting session
    mutate_budget_usd: float | None = 4.0  # per mutation session
    mutate_task_budget: int | None = 80_000  # tokens a designer may spend; it paces itself against the countdown
    designer_tries: int = 5
    paint_timeout: float = 40 * 60
    judge: bool = True  # a model judges each finished painting; off in offline mode
    judge_weight: float = 0.5  # share of the score the judge's verdict carries; the rest is the numeric critic
    scope_views: bool = False  # the painter may set a scope and paint in it with local coordinates
    confirm: int = 3  # paintings a challenger and the champion each stand on before the champion changes
    parents: int = 2
    operator_weights: dict[str, float] = field(default_factory=lambda: {"refine": 0.4, "invent": 0.4, "recombine": 0.2})

    @property
    def n_looks(self) -> int:
        return self.looks if self.looks is not None else max(4, self.actions // 25)

    @property
    def n_patch(self) -> int:
        """Region size for the error tables: 16 px at 128 wide, so the whole canvas stays about 8 x 10 regions
        however big it gets. A view's own table is finer (see `view_patch`)."""
        return self.patch if self.patch is not None else 16 * max(1, round(self.width / 128))


# ---- instruments ----------------------------------------------------------------------------------------------


def probe_source(source: str, width: int, height: int, work_dir: Path) -> tuple[dict, bytes | None]:
    """Probe an instrument in a subprocess, where its code can't touch this process and the time limit works."""
    d = work_dir / "probes" / new_id()
    d.mkdir(parents=True, exist_ok=True)
    (d / "instrument.py").write_text(source)
    try:
        r = subprocess.run([sys.executable, "-m", "conveyor.painting.instrument", str(d / "instrument.py"), str(d),
                            str(width), str(height)], capture_output=True, text=True, timeout=120)
        crash = r.stderr[-800:] if r.returncode else ""
    except subprocess.TimeoutExpired:
        crash = "the probe took longer than 120 s"
    if (d / "probe.json").exists():
        report = json.loads((d / "probe.json").read_text())
    else:
        report = {"ok": False, "errors": [f"the probe crashed: {crash.strip() or 'no output'}"], "warnings": [],
                  "traits": {}, "niche": None}
    sheet = (d / "sheet.png").read_bytes() if (d / "sheet.png").exists() else None
    return report, sheet


def make_instrument(source: str, summary: str, setup: Setup, store: Store | None, height: int, **fields) -> Organism:
    """An instrument organism, probed: niche, traits and demo sheet filled in, or marked not viable with why."""
    report, sheet = probe_source(source, setup.width, height, setup.work_dir)
    org = Organism(node="instrument", genome={"source": source}, summary=summary, niche=report.get("niche"),
                   traits=report.get("traits") or {}, viable=bool(report["ok"]),
                   note="; ".join(report["errors"])[:1000], **fields)
    if sheet is not None and store is not None:
        org.sheet = store.artifact(sheet)
        _SHEETS[org.sheet] = sheet
    return org


def _judge_png(img: np.ndarray) -> bytes:
    """An image as the judge sees it: nearest-neighbour, no grid. The same for the target and every painting."""
    return to_png(img, scale=max(1, min(3, 1024 // max(1, img.shape[1]))))


def instrument_doc(source: str) -> str:
    try:
        return (ast.get_docstring(ast.parse(source)) or "").strip().split("\n\n")[0].replace("\n", " ")[:240]
    except SyntaxError:
        return ""


# ---- painting -------------------------------------------------------------------------------------------------


@dataclass
class Painting:
    instrument_id: str
    prompt_id: str
    score: float
    viable: bool
    critic: dict
    artifacts: dict
    feedback: dict
    session_id: str | None
    error: str | None
    details: dict


@dataclass
class Roles:
    """Which harness runs each kind of model work. Any role may be None in offline mode."""

    paint: ProcessHarness | None = None
    mutate: ProcessHarness | None = None
    judge: ProcessHarness | None = None

    @classmethod
    def of(cls, harness: "Roles | ProcessHarness | None") -> "Roles":
        if isinstance(harness, Roles):
            return harness
        return cls(paint=harness, mutate=harness, judge=harness)

    def all(self) -> list[ProcessHarness]:
        seen: list[ProcessHarness] = []
        for h in (self.paint, self.mutate, self.judge):
            if h is not None and h not in seen:
                seen.append(h)
        return seen


class Painter:
    """Paints (instrument, prompt) pairs. A pair's paintings are numbered and kept, so both nodes read the same
    ones: confirming a new instrument champion paints the strategy champion's next standing too."""

    def __init__(self, setup: Setup, store: Store, harness: Roles | ProcessHarness | None) -> None:
        self.setup = setup
        self.store = store
        roles = Roles.of(harness)
        self.claude = roles.paint  # the painting harness (the name predates Pi)
        self.critic = Critic()
        self.target = load_target(setup.target, width=setup.width, patch=setup.n_patch)
        self._cache: dict[tuple[str, str, int], Painting] = {}
        self._locks: dict[tuple[str, str, int], threading.Lock] = {}
        self._lock = threading.Lock()
        self.target_png = gridded_png(self.target.image)
        self.sheets: dict[str, bytes] = {}  # instrument id -> demo sheet png, for the painter's first message
        self.judge: ClaudeJudge | None = None
        if setup.judge and setup.mode != "offline" and roles.judge is not None:
            self.judge = ClaudeJudge(roles.judge, setup.work_dir / store.run_id)
        self.judge_target_png = _judge_png(self.target.image)

    def paint(self, instrument: Organism, prompt: Organism, sample: int = 0) -> Painting:
        """The pair's painting number `sample`, made now if there isn't one yet."""
        key = (instrument.id, prompt.id, sample)
        with self._lock:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
            result = self._paint(instrument, prompt)
            result.details["sample"] = sample
            if result.viable:  # a crashed or cut-off painting can be tried again later
                self._cache[key] = result
            return result

    def _session_dir(self, kind: str) -> Path:
        d = self.setup.work_dir / self.store.run_id / f"{kind}-{new_id()}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _paint(self, instrument: Organism, prompt: Organism) -> Painting:
        s = self.setup
        d = self._session_dir("paint")
        job = {"source": instrument.genome["source"], "target": s.target, "width": s.width, "actions": s.actions,
               "looks": s.n_looks, "patch": s.n_patch, "seed": random.randrange(1 << 30), "snapshot_every": 5,
               "scope": bool(s.scope_views)}
        (d / "job.json").write_text(json.dumps(job))
        started = time.time()
        ingest = _Ingest(self.store, d)
        interrupted = False
        if s.mode == "offline" or self.claude is None:
            sid = new_id()
            ingest.session_id = sid
            self.store.start_session(sid, node=None, organism_id=instrument.id, purpose="paint (greedy)",
                                     model="greedy", effort=None, request={"job": {**job, "source": "(instrument)"}},
                                     dir=str(d))
            r = subprocess.run([sys.executable, "-m", "conveyor.painting.paintserver", str(d), "--greedy"],
                               capture_output=True, text=True, timeout=s.paint_timeout)
            ingest.pull()
            error = r.stderr[-800:] if r.returncode else None
            self.store.update_session(sid, status="error" if error else "ok", ended=time.time(), cost=0.0, error=error)
            session_error = error
        else:
            inst = Instrument(instrument.genome["source"])
            h, w = self.target.height, self.target.width
            system = (prompt.genome["prompt"].strip() + "\n\n" + prompts.PAINTER_RULES.format(
                w=w, h=h, actions=s.actions, looks="unlimited" if s.n_looks < 0 else s.n_looks, reference=inst.reference(w, h),
                area_cap=Canvas(h, w).area_cap, share=CALL_AREA_SHARE))
            if self.judge is not None:
                system += "\n\n" + prompts.JUDGE_RULE
            if s.scope_views:
                system += "\n\n" + prompts.SCOPE_RULE
            sheet = self._sheet(instrument)
            first, second, third = prompts.PAINTER_FIRST_MESSAGE
            content = [{"type": "text", "text": first + ":"}, {"type": "png", "data": self.target_png}]
            if sheet:
                content += [{"type": "text", "text": second + ":"}, {"type": "png", "data": sheet}]
            content.append({"type": "text", "text": third})
            tools = [t.name for t in inst.spec.tools] + [v.name for v in inst.spec.views] + ["look", "finish"]
            if s.scope_views:
                tools.append("scope")
            outcome = self.claude.run(Job(
                purpose="paint", system_prompt=system, content=content, cwd=d,
                mcp={"name": "canvas", "command": sys.executable, "args": ["-m", "conveyor.painting.paintserver", str(d)]},
                tools=tools, stop_tools=["finish"], max_budget_usd=s.paint_budget_usd, timeout=s.paint_timeout,
                node="painting",
                organism_id=instrument.id, on_start=ingest.start,
                on_event=lambda msg: ingest.pull() if msg.get("type") == "user" else None))
            ingest.pull()
            session_error = outcome.error
            interrupted = outcome.interrupted
        canvas = np.load(d / "canvas.npy") if (d / "canvas.npy").exists() else Canvas(self.target.height, self.target.width).img
        finish = json.loads((d / "finish.json").read_text()) if (d / "finish.json").exists() else {}
        calls = ingest.calls
        applied = [c for c in calls if c.get("status") == "applied" and c.get("tool") not in ("look", "finish")]
        # A session cut off before the painter finished (a stalled or dropped stream, a crash) says nothing about
        # the instrument or the prompt. Scoring its half-painted canvas, or caching it, would charge them for it.
        if interrupted and not finish and len(applied) < s.actions:
            raise SessionFailed(f"the painting session ended after {len(applied)} of {s.actions} actions: {session_error}")
        # A painting that never got going (the session crashed before a single mark) says nothing about either
        # organism, so it doesn't count as an evaluation of them.
        viable = bool(applied)
        critic = self.critic.score(canvas, self.target)
        usage = Counter(c.get("tool") for c in applied)
        rejected = [c for c in calls if c.get("status") == "rejected"]
        stats = {
            "actions_used": len(applied), "actions_budget": s.actions, "looks_used": sum(1 for c in calls if c.get("tool") == "look"),
            "views_used": sum(1 for c in calls if c.get("status") == "view"),
            "tool_use": dict(usage.most_common()), "refused": len(rejected),
            "refusals": [c.get("error", "")[:160] for c in rejected[:5]],
            "ran_dry": sum(1 for c in applied if c.get("dry")), "finished": bool(finish),
        }
        painting_png = to_png(canvas)
        artifacts = {"painting": self.store.artifact(painting_png),
                     "heat": self.store.artifact(heatmap_png(self.critic.pixel_error(canvas, self.target))),
                     "pair": self.store.artifact(side_by_side(self.target.image, canvas, scale=2))}
        feedback = {"pair_png": side_by_side(self.target.image, canvas, scale=2), "note": finish.get("note", ""),
                    "stats": stats, "worst": worst_regions(canvas, self.target), "critic": critic}
        details = {"critic": critic, "note": finish.get("note", ""), "stats": stats, "instrument_id": instrument.id,
                   "prompt_id": prompt.id, "seconds": round(time.time() - started, 1), "session_error": session_error}
        score = critic["total"]
        if viable and self.judge is not None:
            verdict = self._verdict(canvas, instrument.id)
            if verdict is not None:
                w = s.judge_weight
                score = (1 - w) * critic["total"] + w * verdict.score
                judged = {"scores": verdict.scores, "score": round(verdict.score, 4), "critique": verdict.critique,
                          "session_id": verdict.session_id, "weight": w}
                details["judge"] = judged
                feedback["judge"] = judged
            else:
                details["judge"] = {"error": "the judge gave no usable verdict twice; scored by the critic alone"}
        details["score"] = round(score, 5)
        return Painting(instrument_id=instrument.id, prompt_id=prompt.id, score=score if viable else 0.0,
                        viable=viable, critic=critic, artifacts=artifacts, feedback=feedback,
                        session_id=ingest.session_id, error=None if viable else (session_error or "no marks were made"),
                        details=details)

    def _verdict(self, canvas: np.ndarray, organism_id: str):
        """The judge's verdict on a finished painting. An unusable reply is retried once; a session cut off partway
        is retried twice. If the judge never gets through, the painting fails with it: a score from the critic alone
        isn't comparable with one that includes the judge. Rate limits and budget stops propagate."""
        unusable = cut_off = 0
        while unusable < 2 and cut_off < 3:
            try:
                return self.judge.judge(self.judge_target_png, _judge_png(canvas), organism_id=organism_id)
            except JudgeError:
                unusable += 1
            except SessionFailed:
                cut_off += 1
        if cut_off >= 3:
            raise SessionFailed("the judge's session was cut off three times")
        return None

    def _sheet(self, instrument: Organism) -> bytes | None:
        if instrument.sheet and instrument.sheet in _SHEETS:
            return _SHEETS[instrument.sheet]
        if instrument.id not in self.sheets:
            _, sheet = probe_source(instrument.genome["source"], self.setup.width, self.target.height, self.setup.work_dir)
            if sheet:
                self.sheets[instrument.id] = sheet
        return self.sheets.get(instrument.id)

    def evaluation(self, painting: Painting, organism: Organism, partner: Organism | None, reason: str) -> Evaluation:
        return Evaluation(organism_id=organism.id, score=painting.score, viable=painting.viable,
                          partner_id=partner.id if partner else None, reason=reason, details=painting.details,
                          artifacts=painting.artifacts, session_id=painting.session_id, error=painting.error,
                          feedback=painting.feedback)


class _Ingest:
    """Copies a painting's calls.jsonl into the store as strokes, snapshots as artifacts, as the painting runs."""

    def __init__(self, store: Store, session_dir: Path) -> None:
        self.store = store
        self.dir = session_dir
        self.session_id: str | None = None
        self.calls: list[dict] = []
        self._offset = 0
        self._lock = threading.Lock()

    def start(self, session_id: str) -> None:
        self.session_id = session_id

    def pull(self) -> None:
        path = self.dir / "calls.jsonl"
        if not path.exists():
            return
        with self._lock, open(path) as f:
            f.seek(self._offset)
            while True:
                line = f.readline()
                if not line or not line.endswith("\n"):
                    break
                self._offset = f.tell()
                entry = json.loads(line)
                self.calls.append(entry)
                snap = None
                if entry.get("snapshot") and (self.dir / entry["snapshot"]).exists():
                    snap = self.store.artifact((self.dir / entry["snapshot"]).read_bytes())
                if self.session_id:
                    self.store.stroke(self.session_id, entry["i"], entry, snap)


# ---- evaluators -------------------------------------------------------------------------------------------------


def instrument_evaluator(painter: Painter):
    def evaluate(org: Organism, partner: Organism | None, reason: str, sample: int = 0) -> Evaluation:
        painting = painter.paint(org, partner, sample)
        return painter.evaluation(painting, org, partner, reason)
    return evaluate


def prompt_evaluator(painter: Painter):
    def evaluate(org: Organism, partner: Organism | None, reason: str, sample: int = 0) -> Evaluation:
        painting = painter.paint(partner, org, sample)
        return painter.evaluation(painting, org, partner, reason)
    return evaluate


# ---- mutators -------------------------------------------------------------------------------------------------


def _learning_log(lineage: list[dict]) -> str:
    if not lineage:
        return "None yet: this is a seed."
    lines = []
    for e in lineage:
        if not e["viable"]:
            result = f"not viable ({e['note'][:160]})" if e["note"] else "not viable"
        elif e["parent_score"] is not None and e["score"] is not None:
            better = "better" if e["score"] > e["parent_score"] else "worse" if e["score"] < e["parent_score"] else "same"
            result = f"scored {e['score']:.3f} against its parent's {e['parent_score']:.3f} ({better})"
        else:
            result = f"scored {e['score']:.3f}" if e["score"] is not None else "not scored"
        moved = f", moving from {e['parent_niche']} to {e['niche']}" if e.get("niche") and e.get("parent_niche") and e["niche"] != e["parent_niche"] else ""
        lines.append(f"- {e['mutator'] or 'seed'}: {e['summary'][:300]} -> {result}{moved}")
    return "\n".join(lines)


def _evidence(ev: Evaluation | None) -> tuple[str, list[dict]]:
    """What the last painting with this organism showed, as text plus the target/painting image."""
    if ev is None or not ev.feedback:
        return "No painting with it yet.", []
    fb = ev.feedback
    c, st = fb["critic"], fb["stats"]
    worst = "; ".join(f"x {w['x'][0]}-{w['x'][1]}, y {w['y'][0]}-{w['y'][1]}: {w['error']:.2f}" for w in fb["worst"])
    uses = ", ".join(f"{k} x{v}" for k, v in st["tool_use"].items()) or "none"
    single = ev.details.get("score", ev.score)
    lines = [
        (f"Score {ev.score:.3f}, the mean of {ev.samples} paintings; the one described here scored {single:.3f}. "
         if ev.samples > 1 else f"Score {ev.score:.3f}. ") +
        "One painting's score moves by a few hundredths on its own, so read small differences as noise. "
        f"The numeric critic gave {c['total']:.3f} (pixel {c['pixel']:.3f}, style {c['style']:.3f}; "
        "style distances " + ", ".join(f"{k} {v:.2f}" for k, v in c["style_distance"].items()) + ").",
        f"The painter used {st['actions_used']} of {st['actions_budget']} actions ({uses}) and {st['looks_used']} looks. "
        f"{st['refused']} calls were refused" + (f", for example: {st['refusals'][0]}" if st["refusals"] else "") +
        f". {st['ran_dry']} calls ran out of area.",
        f"Worst regions (RMSE): {worst}.",
        "The painter's note at the end: " + (fb["note"].strip() or "(it didn't write one)"),
    ]
    judged = fb.get("judge")
    if judged:
        sc = judged["scores"]
        lines.append(f"An expert judge scored it likeness {sc['likeness']}, colour {sc['colour']}, brushwork "
                     f"{sc['brushwork']}, overall {sc['overall']} (out of 10). The judge's critique: {judged['critique']}")
    return "\n".join(lines), [{"type": "text", "text": "Target (left) and the painting (right):"},
                              {"type": "png", "data": fb["pair_png"]}]


class ClaudeInstrumentMutator(Mutator):
    """Claude Code redesigns the instrument in a workbench where it can run its drafts before submitting."""

    def __init__(self, operator: str, claude: ProcessHarness, setup: Setup, store: Store, height: int, weight: float) -> None:
        self.operator = operator
        self.name = f"{claude.name}:{operator}"
        self.weight = weight
        self.needs_other = operator == "recombine"
        self.claude, self.setup, self.store, self.height = claude, setup, store, height

    def _system(self) -> str:
        canvas = Canvas(self.height, self.setup.width)
        return prompts.DESIGNER_SYSTEM.format(
            w=self.setup.width, h=self.height, contract=prompts.CONTRACT, area_cap=canvas.area_cap,
            share=CALL_AREA_SHARE, max_radius=canvas.max_radius, side=int(2 * canvas.max_radius + 1),
            timeout=CALL_TIMEOUT, max_views=MAX_VIEWS)

    def propose(self, ctx: Context) -> list[Organism]:
        s = self.setup
        d = s.work_dir / self.store.run_id / f"mutate-{self.operator}-{new_id()}"
        d.mkdir(parents=True, exist_ok=True)
        wanted = ctx.wanted_niche if self.operator == "invent" else None
        (d / "job.json").write_text(json.dumps({"width": s.width, "height": self.height, "max_tries": s.designer_tries,
                                                "wanted_niche": wanted}))
        evidence, images = _evidence(ctx.parent_eval)
        content: list[dict] = []
        if self.operator == "recombine" and ctx.other is not None:
            other_evidence, other_images = _evidence(ctx.other_eval)
            content += [{"type": "text", "text": f"Instrument A ({ctx.parent.niche}):\n\n```python\n"
                         f"{ctx.parent.genome['source']}\n```\n\n{evidence}"}, *images,
                        {"type": "text", "text": f"Instrument B ({ctx.other.niche}):\n\n```python\n"
                         f"{ctx.other.genome['source']}\n```\n\n{other_evidence}"}, *other_images,
                        {"type": "text", "text": prompts.RECOMBINE_TASK}]
        else:
            content += [{"type": "text", "text": f"The current instrument ({ctx.parent.niche}):\n\n```python\n"
                         f"{ctx.parent.genome['source']}\n```"},
                        {"type": "text", "text": "What happened when the painter used it:\n" + evidence}, *images]
            if self.operator == "invent":
                filled = "\n".join(f"- {n}: score {e.score:.3f}. {instrument_doc(o.genome['source'])}"
                                   for n, (o, e) in sorted(ctx.niches.items())) or "- none yet"
                empty = ", ".join(ctx.empty_niches) or "none"
                content.append({"type": "text", "text": f"{prompts.NICHE_AXES}\n\nFilled niches:\n{filled}\n\n"
                                f"Empty niches: {empty}.\n\n" + prompts.INVENT_TASK.format(wanted=wanted)})
            else:
                content.append({"type": "text", "text": prompts.REFINE_TASK})
        content.append({"type": "text", "text": "Earlier changes in this instrument's line and what they did:\n"
                        + _learning_log(ctx.lineage)})
        outcome = self.claude.run(Job(
            purpose=f"mutate instrument ({self.operator})", system_prompt=self._system(), content=content, cwd=d,
            mcp={"name": "bench", "command": sys.executable, "args": ["-m", "conveyor.painting.workbench", str(d)]},
            tools=["try_instrument", "submit_instrument"], stop_tools=["submit_instrument"],
            max_budget_usd=s.mutate_budget_usd, node="instrument",
            organism_id=ctx.parent.id, task_budget=s.mutate_task_budget))
        sub = d / "submitted.json"
        if not sub.exists():
            if outcome.interrupted:
                raise SessionFailed(f"the instrument session ended early: {outcome.error}")
            raise RuntimeError(f"no instrument was submitted ({outcome.error or 'the session ended without submitting'})")
        record = json.loads(sub.read_text())
        child = make_instrument(record["source"], record.get("summary", ""), s, self.store, self.height,
                                parent_id=ctx.parent.id, parent2_id=ctx.other.id if ctx.other else None,
                                mutator=self.name, session_id=outcome.session_id)
        if wanted:
            child.traits = {**child.traits, "wanted_niche": wanted}
        return [child]


class ClaudePromptMutator(Mutator):
    """The strategist edits the prompt rather than rewriting it. It returns edits, each quoting the passage it
    replaces, and they're applied to the parent. The wording carries what earlier generations learned, and with
    scores this noisy only a change or two at a time can be credited or blamed. A child that keeps less than
    `MIN_WORDS_KEPT` of its parent's words is refused."""

    name = "claude:strategy"
    SCHEMA = {"type": "object", "properties": {
        "edits": {"type": "array", "description": "One or two edits, applied in order.", "items": {
            "type": "object", "properties": {
                "old": {"type": "string", "description": "A passage copied exactly from the current prompt. "
                        "Empty to add `new` at the end."},
                "new": {"type": "string", "description": "What replaces it. Empty to cut the passage."}},
            "required": ["old", "new"], "additionalProperties": False}},
        "summary": {"type": "string", "description": "One sentence: what you changed and which failure it addresses."}},
        "required": ["edits", "summary"], "additionalProperties": False}

    def __init__(self, claude: ProcessHarness, setup: Setup, store: Store) -> None:
        self.claude, self.setup, self.store = claude, setup, store
        self.name = f"{claude.name}:strategy"

    def propose(self, ctx: Context) -> list[Organism]:
        d = self.setup.work_dir / self.store.run_id / f"mutate-strategy-{new_id()}"
        parent = ctx.parent.genome["prompt"].strip()
        evidence, images = _evidence(ctx.parent_eval)
        instrument = ""
        if ctx.partner is not None:
            instrument = "\n\nThe instrument it painted with, as the painter saw it:\n" + _reference(ctx.partner, self.setup)
        content = [
            {"type": "text", "text": f"The current strategy prompt ({len(parent)} characters):\n<<<\n{parent}\n>>>"},
            {"type": "text", "text": "What happened in the last painting with it:\n" + evidence + instrument}, *images,
            {"type": "text", "text": "Earlier changes to this prompt and what they did:\n" + _learning_log(ctx.lineage)},
            {"type": "text", "text": "Make one or two edits aimed at the most important failure. Don't repeat a "
                                     "change that made things worse."},
        ]
        outcome = self.claude.run(Job(
            purpose="mutate strategy", system_prompt=prompts.STRATEGIST_SYSTEM.format(max_chars=MAX_PROMPT_CHARS),
            content=content, cwd=d, json_schema=self.SCHEMA, max_budget_usd=self.setup.mutate_budget_usd,
            node="painter", organism_id=ctx.parent.id, task_budget=self.setup.mutate_task_budget))
        out = outcome.structured if isinstance(outcome.structured, dict) else _json_in(outcome.result)
        if not out and outcome.interrupted:
            raise SessionFailed(f"the strategy session ended early: {outcome.error}")
        if not out or not isinstance(out.get("edits"), list) or not out["edits"]:
            raise RuntimeError(f"no edits came back ({outcome.error or 'unparseable reply'})")
        text, missed = apply_edits(parent, out["edits"])
        if missed == len(out["edits"]):
            raise RuntimeError(f"none of the {missed} edits matched the prompt")
        text, trimmed = trim_prompt(text)
        if text == parent:
            return []
        kept = words_kept(parent, text)
        if kept < MIN_WORDS_KEPT:
            raise RuntimeError(f"the edits kept {kept:.0%} of the prompt's words; a child has to keep at least "
                               f"{MIN_WORDS_KEPT:.0%}")
        notes = [f"{missed} of {len(out['edits'])} edits didn't match and were skipped" if missed else "",
                 f"trimmed from {trimmed} to {len(text)} characters at a paragraph break" if trimmed else ""]
        summary = str(out.get("summary", ""))[:600] + "".join(f" ({n})" for n in notes if n)
        return [Organism(node="painter", genome={"prompt": text}, summary=summary, session_id=outcome.session_id,
                         traits={"words_kept": round(kept, 3), "edits": len(out["edits"]) - missed})]


MIN_WORDS_KEPT = 0.5


def apply_edits(text: str, edits: list[dict]) -> tuple[str, int]:
    """Apply find-and-replace edits in order. A passage matches exactly, or else with any run of whitespace
    standing for any other; one that matches nowhere or more than once is skipped. Returns the text and the
    number skipped."""
    missed = 0
    for e in edits:
        old, new = str(e.get("old", "")), str(e.get("new", ""))
        if not old.strip():
            text = text.rstrip() + "\n\n" + new.strip() if new.strip() else text
            continue
        if text.count(old) == 1:
            text = text.replace(old, new)
            continue
        hits = list(re.finditer(r"\s+".join(map(re.escape, old.split())), text))
        if len(hits) != 1:
            missed += 1
            continue
        text = text[: hits[0].start()] + new + text[hits[0].end():]
    return re.sub(r"\n{3,}", "\n\n", text).strip(), missed


def words_kept(parent: str, child: str) -> float:
    """The share of the parent's words that survive into the child, in order."""
    a, b = parent.split(), child.split()
    if not a:
        return 1.0
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(a)


def trim_prompt(text: str, limit: int = HARD_PROMPT_CHARS) -> tuple[str, int | None]:
    """The prompt, cut at the last paragraph (or sentence) break under `limit` if it's longer. Returns the
    original length when it was cut, else None."""
    text = text.strip()
    if len(text) <= limit:
        return text, None
    head = text[:limit]
    cut = head.rfind("\n\n")
    if cut < limit // 2:
        cut = head.rfind(". ") + 1
    if cut < limit // 2:
        cut = limit
    return head[:cut].rstrip(), len(text)


def _reference(instrument: Organism, setup: Setup) -> str:
    try:
        inst = Instrument(instrument.genome["source"])
    except InstrumentError:
        return "(the instrument doesn't compile)"
    return inst.reference(setup.width, load_target(setup.target, width=setup.width, patch=setup.n_patch).height)


def _json_in(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class JitterInstrumentMutator(Mutator):
    """Offline stand-in: scale one number inside a tool function. It can only ever retune, never invent."""

    name = "jitter"

    def __init__(self, setup: Setup, store: Store, height: int) -> None:
        self.setup, self.store, self.height = setup, store, height

    def propose(self, ctx: Context) -> list[Organism]:
        rng = random.Random()
        source = ctx.parent.genome["source"]
        tree = ast.parse(source)
        spots = [n for fn in tree.body if isinstance(fn, ast.FunctionDef) for n in ast.walk(fn)
                 if isinstance(n, ast.Constant) and isinstance(n.value, int | float) and not isinstance(n.value, bool)
                 and abs(n.value) > 1e-9 and n.lineno == n.end_lineno]
        if not spots:
            return []
        pick = rng.choice(spots)
        new = round(float(pick.value) * rng.uniform(0.6, 1.5), 3)
        lines = source.splitlines()
        line = lines[pick.lineno - 1]
        lines[pick.lineno - 1] = line[: pick.col_offset] + f"{new:g}" + line[pick.end_col_offset :]
        summary = f"line {pick.lineno}: {pick.value:g} to {new:g}"
        return [make_instrument("\n".join(lines) + "\n", summary, self.setup, self.store, self.height,
                                parent_id=ctx.parent.id)]


# ---- the graph --------------------------------------------------------------------------------------------------


def build(setup: Setup, store: Store, harness: Roles | ProcessHarness | None) -> tuple[list[Node], Painter]:
    roles = Roles.of(harness)
    painter = Painter(setup, store, roles)
    height = painter.target.height
    seeds = [make_instrument(SEEDS[name], f"seed: {name}", setup, store, height) for name in setup.seeds]
    for seed in seeds:
        if not seed.viable:
            raise RuntimeError(f"seed instrument doesn't probe cleanly: {seed.note}")
    strategy = Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY}, summary="")
    offline = setup.mode == "offline" or roles.paint is None or roles.mutate is None
    if offline:
        instrument_mutators: list[Mutator] = [JitterInstrumentMutator(setup, store, height)]
        painter_mutators: list[Mutator] = []
    else:
        w = setup.operator_weights
        instrument_mutators = [ClaudeInstrumentMutator(op, roles.mutate, setup, store, height, w.get(op, 0.0))
                               for op in ("refine", "invent", "recombine") if w.get(op, 0.0) > 0]
        painter_mutators = [ClaudePromptMutator(roles.mutate, setup, store)]
    nodes = [
        Node(name="instrument", seeds=seeds, evaluate=instrument_evaluator(painter), mutators=instrument_mutators,
             partner="painter", archive=True, all_niches=all_niches(), niche_distance=niche_distance,
             parents=setup.parents, confirm=setup.confirm,
             description="The toolkit: a program that defines the painter's tools, their parameters, and any state "
                         "they share. Parents are drawn across niches."),
        Node(name="painter", seeds=[strategy], evaluate=prompt_evaluator(painter), mutators=painter_mutators,
             partner="instrument", fixed=offline, parents=1, confirm=setup.confirm,
             description="The painter's strategy prompt, run by Claude Code with the instrument's tools."
                         if not offline else "Fixed in offline mode: the greedy painter ignores prompts."),
    ]
    return nodes, painter
