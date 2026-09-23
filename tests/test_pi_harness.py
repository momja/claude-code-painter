"""The real Pi sidecar, driven by Pi's fake provider: the agent loop, the protocol, and what gets recorded."""

import json
import shutil
from unittest import mock

import numpy as np
import pytest

from conveyor.events import EventSink
from conveyor.events import connect
from conveyor.events import observing
from conveyor.llm import LLMClient
from conveyor.painting.canvas import TARGETS_DIR
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import load_target
from conveyor.painting.llm_agent import INITIAL_PROMPT
from conveyor.painting.llm_agent import PromptStrategy
from conveyor.painting.pi_harness import PI_DIR
from conveyor.painting.pi_harness import PiHarness
from conveyor.painting.toolkit import initial_toolkit

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (PI_DIR / "node_modules" / "@earendil-works" / "pi-agent-core").is_dir(),
    reason="needs node and `npm install` in pi-painter/",
)

STROKE = {"x": 20, "y": 30, "angle": 0, "color": "#806040"}
SCRIPT = [
    {"blocks": [{"thinking": "Block in the body first."},
                {"tool": "flat_wash", "args": STROKE},
                {"tool": "round_mid", "args": {**STROKE, "x": 30}},
                {"tool": "look", "args": {}}]},
    {"blocks": [{"thinking": "The body is in. Fix the ear, then stop."},
                {"tool": "round_mid", "args": {**STROKE, "y": 10}},
                {"tool": "no_such_brush", "args": STROKE},
                {"tool": "finish", "args": {"note": "done"}}]},
    {"blocks": [{"text": "This turn should never be requested."}]},
]


def _paint(tmp_path, script=SCRIPT, n_strokes=10, record=True, min_stroke_fraction=0.0):
    sink = EventSink(tmp_path / "pi.db")
    client = LLMClient(transport=lambda payload: {}, sink=sink)
    harness = PiHarness(client, faux=script, max_looks=2, min_stroke_fraction=min_stroke_fraction)
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=48)
    with sink.trace() as trace, observing(node="agent", organism_id="org1"):
        canvas = harness(PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=n_strokes), initial_toolkit(), target,
                         seed=1, record=record)
    sink.close()
    return canvas, target, client, trace, connect(tmp_path / "pi.db", readonly=True)


def test_one_conversation_with_strokes_look_and_finish(tmp_path):
    canvas, target, client, trace, conn = _paint(tmp_path)
    assert not np.allclose(canvas, blank(target))

    calls = conn.execute("SELECT * FROM llm_calls ORDER BY started").fetchall()
    assert len(calls) == 2  # finish stopped the loop before the third scripted reply
    first, second = (json.loads(c["request"]) for c in calls)
    assert first["conversation"] == second["conversation"]
    assert (first["turn"], second["turn"]) == (1, 2)
    assert all(c["node"] == "agent" and c["organism_id"] == "org1" and c["trace_id"] == trace.id for c in calls)

    # Turn 1 sends the system prompt and the opening message with the target and the blank canvas.
    assert first["messages"][0]["role"] == "system" and "look" in first["messages"][0]["content"]
    images = [p for p in first["messages"][1]["content"] if p["type"] == "image"]
    assert len(images) == 2
    assert {t["function"]["name"] for t in first["tools"]} == {"flat_wash", "round_mid", "look", "finish"}
    # Turn 2 sends only what's new: the three tool results, the look result carrying the canvas image.
    assert [m["role"] for m in second["messages"]] == ["tool", "tool", "tool"]
    look = second["messages"][2]
    assert look["name"] == "look" and any(p["type"] == "image" for p in look["content"])
    stored = conn.execute("SELECT count(*) FROM artifacts WHERE name=?",
                          (next(p["artifact"] for p in look["content"] if p["type"] == "image"),)).fetchone()[0]
    assert stored == 1

    assert calls[0]["reasoning"] == "Block in the body first."
    outcomes = [[r["status"] for r in json.loads(c["tool_results"])] for c in calls]
    assert outcomes == [["applied", "applied", "applied"], ["applied", "rejected", "applied"]]
    assert json.loads(calls[1]["tool_results"])[1]["error"]  # Pi's own error for the unknown tool


def test_cache_reads_are_recorded_and_counted(tmp_path):
    _, _, client, _, conn = _paint(tmp_path)
    second = json.loads(conn.execute("SELECT usage FROM llm_calls ORDER BY started LIMIT 1 OFFSET 1").fetchone()[0])
    assert second["prompt_tokens_details"]["cached_tokens"] > 0  # the fake provider simulates prefix caching
    assert client.usage.cached_tokens > 0 and client.usage.calls == 2
    events = [json.loads(r[0]) for r in conn.execute("SELECT data FROM events WHERE kind='llm_call'")]
    assert sum(e["cached_tokens"] for e in events) == client.usage.cached_tokens


def test_replay_interleaves_model_calls_strokes_and_looks(tmp_path):
    _, _, _, trace, conn = _paint(tmp_path)
    names = [r[0] for r in conn.execute("SELECT name FROM spans WHERE trace_id=? ORDER BY idx", (trace.id,))]
    # Each model call's span lands when its reply ends, right before the tool calls it made.
    assert names == ["model call (paint)", "flat_wash", "round_mid", "look",
                     "model call (paint)", "round_mid", "finish"]


def test_finish_is_refused_until_most_strokes_are_used(tmp_path):
    early = {"blocks": [{"tool": "flat_wash", "args": STROKE}, {"tool": "flat_wash", "args": {**STROKE, "x": 5}},
                        {"tool": "finish", "args": {"note": "done already"}}]}
    rest = {"blocks": [{"tool": "round_mid", "args": {**STROKE, "y": y}} for y in (5, 15)]}
    _, _, client, trace, conn = _paint(tmp_path, script=[early, rest], n_strokes=4, min_stroke_fraction=0.9)
    turns = conn.execute("SELECT tool_results FROM llm_calls ORDER BY started").fetchall()
    first, second = (json.loads(t[0]) for t in turns)
    assert [r["status"] for r in first] == ["applied", "applied", "rejected"]
    assert first[2]["error"].startswith("refused, 2 of the 4")
    assert [r["status"] for r in second] == ["applied", "applied"]  # the fourth stroke ends the painting
    spans = [r[0] for r in conn.execute("SELECT name FROM spans WHERE trace_id=? ORDER BY idx", (trace.id,))]
    assert "finish refused" in spans and "finish" not in spans


def test_finish_is_accepted_after_repeated_refusals(tmp_path):
    stubborn = {"blocks": [{"tool": "finish", "args": {}}]}
    _, _, _, _, conn = _paint(tmp_path, script=[stubborn] * 5, n_strokes=10, min_stroke_fraction=0.9)
    statuses = [json.loads(t[0])[0]["status"] for t in conn.execute("SELECT tool_results FROM llm_calls ORDER BY started")]
    assert statuses == ["rejected", "rejected", "rejected", "applied"]


def test_the_painter_can_run_a_different_model_from_the_mutators(tmp_path):
    """
    Pi dispatches on the model's own API, so a responses-API model (muse-spark) can paint while `LLMClient`,
    which only speaks chat completions, keeps running the mutators on its own model.
    """
    client = LLMClient(transport=lambda payload: {}, provider="opencode-go", api_key="test",
                       model="glm-5.3-flash")
    assert PiHarness(client, faux=[], model="muse-spark-1.3-contributor").model == "muse-spark-1.3-contributor"
    assert PiHarness(client, faux=[]).model == "glm-5.3-flash"  # no override, no change
    assert client.model == "glm-5.3-flash"  # the mutators are untouched either way

    # A real harness (no faux script) looks the paint model up in the local catalog for the sidecar.
    real = PiHarness(client, model="muse-spark-1.3-contributor")
    assert real.model_def["id"] == "muse-spark-1.3-contributor"
    assert "image" in real.model_def["input"]


def test_each_sidecar_gets_a_bounded_heap(tmp_path):
    """Node defaults to a 2 GB heap per process, and paintings run several sidecars at once."""
    client = LLMClient(transport=lambda payload: {}, provider="opencode-go", api_key="test")
    assert PiHarness(client, faux=[]).heap_mb == 256
    seen = {}

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        raise RuntimeError("stop here: the spawn is all this test needs")

    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=48)
    harness = PiHarness(client, faux=[{"blocks": []}], heap_mb=128)
    with mock.patch("conveyor.painting.pi_harness.subprocess.Popen", fake_popen), pytest.raises(RuntimeError):
        harness(PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=2), initial_toolkit(), target, seed=1)
    assert "--max-old-space-size=128" in seen["argv"]
    assert seen["argv"][-1].endswith("painter.mjs")


def test_an_in_flight_call_names_the_painting_model(tmp_path):
    """The live view reads rows while they run, so a row must name the model being called from the start."""
    sink = EventSink(tmp_path / "m.db")
    client = LLMClient(transport=lambda payload: {}, provider="opencode-go", api_key="test",
                       model="glm-5.3-flash", sink=sink)
    client.external_start("call1", purpose="paint", request={"turn": 1}, model="muse-spark-1.3-contributor")
    client.external_start("call2", purpose="mutate", request={})  # no override: the client's own model
    sink.close()
    rows = dict(connect(tmp_path / "m.db", readonly=True).execute("SELECT id, model FROM llm_calls").fetchall())
    assert rows["call1"] == "muse-spark-1.3-contributor"
    assert rows["call2"] == "glm-5.3-flash"


def test_the_image_limit_trims_what_is_sent_rather_than_cutting_looks(tmp_path):
    """
    A provider's image limit belongs to the sidecar's trim, not to the look budget. Cutting looks cost half the
    painter's checking and didn't even avoid the fault: in one run 15 of 21 paintings sent 10 to 15 images
    fine, while 7 were rejected at the 9th.
    """
    from conveyor.llm import PROVIDERS
    from conveyor.painting.pi_harness import _PaintSession

    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=48)
    strategy = PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=500)  # one look per 40 strokes

    def session(provider: str) -> _PaintSession:
        client = LLMClient(transport=lambda payload: {}, provider=provider, api_key="test")
        return _PaintSession(PiHarness(client, faux=[]), strategy, initial_toolkit(), target, seed=1, record=False)

    assert session("opencode-go").max_looks == 13  # the stroke budget decides, not the image limit
    assert session("openrouter").max_looks == 13
    assert PROVIDERS["opencode-go"].max_images == 8  # the sidecar trims to this, keeping target plus newest
    assert PROVIDERS["openrouter"].max_images is None


def test_pressure_is_recorded_and_bad_pressure_rejected(tmp_path):
    script = [{"blocks": [{"tool": "flat_wash", "args": {**STROKE, "pressure": 0.3}},
                          {"tool": "flat_wash", "args": {**STROKE, "pressure": "hard"}},
                          {"tool": "finish", "args": {}}]}]
    _, _, _, trace, conn = _paint(tmp_path, script=script)
    results = json.loads(conn.execute("SELECT tool_results FROM llm_calls").fetchone()[0])
    assert [r["status"] for r in results] == ["applied", "rejected", "applied"]
    args = json.loads(conn.execute("SELECT args FROM spans WHERE trace_id=? AND name='flat_wash'", (trace.id,)).fetchone()[0])
    assert args["pressure"] == 0.3


def test_running_out_of_strokes_ends_the_painting(tmp_path):
    many = {"blocks": [{"tool": "flat_wash", "args": {**STROKE, "x": i}} for i in range(5)]}
    _, _, client, _, conn = _paint(tmp_path, script=[many, many], n_strokes=3)
    results = json.loads(conn.execute("SELECT tool_results FROM llm_calls").fetchone()[0])
    assert [r["status"] for r in results] == ["applied"] * 3 + ["ignored"] * 2
    assert client.usage.calls == 1
