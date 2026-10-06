"""The painter works from images; numeric quality metrics remain available to diagnostics."""

import json

import pytest

from conveyor.painting import prompts
from conveyor.painting.paintserver import PaintServer, PaintSession
from conveyor.painting.seeds import PEN


@pytest.fixture
def server(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({
        "source": PEN, "target": "self_portrait", "width": 64,
        "actions": 10, "looks": 1, "scope": True,
    }))
    return PaintServer(PaintSession(tmp_path))


@pytest.mark.parametrize("name,args", [
    ("start", {"x": 10, "y": 10, "color": "#223344"}),
    ("paint_batch", {"calls": [["start", {"x": 10, "y": 10, "color": "#223344"}]]}),
    ("look", {}),
    ("detail", {"x": 32, "y": 32, "span": 16}),
    ("scope", {"x": 32, "y": 32, "span": 16}),
    ("scope", {"clear": True}),
    ("finish", {"note": "done"}),
    ("move", {"dx": "bad"}),
])
def test_no_metrics_in_text_or_structured_state(server, name, args):
    result = server.handle({"id": 1, "method": "tools/call", "params": {
        "name": name, "arguments": args, "_meta": {"conveyor/paintingState": True},
    }})["result"]
    visible = " ".join(b["text"] for b in result["content"] if b["type"] == "text").lower()
    for metric in ("score", "pixel error", "pixel_error", "rmse", "pixel match", "style_distance"):
        assert metric not in visible
    state = result["structuredContent"]["painting_state"]
    assert "score" not in state and "pixel_error" not in state
    assert "actions_left" in state and "pen" in state
    if name in ("look", "detail") or name == "scope" and not args.get("clear"):
        assert any(b["type"] == "image" for b in result["content"])


def test_metrics_still_recorded_for_diagnostics(server):
    s = server.s
    s.apply("start", {"x": 10, "y": 10, "color": "#223344"})
    s.apply("move", {"dx": 20, "dy": 0})
    s.finish("done")
    calls = [json.loads(line) for line in (s.dir / "calls.jsonl").read_text().splitlines()]
    for call in calls[:2]:
        assert all(key in call for key in ("error_before", "error_after", "score_before", "score_after"))
    finish = json.loads((s.dir / "finish.json").read_text())
    assert finish["score"] == round(s.score, 4)
    assert finish["error"] == round(s.error, 2)


def test_exhausted_looks_does_not_direct_painter_to_scores(server):
    server.s.look()
    result = server.handle({"id": 1, "method": "tools/call", "params": {"name": "look"}})["result"]
    assert result["isError"]
    text = result["content"][0]["text"]
    assert "free views" in text and "score" not in text.lower()


def test_prompt_explains_visual_completion_without_promising_live_scores():
    rules = prompts.PAINTER_RULES
    assert "not shown while you work" in rules
    assert "Complete the major regions and defining shapes" in rules
    assert "no useful visual" in rules
    for rule in (rules, prompts.BATCH_RULE, prompts.SCOPE_RULE, prompts.JUDGE_RULE):
        assert "score after each call" not in rule
        assert "error table" not in rule
        assert "60%" not in rule
