"""The critic's per-scale report is the same arithmetic as its score."""

import io

import pytest
from PIL import Image

from conveyor.painting.agent import Strategy
from conveyor.painting.agent import paint
from conveyor.painting.canvas import TARGETS_DIR
from conveyor.painting.canvas import load_target
from conveyor.painting.critic import SCALES
from conveyor.painting.critic import Critic
from conveyor.painting.toolkit import initial_toolkit


@pytest.fixture
def painted():
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=48)
    return paint(Strategy(), initial_toolkit(), target, seed=1), target


def test_report_adds_up_to_the_score(painted):
    canvas, target = painted
    critic = Critic()
    stored: dict[str, bytes] = {}

    def store(png: bytes) -> str:
        name = f"a{len(stored)}"
        stored[name] = png
        return name

    report = critic.report(canvas, target, store)
    score = critic.score(canvas, target)
    assert report["pixel"] == pytest.approx(score["pixel"], abs=1e-4)
    assert report["style"] == pytest.approx(score["style"], abs=1e-4)
    assert report["total"] == pytest.approx(score["total"], abs=1e-4)
    assert report["pixel"] == pytest.approx(sum(s["contribution"] for s in report["scales"]), abs=1e-4)
    assert sum(w for _, w in SCALES) == pytest.approx(1.0)

    assert [(s["factor"], s["size"]) for s in report["scales"]] == [(1, [48, 48]), (2, [24, 24]), (4, [12, 12])]
    for s in report["scales"]:
        assert s["score"] == pytest.approx(max(0.0, 1 - s["rmse"] / report["rmse_ceiling"]), abs=1e-4)
        for key in ("target", "canvas", "error"):
            assert Image.open(io.BytesIO(stored[s[key]])).size == tuple(s["size"])


def test_report_without_a_recorder_still_scores(painted):
    canvas, target = painted
    report = Critic().report(canvas, target, lambda png: None)
    assert all(s["error"] is None for s in report["scales"])
    assert report["total"] > 0
