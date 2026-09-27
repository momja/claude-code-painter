import numpy as np
import pytest

from conveyor.painting.canvas import Canvas
from conveyor.painting.canvas import CanvasError
from conveyor.painting.canvas import load_target
from conveyor.painting.canvas import parse_color
from conveyor.painting.critic import Critic


def test_one_call_cannot_cover_more_than_its_area():
    c = Canvas(160, 128)
    c.begin_call()
    laid = sum(c.dab(x, y, 10, "#000000") for x in range(0, 128, 8) for y in range(0, 160, 8))
    assert c.dry and c.area_used <= c.area_cap
    assert laid < 16 * 20
    touched = (np.abs(c.img - Canvas(160, 128).img).max(axis=-1) > 0.02).sum()
    assert touched <= c.area_cap * 1.05
    c.begin_call()
    assert not c.dry and c.dab(64, 80, 3, "#000000")


def test_stamp_smudge_and_pick():
    c = Canvas(64, 64)
    c.begin_call()
    assert c.stamp(20, 20, np.ones((5, 9)), "#ff0000")
    assert c.pick(20, 20) == pytest.approx((1.0, 0.0, 0.0))
    with pytest.raises(CanvasError):
        c.stamp(20, 20, np.ones((200, 3)), "#ff0000")
    c.begin_call()
    c.smudge(20, 20, 3, 6, 0, strength=1.0)
    assert c.pick(26, 20)[0] > 0.9  # red dragged right


def test_parse_color():
    assert parse_color("#fff") == (1.0, 1.0, 1.0)
    assert parse_color([255, 0, 0]) == (1.0, 0.0, 0.0)
    assert parse_color((0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5)
    with pytest.raises(CanvasError):
        parse_color("nope")


def test_critic_prefers_the_target_to_paper():
    t = load_target("self_portrait", width=64)
    critic = Critic()
    assert critic.score(t.image, t)["total"] > 0.95
    assert critic.score(Canvas(t.height, t.width).img, t)["total"] < 0.7
