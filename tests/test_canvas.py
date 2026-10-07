import numpy as np
import pytest

from conveyor.painting.canvas import Canvas
from conveyor.painting.canvas import CanvasError
from conveyor.painting.canvas import error_table
from conveyor.painting.canvas import gridded_png
from conveyor.painting.canvas import load_target
from conveyor.painting.canvas import parse_color
from conveyor.painting.canvas import view_patch
from conveyor.painting.canvas import view_png
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


def test_the_brush_limit_is_a_radius_one_call_can_lay():
    for h, w in ((64, 64), (160, 128), (640, 512)):
        c = Canvas(h, w)
        c.begin_call()
        assert c.dab(w / 2, h / 2, c.max_radius, "#000000") and not c.dry


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


def test_view_cuts_a_square_window_and_zooms_it():
    c = Canvas(640, 512)
    assert c.view(10, 10, 64).rect == (0, 0, 64, 64)  # kept inside the canvas near a corner
    v = c.view(256, 320, 33)
    assert v.rect == (240, 304, 273, 337) and v.img.shape == (33, 33, 3) and v.span == 33
    assert c.view(0, 0, 9999).rect == (0, 0, 512, 512)  # a span is clamped to the canvas's smaller side
    assert c.view(0, 0, 512, 64).scale == 2  # zoom is clamped so the picture stays within 1024 px a side
    assert c.view(0, 0, 16, 3).scale == 3 and c.view(0, 0, 16, 0).scale == 1


def test_views_and_whole_canvas_pictures_keep_their_scale():
    import io
    from PIL import Image

    def size(png):
        return Image.open(io.BytesIO(png)).size

    c = Canvas(640, 512)
    assert size(gridded_png(c.img)) == (568, 696)  # 512 x 640 at 1 px per canvas px, 28 px margins
    assert size(gridded_png(Canvas(160, 128).img)) == (568, 696)  # 128 wide at 4 px per canvas px
    v = c.view(256, 320, 64)
    assert size(view_png(v)) == (300, 300)  # 64 px window at 4x, 22 px margins


def test_error_table_can_follow_a_window():
    t = load_target("self_portrait", width=128)
    paper = Canvas(t.height, t.width).img
    whole = error_table(paper, t)
    assert "Error per 16 x 16 px region (0 is a perfect match)" in whole
    assert len(whole.splitlines()) == 2 + t.height // 16
    assert view_patch(64) == 8 and view_patch(128) == 16 and view_patch(33) == 8
    window = error_table(paper, t, rect=(32, 48, 96, 112), patch=view_patch(64))
    assert "Error per 8 x 8 px region in x 32-96, y 48-112" in window
    assert len(window.splitlines()) == 10  # header, column labels, and eight 8 px rows


def test_load_target_handles_any_image(tmp_path):
    from PIL import Image

    from conveyor.painting.canvas import MAX_TARGET_HEIGHT

    rgba = tmp_path / "logo.png"
    Image.new("RGBA", (300, 200), (0, 0, 0, 0)).save(rgba)  # fully transparent: flattens to white, not black
    t = load_target(rgba, width=64)
    assert t.image.shape == (32, 64, 3) and t.image.min() > 0.99

    tall = tmp_path / "tall.png"
    Image.new("RGB", (50, 20000), (10, 200, 10)).save(tall)  # a scroll: cropped, not a 25000 row canvas
    t = load_target(tall, width=128)
    assert t.height == MAX_TARGET_HEIGHT and t.height % t.patch == 0

    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (100, 1), (1, 2, 3)).save(tiny)  # absurdly flat: still one patch row
    assert load_target(tiny, width=64).height == 16
