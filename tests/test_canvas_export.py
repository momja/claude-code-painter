"""Paint-history exports: committed pixels, fixed crops, spaced stages, limits, HTTP and the command line."""

import io
import json
import urllib.request

import numpy as np
import pytest
from PIL import Image
from PIL import ImageSequence

from conveyor.__main__ import build_parser
from conveyor.commons import export
from conveyor.commons.tiles import PAPER_RGB
from conveyor.commons.tiles import TILE
from conveyor.commons.tiles import SharedCanvas
from conveyor.commons.tiles import create_canvas
from conveyor.store import connect
from test_commons import served  # noqa: F401 - fixture
from test_launch import call as http

PAPER = tuple(int(c) for c in PAPER_RGB)
RED = (200, 40, 20)
BLUE = (30, 70, 200)


def paint(canvas, color, tx=0, ty=0):
    with canvas.transaction():
        return canvas.record(agent_id=None, tool="paint", status="painted", x=tx * TILE, y=ty * TILE,
                             tiles={(tx, ty): np.full((TILE, TILE, 3), color, dtype=np.uint8)})


def look(canvas):
    return canvas.record(agent_id=None, tool="look", status="viewed", x=0, y=0)


def opened(data):
    return Image.open(io.BytesIO(data))


@pytest.fixture
def history(tmp_path):
    db = tmp_path / "canvas.db"
    row = create_canvas(db, "Shore line", viewport=128, frame=(256, 128))
    canvas = SharedCanvas(db, row["id"])
    first = paint(canvas, RED)
    second = paint(canvas, BLUE, tx=1)
    last = paint(canvas, BLUE)
    conn = connect(db, readonly=True)
    yield db, row, canvas, conn, [first, second, last]
    conn.close()
    canvas.close()


def test_gif_replays_pixels_from_blank_paper_to_the_result_and_loops(history):
    _, row, _, conn, ops = history
    with opened(export.export_gif(conn, row["id"], frames=4, fps=5)) as gif:
        assert gif.format == "GIF" and gif.info["loop"] == 0
        frames = [im.convert("RGB") for im in ImageSequence.Iterator(gif)]
        assert len(frames) == 4 and {im.size for im in frames} == {(256, 128)}
        assert [im.getpixel((40, 40)) for im in frames] == [PAPER, RED, RED, BLUE]
        assert [im.getpixel((170, 40)) for im in frames] == [PAPER, PAPER, BLUE, BLUE]
        gif.seek(0)
        assert gif.info["duration"] == 500
        gif.seek(1)
        assert gif.info["duration"] == 200
        gif.seek(3)
        assert gif.info["duration"] == 1500
    with opened(export.export_gif(conn, row["id"], side=128, seq=ops[0])) as gif:
        assert gif.size == (128, 64) and gif.n_frames == 2
        gif.seek(1)
        assert gif.convert("RGB").getpixel((100, 20)) == PAPER
    with opened(export.export_png(conn, row["id"], seq=0)) as png:
        assert png.getpixel((40, 40)) == PAPER


def test_gif_replaces_erased_paint_and_preserves_unchanged_call_delays(tmp_path):
    db = tmp_path / "canvas.db"
    row = create_canvas(db, "Erased", viewport=128)
    canvas = SharedCanvas(db, row["id"])
    conn = connect(db, readonly=True)
    try:
        paint(canvas, RED)
        look(canvas)
        paint(canvas, PAPER)
        with opened(export.export_gif(conn, row["id"], frames=4, fps=30)) as gif:
            frames = [im.convert("RGB") for im in ImageSequence.Iterator(gif)]
            assert [im.getpixel((40, 40)) for im in frames] == [PAPER, RED, PAPER]
            gif.seek(1)
            assert gif.info["duration"] == 70  # the identical look frame keeps its share of the playback time
            gif.seek(2)
            assert gif.info["duration"] == 1500
    finally:
        conn.close()
        canvas.close()


def test_stages_sample_this_canvass_call_ranks_not_database_sequence_numbers(tmp_path):
    db = tmp_path / "canvas.db"
    a = create_canvas(db, "A", viewport=128)
    b = create_canvas(db, "B", viewport=128)
    canvas, other = SharedCanvas(db, a["id"]), SharedCanvas(db, b["id"])
    conn = connect(db, readonly=True)
    try:
        first = paint(canvas, RED)
        for _ in range(20):
            look(other)
        second = look(canvas)  # calls that do not paint still count, as in replay
        third = paint(canvas, BLUE)
        for _ in range(20):
            look(other)
        last = look(canvas)
        assert export.stages(conn, a["id"], frames=3) == [(0, 0), (second, 2), (last, 4)]
        assert export.stages(conn, a["id"], frames=60) == [(0, 0), (first, 1), (second, 2), (third, 3), (last, 4)]
        assert export.stages(conn, a["id"], frames=3, seq=third) == [(0, 0), (second, 2), (third, 3)]
        assert export.stages(conn, a["id"], frames=3, seq=0) == [(0, 0)]
    finally:
        conn.close()
        canvas.close()
        other.close()


def test_progress_sheet_has_spaced_panels_with_the_same_crop_and_exact_op_metadata(history):
    _, row, _, conn, ops = history
    with opened(export.export_sheet(conn, row["id"], side=256, frames=4, columns=2)) as sheet:
        assert sheet.format == "PNG" and sheet.mode == "RGB"
        assert sheet.size == (560, 416)
        assert json.loads(sheet.info["op_sequences"]) == [0, *ops]
        assert json.loads(sheet.info["canvas_window"]) == [0, 0, 256, 128]
        assert sheet.info["canvas"] == row["id"]
        # Panels begin at (16, 64), then (288, 64), (16, 240), (288, 240).
        assert [sheet.getpixel((x + 40, y + 40)) for x, y in [(16, 64), (288, 64), (16, 240), (288, 240)]] == [
            PAPER, RED, RED, BLUE]
        assert sheet.getpixel((16 + 170, 240 + 40)) == BLUE
    with opened(export.export_sheet(conn, row["id"], frames=9, seq=ops[0])) as sheet:
        assert json.loads(sheet.info["op_sequences"]) == [0, ops[0]]
        assert sheet.size == (560, 240)  # no empty columns just to reach the requested count


def test_unbounded_histories_keep_negative_coordinates_and_a_fixed_final_crop(tmp_path):
    db = tmp_path / "canvas.db"
    row = create_canvas(db, "Open", viewport=128)
    canvas = SharedCanvas(db, row["id"])
    conn = connect(db, readonly=True)
    try:
        first = paint(canvas, RED, tx=-1, ty=-1)
        paint(canvas, BLUE, tx=1, ty=0)
        with opened(export.export_gif(conn, row["id"], frames=3)) as gif:
            assert gif.size == (384, 256)
            gif.seek(1)
            assert gif.convert("RGB").getpixel((40, 40)) == RED
            assert gif.convert("RGB").getpixel((300, 170)) == PAPER
            gif.seek(2)
            assert gif.convert("RGB").getpixel((40, 40)) == RED
            assert gif.convert("RGB").getpixel((300, 170)) == BLUE
        with opened(export.export_gif(conn, row["id"], seq=first)) as gif:
            assert gif.size == (128, 128)  # later paint cannot expand a historical export's crop
        with opened(export.export_sheet(conn, row["id"], frames=3)) as sheet:
            assert json.loads(sheet.info["canvas_window"]) == [-128, -128, 256, 128]
    finally:
        conn.close()
        canvas.close()


def test_empty_and_nonpainting_histories_export_blank_paper(tmp_path):
    db = tmp_path / "canvas.db"
    row = create_canvas(db, "Empty", viewport=128)
    canvas = SharedCanvas(db, row["id"])
    conn = connect(db, readonly=True)
    try:
        for nonpainting in (False, True):
            op = look(canvas) if nonpainting else None
            with opened(export.export_gif(conn, row["id"], seq=op)) as gif:
                assert gif.size == (128, 128) and gif.n_frames == 1
                assert gif.convert("RGB").getpixel((20, 20)) == PAPER
            with opened(export.export_sheet(conn, row["id"], seq=op)) as sheet:
                assert json.loads(sheet.info["op_sequences"]) == ([0] if op is None else [0, op])
                assert sheet.getpixel((36, 84)) == PAPER
    finally:
        conn.close()
        canvas.close()


@pytest.mark.parametrize("renderer", [export.export_gif, export.export_sheet])
def test_live_export_pins_the_end_before_rendering_and_does_not_block_a_painter(tmp_path, monkeypatch, renderer):
    db = tmp_path / "canvas.db"
    row = create_canvas(db, "Live", viewport=128)
    canvas = SharedCanvas(db, row["id"])
    first = paint(canvas, RED)
    conn = connect(db, readonly=True)
    compose = export.compose
    calls = []

    def during_export(conn, cid, box, scale, seq):
        calls.append(seq)
        if len(calls) == 1:
            paint(canvas, BLUE, tx=3)
            paint(canvas, BLUE)
        return compose(conn, cid, box, scale, seq)

    monkeypatch.setattr(export, "compose", during_export)
    try:
        with opened(renderer(conn, row["id"], frames=2)) as image:
            assert calls == [0, first]
            if image.format == "GIF":
                assert image.size == (128, 128)
                image.seek(1)
                assert image.convert("RGB").getpixel((40, 40)) == RED
            else:
                assert json.loads(image.info["canvas_window"]) == [0, 0, 128, 128]
                assert json.loads(image.info["op_sequences"]) == [0, first]
    finally:
        conn.close()
        canvas.close()


@pytest.mark.parametrize("renderer, options", [
    (export.export_gif, {"side": 8}),
    (export.export_gif, {"side": 2049}),
    (export.export_gif, {"frames": 1}),
    (export.export_gif, {"frames": 241}),
    (export.export_gif, {"fps": 0}),
    (export.export_gif, {"fps": 31}),
    (export.export_gif, {"seq": -1}),
    (export.export_gif, {"seq": 2**65}),
    (export.export_sheet, {"side": 127}),
    (export.export_sheet, {"side": 1025}),
    (export.export_sheet, {"frames": 1}),
    (export.export_sheet, {"frames": 25}),
    (export.export_sheet, {"columns": 0}),
    (export.export_sheet, {"columns": 7}),
    (export.export_sheet, {"seq": -1}),
    (export.export_sheet, {"seq": 2**65}),
])
def test_bad_options_fail_before_rendering(history, monkeypatch, renderer, options):
    _, row, _, conn, _ = history
    monkeypatch.setattr(export, "compose", lambda *a: pytest.fail("should not render invalid options"))
    with pytest.raises(ValueError):
        renderer(conn, row["id"], **options)
    with pytest.raises(KeyError):
        renderer(conn, "missing")


def test_large_exports_are_rejected_before_allocating_images(tmp_path, monkeypatch):
    db = tmp_path / "canvas.db"
    row = create_canvas(db, "Large", viewport=128, frame=(8192, 8192))
    canvas = SharedCanvas(db, row["id"])
    for _ in range(23):
        look(canvas)
    conn = connect(db, readonly=True)
    monkeypatch.setattr(export, "compose", lambda *a: pytest.fail("should check size before composing"))
    try:
        with pytest.raises(ValueError, match="Animation is too large"):
            export.export_gif(conn, row["id"], side=2048, frames=24)
        with pytest.raises(ValueError, match="Progress sheet is too large"):
            export.export_sheet(conn, row["id"], side=1024, frames=24, columns=1)
    finally:
        conn.close()
        canvas.close()


def test_history_downloads_over_http_have_the_right_type_name_cache_and_validation(served):  # noqa: F811
    base, db, _ = served
    row = create_canvas(db, "Shore line", viewport=128, frame=(256, 128))
    canvas = SharedCanvas(db, row["id"])
    try:
        first = paint(canvas, RED)
        paint(canvas, BLUE, tx=1)
        for route, ctype, name in [("export.gif", "image/gif", f"Shore-line-op{first}.gif"),
                                   ("progress.png", "image/png", f"Shore-line-op{first}-progress.png")]:
            with urllib.request.urlopen(f"{base}/api/canvases/{row['id']}/{route}?seq={first}&frames=3") as response:
                assert response.headers["Content-Type"] == ctype
                assert name in response.headers["Content-Disposition"]
                assert response.headers["Cache-Control"] == "no-store"
                with opened(response.read()) as image:
                    if ctype == "image/gif":
                        image.seek(image.n_frames - 1)
                        assert image.convert("RGB").getpixel((170, 40)) == PAPER
                    else:
                        assert json.loads(image.info["op_sequences"]) == [0, first]
            assert http(base, "GET", f"/api/canvases/missing/{route}")[0] == 404
            assert http(base, "GET", f"/api/canvases/{row['id']}/{route}?frames=1")[0] == 400
            assert http(base, "GET", f"/api/canvases/{row['id']}/{route}?seq=-1")[0] == 400
        assert http(base, "GET", f"/api/canvases/{row['id']}/export.gif?fps=0")[0] == 400
        assert http(base, "GET", f"/api/canvases/{row['id']}/export.gif?frames=lots")[0] == 400
        assert http(base, "GET", f"/api/canvases/{row['id']}/progress.png?columns=7")[0] == 400
        with (urllib.request.urlopen(f"{base}/api/canvases/{row['id']}/export.gif?seq=0") as response,
              opened(response.read()) as image):
            assert image.n_frames == 1 and image.convert("RGB").getpixel((40, 40)) == PAPER
    finally:
        canvas.close()


def test_cli_exports_gif_and_sheet_and_infers_gif_from_the_output_name(history, tmp_path, monkeypatch):
    db, row, _, _, ops = history
    parser = build_parser()
    monkeypatch.chdir(tmp_path)
    for options, expected, format in [
        (["--format", "gif", "--frames", "3", "--fps", "5"], f"Shore-line-op{ops[1]}.gif", "GIF"),
        (["--format", "sheet", "--frames", "3", "--columns", "2"], f"Shore-line-op{ops[1]}-progress.png", "PNG"),
        (["-o", "paint.gif"], "paint.gif", "GIF"),
    ]:
        args = parser.parse_args(["export", row["id"], "--db", str(db), "--seq", str(ops[1]), *options])
        args.func(args)
        with Image.open(tmp_path / expected) as image:
            assert image.format == format
            if format == "PNG":
                assert json.loads(image.info["op_sequences"])[-1] == ops[1]
    args = parser.parse_args(["export", row["id"], "--db", str(db), "--format", "gif", "-o", "wrong.png"])
    with pytest.raises(SystemExit, match=r"needs a \.gif"):
        args.func(args)
    args = parser.parse_args(["export", row["id"], "--db", str(db), "--format", "sheet", "--columns", "0"])
    with pytest.raises(SystemExit, match="columns"):
        args.func(args)


def test_export_filenames_remain_safe():
    assert export.filename('" /?\n') == "canvas.png"
    assert export.filename("Shore line", seq=0, kind="gif") == "Shore-line-op0.gif"
    assert export.filename("Shore line", kind="sheet") == "Shore-line-progress.png"
    with pytest.raises(ValueError):
        export.filename("C", kind="other")
