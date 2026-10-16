"""Optional responsive UI gate; no model calls.

uv run --with playwright python -m pytest tests/test_mobile_browser.py -q
Set PLAYWRIGHT_CHROMIUM_EXECUTABLE to use a system Chromium.
"""
import io
import os
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from conveyor.__main__ import build_parser
from conveyor.brushlab import create_lab
from conveyor.commons import bench
from conveyor.commons.tiles import create_canvas
from conveyor.launch import Launcher
from conveyor.painting.instrument import all_niches
from conveyor.painting.seeds import ROUND
from conveyor.server import make_server
from conveyor.store import Store, connect

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE"), headless=True)
        except playwright.Error as error:
            pytest.skip(f"Chromium unavailable: {error}")
        yield browser
        browser.close()


@pytest.fixture
def mobile_app(tmp_path):
    db = tmp_path / "mobile.db"
    with_image = io.BytesIO()
    Image.new("RGB", (64, 80), "#556677").save(with_image, "PNG")
    store = Store(db, run_name="Responsive browser fixture " + "long-name-" * 12, config={"width": 64})
    try:
        image = store.artifact(with_image.getvalue())
        for oid, node, genome in (("instrument", "instrument", {"source": ROUND}),
                                  ("prompt", "painter", {"prompt": "Paint big."})):
            store.organism({"id": oid, "node": node, "created": time.time(), "genome": genome,
                            "summary": "LongUnbrokenSummary" * 8, "viable": 1,
                            "niche": all_niches()[0] if node == "instrument" else None, "sheet": image})
        store.emit("graph", nodes=[{"name": "instrument", "archive": True, "all_niches": all_niches()},
                                  {"name": "painter", "archive": False}])
        store.start_session("painting", node="painting", organism_id="instrument", purpose="paint",
                            model="fixture", effort="low", request={}, dir=None)
        store.stroke("painting", 1, {"tool": "stroke", "args": {"x": 10}, "i": 1}, image)
        store.update_session("painting", status="ok", ended=time.time())
        for oid, partner, node in (("instrument", "prompt", "instrument"), ("prompt", "instrument", "painter")):
            store.evaluation({"id": oid + "-eval", "node": node, "organism_id": oid, "partner_id": partner,
                              "score": .7, "viable": 1, "started": time.time() - 1, "ended": time.time(),
                              "session_id": "painting", "artifacts": {"painting": image},
                              "details": {"instrument_id": "instrument", "prompt_id": "prompt"}})
        store.emit("run_finished")
        store.flush()
    finally:
        store.close()
    canvas = create_canvas(db, "Mobile canvas", viewport=256, max_calls=20, frame=(768, 512))
    create_lab(db, {"name": "Mobile lab", "brief": "Explore big gestures."})
    benchmark = bench.create(db, {"name": "Mobile benchmark", "canvas": {"viewport": 256, "max_calls": 10},
                                 "agents": [{"x": 0, "y": 0}], "variants": {"fixture": "claude"}})
    # Save a populated result without starting the launcher's model worker watcher.
    conn = connect(db)
    conn.execute("UPDATE canvas_agents SET status='finished'")
    conn.execute("UPDATE benchmarks SET status='finished'")
    conn.commit()
    conn.close()
    launcher = Launcher(db, build_parser)
    server = make_server(db, "127.0.0.1", 0, launcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", canvas["id"], benchmark["id"]
    finally:
        launcher.shutdown(grace=1)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def fits(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), page.evaluate("""() =>
        [...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().right > innerWidth + 1)
          .map(e => ({tag:e.tagName,id:e.id,cls:e.className,parent:e.parentElement.className,
            width:e.getBoundingClientRect().width,grid:getComputedStyle(e).gridTemplateColumns})).slice(0,16)""")


@pytest.mark.parametrize("width,height", [(320, 640), (390, 844), (768, 1024), (1440, 1000)])
def test_populated_interfaces_fit_viewport(browser, mobile_app, width, height):
    base, cid, bid = mobile_app
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=width < 1000)
    page = context.new_page()
    page.set_default_timeout(7000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        for path, ready in (("/", ".archive .cell"), ("/studio", "#graph .card"),
                            (f"/canvas?canvas={cid}", "#pen:not([hidden])"),
                            (f"/bench?bench={bid}", ".run"), ("/brush-lab", "#lab-name")):
            page.goto(base + path)
            page.wait_for_selector(ready, state="attached")
            fits(page)
        page.click("#new-lab")
        fits(page)
        assert page.locator("dialog[open]").evaluate("e => e.scrollHeight > 0 && e.clientHeight <= innerHeight")
        page.keyboard.press("Escape")
        page.goto(base + "/bench")
        page.click("#new")
        page.wait_for_selector("#map")
        fits(page)
        page.click("#painters-summary")
        fits(page)
        page.click("#check")
        page.wait_for_selector("#plan:not([hidden])")
        fits(page)
        # The standalone viewer's archive can be rendered without fetching its CDN SQLite engine.
        page.route("https://cdnjs.cloudflare.com/**", lambda route: route.abort())
        page.route(base + "/viewer.html", lambda route: route.fulfill(path=ROOT / "viewer.html", content_type="text/html"))
        page.goto(base + "/viewer.html")
        page.evaluate("""() => {
          state.orgs=[{id:'fixture',niche:'stateless/scalar/short',viable:1,summary:'A saved instrument'}];
          state.bestByOrg={}; renderArchive(); renderOrgs();
          document.querySelector('#archSec').hidden=false; document.querySelector('#orgSec').hidden=false;
        }""")
        fits(page)
        if width < 760:
            assert page.locator("#archive-scroll").evaluate("e => e.scrollWidth > e.clientWidth")
        assert not errors
    finally:
        context.close()


@pytest.mark.parametrize("width,height", [(320, 640), (390, 844)])
def test_mobile_controls_drawer_list_dialog_and_touch_sketch(browser, mobile_app, width, height):
    base, cid, _ = mobile_app
    context = browser.new_context(viewport={"width": width, "height": height}, has_touch=True, is_mobile=True)
    page = context.new_page()
    page.set_default_timeout(7000)
    try:
        page.goto(base)
        page.locator(".archive [data-org]").first.click()
        page.wait_for_selector("#drawer.open")
        assert page.locator("#drawer-close").evaluate("e => e === document.activeElement")
        assert page.locator("#main").evaluate("e => e.inert")
        fits(page)
        page.keyboard.press("Shift+Tab")
        assert page.evaluate("document.querySelector('#drawer').contains(document.activeElement)")
        page.keyboard.press("Escape")
        assert page.locator("#drawer").evaluate("e => e.inert")
        page.click("#new-run")
        page.wait_for_selector("#nr[open]")
        fits(page)
        page.click("#nr-cancel")

        page.goto(base + "/studio")
        page.wait_for_selector("#node-list .card")
        assert page.locator("#view-mode").input_value() == "list"
        page.fill("#search", "instrument:instrument")
        assert page.locator("#node-list .card").count() == 1
        page.locator("#node-list .card").click()
        assert page.locator("#instrument").input_value() == "instrument"
        page.select_option("#prompt", "prompt")
        page.fill("#brief", "A blue mountain")
        assert page.locator("#submit").is_enabled()
        page.select_option("#view-mode", "graph")
        assert page.evaluate("scale > .1"), "Graph zoom must use the visible graph area when List is the default"
        page.click("#fit")
        fits(page)

        page.goto(base + f"/canvas?canvas={cid}")
        page.wait_for_selector("#pen:not([hidden])")
        page.click("#choose-painters")
        page.wait_for_selector("#painters-dialog[open] .pcard")
        fits(page)
        page.click("#painters-cancel")
        page.click("#new-canvas")
        fits(page)
        page.keyboard.press("Escape")
        page.click("#pen")
        page.locator("#wrap").scroll_into_view_if_needed()
        box = page.locator("#wrap").bounding_box()
        session = context.new_cdp_session(page)
        x, y = box["x"] + 50, box["y"] + 50
        session.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
        session.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x + 40, "y": y + 40}]})
        session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        page.wait_for_function("S.sketches.length === 1 && S.pending.length === 0")
        page.click("#undo")
        page.wait_for_function("sketchLines().length === 0")
        fits(page)
    finally:
        context.close()
