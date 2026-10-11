"""Optional browser gate: uv run --with playwright python -m pytest tests/test_brushlab_browser.py -q.

Uses a fake model CLI. Install Playwright's Chromium, or set PLAYWRIGHT_CHROMIUM_EXECUTABLE.
"""
import os
import threading

import pytest

from conveyor.__main__ import build_parser
from conveyor.brushlab import create_lab
from conveyor.launch import Launcher
from conveyor.painting.seeds import PEN
from conveyor.server import make_server

playwright = pytest.importorskip("playwright.sync_api")


def test_create_design_replay_branch_audition_and_publish(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("PATH", str(fake_claude.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", PEN)
    db = tmp_path / "lab.db"
    create_lab(db, {"name": "Earlier work", "brief": "Explore unusual gestures."})
    launcher = Launcher(db, build_parser)
    server = make_server(db, "127.0.0.1", 0, launcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with playwright.sync_playwright() as p:
            try:
                browser = p.chromium.launch(executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE"), headless=True)
            except playwright.Error as error:
                pytest.skip(f"Chromium unavailable: {error}")
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/brush-lab")
                page.click("#new-lab")
                page.fill("#new-name", "Echoes and broken contact")
                page.fill("#new-brief", "Remember earlier gestures and echo fragments of them.")
                page.click("#new-form button.primary")
                page.wait_for_function("document.querySelector('#lab-name').textContent==='Echoes and broken contact'")
                page.select_option("#source", "seed:pen")
                page.fill("#locks", "Keep the dry-brush gaps")
                page.check("#lock-interface")
                page.select_option("#count", "2")
                page.click("#generate")
                page.wait_for_function("document.querySelectorAll('.card .tag.ready').length===2", timeout=45000)
                page.locator("[data-inspect]").first.click()
                page.select_option("#preview-mode", "replay")
                page.locator("#frame").fill("1")
                assert "Pen state" in page.locator("#call").inner_text()
                page.wait_for_function("document.querySelector('#preview').complete && document.querySelector('#preview').naturalWidth>0")
                page.click("#branch")
                assert page.locator("#count").input_value() == "1"
                assert page.locator("#locks").input_value() == "Keep the dry-brush gaps"
                assert page.locator("#lock-interface").is_checked()
                page.fill("#painter-prompt", "Explore repeated gestures and preserve broken marks.")
                page.click("#audition")
                page.wait_for_function("document.querySelector('#auditions img')!==null", timeout=45000)
                page.click("#publish")
                page.wait_for_function("document.querySelector('#message').textContent.includes('confirm')")
                page.check("#confirm-traits")
                page.click("#publish")
                page.wait_for_selector("#published-link", state="visible")
                assert page.locator("#painter-prompt").evaluate("e=>e.readOnly")
                result = page.evaluate("fetch('/api/canvas-catalog').then(r=>r.json())")
                assert any(pair.get("published") for pair in result["pairs"])
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
                assert not errors
            finally:
                browser.close()
    finally:
        launcher.shutdown(grace=5)
        server.shutdown()
        server.server_close()
