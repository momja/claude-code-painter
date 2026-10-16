"""Real Chromium gate. Install playwright; missing browser is a failure, not a skip.

UV_CACHE_DIR=/tmp/uv-cache uv run --with playwright --group dev pytest tests/test_canvas_loading_browser.py
"""
import importlib.util
import os
from pathlib import Path

import pytest

playwright = pytest.importorskip('playwright.sync_api')
spec = importlib.util.spec_from_file_location('measurement', Path(__file__).parents[1] / 'scripts/measure_canvas_loading.py')
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


@pytest.mark.parametrize("bitmap", [True, False])
def test_viewport_priority_cancellation_replay_switch_and_cache(bitmap):
    with measurement.FixtureServer() as server, playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE', '/usr/bin/chromium'))
        page = browser.new_page(viewport={'width':1440, 'height':1100})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(server.url)
        if not bitmap:
            page.evaluate('window.createImageBitmap=undefined')
        cdp = page.context.new_cdp_session(page)
        cdp.send('Network.enable')
        cdp.send('Network.emulateNetworkConditions', dict(offline=False, latency=150, downloadThroughput=200000, uploadThroughput=200000))
        page.evaluate(measurement.SETUP)
        page.wait_for_function('loading.size===6')
        first = page.evaluate('[...loading.keys()]')
        assert all(any(f'/{x},{y},' in k for x in range(2, 6) for y in range(2, 6)) for k in first)
        # Pan while six requests are outstanding; no old completion may enter the cache.
        page.evaluate('S.cam.x=2048; S.heads=new Map([["16,0",2]]); draw()')
        page.wait_for_function('images.has("fixture/16,0,2") && loading.size===0')
        assert page.evaluate('[...images.keys()]') == ['fixture/16,0,2']
        # Zoom out, then replay backwards. Future tiles are never fallback for older ops.
        page.evaluate('''S.cam={x:0,y:0,z:.5}; S.mode='replay';
          S.tracks=new Map(); S.history={ops:[[1],[2]]}; S.byTile=new Map([['0,0',[1,2]]]); S.idx=1; draw();''')
        page.wait_for_function('images.has("fixture/0,0,2")')
        page.evaluate('S.idx=0; draw()')
        assert page.evaluate('fallback(0,0,1)===null')
        page.wait_for_function('images.has("fixture/0,0,1")')
        # Identical coordinates and sequence on another canvas cannot alias.
        page.evaluate('S.cid="other"; shown.clear(); draw()')
        page.wait_for_function('images.has("other/0,0,1")')
        assert any('/other/tiles/0/0/1' in path for path, _ in server.requests)
        # Keep hundreds of versions, verifying eviction and the current working set.
        page.evaluate('''for(let i=10;i<600;i++) images.set(`old/0,0,${i}`,new Image()); trimImages();''')
        assert page.evaluate('images.size') == 256
        assert page.evaluate('images.has("other/0,0,1")')
        count = len(server.requests)
        page.evaluate('draw(); draw(); draw()')
        page.wait_for_timeout(100)
        assert len(server.requests) == count
        assert not errors
        # Reload drops decoded objects but must reuse the immutable HTTP response.
        page.reload()
        page.evaluate("S.cid='other'; S.cam={x:0,y:0,z:1}; S.heads=new Map([['0,0',1]]); draw()")
        page.wait_for_function('images.has("other/0,0,1")')
        assert len(server.requests) == count
        browser.close()


def test_benchmark_poll_preserves_connected_image():
    html = (measurement.ROOT / 'src/conveyor/bench.html').read_text()
    # Exercise show() with its real DOM, without starting benchmark polling or a model.
    html = html.replace('\ninit();', '\n// test initialization')
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE', '/usr/bin/chromium'))
        page = browser.new_page()
        page.route('**/*', lambda route: route.fulfill(body=html, content_type='text/html'))
        page.goto('http://fixture.test/bench')
        page.evaluate('''window.fixture={id:'bench', name:'Test', status:'done', order:'parallel', seed:1, cost:0, spec:{canvas:{}},
          runs:[{canvas_id:'one',label:'One',variant:'v',state:'done',models:[],last_op:7,cost:0,paints:0,seconds:0,input_tokens:0,output_tokens:0,statuses:{},by_model:[]}]};
          show(fixture); window.original=document.querySelector('#runs img'); show(fixture);''')
        assert page.evaluate("original===document.querySelector('#runs img') && original.isConnected")
        assert page.locator('#runs img').get_attribute('src').endswith('/preview.png?seq=7')
        page.evaluate('fixture.runs[0].last_op=8; show(fixture)')
        assert page.evaluate("original===document.querySelector('#runs img')")
        assert page.locator('#runs img').get_attribute('src').endswith('seq=8')
        assert page.locator('#runs a[download]').get_attribute('href').endswith('/export.png')
        browser.close()
