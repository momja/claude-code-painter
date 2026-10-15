"""Disposable, model-free codec and Chromium viewport benchmark.

UV_CACHE_DIR=/tmp/uv-cache uv run --with playwright python scripts/measure_canvas_loading.py
Uses installed Chromium (override PLAYWRIGHT_CHROMIUM_EXECUTABLE). No production data.
"""
import io
import base64
import json
import os
import threading
import time
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    rng = np.random.default_rng(2026)
    sparse = Image.new('RGB', (128, 128), '#f5f1e8')
    ImageDraw.Draw(sparse).line([(12, 20), (85, 98), (110, 30)], fill='#914a60', width=4)
    y, x = np.mgrid[:128, :128]
    gradient = Image.fromarray(np.stack([x * 2, y * 2, (x + y)], -1).astype('uint8'))
    texture = Image.fromarray(np.clip(np.asarray(gradient).astype(float) * .6 + rng.normal(80, 24, (128, 128, 3)), 0, 255).astype('uint8'))
    lettering = sparse.copy()
    d = ImageDraw.Draw(lettering)
    for row in range(0, 128, 16):
        d.text((4, row), 'Conveyor paint 123', fill='#1d2a3a')
    return dict(sparse=sparse, gradient=gradient, texture=texture, lettering=lettering)


def encoded(im, fmt='PNG', **kw):
    out = io.BytesIO()
    im.save(out, format=fmt, **kw)
    return out.getvalue()


def codec_options():
    options = [('png', 'PNG', {}), ('webp_lossless', 'WEBP', {'lossless': True, 'method': 4}),
               ('webp_q85', 'WEBP', {'quality': 85}),
               ('avif_q85', 'AVIF', {'quality': 85, 'speed': 6, 'subsampling': '4:4:4'}),
               ('avif_q100', 'AVIF', {'quality': 100, 'speed': 6, 'subsampling': '4:4:4'})]
    try:
        import pillow_jxl  # optional benchmark dependency, never a production dependency
        options.append(('jxl_lossless', 'JXL', {'lossless': True, 'effort': 7}))
    except ImportError:
        pass
    return options


def browser_codecs():
    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE', '/usr/bin/chromium'))
        page = browser.new_page()
        out['browser'] = browser.version
        for label, fmt, opts in codec_options():
            data = encoded(fixtures()['texture'], fmt, **opts)
            url = 'data:image/' + fmt.lower() + ';base64,' + base64.b64encode(data).decode()
            out[label] = page.evaluate("""async url => {const t=performance.now(); const img=new Image(); img.src=url;
                try{await img.decode(); return {supported:true,decode_ms:performance.now()-t};}
                catch(e){return {supported:false};}}""", url)
        browser.close()
    return out


def codecs():
    result = {}
    for name, im in fixtures().items():
        result[name] = {}
        for label, fmt, opts in codec_options():
            start = time.perf_counter()
            for _ in range(5):
                data = encoded(im, fmt, **opts)
            encode = (time.perf_counter() - start) * 200
            start = time.perf_counter()
            for _ in range(5):
                decoded = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
            decode = (time.perf_counter() - start) * 200
            result[name][label] = dict(bytes=len(data), encode_ms=round(encode, 3), decode_ms=round(decode, 3),
                                       max_error=int(np.abs(decoded.astype(int) - np.asarray(im).astype(int)).max()))
    return result


class FixtureServer:
    def __enter__(self):
        self.requests = []
        self.tiles = [encoded(im) for im in fixtures().values()]
        owner = self
        self.html = Path(os.getenv('CANVAS_HTML', ROOT / 'src/conveyor/commons.html')).read_text().replace('\ninit();', '\n// fixture: manual initialization').encode()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                if '/tiles/' in self.path:
                    tx, ty, seq = map(int, self.path.split('/')[-3:])
                    body = owner.tiles[(tx + ty + seq) % 4]
                    owner.requests.append((self.path, len(body)))
                    mime = 'image/png'
                else:
                    body = owner.html
                    mime = 'text/html'
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        return self
    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()


SETUP = r"""() => {
S.cid='fixture'; S.cam={x:0,y:0,z:1}; cv.width=1023; cv.height=1023;
S.heads=new Map(Array.from({length:64},(_,i)=>[`${i%8},${Math.floor(i/8)}`,1]));
window.stats={start:performance.now(),first:null,center:null,complete:null,draws:0,seen:new Set()};
const original=ctx.drawImage.bind(ctx);
ctx.drawImage=(img,...args)=>{original(img,...args); stats.draws++;
 if(!stats.first) stats.first=performance.now()-stats.start;
 const key = [...images].find(([k,v])=>v===img)?.[0];
 if(key?.endsWith(','+window.targetSeq)){ stats.seen.add(key);
 if(/(?:^|\/)3,3,/.test(key) && !stats.center) stats.center=performance.now()-stats.start; }
 if(stats.seen.size===64 && !stats.complete) stats.complete=performance.now()-stats.start;
};
window.targetSeq=1; draw();
}"""


def viewport():
    out = {}
    with FixtureServer() as server, sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv('PLAYWRIGHT_CHROMIUM_EXECUTABLE', '/usr/bin/chromium'), headless=True)
        for scenario in ['cold', 'warm', 'scrub']:
            page = browser.new_page(viewport={'width':1440, 'height':1100}) if scenario != 'warm' else page
            if scenario != 'warm':
                page.goto(server.url)
                session = page.context.new_cdp_session(page)
                session.send('Network.enable')
                session.send('Network.emulateNetworkConditions', dict(offline=False, latency=200, downloadThroughput=64000, uploadThroughput=32000))
            else:
                page.reload()
            start = len(server.requests)
            page.evaluate(SETUP)
            if scenario == 'scrub':
                # Eight superseded versions while the first slow requests are in flight.
                page.evaluate("""async () => {for(let seq=2;seq<=9;seq++){
                  await new Promise(r=>setTimeout(r,30)); window.targetSeq=seq; stats.seen.clear(); stats.complete=null;
                  S.heads=new Map([...S.heads].map(([k])=>[k,seq])); draw();}}""")
            page.wait_for_function('stats.complete!==null', timeout=180000)
            out[scenario] = page.evaluate('({first_ms:stats.first,center_ms:stats.center,complete_ms:stats.complete,draw_calls:stats.draws,decoded:images.size})')
            out[scenario].update(requests=len(server.requests)-start, bytes=sum(n for _,n in server.requests[start:]))
        browser.close()
    return out


def preview_cost():
    from conveyor.commons.tiles import create_canvas
    from conveyor.commons.export import export_png
    from conveyor.commons.preview import PreviewCache
    from conveyor.store import connect
    with tempfile.TemporaryDirectory() as directory:
        db=Path(directory)/'fixture.db'
        cid=create_canvas(db,'Preview',frame=(1024,1024))['id']
        conn=connect(db)
        tiles=list(fixtures().values())
        for version in range(9):
            seq=conn.execute("INSERT INTO canvas_ops(canvas_id,ts,tool,status) VALUES (?,0,'paint','ok')",(cid,)).lastrowid
            for y in range(8):
                for x in range(8):
                    conn.execute('INSERT INTO canvas_tiles VALUES (?,?,?,?,?)',(cid,x,y,seq,encoded(tiles[(x+y+version)%4])))
                    conn.execute('INSERT OR REPLACE INTO canvas_heads VALUES (?,?,?,?)',(cid,x,y,seq))
        conn.commit()
        result={}
        cache=PreviewCache()
        for label,f in [('old',lambda:export_png(conn,cid,900)),('new',lambda:cache.get(conn,cid,seq))]:
            values=[]
            for _ in range(3):
                t=time.perf_counter(); data=f(); values.append(round((time.perf_counter()-t)*1000,2))
            result[label]={'ms_per_call':values,'bytes':len(data)}
        conn.close()
        return result


if __name__ == '__main__':
    print(json.dumps(dict(codecs=codecs(), browser_codecs=browser_codecs(), preview=preview_cost(), viewport={} if os.getenv('CODECS_ONLY') else viewport()), indent=2))
