"""Preview derivatives preserve historical pixels and bound cache memory."""
import io
import threading
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from conveyor.commons.export import export_png
from conveyor.commons.preview import PreviewCache
from conveyor.commons.tiles import create_canvas, encode_tile
from conveyor.server import Views, make_server
from conveyor.store import connect


def add_version(conn, cid, color):
    seq = conn.execute("INSERT INTO canvas_ops(canvas_id,ts,tool,status) VALUES (?,0,'paint','ok')", (cid,)).lastrowid
    data = encode_tile(np.full((128,128,3), color, dtype=np.uint8))
    conn.execute('INSERT INTO canvas_tiles VALUES (?,0,0,?,?)', (cid,seq,data))
    conn.execute('INSERT OR REPLACE INTO canvas_heads VALUES (?,0,0,?)', (cid,seq))
    conn.commit()
    return seq


def test_snapshot_cache_eviction_validation_and_exports(tmp_path):
    db = tmp_path / 'preview.db'
    cid = create_canvas(db, 'Preview')['id']
    conn = connect(db)
    first = add_version(conn,cid,30)
    views = Views(db)
    parts = ['api','canvases',cid,'preview.png']
    with patch('conveyor.commons.preview.export_png', wraps=export_png) as encode:
        one, kind = views.canvas_get(parts, {'seq':str(first)})
        assert kind == 'png'
        assert views.canvas_get(parts, {'seq':str(first)})[0] is one
        assert encode.call_count == 1
        second = add_version(conn,cid,190)
        two = views.canvas_get(parts, {'seq':str(second)})[0]
        assert two != one
        assert views.canvas_get(parts, {'seq':str(first)})[0] == one
        assert np.asarray(Image.open(io.BytesIO(one)))[0,0,0] == 30
        assert np.asarray(Image.open(io.BytesIO(two)))[0,0,0] == 190
    blank = views.canvas_get(parts, {'seq':'0'})[0]
    assert blank == export_png(conn, cid, side=900, seq=0)
    assert blank != one
    for seq in [-1,second+1]:
        with pytest.raises(ValueError):
            views.canvas_get(parts, {'seq':str(seq)})
    with pytest.raises(ValueError):
        views.canvas_get(parts, {})
    with pytest.raises(KeyError):
        views.previews.get(conn,'missing',0)
    small = PreviewCache(max_bytes=len(one))
    small.get(conn,cid,first)
    small.get(conn,cid,second)
    assert small.bytes <= small.max_bytes
    assert len(small.entries) <= 1
    too_small = PreviewCache(max_bytes=1)
    assert too_small.get(conn,cid,first) == one
    assert not too_small.entries
    server = make_server(db,'127.0.0.1',0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f'http://127.0.0.1:{server.server_port}/api/canvases/{cid}'
    try:
        with urllib.request.urlopen(f'{root}/preview.png?seq={first}') as r:
            assert 'immutable' in r.headers['Cache-Control']
            assert r.read() == one
        with urllib.request.urlopen(f'{root}/export.png') as r:
            assert r.headers['Cache-Control'] == 'no-store'
            assert 'filename="Preview.png"' in r.headers['Content-Disposition']
            assert r.read() == two
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f'{root}/preview.png?seq={second+1}')
        assert e.value.code == 400
        assert 'immutable' not in e.value.headers['Cache-Control']
    finally:
        server.shutdown()
        server.server_close()
        conn.close()
