"""Bounded, sequence-bound display derivatives; authoritative PNG exports stay untouched."""
from collections import OrderedDict
import threading

from conveyor.commons.export import export_png


class PreviewCache:
    """Per-server byte LRU. Serialize misses so concurrent polls don't encode duplicates."""

    def __init__(self, max_bytes: int = 16 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def get(self, conn, canvas_id: str, seq: int) -> bytes:
        row = conn.execute("SELECT 1 FROM canvases WHERE id=?", (canvas_id,)).fetchone()
        if row is None:
            raise KeyError(canvas_id)
        latest = conn.execute("SELECT coalesce(max(seq), 0) FROM canvas_ops WHERE canvas_id=?", (canvas_id,)).fetchone()[0]
        if seq < 0 or seq > latest:
            raise ValueError("preview seq must be an existing canvas moment (0 through latest op)")
        key = (canvas_id, seq)
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key]
            data = export_png(conn, canvas_id, side=900, seq=seq)
            if len(data) <= self.max_bytes:
                while self.entries and self.bytes + len(data) > self.max_bytes:
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= len(old)
                self.entries[key] = data
                self.bytes += len(data)
            return data
