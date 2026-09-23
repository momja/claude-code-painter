import json

from conveyor.events import EventSink
from conveyor.events import artifact
from conveyor.events import connect
from conveyor.events import current_trace


def test_events_spans_and_artifacts(tmp_path):
    sink = EventSink(tmp_path / "e.db", run_name="t")
    sink.set_iteration("a", 3)
    sink.emit("a", "evaluated", "org1", score=0.5)
    with sink.trace() as t:
        assert current_trace() is t
        name = artifact(b"hello")
        t.span("liner_1", args={"x": 1}, result={"kept": True}, image=b"img")
        t.span("skip")
    assert current_trace() is None
    assert artifact(b"nobody is watching") is None
    sink.close()

    conn = connect(tmp_path / "e.db", readonly=True)
    row = conn.execute("SELECT * FROM events WHERE kind='evaluated'").fetchone()
    assert row["iteration"] == 3
    assert row["organism_id"] == "org1"
    assert json.loads(row["data"]) == {"score": 0.5}
    spans = conn.execute("SELECT * FROM spans ORDER BY idx").fetchall()
    assert [s["name"] for s in spans] == ["liner_1", "skip"]
    assert spans[0]["artifact"] is not None and spans[1]["artifact"] is None
    stored = conn.execute("SELECT data FROM artifacts WHERE name=?", (name,)).fetchone()["data"]
    assert stored == b"hello"


def test_artifacts_are_content_addressed(tmp_path):
    sink = EventSink(tmp_path / "e.db")
    a = sink.store_artifact(b"same")
    b = sink.store_artifact(b"same")
    c = sink.store_artifact(b"different")
    sink.close()
    assert a == b != c
    conn = connect(tmp_path / "e.db", readonly=True)
    assert conn.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 2


def test_flush_makes_events_visible(tmp_path):
    sink = EventSink(tmp_path / "e.db")
    sink.emit(None, "cycle", cycle=0)
    sink.flush()
    conn = connect(tmp_path / "e.db", readonly=True)
    assert conn.execute("SELECT count(*) FROM events WHERE kind='cycle'").fetchone()[0] == 1
    sink.close()
