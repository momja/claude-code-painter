"""End to end: one cycle of the painting graph, then every dashboard endpoint against the result."""

import json
import threading
import urllib.request

import pytest

from conveyor.events import EventSink
from conveyor.graph import Conductor
from conveyor.painting.problem import build_painting_graph
from conveyor.server import make_server


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    # 48 px leaves only 9 patches, and on this target the oracle misses most of them, so the critic blames
    # the toolkit for everything and the agent has no failures to evolve from. 64 gives it something.
    board, nodes, edges, schedule = build_painting_graph(width=64)
    sink = EventSink(tmp / "p.db", run_name="smoke")
    Conductor(nodes, edges, sink, board, schedule).run(cycles=1)
    sink.close()
    server = make_server(tmp / "p.db", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", sink.run_id
    server.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        body = r.read()
        return json.loads(body) if r.headers["Content-Type"] == "application/json" else body


def test_dashboard_endpoints(served):
    base, run = served
    assert b"<title>Conveyor</title>" in get(base + "/")
    assert get(base + "/api/runs")[0]["id"] == run

    overview = get(f"{base}/api/runs/{run}")
    names = [n["name"] for n in overview["graph"]["nodes"]]
    assert names == ["toolkit", "agent", "critic"]
    toolkit = overview["graph"]["nodes"][0]
    assert toolkit["champion"]["score"] > 0 and toolkit["thumbnail"]
    assert set(overview["blame"]["keys"]) == {"toolkit", "agent"}

    for node in ("toolkit", "agent"):
        detail = get(f"{base}/api/runs/{run}/nodes/{node}")
        assert detail["iterations"][0]["iteration"] == 0
        assert detail["lineage"]
        assert detail["mutators"]

    champion = overview["graph"]["nodes"][1]["champion"]["id"]
    org = get(f"{base}/api/runs/{run}/organisms/{champion}")
    assert org["node"] == "agent" and org["evaluations"]
    assert "strokes" in org["text"]

    ev = org["evaluations"][0]
    reports = ev["details"]["critic"]
    assert [(r["target"], r["train"]) for r in reports] == [("self_portrait", True)]  # held-out is champions only
    held = org["holdouts"][0]["details"]["critic"]
    assert [(r["target"], r["train"]) for r in held] == [("starry_night", False)]
    assert [s["factor"] for s in reports[0]["scales"]] == [1, 2, 4]
    assert get(f"{base}/artifacts/{reports[0]['scales'][2]['error']}")[:4] == b"\x89PNG"
    trace = get(f"{base}/api/traces/{ev['trace_id']}")
    assert len(trace["spans"]) > 10
    assert any(s["artifact"] for s in trace["spans"])

    png = get(f"{base}/artifacts/{ev['artifacts']['thumb']}")
    assert png[:4] == b"\x89PNG"


def test_bad_artifact_name_is_rejected(served):
    base, _ = served
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/artifacts/..%2F..%2Fetc%2Fpasswd")
    assert e.value.code in (400, 404)
