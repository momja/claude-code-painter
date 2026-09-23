"""Full model-call records: streaming, retries, failures, tool outcomes, organism links, and the endpoints."""

import json
import threading
import time
import urllib.request

import httpx
import pytest

from conveyor.events import EventSink
from conveyor.events import connect
from conveyor.events import observing
from conveyor.llm import LLMClient
from conveyor.llm import LLMError
from conveyor.llm import function_tool
from conveyor.llm import image_part
from conveyor.llm import text_part
from conveyor.observe import ObservedMutator
from conveyor.painting.canvas import TARGETS_DIR
from conveyor.painting.canvas import load_target
from conveyor.painting.canvas import to_png
from conveyor.painting.llm_agent import INITIAL_PROMPT
from conveyor.painting.llm_agent import PromptStrategy
from conveyor.painting.llm_agent import llm_paint
from conveyor.painting.llm_mutators import LLMPromptMutator
from conveyor.painting.toolkit import initial_toolkit
from conveyor.server import make_server
from test_llm import FakeTransport
from test_llm import _failure
from test_llm import reply

STREAM = [
    {"model": "z-ai/glm-5.3-flash", "choices": [{"index": 0, "delta": {
        "role": "assistant", "reasoning": "The coat is blue. ",
        "reasoning_details": [{"type": "reasoning.text", "text": "The coat is blue. "}]}}]},
    {"choices": [{"index": 0, "delta": {"reasoning_details": [{"type": "reasoning.text", "text": "Start with a wash."}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                        "function": {"name": "flat_wash", "arguments": '{"x": 20, "y"'}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {
        "arguments": ': 30, "angle": 0, "color": "#806040"}'}}]}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
     "usage": {"prompt_tokens": 900, "completion_tokens": 120, "cost": 0.0002,
               "completion_tokens_details": {"reasoning_tokens": 60}}},
]
BRUSH_TOOL = function_tool("flat_wash", "stroke", {"x": {"type": "number"}}, ["x"])


def sse(*chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)


def streaming_client(handler, **kw):
    kw.setdefault("provider", "openrouter")
    kw.setdefault("model", "z-ai/glm-5.3-flash")
    return LLMClient(api_key="test", http_transport=httpx.MockTransport(handler), **kw)


def test_streamed_call_records_thinking_tool_calls_and_the_images_sent(tmp_path):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        body = ": OPENROUTER PROCESSING\n\n" + sse(*STREAM) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    sink = EventSink(tmp_path / "s.db")
    client = streaming_client(handler, sink=sink)
    png = to_png(load_target(TARGETS_DIR / "self_portrait.jpg", width=32).image)
    with observing(node="agent", organism_id="org1"):
        reply = client.chat(
            [{"role": "system", "content": "Paint."}, {"role": "user", "content": [text_part("Target:"), image_part(png)]}],
            purpose="paint", tools=[BRUSH_TOOL],
        )
    sink.close()

    assert seen[0]["stream"] is True and seen[0]["reasoning"] == {"effort": "low"}
    assert reply.reasoning == "The coat is blue. Start with a wash."
    assert reply.tool_calls[0].arguments == {"x": 20, "y": 30, "angle": 0, "color": "#806040"}
    assert reply.cost == 0.0002 and reply.reasoning_tokens == 60

    conn = connect(tmp_path / "s.db", readonly=True)
    row = conn.execute("SELECT * FROM llm_calls").fetchone()
    assert row["id"] == reply.call_id
    assert (row["status"], row["node"], row["organism_id"]) == ("ok", "agent", "org1")
    assert row["reasoning"] == reply.reasoning and row["first_token"] is not None
    assert json.loads(row["tool_calls"])[0] == {"id": "call_1", "name": "flat_wash",
                                                 "arguments": {"x": 20, "y": 30, "angle": 0, "color": "#806040"}}
    assert json.loads(row["usage"])["completion_tokens_details"]["reasoning_tokens"] == 60
    image = json.loads(row["request"])["messages"][1]["content"][1]
    assert image["type"] == "image"
    assert conn.execute("SELECT data FROM artifacts WHERE name=?", (image["artifact"],)).fetchone()["data"] == png


def test_opencode_go_sends_a_stable_session_header():
    """OpenCode Go answers 400 MissingSessionID without it, and routes and caches on its value."""
    seen = []

    def handler(request):
        seen.append(request.headers)
        return httpx.Response(200, content=sse(*STREAM).encode())

    msg = [{"role": "user", "content": "hi"}]
    client = streaming_client(handler, provider="opencode-go", model="glm-5.3-flash")
    client.chat(msg, purpose="test")
    client.chat(msg, purpose="test")
    client.chat(msg, purpose="test", session_id="one-painting")
    assert seen[0]["x-opencode-session"] == seen[1]["x-opencode-session"] == client.session_id
    assert seen[2]["x-opencode-session"] == "one-painting"  # one conversation keeps one id

    streaming_client(handler).chat(msg, purpose="test")  # openrouter by default
    assert "x-opencode-session" not in seen[3]


def test_rate_limits_are_retried_before_the_stream_starts(monkeypatch):
    monkeypatch.setattr("conveyor.llm.time.sleep", lambda s: None)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, content=sse(*STREAM).encode())

    reply = streaming_client(handler).chat([{"role": "user", "content": "hi"}], purpose="test")
    assert len(attempts) == 3 and reply.tool_calls[0].name == "flat_wash"


def test_mid_stream_error_fails_the_call_without_retrying(tmp_path):
    attempts = []

    def handler(request):
        attempts.append(1)
        err = {"error": {"code": 502, "message": "provider died"},
               "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
        return httpx.Response(200, content=sse(STREAM[0], err).encode())

    sink = EventSink(tmp_path / "e.db")
    with pytest.raises(LLMError, match="provider died"):
        streaming_client(handler, sink=sink).chat([{"role": "user", "content": "hi"}], purpose="test")
    sink.close()
    assert len(attempts) == 1
    row = connect(tmp_path / "e.db", readonly=True).execute("SELECT * FROM llm_calls").fetchone()
    assert row["status"] == "error" and "provider died" in row["error"]
    assert row["reasoning"] == "The coat is blue. "  # what streamed before the failure is kept


def test_painter_records_what_happened_to_each_tool_call(tmp_path):
    sink = EventSink(tmp_path / "p.db")
    client = LLMClient(transport=FakeTransport(), sink=sink)
    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=48)
    with sink.trace(), observing(node="agent"):
        # The fake replies with 6 real strokes and one unknown brush. With 7 allowed, all 7 are handled.
        llm_paint(PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=7, strokes_per_turn=7), initial_toolkit(), target,
                  seed=1, record=True, client=client)
        # With 5 allowed, the last two are ignored.
        llm_paint(PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=5, strokes_per_turn=5), initial_toolkit(), target,
                  seed=1, record=False, client=client)
    sink.close()
    rows = connect(tmp_path / "p.db", readonly=True).execute(
        "SELECT tool_results FROM llm_calls ORDER BY started").fetchall()
    first, second = ([r["status"] for r in json.loads(row[0])] for row in rows)
    assert first == ["applied"] * 6 + ["rejected"]
    assert second == ["applied"] * 5 + ["ignored"] * 2


def test_painter_retries_once_when_thinking_eats_the_whole_reply():
    payloads = []
    normal = FakeTransport()

    def transport(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            r = reply([])
            r["choices"][0]["finish_reason"] = "length"
            return r
        return normal(payload)

    target = load_target(TARGETS_DIR / "self_portrait.jpg", width=48)
    llm_paint(PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=6, strokes_per_turn=6), initial_toolkit(), target,
              seed=1, client=LLMClient(transport=transport, provider="openrouter"))
    assert len(payloads) == 2
    assert payloads[1]["reasoning"] == {"effort": "minimal"}
    assert "ran out of room" in payloads[1]["messages"][1]["content"][-1]["text"]
    assert payloads[1]["max_tokens"] == 8192


def test_child_organism_points_to_the_call_that_wrote_it(tmp_path):
    sink = EventSink(tmp_path / "m.db")
    client = LLMClient(transport=FakeTransport(), sink=sink)
    mutator = ObservedMutator(LLMPromptMutator(client), "agent", sink)
    [child] = mutator.mutate(PromptStrategy(prompt=INITIAL_PROMPT), [_failure("agent")], [])
    sink.close()
    conn = connect(tmp_path / "m.db", readonly=True)
    organism = json.loads(conn.execute("SELECT data FROM events WHERE kind='organism'").fetchone()[0])
    call = conn.execute("SELECT * FROM llm_calls").fetchone()
    assert organism["llm_call_ids"] == [call["id"]]
    assert (call["node"], call["mutator"], call["purpose"]) == ("agent", "LLMPromptMutator", "mutate")
    assert json.loads(call["tool_results"])[0]["status"] == "applied"


def test_server_lists_calls_serves_one_in_full_and_shows_live_ones(tmp_path):
    sink = EventSink(tmp_path / "v.db")
    client = LLMClient(transport=FakeTransport(), sink=sink)
    with observing(node="agent"):
        client.chat([{"role": "user", "content": "hi"}], purpose="test")
    sink.start_call("live1", node="toolkit", mutator="LLMToolkitMutator", organism_id=None, trace_id=None,
                    purpose="mutate", model="z-ai/glm-5.3-flash", request={"messages": []})
    sink.update_call("live1", reasoning="Thinking about brushes " * 100, first_token=time.time())
    sink.close()

    server = make_server(tmp_path / "v.db", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        return json.loads(urllib.request.urlopen(base + path).read())

    try:
        calls = get(f"/api/runs/{sink.run_id}/calls?node=agent")
        assert [c["status"] for c in calls] == ["ok"]
        full = get(f"/api/calls/{calls[0]['id']}")
        assert full["request"]["messages"][0]["content"] == "hi" and full["content"] == "Hello."
        live = get(f"/api/runs/{sink.run_id}/live")
        assert [c["id"] for c in live["running"]] == ["live1"]
        assert len(live["running"][0]["reasoning_tail"]) == 1500
        assert live["running"][0]["reasoning_chars"] == len("Thinking about brushes " * 100)
    finally:
        server.shutdown()


def test_the_plain_openai_path_asks_for_usage_through_stream_options():
    """OpenRouter reports usage on a stream unasked; OpenAI-compatible providers only do with this flag."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        body = sse(*STREAM) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    LLMClient(api_key="test", http_transport=httpx.MockTransport(handler)).chat(
        [{"role": "user", "content": "hi"}], purpose="t")
    streaming_client(handler).chat([{"role": "user", "content": "hi"}], purpose="t")

    go, router = seen
    assert go["stream"] is True and go["stream_options"] == {"include_usage": True}
    assert "stream_options" not in router
