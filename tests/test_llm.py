"""The provider integrations against a fake transport: painter, mutators, cost recording, budget, key loading."""

import json
import threading
import urllib.request

import numpy as np
import pytest

from conveyor.events import EventSink
from conveyor.events import connect
from conveyor.events import observing
from conveyor.graph import Conductor
from conveyor.llm import BudgetExceeded
from conveyor.llm import LLMClient
from conveyor.llm import load_api_key
from conveyor.painting.canvas import blank
from conveyor.painting.canvas import physics_violations
from conveyor.painting.canvas import load_target
from conveyor.painting.canvas import TARGETS_DIR
from conveyor.painting.llm_agent import INITIAL_PROMPT
from conveyor.painting.llm_agent import PromptStrategy
from conveyor.painting.llm_agent import llm_paint
from conveyor.painting.llm_mutators import LLMPromptMutator
from conveyor.painting.llm_mutators import LLMToolkitMutator
from conveyor.painting.problem import PatchFailure
from conveyor.painting.problem import build_painting_graph
from conveyor.painting.toolkit import initial_toolkit
from conveyor.server import make_server

COST = 0.0003

# What a model sends back for a brush now: the code that draws the mark, not a row of knobs.
_BRUSH_SOURCE = """def alpha(u, v, rng, radius, length):
    d = np.hypot(u - np.clip(u, 0.0, length), v)
    return np.clip((radius - d) / 2.0, 0.0, 1.0) * 0.4
"""
_DOTTED_SOURCE = """def alpha(u, v, rng, radius, length):
    phase = np.abs(np.mod(u, 2.5) - 1.25)
    d = np.hypot(phase, v)
    return np.clip((radius * 0.6 - d) / 1.5 + 0.5, 0.0, 1.0) * 0.8
"""


def reply(calls, text=None):
    return {
        "model": "z-ai/glm-5.3-flash",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": text,
                "tool_calls": [
                    {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                    for i, (n, a) in enumerate(calls)
                ],
            },
        }],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost": COST},
    }


class FakeTransport:
    def __init__(self, toolkit_reply=None):
        self.payloads = []
        self.toolkit_reply = toolkit_reply
        self.lock = threading.Lock()

    def __call__(self, payload):
        with self.lock:
            self.payloads.append(payload)
        names = [t["function"]["name"] for t in payload.get("tools") or []]
        if not names:
            return reply([], text="Hello.")
        if "revise_prompt" in names:
            return reply([("revise_prompt", {"prompt": "Block in the fur, then hatch along the hair.",
                                             "summary": "Hatch along the hair direction."})])
        if "revise_toolkit" in names:
            return reply([("revise_toolkit", self.toolkit_reply or {
                "brushes": [
                    {"name": "wash", "radius": 6, "length": 8, "doc": "Broad soft sweep.",
                     "source": _BRUSH_SOURCE},
                    # The model is free to write a mark the old parameter set had no way to describe.
                    {"name": "fine liner", "radius": 1.0, "length": 2, "doc": "A broken line of dots.",
                     "source": _DOTTED_SOURCE},
                ],
                "summary": "Add a fine liner for fur.",
            })])
        # Painting: a handful of strokes with the first brush, plus one call to a brush that doesn't exist.
        calls = [(names[0], {"x": 8 + 4 * i, "y": 12, "angle": 30, "color": "#7a6045"}) for i in range(6)]
        calls.append(("no_such_brush", {"x": 1, "y": 1, "angle": 0, "color": "#000000"}))
        return reply(calls, text="Blocking in the body.")


@pytest.fixture
def target():
    return load_target(TARGETS_DIR / "self_portrait.jpg", width=48)


def test_llm_paint_applies_strokes_and_records_everything(tmp_path, target):
    sink = EventSink(tmp_path / "l.db")
    transport = FakeTransport()
    client = LLMClient(transport=transport, sink=sink)
    strategy = PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=14, strokes_per_turn=7)
    with sink.trace() as t, observing(node="agent", organism_id="org1"):
        canvas = llm_paint(strategy, initial_toolkit(), target, seed=1, record=True, client=client)
    sink.close()

    assert not np.allclose(canvas, blank(target))
    assert len(transport.payloads) == 2  # 14 strokes at 7 per turn
    first = transport.payloads[0]
    assert first["model"] == "glm-5.3-flash"  # the default provider's default model
    assert first["tool_choice"] == "required"
    assert {t["function"]["name"] for t in first["tools"]} == {"flat_wash", "round_mid"}
    images = [p for p in first["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 2 and images[0]["image_url"]["url"].startswith("data:image/png;base64,")

    conn = connect(tmp_path / "l.db", readonly=True)
    calls = [json.loads(r[0]) for r in conn.execute("SELECT data FROM events WHERE kind='llm_call' AND node='agent'")]
    assert len(calls) == 2 and all(c["cost"] == COST and c["trace_id"] == t.id for c in calls)
    names = [r[0] for r in conn.execute("SELECT name FROM spans WHERE trace_id=? ORDER BY idx", (t.id,))]
    assert names.count("model call (paint)") == 2
    assert names.count("flat_wash") == 12
    assert names.count("no_such_brush") == 2
    assert client.usage.calls == 2 and client.usage.cost == pytest.approx(2 * COST)


def _failure(blame):
    return PatchFailure(data_point_id="self_portrait/r1c2", failure_type="fine_detail", target="self_portrait",
                        row=1, col=2, blame=blame, oracle_err=0.05, agent_err=0.11)


def test_llm_mutators_return_children_and_force_the_tool(tmp_path):
    transport = FakeTransport()
    client = LLMClient(transport=transport)
    child = LLMPromptMutator(client).mutate(PromptStrategy(prompt=INITIAL_PROMPT), [_failure("agent")], [])[0]
    assert child.prompt.startswith("Block in the fur")
    assert child.from_change_summary == "[llm] Hatch along the hair direction."
    assert transport.payloads[-1]["tool_choice"] == {"type": "function", "function": {"name": "revise_prompt"}}

    tk = LLMToolkitMutator(client).mutate(initial_toolkit(), [_failure("toolkit")], [])[0]
    assert [b.name for b in tk.brushes] == ["wash", "fine_liner"]
    assert physics_violations(tk.brushes) == []  # the code the model sent compiles and draws a mark


def test_unusable_toolkit_reply_raises(tmp_path):
    client = LLMClient(transport=FakeTransport(toolkit_reply={"brushes": [{"name": "x"}], "summary": "?"}))
    with pytest.raises(ValueError, match="bad brush"):
        LLMToolkitMutator(client).mutate(initial_toolkit(), [_failure("toolkit")], [])


def test_budget_announces_once_then_hard_caps():
    spent = []
    client = LLMClient(transport=FakeTransport(), budget_usd=COST * 2, on_budget=spent.append)
    msg = [{"role": "user", "content": "hi"}]
    for _ in range(3):
        client.chat(msg, purpose="test")
    assert len(spent) == 1
    with pytest.raises(BudgetExceeded):
        client.chat(msg, purpose="test")


@pytest.mark.parametrize("provider, var, value", [
    ("opencode-go", "OPENCODE_API_KEY", "oc-"),
    ("openrouter", "OPENROUTER_API_KEY", "sk-or-"),
])
def test_load_api_key_prefers_env_then_dotenv(tmp_path, monkeypatch, provider, var, value):
    monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text(f'# comment\nexport {var}="{value}from-file"\n')
    assert load_api_key(env, provider) == f"{value}from-file"
    monkeypatch.setenv(var, f"{value}from-env")
    assert load_api_key(env, provider) == f"{value}from-env"
    assert load_api_key(tmp_path / "missing", provider) == f"{value}from-env"


def test_load_api_key_ignores_the_other_provider_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-or-only\n")
    assert load_api_key(env, "opencode-go") is None
    assert load_api_key(env, "openrouter") == "sk-or-only"


def test_llm_graph_runs_and_dashboard_reports_spend(tmp_path):
    sink = EventSink(tmp_path / "g.db", run_name="llm smoke")
    client = LLMClient(transport=FakeTransport(), sink=sink)
    # Stateless on purpose: the Pi harness needs node, npm packages and a model catalog lookup over the network.
    board, nodes, edges, schedule = build_painting_graph(width=48, llm=client, harness="stateless", strokes=24)
    Conductor(nodes, edges, sink, board, schedule).run(cycles=1)
    sink.close()

    server = make_server(tmp_path / "g.db", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/api/runs/{sink.run_id}"
    try:
        overview = json.loads(urllib.request.urlopen(base).read())
        assert overview["llm_total"]["calls"] == client.usage.calls > 0
        assert overview["llm_total"]["cost"] == pytest.approx(client.usage.cost)
        by_node = {n["name"]: n for n in overview["graph"]["nodes"]}
        assert by_node["agent"]["llm"]["calls"] > 0

        agent = json.loads(urllib.request.urlopen(base + "/nodes/agent").read())
        prompt_mutator = next(m for m in agent["mutators"] if m["mutator"] == "LLMPromptMutator")
        assert prompt_mutator["model_calls"] >= 1 and prompt_mutator["model_cost"] > 0
    finally:
        server.shutdown()


# ---- providers --------------------------------------------------------------------------------------------

def _no_cost_reply():
    """What a plain OpenAI-compatible provider sends back: usage, but no price for it."""
    r = reply([("flat_wash", {"x": 1, "y": 1, "angle": 0, "color": "#806040"})])
    r["usage"] = {"prompt_tokens": 1000, "completion_tokens": 200,
                  "prompt_tokens_details": {"cached_tokens": 400}}
    return r


def test_opencode_go_sends_plain_openai_fields_and_openrouter_sends_its_own():
    seen = []

    def transport(payload):
        seen.append(payload)
        return _no_cost_reply()

    LLMClient(transport=transport).chat([{"role": "user", "content": "hi"}], purpose="t")
    LLMClient(transport=transport, provider="openrouter", model="z-ai/glm-5.3-flash").chat(
        [{"role": "user", "content": "hi"}], purpose="t")

    go, router = seen
    assert go["reasoning_effort"] == "low" and "reasoning" not in go and "usage" not in go
    assert router["reasoning"] == {"effort": "low"} and router["usage"] == {"include": True}
    assert "reasoning_effort" not in router


@pytest.mark.parametrize("effort", ["none", "minimal", "medium"])
def test_an_effort_the_model_does_not_publish_is_dropped_rather_than_sent(effort):
    """glm-5.3-flash takes low, high and max. Sending 'medium' would be rejected, so it goes unasked."""
    seen = []

    def transport(payload):
        seen.append(payload)
        return _no_cost_reply()

    LLMClient(transport=transport, reasoning_effort=effort).chat([{"role": "user", "content": "hi"}], purpose="t")
    assert "reasoning_effort" not in seen[0]


def test_spend_is_worked_out_from_the_catalog_when_the_reply_does_not_price_itself():
    client = LLMClient(transport=lambda payload: _no_cost_reply())
    got = client.chat([{"role": "user", "content": "hi"}], purpose="t")
    # 600 fresh input at $0.15/M, 400 cached at $0.03/M, 200 output at $0.50/M.
    assert got.cost == pytest.approx((600 * 0.15 + 400 * 0.03 + 200 * 0.5) / 1e6)
    assert client.usage.cost == pytest.approx(got.cost)
    assert client.usage.cached_tokens == 400


def test_a_price_in_the_reply_wins_over_the_catalog():
    client = LLMClient(transport=FakeTransport(), provider="openrouter", model="z-ai/glm-5.3-flash")
    assert client.chat([{"role": "user", "content": "hi"}], purpose="t").cost == COST


def test_an_uncatalogued_model_reports_no_cost_rather_than_a_wrong_one():
    client = LLMClient(transport=lambda payload: _no_cost_reply(), model="not-a-model")
    assert not client.info.known
    assert client.chat([{"role": "user", "content": "hi"}], purpose="t").cost is None


def test_the_budget_still_bites_on_a_provider_that_sends_no_prices():
    spent = []
    client = LLMClient(transport=lambda payload: _no_cost_reply(), budget_usd=0.0002, on_budget=spent.append)
    msg = [{"role": "user", "content": "hi"}]
    client.chat(msg, purpose="t")
    client.chat(msg, purpose="t")
    assert spent  # the catalog priced it, so the run knows to stop
    with pytest.raises(BudgetExceeded):
        for _ in range(10):
            client.chat(msg, purpose="t")
