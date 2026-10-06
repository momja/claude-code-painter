"""The Pi harness, run on Pi's scripted faux provider: the real agent loop, MCP plumbing and recording, no network."""

import argparse
import json
import sqlite3

import pytest

from conveyor.evolve import Context
from conveyor.evolve import Organism
from conveyor.harness import Meter
from conveyor.harness import RateLimited
from conveyor.painting import prompts
from conveyor.painting.judge import ClaudeJudge
from conveyor.painting.problem import ClaudeInstrumentMutator
from conveyor.painting.problem import ClaudePromptMutator
from conveyor.painting.problem import Painter
from conveyor.painting.problem import Setup
from conveyor.painting.problem import make_instrument
from conveyor.painting.seeds import PEN
from conveyor.painting.seeds import ROUND
from conveyor.pi import PiAgent
from conveyor.pi import available
from conveyor.pi import thinking_level
from conveyor.store import Store

needs_pi = pytest.mark.skipif(available() is not None, reason=f"Pi sidecar not installed: {available()}")


def stroke(x, y):
    return {"tool": "stroke", "args": {"x": x, "y": y, "angle": 0, "length": 30, "size": 6, "color": "#6f8fb5"}}


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "pi.db")
    yield store, Meter(), Setup(width=64, actions=10, work_dir=tmp_path / "sessions", judge=False)
    store.close()


def rows(store, sql, *args):
    store.flush()
    conn = sqlite3.connect(store.path)
    return conn.execute(sql, args).fetchall()


@needs_pi
def test_pi_compacts_after_n_looks(env):
    store, meter, setup = env
    look = {"blocks": [stroke(10, 10), {"tool": "look", "args": {}}]}
    faux = [look, look, {"blocks": [{"text": "SUMMARY: sky is done."}]},  # the third reply is the summarizer's
            {"blocks": [{"tool": "finish", "args": {"note": "ok"}}]}, {"blocks": [{"text": "Done."}]}]
    pi = PiAgent("faux-model", store=store, meter=meter, faux=faux, compact_every_looks=2)
    painter = Painter(setup, store, pi)
    inst = make_instrument(ROUND, "round", setup, store, painter.target.height)
    p = painter.paint(inst, Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY}))
    assert p.viable and p.details["note"] == "ok"
    [(status, turns)] = rows(store, "SELECT status, num_turns FROM sessions")
    assert status == "ok" and turns == 3  # look, look, finish; the summary is not a turn
    events = [json.loads(d) for (d,) in rows(store, "SELECT data FROM session_events WHERE kind='other'")]
    assert [e["subtype"] for e in events].count("compact_boundary") == 1
    starts = [e for e in events if e["subtype"] == "request_start"]
    assert len(starts) == 3 and all(e["images"] >= 1 for e in starts)  # each request logs what it sent
    assert [e["messages"] for e in starts] == [1, 4, 5]  # the third goes out compacted: 7 messages became 5
    assert sum(e["subtype"] == "request_end" for e in events) == 3


@needs_pi
def test_a_painting_on_pi(env):
    store, meter, setup = env
    faux = [
        {"blocks": [{"thinking": "Sky first."}, stroke(10, 10), stroke(10, 30), {"tool": "look", "args": {}}]},
        {"blocks": [{"tool": "finish", "args": {"note": "Needed curves."}}]},
        {"blocks": [{"text": "Done."}]},
        {"blocks": [{"text": "This reply should never be asked for."}]},
    ]
    painter = Painter(setup, store, PiAgent("faux-model", store=store, meter=meter, faux=faux))
    inst = make_instrument(ROUND, "round", setup, store, painter.target.height)
    p = painter.paint(inst, Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY}))
    assert p.viable and p.details["stats"]["tool_use"] == {"stroke": 2} and p.details["note"] == "Needed curves."
    [(status, model, turns)] = rows(store, "SELECT status, model, num_turns FROM sessions")
    assert status == "ok" and model == "faux-model" and turns == 2  # it stops after the turn that called finish
    kinds = [k for (k,) in rows(store, "SELECT kind FROM session_events ORDER BY idx")]
    assert kinds[0] == "init" and "thinking" in kinds and kinds.count("tool_use") == 4 and kinds[-1] == "result"
    joined = rows(store, "SELECT count(*) FROM strokes WHERE json_extract(data, '$.tool_use_id') IS NOT NULL")[0][0]
    assert joined == 4  # every paint-server log line carries the tool call id from the transcript
    images = rows(store, "SELECT data FROM session_events WHERE kind='tool_result'")
    assert any(json.loads(d)["images"] for (d,) in images)  # the look came back as an image


@needs_pi
def test_structured_output_through_respond(env):
    store, meter, setup = env
    answer = {"edits": [{"old": "Block in the large areas of colour first", "new": "Paint the darks first"}],
              "summary": "darks first"}
    pi = PiAgent("faux-model", store=store, meter=meter, faux=[{"blocks": [{"tool": "respond", "args": answer}]},
                                                                {"blocks": [{"text": "ok"}]}])
    parent = Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY})
    ctx = Context(node="painter", parent=parent, parent_eval=None, partner=None, lineage=[], niches={}, empty_niches=[])
    mutator = ClaudePromptMutator(pi, setup, store)
    [child] = mutator.propose(ctx)
    assert mutator.name == "pi:strategy" and "Paint the darks first, then work" in child.genome["prompt"]

    verdict = {"likeness": 6, "colour": 5, "brushwork": 4, "overall": 5, "critique": "Too blue."}
    judge = ClaudeJudge(PiAgent("faux-model", store=store, meter=meter,
                                faux=[{"blocks": [{"tool": "respond", "args": verdict}]}, {"blocks": [{"text": "ok"}]}]),
                        setup.work_dir)
    v = judge.judge(b"\x89PNG\r\n\x1a\n", b"\x89PNG\r\n\x1a\n")
    assert v.scores["likeness"] == 6 and v.critique == "Too blue."


@needs_pi
def test_an_instrument_mutation_on_pi(env):
    store, meter, setup = env
    faux = [{"blocks": [{"tool": "try_instrument", "args": {"source": PEN}}]},
            {"blocks": [{"tool": "submit_instrument", "args": {"source": PEN, "summary": "a pen"}}]},
            {"blocks": [{"text": "Submitted."}]}]
    parent = make_instrument(ROUND, "seed", setup, store, 80)
    mutator = ClaudeInstrumentMutator("invent", PiAgent("faux-model", store=store, meter=meter, faux=faux),
                                      setup, store, 80, 1.0)
    ctx = Context(node="instrument", parent=parent, parent_eval=None, partner=None, lineage=[], niches={},
                  empty_niches=[], wanted_niche="stateful/scalar/medium")
    [child] = mutator.propose(ctx)
    assert mutator.name == "pi:invent" and child.viable and child.traits["stateful"] and child.summary == "a pen"


def test_rate_limits_are_per_harness():
    meter = Meter(max_usage=0.85)
    meter.note_rate_limit({"status": "rejected", "resetsAt": 4102444800}, "claude")  # year 2100
    meter.check("pi")  # Pi keeps running while Claude's window is spent
    with pytest.raises(RateLimited, match="claude hit its usage limit"):
        meter.check("claude")


def test_thinking_levels_move_to_what_the_model_publishes():
    assert thinking_level("medium", ["low", "high", "max"]) == "high"  # ties go to the higher level
    assert thinking_level("xhigh", ["low", "high", "max"]) == "max"
    assert thinking_level("high", None) == "high"
    assert thinking_level("none", ["low"]) == "off"


def test_openai_codex_provider_uses_the_chatgpt_subscription_api(monkeypatch, tmp_path):
    from conveyor.pi import AUTH_FILE_ENV
    from conveyor.pi import PROVIDERS
    from conveyor.pi import model_api
    from conveyor.pi import provider_authenticated

    provider = PROVIDERS["openai-codex"]
    assert provider.default_model == "gpt-5.4"
    assert provider.oauth and provider.env_var is None
    assert model_api(provider, provider.default_model) == "openai-codex-responses"
    monkeypatch.setenv(AUTH_FILE_ENV, str(tmp_path / "pi-auth.json"))
    assert not provider_authenticated(provider)
    (tmp_path / "pi-auth.json").write_text(json.dumps({
        "openai-codex": {"type": "oauth", "access": "test", "refresh": "test", "expires": 1},
    }))
    assert provider_authenticated(provider)


def test_openai_auth_cli_and_provider_option_parse():
    from conveyor.__main__ import build_parser

    parser = build_parser()
    assert parser.parse_args(["run", "--provider", "openai-codex", "--offline"]).provider == "openai-codex"
    auth = parser.parse_args(["auth", "login", "openai"])
    assert auth.auth_action == "login" and auth.provider == "openai"


@needs_pi
def test_roles_mix_harnesses(env, fake_claude, monkeypatch, capsys):
    from conveyor.__main__ import _roles

    store, meter, _ = env
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    args = argparse.Namespace(harness="claude", model="claude-opus-5-5", effort="high", provider="opencode-go",
                              claude=str(fake_claude), lanes=1, judge=True,
                              paint_harness="pi", paint_model=None, paint_effort=None,
                              mutate_harness=None, mutate_model=None, mutate_effort="low",
                              judge_harness=None, judge_model=None, judge_effort=None)
    roles = _roles(args, store, meter)
    assert roles.paint.name == "pi" and roles.paint.model == "glm-5.3-flash"  # a Claude model id isn't inherited
    assert roles.mutate.name == "claude" and roles.mutate.effort == "low"
    assert roles.judge.name == "claude" and roles.judge.effort == "high" and roles.judge is not roles.mutate
    assert "paint   pi: glm-5.3-flash on OpenCode Go" in capsys.readouterr().out
