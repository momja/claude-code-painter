"""The host side of Claude Code, run against the fake CLI: recording, painting, and both mutators."""

import json
import sqlite3

import pytest

from conveyor.claude import ClaudeCode
from conveyor.claude import Job
from conveyor.claude import Meter
from conveyor.claude import Settings
from conveyor.evolve import Context
from conveyor.evolve import Organism
from conveyor.painting import prompts
from conveyor.painting.problem import ClaudeInstrumentMutator
from conveyor.painting.problem import ClaudePromptMutator
from conveyor.painting.problem import Painter
from conveyor.painting.problem import Setup
from conveyor.painting.instrument import probe
from conveyor.painting.problem import make_instrument
from conveyor.painting.seeds import PEN
from conveyor.painting.seeds import ROUND
from conveyor.store import Store


@pytest.fixture
def env(tmp_path, fake_claude, monkeypatch):
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    store = Store(tmp_path / "run.db", run_name="test")
    meter = Meter(budget_usd=10)
    claude = ClaudeCode(Settings(binary=str(fake_claude)), store, meter, lanes=2)
    setup = Setup(width=64, actions=10, work_dir=tmp_path / "sessions")
    yield store, claude, setup, log, meter
    store.close()


def rows(store, sql, *args):
    store.flush()
    conn = sqlite3.connect(store.path)
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, args)]


def test_argv_isolates_the_session(env):
    _, claude, _, _, _ = env
    argv = claude.argv(Job(purpose="x", system_prompt="S", content=[], cwd=env[2].work_dir,
                           mcp={"name": "canvas", "command": "py", "args": ["-m", "srv"]}, tools=["stroke", "look"]))
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5" and argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--allowedTools") + 1] == "mcp__canvas__stroke,mcp__canvas__look"


def test_a_painting_is_recorded_end_to_end(env):
    store, claude, setup, log, meter = env
    painter = Painter(setup, store, claude)
    inst = make_instrument(PEN, "pen", setup, store, painter.target.height)
    prompt = Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY})
    p = painter.paint(inst, prompt)
    assert p.viable and 0 < p.score < 1
    assert p.details["stats"]["tool_use"] == {"start": 2, "move": 2, "stop": 2}
    assert p.details["note"] == "The fake painter wanted thinner lines."
    # The judge's verdict (5, 6, 4, 5: a mean of 5, so (5 - 1) / 9) carries half the score.
    judged = p.details["judge"]
    assert judged["critique"] == "The face sits too far left." and judged["score"] == pytest.approx(4 / 9, abs=1e-3)
    assert p.score == pytest.approx(0.5 * p.critic["total"] + 0.5 * 4 / 9, abs=1e-3)
    from conveyor.painting.problem import _evidence
    text, _ = _evidence(painter.evaluation(p, inst, prompt, "new"))
    assert "expert judge scored it likeness 5, colour 6, brushwork 4, overall 5" in text and "too far left" in text
    assert painter.paint(inst, prompt) is p  # the same pair is painted once, and judged once
    assert meter.sessions == 2 and meter.spent == pytest.approx(0.02) and meter.rate_limit["status"] == "allowed"

    first = json.loads(log.read_text().splitlines()[0])
    assert first["first"] == ["text", "image", "text", "image", "text"]  # target and demo sheet go in as images
    judge_call = json.loads(log.read_text().splitlines()[1])
    assert judge_call["first"] == ["text", "image", "text", "image"]  # the judge sees the target and the painting only
    painter_prompt = first["argv"][first["argv"].index("--system-prompt") + 1]
    assert "an expert judge looks at your painting" in painter_prompt
    [session] = rows(store, "SELECT * FROM sessions WHERE purpose='paint'")
    assert session["status"] == "ok" and session["purpose"] == "paint" and session["cost"] == pytest.approx(0.01)
    kinds = [r["kind"] for r in rows(store, "SELECT kind FROM session_events WHERE session_id=? ORDER BY idx", session["id"])]
    assert kinds.count("tool_use") == 8 and kinds.count("tool_result") == 8 and "thinking" in kinds and kinds[-1] == "result"
    strokes = rows(store, "SELECT data FROM strokes WHERE session_id=?", session["id"])
    assert len(strokes) == 8 and json.loads(strokes[0]["data"])["tool_use_id"] == "toolu_fake_1"


def test_instrument_mutator_submits_through_the_workbench(env, monkeypatch):
    store, claude, setup, _, _ = env
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", PEN)
    parent = make_instrument(ROUND, "seed", setup, store, 80)
    pen_niche = probe(PEN, 64, 80)["niche"]  # reach is relative to the canvas, so the niche depends on its size
    mutator = ClaudeInstrumentMutator("invent", claude, setup, store, 80, 1.0)
    ctx = Context(node="instrument", parent=parent, parent_eval=None, partner=None, lineage=[], niches={},
                  empty_niches=[pen_niche], wanted_niche=pen_niche)
    [child] = mutator.propose(ctx)
    assert child.viable and child.niche == pen_niche and child.parent_id == parent.id
    assert child.traits["wanted_niche"] == pen_niche and child.sheet
    [session] = rows(store, "SELECT * FROM sessions WHERE purpose LIKE 'mutate%'")
    assert f"Aim for the niche **{pen_niche}**" in json.dumps(json.loads(session["request"])["content"])


def test_instrument_mutator_fails_loudly_without_a_submission(env, monkeypatch):
    store, claude, setup, _, _ = env
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", "import os")
    parent = make_instrument(ROUND, "seed", setup, store, 80)
    ctx = Context(node="instrument", parent=parent, parent_eval=None, partner=None, lineage=[], niches={}, empty_niches=[])
    with pytest.raises(RuntimeError, match="no instrument was submitted"):
        ClaudeInstrumentMutator("refine", claude, setup, store, 80, 1.0).propose(ctx)


def test_prompt_mutator_applies_edits(env, monkeypatch):
    store, claude, setup, _, _ = env
    edits = [{"old": "Block in the large areas of colour first", "new": "Paint the darks first"},
             {"old": "", "new": "Keep twenty actions for the eyes."},
             {"old": "a sentence that isn't there", "new": "anything"}]
    monkeypatch.setenv("FAKE_CLAUDE_STRUCTURED", json.dumps({"edits": edits, "summary": "darks first"}))
    parent = Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY})
    ctx = Context(node="painter", parent=parent, parent_eval=None, partner=None, lineage=[], niches={}, empty_niches=[])
    [child] = ClaudePromptMutator(claude, setup, store).propose(ctx)
    text = child.genome["prompt"]
    assert "Paint the darks first, then work toward edges" in text and text.endswith("Keep twenty actions for the eyes.")
    assert "Block in" not in text and child.summary.startswith("darks first (1 of 3 edits didn't match")
    assert 0.8 < child.traits["words_kept"] < 1


def test_prompt_mutator_refuses_a_rewrite(env, monkeypatch):
    store, claude, setup, _, _ = env
    rewrite = [{"old": prompts.INITIAL_STRATEGY, "new": "Paint whatever you like, quickly."}]
    monkeypatch.setenv("FAKE_CLAUDE_STRUCTURED", json.dumps({"edits": rewrite, "summary": "a fresh start"}))
    parent = Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY})
    ctx = Context(node="painter", parent=parent, parent_eval=None, partner=None, lineage=[], niches={}, empty_niches=[])
    with pytest.raises(RuntimeError, match="kept 0% of the prompt's words"):
        ClaudePromptMutator(claude, setup, store).propose(ctx)


def test_edits_match_across_reflowed_whitespace():
    from conveyor.painting.problem import apply_edits

    text, missed = apply_edits("Look at the canvas\nevery so often.\n\nMatch colours.", [
        {"old": "the canvas every so often", "new": "the canvas after every ten actions"},
        {"old": "Match colours.", "new": ""}])
    assert text == "Look at the canvas after every ten actions." and missed == 0


def test_budget_stops_new_sessions(env):
    from conveyor.claude import BudgetExhausted

    _, claude, _, _, meter = env
    meter.add(100)
    with pytest.raises(BudgetExhausted):
        claude.run(Job(purpose="x", system_prompt="s", content=[{"type": "text", "text": "hi"}], cwd=env[2].work_dir))


def test_usage_window_guard():
    from conveyor.claude import RateLimited

    meter = Meter(max_usage=0.85)
    meter.note_rate_limit({"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.6}}})
    meter.check()
    assert not meter.should_stop()
    meter.note_rate_limit({"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.86},
                                                                    "seven_day": {"utilization": 0.3}}})
    assert meter.should_stop()
    with pytest.raises(RateLimited, match="five-hour usage window is at 86%"):
        meter.check()


def test_a_waiting_meter_holds_sessions_until_the_window_resets(monkeypatch):
    import time

    from conveyor import harness

    monkeypatch.setattr(harness, "RESET_GRACE", 0.0)
    waits, resumes = [], []
    meter = Meter(max_usage=0.85, wait_hours=1, on_wait=lambda reason, until: waits.append((reason, until)),
                  on_resume=lambda: resumes.append(True))
    resets = time.time() + 1.5
    meter.note_rate_limit({"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.9, "resetsAt": resets}}})
    assert meter.should_stop() is None  # the run carries on; its next session waits
    started = time.time()
    meter.check("claude")
    assert resets <= time.time() < started + 5
    assert [until for _, until in waits] == [resets] and "five-hour usage window is at 90%" in waits[0][0]
    assert resumes == [True]


def test_a_limit_that_resets_past_the_wait_still_stops_the_run():
    import time

    from conveyor.claude import RateLimited

    meter = Meter(max_usage=0.85, wait_hours=1)
    meter.note_rate_limit({"status": "allowed", "unifiedWindows": {
        "seven_day": {"utilization": 0.9, "resetsAt": time.time() + 3 * 86400}}})
    assert "past the 1-hour wait" in meter.should_stop()
    with pytest.raises(RateLimited, match="seven-day usage window is at 90%.*past the 1-hour wait"):
        meter.check()


def test_stopping_ends_a_wait():
    import threading
    import time

    meter = Meter(max_usage=0.85, wait_hours=1)
    meter.note_rate_limit({"status": "allowed", "unifiedWindows": {
        "five_hour": {"utilization": 0.9, "resetsAt": time.time() + 600}}})
    stop = threading.Event()
    waiter = threading.Thread(target=meter.check, args=("claude", stop))
    waiter.start()
    time.sleep(0.2)
    stop.set()
    waiter.join(timeout=5)
    assert not waiter.is_alive()


def test_a_painting_the_limit_cuts_off_raises_and_the_next_waits_for_the_reset(tmp_path, fake_claude, monkeypatch):
    import time

    from conveyor import harness
    from conveyor.harness import SessionCutOff

    monkeypatch.setattr(harness, "RESET_GRACE", 0.0)
    flag = tmp_path / "cut-off"
    flag.touch()
    monkeypatch.setenv("FAKE_CLAUDE_CUT_OFF", str(flag))
    monkeypatch.setenv("FAKE_CLAUDE_RESETS_IN", "2")
    store = Store(tmp_path / "run.db", run_name="test")
    meter = Meter(budget_usd=10, max_usage=0.85, wait_hours=1)
    claude = ClaudeCode(Settings(binary=str(fake_claude)), store, meter, lanes=1)
    painter = Painter(Setup(width=64, actions=10, work_dir=tmp_path / "sessions", judge=False), store, claude)
    inst = make_instrument(PEN, "pen", painter.setup, store, painter.target.height)
    prompt = Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY})
    with pytest.raises(SessionCutOff, match="usage limit ended a paint session partway"):
        painter.paint(inst, prompt)
    assert meter.limited("claude")
    resets = meter.limited_until["claude"]
    p = painter.paint(inst, prompt)  # held until the reset, then painted in full
    assert time.time() >= resets and p.viable
    assert p.details["stats"]["tool_use"] == {"start": 2, "move": 2, "stop": 2}
    assert not meter.limited("claude")
    store.close()


def test_recombine_sees_both_parents(env, monkeypatch):
    store, claude, setup, _, _ = env
    monkeypatch.setenv("FAKE_CLAUDE_SUBMIT", PEN)
    a = make_instrument(ROUND, "seed a", setup, store, 80)
    b = make_instrument(PEN, "seed b", setup, store, 80)
    mutator = ClaudeInstrumentMutator("recombine", claude, setup, store, 80, 1.0)
    assert mutator.needs_other
    ctx = Context(node="instrument", parent=a, parent_eval=None, partner=None, lineage=[], niches={}, empty_niches=[],
                  other=b, other_eval=None)
    [child] = mutator.propose(ctx)
    assert child.parent_id == a.id and child.parent2_id == b.id and child.mutator == "claude:recombine"
    [session] = rows(store, "SELECT request FROM sessions WHERE purpose LIKE 'mutate%'")
    content = json.dumps(json.loads(session["request"])["content"])
    assert "Instrument A" in content and "Instrument B" in content and "Pen plotter" in content and "Round brush" in content


def test_long_prompts_are_cut_at_a_paragraph_and_say_so():
    from conveyor.painting.problem import trim_prompt

    short = "Paint darks first."
    assert trim_prompt(short) == (short, None)
    long = "\n\n".join(f"Section {i}. " + "word " * 60 for i in range(30))
    text, original = trim_prompt(long, limit=1000)
    assert original == len(long.strip()) and len(text) <= 1000 and text.endswith("word") and not text.endswith(" ")
    assert text.count("Section") == text.count("\n\n") + 1  # whole sections only


def test_a_judge_that_returns_nothing_leaves_the_critic_score(env, monkeypatch):
    store, claude, setup, log, meter = env
    monkeypatch.setenv("FAKE_CLAUDE_JUDGE", "{}")
    painter = Painter(setup, store, claude)
    inst = make_instrument(ROUND, "round", setup, store, painter.target.height)
    p = painter.paint(inst, Organism(node="painter", genome={"prompt": prompts.INITIAL_STRATEGY}))
    assert p.viable and p.score == pytest.approx(p.critic["total"])
    assert "scored by the critic alone" in p.details["judge"]["error"]
    assert meter.sessions == 3  # the painting, then the judge twice
