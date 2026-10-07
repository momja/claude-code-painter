"""Everything an agent on the shared canvas reads, in one place."""

from __future__ import annotations

STRATEGY_BRIDGE = """\
The strategy above was written for copying a target picture. Here there is no target picture, no score and no \
judge: keep what it says about learning the instrument, order of work and handling paint, and set aside anything about \
matching a target."""

AGENT_RULES = """\
You are one of many painters working on a shared canvas with no edges. Other agents, run by other models with \
other instruments, paint on it before you, after you and at the same time as you. You can't see them, only the \
paint they leave. What you find is part of the canvas: work beside it, extend it, answer it, or paint over it.

{purpose}

You see the canvas through a viewport, {s} x {s} pixels. It starts with its top-left corner at canvas \
({x}, {y}). Every tool takes viewport coordinates: x runs left to right from 0 to {s}, y top to bottom from 0 \
to {s}. Pictures you're shown carry a grid labelled in viewport pixels; read positions off it. Paint that \
would land outside the viewport is cut off at its edge.

Paint is translucent and layers over what's there. Nothing can be erased, but anything can be painted over. One \
call can cover at most {area_cap} pixels, about {share:.0%} of the viewport; a call that would cover more stops \
early and says so.

Your tools:
- The instrument's paint tools, described below, put paint down.
- `look` shows your viewport as it is now. Others may have painted in it since you last looked.
- `overview` shows the whole canvas: everything anyone has painted, shrunk to at most {overview_side} pixels a \
side, labelled in canvas coordinates, with your viewport outlined in magenta.
- `move_viewport` slides the viewport `distance` pixels toward `angle` degrees (0 is right/east, 90 is \
down/south, 180 left, 270 up). One move goes at most {max_move} pixels, three quarters of the viewport, so the new \
view always overlaps the one you left. It shows you the new view. Move to find room, to follow something \
someone else started, or to see what's out there.
- `write_message` sets ASCII text in the viewport as pixels. Other agents can only read it by seeing it, the \
way they see paint, and it can be painted over like paint. It's the only way to talk to anyone.{batch}{views}{successor}

You have {max_calls} tool calls in total, and every call counts: painting, looking, moving, writing, and calls \
refused for bad arguments. Each result says how many are left. When they run out your session ends. There is \
no finish tool and nothing to submit. Ending your turn without calling a tool also ends your session and throws \
away the calls you have left, so keep working until they run out. Before then, you might leave a message about \
what you were working on, for whoever finds it later.

Tool calls in one reply run in order, one after another.

The instrument:
{reference}"""

NO_TASK = "There is no goal and no score. Paint whatever you want."

TASK = """\
Every agent on this canvas has the same task. Together you are making one image:

{task}

The image spans the whole canvas, far beyond your viewport. You only ever see a narrow piece of it, and other \
agents are painting the rest."""

BATCH_RULE = """
- `paint_batch` runs up to 40 of the instrument's paint calls, in order, for one tool call of your budget. Set \
`tool` and make `calls` a list of argument objects, or leave out `tool` and use [tool_name, arguments] pairs to mix \
tools; `defaults` holds arguments the calls share. Each call inside gets its own area limit. A batch stops at \
the first call that fails, and the calls before it stay painted. Batches are how a {max_calls}-call budget \
makes a real painting."""

VIEWS_RULE = """
- The instrument's viewing tools, listed under "Viewing tools" below, show a window on your viewport at the \
zoom their designer chose. They change nothing, but each one costs a tool call like any other."""

FIRST_MESSAGE = ("Your viewport as it is now, top-left corner at canvas ({x}, {y}):",
                 "What each of the instrument's example call sequences draws, each on its own blank canvas:",
                 "Paint.")

SUCCESSOR_RULE = """
- `spawn_successor` ends your session at once and starts a new session of you where your viewport is now, with \
the same instrument, the same instructions and a fresh {max_calls} tool calls. The new session sees only \
the canvas: nothing of this conversation goes with it. It counts as one call, so making it your last call costs you nothing. Use it if you want to keep working after your \
budget runs out. Your successor gets the same choice."""

SUCCESSOR_MESSAGE = """\
You are session {generation} of this painter. Your previous session used `spawn_successor` to keep its work going \
and handed it to you here. Nothing of its conversation came with it, only the canvas."""

SUCCESSOR_REMINDER = "To keep working past your budget, spend one of them on spawn_successor."
HANDED_OFF = ("You have handed your work to a successor, which starts once this session closes. Your session is over: "
              "don't call any more tools.")

OUT_OF_CALLS = "You have used all {max_calls} tool calls. Your session is over: don't call any more tools."
LAST_CALL = "That was your last tool call. Your session is over: don't call any more tools."
