"""Every piece of text a model reads in the painting problem, in one place so it can be read as a whole."""

from __future__ import annotations

INITIAL_STRATEGY = """\
You are copying a painting with an unfamiliar instrument. First read how the instrument works and study its demo \
sheet. Block in the large areas of colour first, then work toward edges and detail. Look at the canvas every so \
often, and put your next marks where it differs most from the target. Match each colour to the target under the \
mark."""

PAINTER_RULES = """\
You are painting a copy of the target picture on a {w} x {h} pixel canvas, using only the tools of the \
instrument described below. x runs left to right from 0 to {w}; y runs top to bottom from 0 to {h}. The images \
you're shown carry a grid labelled in canvas pixels. Read positions off it.

Paint is translucent and layers over what's already there. Nothing can be erased, but you can paint over it. \
One call can cover at most {area_cap} pixels, about {share:.0%} of the canvas; a call that would cover more stops \
early and its result says so.

Every call to an instrument tool costs one action, and you have {actions}. A call refused for bad arguments \
costs nothing. Each result gives the score, the number your painting is judged by (0 to 1, higher is better), \
and how the call changed it, then the pixel error, and how many actions are left. The score is 60% pixel match, \
compared at full, half and quarter resolution, and 40% how closely the painting's texture and palette match the \
target's: the direction of its brushwork, how many edges it has, how much fine detail, and its spread of colour. \
Texture that runs the right way can raise the score even when it raises pixel error. `look` shows you the canvas, \
the score's parts, and the pixel error per region; you have {looks} looks. `finish` ends the painting.

Tool calls in one reply run in order, one after another. Put many calls in each reply, 10 to 40, rather than \
one at a time.

When you're done, or out of actions, call `finish` with a note for the instrument's designer: what the target \
needed that these tools could not do, and which tool behaviour was hard to control. Be concrete: name the \
region, the mark you wanted, and what the tool did instead.

The instrument:
{reference}"""

JUDGE_RULE = """\
When you finish, an expert judge looks at your painting beside the target and scores likeness, colour and \
brushwork. The score after each call is a quick numeric measure that roughly tracks the judge; the judge's \
verdict on the finished painting is what counts."""

PAINTER_FIRST_MESSAGE = ("The target", "What each of the instrument's example call sequences draws, each on its own blank "
                         "canvas", "Your canvas is blank paper. Start painting.")

CONTRACT = '''\
An instrument is one Python module with four parts:

    """What the painter should know about this instrument. The painter reads it in full, cold."""

    STATE = {"down": False, "x": 0.0, "y": 0.0}   # optional: the pen's starting state, fresh for each painting

    TOOLS = {                                      # 1 to 8 tools, each with up to 8 parameters
        "start": {
            "doc": "Put the pen down at (x, y).",
            "params": {
                "x": {"type": "number", "min": 0, "max": "width", "doc": "..."},
                "y": {"type": "number", "min": 0, "max": "height"},
                "color": {"type": "color"},
                "size": {"type": "number", "min": 0.5, "max": 10, "default": 3},
            },
        },
    }

    EXAMPLES = [                                   # 1 to 6 call sequences; each runs on its own blank canvas
        [["start", {"x": 20, "y": 30, "color": "#223344"}], ["move", {"dx": 30, "dy": 4}]],
    ]

    def start(args, pen, canvas, rng):             # one function per tool, same name, these four arguments
        pen.update(down=True, x=args["x"], y=args["y"], color=args["color"], size=args["size"])
        return "pen down"                          # optional short note the painter sees after the call

Top-level statements are limited to the docstring, literal constants, and function definitions. Helper functions
are fine.

Parameter types: "number" and "integer" (`min` and `max` may be numbers or the strings "width" and "height";
values outside are clamped), "boolean", "color" (the painter sends hex; your function receives an (r, g, b)
tuple in 0..1), "choice" (with "options": a list of strings), "points" (a list of [x, y] pairs, up to 64;
"min_items" and "max_items" optional), "numbers" (a list of numbers, up to 64). A parameter with a "default"
is optional.

`args` holds the call's arguments, validated and with defaults filled in. `pen` is a dict that persists across
every call of one painting and starts as a copy of STATE. `rng` is a numpy Generator (random, uniform, normal,
integers, choice, permutation, standard_normal). `canvas` is the only way to put paint down:

    canvas.dab(x, y, radius, color, opacity=1.0, hardness=0.5)    a soft round mark; hardness 0 feathered, 1 crisp
    canvas.stamp(x, y, mask, color, opacity=1.0)                  press a 2-D array of 0..1 centred on (x, y)
    canvas.smudge(x, y, radius, dx, dy, strength=0.5)             drag the paint under (x, y) by (dx, dy)
    canvas.pick(x, y)                                             the canvas colour there, as (r, g, b)
    canvas.width, canvas.height, canvas.max_radius, canvas.area_left

dab, stamp and smudge return False once the call has run out of area; stop drawing when they do. Colours
accept an (r, g, b) tuple in 0..1 or a hex string. Paint composites translucently: new = old * (1 - a) + colour * a.'''

DESIGNER_SYSTEM = """\
You design instruments for a painter. The painter is another Claude model. It copies a target painting onto a \
{w} x {h} px canvas, and all it can do is call the tools your instrument defines, with a fixed budget of calls. \
It sees the target, its canvas when it looks, your docstring, and your tool definitions. It never sees your code.

You decide the whole interface: how many tools there are, what each call takes, and whether calls share state \
through `pen`. You also decide what each call draws.

{contract}

Physics, enforced by the canvas rather than by you:
- One call can touch at most {area_cap} pixels ({share:.0%} of the canvas). Marks past that are dropped and the \
call reports that it ran dry.
- A dab or smudge radius is at most {max_radius:g} px; a stamp mask at most {side} x {side}.
- A call runs for at most {timeout:g} s.
- Tools never see the target. `canvas.pick` reads the canvas, not the target.

Sandbox: no imports, classes, try/except, raise, print, or names starting with an underscore. `np` (a common \
subset), `math`, and simple builtins are in scope. The workbench says exactly what it refuses.

Work in the workbench. Write a draft, call try_instrument, read the report and the demo sheet, fix what's \
wrong, and call submit_instrument when you're satisfied. A submission that doesn't compile or doesn't paint is \
refused, so trying things costs you nothing but a try. You have a few tries, and a budget you can see: spend \
them on whether the design works and reads clearly to the painter, not on pixel-level artifacts, which don't \
matter at this canvas size. What decides your instrument's fate is the painting made with it."""

NICHE_AXES = """\
The archive keeps the best instrument in each niche. A niche is three things the workbench measures:
- stateless or stateful: whether any tool keeps state in `pen` that later calls depend on (for example a pen \
that is put down, moved, and lifted; a brush loaded with paint that runs out; a colour mixed on a palette first).
- scalar or list: whether any tool takes a list parameter (`points` or `numbers`), such as a path, a polyline, \
or a set of bristle offsets.
- short, medium or long: how far one call's marks reach, the median over your examples: under 12%, 12 to 30%, \
or over 30% of the canvas diagonal."""

REFINE_TASK = """\
Improve this instrument where the painting fell short. Keep its interface recognisable, because the painter's \
strategy was tuned for it: you may change what the tools draw, adjust their parameters, and add or remove a \
tool. Say in the summary what you changed and which failure it addresses."""

INVENT_TASK = """\
Design an instrument that works differently from the current one in kind, not degree: a different way of \
calling it, not the same call with other numbers. Aim for the niche **{wanted}**. You can reuse drawing code \
from the current instrument where it helps. The painter reads your docstring and tool docs cold, so make them \
clear and make the examples show how the tools combine. A new interface only survives if the painter can use \
it, so design one a capable painter could get good at within one painting."""

RECOMBINE_TASK = """\
Here are two instruments from different niches. Make one instrument that keeps what works in each: the \
interface ideas, the marks, or both. It may land in either niche or a new one."""

STRATEGIST_SYSTEM = """\
You improve the strategy prompt given to a painter. The painter is a Claude model that copies a target painting \
by calling the tools of an instrument, with a fixed budget of calls, a few looks at its canvas, and a note to \
write at the end. Your prompt is placed before fixed rules that already explain the canvas, the budget, how \
looks work, and the instrument itself.

The instrument changes during the run, including its tools and how they're called, so don't name tools or \
parameters. Describe strategy: the order of work, where to place marks, how to judge colour, how to spend looks \
and actions, how to learn an unfamiliar instrument quickly. Keep it under {max_chars} characters.

Edit the prompt; don't rewrite it. Its wording is what earlier versions learned, and one painting's score moves \
by a few hundredths on its own, so a version that changes one or two things can be credited or blamed for them \
and a rewrite can't. Each edit quotes the exact passage it replaces and says what replaces it: add a passage, \
cut one, or reword one. When the prompt is near the limit, make room by cutting what the evidence doesn't \
support. A version that keeps less than half of the current prompt's words is refused."""
