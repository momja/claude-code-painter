# conveyor

Run a graph of [darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver) nodes and watch what each node is doing.

Each evolving node owns its own `Evolver` and population. Nodes score their organisms against the current champions of their partner nodes. When a champion changes, the nodes that depend on it get their top organisms rescored, because their stored scores were measured against a partner that's gone. Every mutation, evaluation, rescore, and tool call goes into one SQLite event log, and a local dashboard reads that log while the run goes.

The repo ships with a working example: the two-component robotic painting problem, with van Gogh's *Self-Portrait* (1889) as the training target and *The Starry Night* (1889) held out. It runs on numpy and needs no API keys.

## Quick start

```sh
uv sync
uv run conveyor demo            # runs 25 cycles and serves http://127.0.0.1:8765 while it runs
uv run conveyor serve           # browse earlier runs in runs/conveyor.db
uv run --group dev pytest       # tests
```

A 30-cycle demo takes about a minute and a half on a laptop.

## Painting with an LLM

The painter runs through [OpenCode Go](https://opencode.ai/docs/go/) by default, and OpenRouter is still
wired up. Put the key for whichever you use in `.env` at the project root (it's gitignored):

```
OPENCODE_API_KEY=...        # default
OPENROUTER_API_KEY=sk-or-...   # for --provider openrouter
```

Then:

```sh
(cd pi-painter && npm install)                              # once: the Pi harness needs Node 20+
uv run conveyor demo --painter llm                          # glm-5.3-flash on OpenCode Go, Pi harness, 20 cycles, $5 budget
uv run conveyor demo --painter llm --parents 6 --concurrency 12   # wider: more paintings at once
uv run conveyor demo --painter llm --harness stateless      # the older fresh-request-per-turn painter
uv run conveyor compare --repeats 2                         # paint one target with each harness, print a table
uv run conveyor demo --painter llm --cycles 6 --budget 3 --reasoning none
uv run conveyor demo --painter llm --model qwen3.8-max      # any catalogued model that takes images and tools
uv run conveyor demo --painter llm --provider openrouter   # same run, called through OpenRouter instead
uv run conveyor demo --painter llm --paint-model muse-spark-1.3-contributor   # paint and mutate on different models
```

Both providers speak OpenAI-compatible `/chat/completions`, but not identically, and `llm.py:Provider` holds
the three differences: OpenRouter takes `reasoning: {"effort": ...}` and prices each reply in `usage.cost`
when asked; OpenCode Go takes a plain `reasoning_effort`, wants `stream_options` to report usage at all, and
sends no prices. So spend on OpenCode Go is worked out here, from tokens and the per-million rates in
[models.dev](https://models.dev) (`catalog.py`). `--budget` means the same thing on both. A model the catalog
doesn't list runs fine but reports no cost, and the CLI says so rather than letting `--budget` look enforced.

Efforts differ too. Each model publishes the ones it takes (`glm-5.3-flash` takes low, high and max), and an
effort outside that list is dropped rather than sent and rejected, leaving the model to decide. Only
OpenRouter reads `--reasoning none` as thinking off.

In this mode:

- **The agent node is a real LLM painter.** Each turn it sees the target and its canvas as images, plus the error per region, and answers with brush calls. Each brush in the current toolkit is one tool. Each painting gets 500 strokes (`--strokes`). The cap stays fixed during a run: extra strokes almost always lower pixel error, so a budget evolution could change would only grow. The oracle fitter gets at least the same budget per patch, so it never loses to the agent it's judging.
- **The LLM painter starts with fine brushes.** Besides the wash and the round, the toolkit starts with an edge brush and a liner. All four are plain capsules, so a run starts from the only mark the old parameter-only toolkit could make and has to write the rest. At 64 px the eyes and the dark lapel lines are 2 to 3 px wide. With only 7 and 14 px brushes, a live run found that every dark stroke raised the error, and it washed the painting out instead. The scripted demo keeps the thin toolkit so it can show the toolkit evolving liners.
- **The held-out painting is for champions only.** Ordinary evaluations paint just the training target. When an organism becomes champion, the runner paints *The Starry Night* for it once and records a `holdout` event, which the overfitting alarm reads. It checks the first champion and then every `holdout_every`-th one (3 for the LLM painter, since champions change often early and each check costs a full painting). Before this, every evaluation painted both targets, which doubled the cost for a number only the champion alarm used.
- **Error regions are labelled with pixel ranges** ("x 32-48, y 0-16"), both in the grid the painter sees and in the worst-regions text.
- **The canvas is 128 px for the LLM painter** (`--width`), and brush sizes and their limits scale with it, so a wash covers the same fraction of the picture as it did at 64 px and 500 strokes still cover the canvas.
- **The model's images carry a coordinate grid.** The target, the starting canvas, and every `look` get about eight labelled divisions across (16 px apart at 128), in a margin whose size and font scale with the image. At 64 px the labels were too cramped to read, and one live run placed a 4 px ear 10 px off.
- **Strokes take a `pressure`** from 0.1 to 1 (default 1), which scales how much pigment goes down, so detail can be a light touch instead of a dark bar. The oracle fitter samples pressure too, so it still covers everything the painter can do.
- **The agent's organism is its prompt.** `LLMPromptMutator` rewrites the prompt using the regions the painter got wrong and the learning log.
- **The toolkit node mixes mutators.** `LLMToolkitMutator` writes brush code and competes with the rule-based ones. The mutator table shows both what each one's children achieve and what it costs.
- **Every model call is recorded** with tokens, cost, latency, and errors. The header shows total spend, each node card shows its own, and the replay shows each call's reasoning next to the strokes it made. An alarm fires when more than 10% of a node's calls fail.
- **You can watch the model work.** The client streams every call and stores it in full: the exact request (the images the model saw are kept), the complete thinking, the reply, every tool call with its arguments, what happened to each one (applied and how it changed error, rejected and why, or ignored for going over the per-turn limit), and tokens, thinking tokens, cost, time to first token, and latency.
  - "Model calls in flight" on the main page shows each running call with its thinking as it streams.
  - Any call opens as a transcript. You can get there from the node's recent-calls list, from an organism's evaluations, from any stroke in the replay (it links to the call that made it), and from an organism an LLM mutator wrote (it links to the call that wrote it).
  - `--reasoning` controls thinking effort. With `none` there's no thinking to show.
- **`--budget` is soft.** Once spend passes it, the run stops after the current iteration. At 1.5 times the budget, calls fail outright.

### The harnesses

**Pi (default).** Each painting is one conversation run by [Pi](https://github.com/badlogic/pi-mono)'s `Agent`, in a Node sidecar (`pi-painter/painter.mjs`, one process per painting). The model gets one tool per brush, plus `look` and `finish`. A stroke's result says how it changed the total error. `look` returns the canvas as an image, along with the error for each region. A painting gets one look per 40 strokes, and at least 6, so 13 at 500 strokes. A provider's image limit trims that: the conversation keeps every image it sends, and GLM through OpenCode Go takes 8 per request (400 `too_many_images` above it), so with the target and the starting canvas always present a painting there gets 6 looks. The sidecar also drops the oldest images if a conversation somehow goes over. The rules ask the model to put up to 40 brush calls in each reply, because every turn re-sends the whole conversation. `PiHarness(min_stroke_fraction=...)` can refuse `finish` until most strokes are used, but it's off by default. With blunt brushes it only bought filler: in one live run the model spent its last 150 required strokes on beige washes that faded the painting. `finish`, or using the last stroke, ends the conversation. Python (`painting/pi_harness.py`) owns the canvas: the sidecar sends each tool call over stdio and waits for the result. Each model turn streams back and gets recorded as a call in the same conversation.

Caching comes from Pi's `sessionId`, which Pi sends upstream as `x-session-id`. That keeps every turn of a painting on the same provider, so the provider's prefix cache can hit. The conversation is never trimmed, since dropping old canvas images would change the prefix and throw the cache away. Every transcript shows how many input tokens came from the cache, and the header shows the run's total share. Pi ships its own table for both providers; for a model the installed pi-ai predates, `pi_harness.py` builds the definition, prices included, from the same models.dev catalog the Python client uses.

**Stateless.** `painting/llm_agent.py:llm_paint` sends a fresh request every turn, containing the prompt, the target, the canvas, the error grid, and recent strokes, and offers one tool per brush with `tool_choice: "required"`. It's simpler, but the model re-derives the whole scene every turn and only sees what its strokes did when the next turn starts.

**Comparing them.** `conveyor compare` paints the same target with both harnesses under the same prompt, brushes, and stroke budget. It prints score, time, calls, cost, cache share, and thinking tokens, and the dashboard shows the same table with each painting and a link to its transcript.

The mutators each make one forced tool call (`revise_prompt`, `revise_toolkit`) through `conveyor/llm.py`, which handles HTTP, streaming, retries, and recording for everything that isn't Pi.

## Brushes are programs

A brush is not a row of sliders. It is a name, how far its mark reaches from the stroke path (`radius`), how
long that path is (`length`), and a small Python program that draws the mark:

```python
def alpha(u, v, rng, radius, length):
    """A broken line of separate dots."""
    period = max(radius * 2.20, 0.5)
    phase = np.abs(np.mod(u, period) - period * 0.5)   # distance to the nearest dot centre
    d = np.hypot(phase, v)
    ramp = max(radius * 0.35, 0.5)
    a = np.clip((max(radius * 0.55, 0.5) - d) / ramp + 0.5, 0.0, 1.0)
    a = a * ((u > -radius) & (u < length + radius))     # dots only along the path, not past its ends
    return np.clip(a * 0.80, 0.0, 1.0)
```

`u` is distance along the path, `v` is distance across it, both in canvas pixels, and the return value is how
much pigment lands at each one. Everything else stays with the caller: where the stroke goes, its angle, the
bounding box, clipping, compositing. This is the whole reason a mutator can invent a mark rather than retune
one. Modulating on `u` breaks a line into dots. Modulating on `v` gives the parallel bristles that make up
most of a van Gogh surface. Scaling the reach by `u / length` tapers the stroke to a point. None of those
were reachable when a brush was six numbers fed to one fixed capsule.

- **The scripted mutators write code too**, so the no-API-key demo still evolves. `TargetedToolkitMutator`
  fills templates (`brushcode.py:TEMPLATES`) for the mark the failure type calls for, and keeps which
  template and which numbers it used, so a later edit can still change "softness" by name.
  `RandomToolkitMutator` mostly scales one numeric literal in the source, which works just as well on code an
  LLM wrote, where there are no named knobs at all.
- **`LLMToolkitMutator` writes free-form source.** It gets the contract, the sandbox rules, and a plain
  capsule to work from, and returns whole brushes.
- **The sandbox is not a security boundary.** Brush code runs in this process with no imports, no `while`, no
  underscored names, and nothing in scope but `np`, `math`, `rng` and a few builtins, with module attributes
  on an allowlist because numpy has plain-named doors out (`np.ctypeslib.ctypes`). That is enough to stop a
  mutator wandering somewhere it shouldn't by accident. It would not stop someone trying. Don't point it at
  brush code you didn't generate.
- **A brush that misbehaves costs its toolkit, not the run.** `physics_violations` compiles every brush and
  runs it once on a small grid, so code that won't compile, returns the wrong shape, leaves 0 to 1, or lays
  no pigment marks the toolkit non-viable with a reason the dashboard shows. A brush that only breaks on some
  later grid shape lays nothing and gets selected out. The size limits still bite: whatever the program does,
  it only draws inside the box `radius` and `length` bought it.
- **It costs about 9% per stroke** over the fixed capsule it replaced, measured against the old inline
  version of the same maths.

## What the dashboard shows

- **Graph.** One card per node, showing what its current champion produces. The toolkit node shows a swatch sheet of its brushes, the agent node its painting, and the critic its error map. Cards also show score, iteration count, and live status (evolving, rescoring after a partner changed, waiting). Edges show artifact flow and failure feedback.
- **Alarms.** Stalled champions, mutators whose children rarely beat their parent, mutators whose proposals fail verification, mutators that throw, bursts of non-viable organisms, training score rising while held-out falls, and rescores that drop because a partner changed.
- **Who gets the blame.** How many patches the critic blamed on the toolkit and on the agent over time.
- **Per node.** Score percentiles with champion changes marked, failures by type, the full lineage tree (crossover drawn dashed, non-viable organisms hollow), a mutator yield table, the learning log, and the rescore history.
- **Per organism.** Click any organism. The panel shows its sub-scores, what changed and why, a diff against its parent, output images, the critic's breakdown at each resolution (target, painting, and error map side by side, with the numbers that add up to the score), the failing patches (target, what the oracle managed with these tools, the painting), every evaluation with the partner versions it ran against, and a replay of each tool call in the painting run.

## Wiring your own nodes

```python
from conveyor import Board, Conductor, Edge, EventSink, Node, artifact, current_trace

board = Board()   # evaluators read partner champions from here: board.elites("agent", k=2)

nodes = [
    Node("toolkit", initial_organism=..., evaluator=ToolkitEvaluator(board), mutators=[...],
         partners=["agent"], partner_k=2, verify_mutations=True),
    Node("agent", initial_organism=..., evaluator=AgentEvaluator(board), mutators=[...],
         partners=["toolkit"], partner_k=1),
    Node("critic", fixed=True, version="v1", mirror="agent", thumbnail_artifact="heat"),
]
edges = [Edge("toolkit", "agent", "artifact", "brush functions"), Edge("critic", "toolkit", "feedback", "failures")]

sink = EventSink("runs/conveyor.db", run_name="my run")
Conductor(nodes, edges, sink, board, schedule=[("agent", 2), ("toolkit", 1)]).run(cycles=40)
sink.close()
```

Your organisms, evaluators, and mutators stay plain darwinian_evolver classes. For the dashboard to show more, they can add the following. All of it is optional.

| Hook | Where it shows up |
|---|---|
| `Organism.render_text()` | The diff against the parent. Without it the diff uses the problem fields as JSON. |
| `EvaluationResult.visualizer_props` | Sub-scores in the organism panel. Keys starting with `blame_` feed the blame chart. A `holdout` key turns on the overfitting alarm. |
| An `artifacts: dict[str, str]` field on your result | Images. `thumb` goes on the graph card. Keys named `canvas_*`, `oracle_*`, and `heat_*` show up in the organism panel. |
| `artifact(png_bytes)` inside `evaluate()` | Stores the image and returns the name to put in `artifacts`. |
| `current_trace().span(name, args, result, image=...)` inside `evaluate()` | One tool call in the replay. |
| `change_summary` starting with `[tag]` | The learning log. The painting mutators use tags to avoid repeating a change that made an ancestor worse. |

## Layout

```
src/conveyor/
  events.py       event log, spans, content-addressed images, all in one SQLite file per database
  observe.py      ObservedMutator / ObservedEvaluator wrappers around your classes
  population.py   RescorablePopulation: WeightedSamplingPopulation whose stored scores can be replaced
  graph.py        Node, Edge, Board, Conductor (scheduling, champion tracking, rescoring)
  server.py       read-only HTTP API + alarms, standard library only
  dashboard/      the single-page dashboard
  painting/       the example problem: canvas physics, critic, toolkit node, agent node
    brushcode.py  the brush sandbox: what a brush program may do, and the templates the scripted mutators fill
```

## The painting example, and where it cuts corners

- **The agent isn't an LLM.** `painting/agent.py:paint` is a scripted greedy painter, and `Strategy` plays the part of its prompt. To use a real agent, replace `paint()` with a tool-use loop that calls the toolkit's brushes and records a span per call. The graph, the blame split, and the dashboard stay the same.
- **The mutators aren't LLMs either.** They apply rules: they read the failure type and try the change a person would try first, then write it as brush code from a template. Swap in LLM calls that get the failing patches and the learning log.
- **The critic is a deterministic proxy.** The pixel score compares RMSE at three scales. The style score compares texture statistics. For real use, replace the style half with pairwise VLM judgments.
- **Blame uses an oracle fitter.** A greedy search finds the best each patch can look with the current brushes. If even that is far from the target, the toolkit is at fault. If the oracle gets close and the painting doesn't, the agent is at fault. The thresholds sit at the top of `painting/problem.py`.
- **The oracle tries every brush, the same number of times each** (`TRIES_PER_BRUSH`). It used to draw one brush at random and try a fixed six placements however many brushes the toolkit held, so each extra brush got fewer tries and a larger toolkit fit worse. Blame then ran backwards: adding a rake, dots and a taper to the four seed brushes moved the toolkit's blamed patches from 19 up to 21, so improving the toolkit raised its own blame and the blame chart sat flat all run. It now costs more to fit a big toolkit, which is the price of the number meaning something.

## Known limits

- A rescore covers the top `rescore_top_k` organisms, then anything with a stale score that climbs to the top. Everything else keeps its stale score until it gets resampled.
- When a node is evaluated against several partner elites, blame comes from the first pairing only, which is the champion pairing.
- The dashboard's "beat parent" counts only children whose parent was last scored against the same partner versions. The rest show as "not comparable". darwinian_evolver's own learning log has no such check. It compares a child with the parent's stored result, and that result can be stale for parents outside the rescored top-k. A mutator reading the log can therefore see "better than the parent" when the partner did the improving.
- **The blame split reads as fault, but the agent's share is inflated.** The oracle fits each patch on its own, samples the ideal colour from the target for every stroke, and gets `OracleFitter.STROKES` (18) per patch where the scripted agent spends about 2. It outspends the agent roughly 9 to 1 on the very patches it then rules the tools can handle. `GAP_THRESHOLD` is an absolute gap against that, so a good share of "the agent's fault" is really the funding gap. Fixing it means either a threshold that accounts for the gap, or comparing the agent against a second fit given the agent's own per-patch spend, which cannot share the per-toolkit cache.
- **Toolkit blame still moves with `OracleFitter.STROKES`.** At 2 strokes per patch the toolkit is blamed for every patch, at 96 for almost none, with nothing about the brushes or the painting changing. The constant is a real assumption about how many strokes a patch deserves, not a neutral default.
- One process, one SQLite writer per run. The dashboard opens the database read-only, so it can run from another process.
