from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from conveyor.events import EventSink
from conveyor.graph import Conductor


def _llm_client(args: argparse.Namespace):
    from conveyor.llm import PROVIDERS
    from conveyor.llm import LLMClient
    from conveyor.llm import load_api_key

    provider = PROVIDERS[args.provider]
    model = args.model or provider.default_model
    key = load_api_key(provider=provider.key)
    if not key:
        print(
            f"No {provider.label} key found. Put {provider.env_var}=... in .env at the project root, "
            "or export it in your shell.",
            file=sys.stderr,
        )
        sys.exit(2)
    # A few more model lanes than painting lanes, so mutator calls don't queue behind paintings.
    lanes = (getattr(args, "concurrency", None) or 8) + 4
    client = LLMClient(model, key, provider=provider.key, reasoning_effort=args.reasoning,
                       budget_usd=args.budget, max_concurrency=lanes)
    if not client.info.known:
        print(f"Warning: the catalog doesn't list {model} on {provider.label}, so spend reads $0 and "
              "--budget won't stop the run.", file=sys.stderr)
    elif not client.info.takes_images:
        print(f"Warning: {model} doesn't take images, and the painter shows the model its canvas.",
              file=sys.stderr)
    return client


def _run_painting(args: argparse.Namespace, on_ready=None) -> None:
    from conveyor.painting.problem import build_painting_graph

    llm = _llm_client(args) if args.painter == "llm" else None
    model = llm.model if llm else None
    width = args.width or (128 if llm else 80)
    cycles = args.cycles or (20 if llm else 25)
    board, nodes, edges, schedule = build_painting_graph(width=width, llm=llm, harness=args.harness,
                                                         strokes=args.strokes, parents=args.parents,
                                                         concurrency=args.concurrency,
                                                         paint_model=args.paint_model)
    name = args.name or (f"painting, {model}, {args.harness}" if llm else "painting, scripted")
    sink = EventSink(
        args.db,
        run_name=name,
        config={"cycles": cycles, "width": width, "schedule": schedule, "painter": args.painter,
                "harness": args.harness if llm else None, "model": model,
                "paint_model": (args.paint_model or model) if llm else None,
                "provider": args.provider if llm else None,
                "strokes": args.strokes if llm else None,
                "budget_usd": args.budget if llm else None},
    )
    conductor = Conductor(nodes, edges, sink, board, schedule, log=print)
    if llm:
        llm.sink = sink

        def over_budget(spent: float) -> None:
            print(f"Spent ${spent:.2f}, past the ${args.budget:.2f} budget. Stopping after the current iteration.")
            conductor.stop()

        llm.on_budget = over_budget
        painter_note = f"{args.paint_model} painting, {model} mutating" if args.paint_model else model
        print(f"Painting with {painter_note} through {llm.provider.label}, {args.harness} harness, "
              f"{args.strokes} strokes, reasoning effort {args.reasoning}, budget ${args.budget:.2f}, "
              f"{cycles} cycles.")
    print(f"Run {sink.run_id} writing to {Path(args.db).resolve()}")
    if on_ready:
        on_ready(sink.run_id)
    try:
        conductor.run(cycles)
    except KeyboardInterrupt:
        conductor.stop()
        print("Stopped.")
    finally:
        sink.close()
    for node_name, pop in conductor.populations.items():
        best, result = pop.top(1)[0]
        print(f"{node_name}: champion {str(best.id)[:8]} scored {result.score:.3f} ({len(pop.organisms)} organisms)")
    if llm:
        _print_usage(llm.usage)


def _print_usage(u) -> None:
    share = f", {u.cached_tokens / u.prompt_tokens:.0%} of input cached" if u.prompt_tokens else ""
    print(f"Model spend ${u.cost:.4f} over {u.calls} calls ({u.errors} failed), {u.prompt_tokens} tokens in"
          f"{share}, {u.completion_tokens} out ({u.reasoning_tokens} thinking).")


def _compare(args: argparse.Namespace) -> None:
    """Paint the same target with each harness under the same prompt, brushes, and stroke budget."""
    from functools import partial

    from conveyor.events import connect
    from conveyor.events import observing
    from conveyor.llm import Usage
    from conveyor.painting.canvas import default_targets
    from conveyor.painting.canvas import set_canvas_scale
    from conveyor.painting.canvas import to_png
    from conveyor.painting.critic import Critic
    from conveyor.painting.llm_agent import INITIAL_PROMPT
    from conveyor.painting.llm_agent import PromptStrategy
    from conveyor.painting.llm_agent import llm_paint
    from conveyor.painting.pi_harness import PiHarness
    from conveyor.painting.toolkit import initial_toolkit

    client = _llm_client(args)
    harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()]
    sink = EventSink(args.db, run_name=f"harness comparison, {client.model}",
                     config={"painter": "llm", "model": client.model, "provider": args.provider,
                             "harnesses": harnesses, "strokes": args.strokes,
                             "width": args.width, "reasoning": args.reasoning, "budget_usd": args.budget})
    client.sink = sink
    target = default_targets(width=args.width)[0][0]
    toolkit = initial_toolkit(fine=True, scale=set_canvas_scale(args.width))
    critic = Critic()
    strategy = PromptStrategy(prompt=INITIAL_PROMPT, n_strokes=args.strokes)
    painters = {"pi": PiHarness(client), "stateless": partial(llm_paint, client=client)}
    rows = []
    print(f"Run {sink.run_id}: {target.name} at {args.width}px, {args.strokes} strokes, {client.model} on "
          f"{client.provider.label}, reasoning {args.reasoning}.")
    for repeat in range(args.repeats):
        for harness in harnesses:
            before = Usage(**vars(client.usage))
            started = time.time()
            with sink.trace() as trace, observing(node=harness, organism_id=f"{harness}-{repeat}"):
                canvas = painters[harness](strategy, toolkit, target, seed=1000 + repeat, record=True)
            wall = time.time() - started
            report = critic.report(canvas, target, sink.store_artifact)
            after = client.usage
            row = dict(
                harness=harness, repeat=repeat, score=report["total"], pixel=report["pixel"], style=report["style"],
                wall=wall, calls=after.calls - before.calls, cost=after.cost - before.cost,
                tokens_in=after.prompt_tokens - before.prompt_tokens,
                cached=after.cached_tokens - before.cached_tokens,
                tokens_out=after.completion_tokens - before.completion_tokens,
                thinking=after.reasoning_tokens - before.reasoning_tokens, trace_id=trace.id,
            )
            sink.emit(harness, "comparison", None, **row, canvas=sink.store_artifact(to_png(canvas, 2)),
                      critic={**report, "train": True})
            rows.append(row)
            print(f"  {harness:9s} #{repeat}: score {row['score']:.3f} in {wall:.0f}s, {row['calls']} calls, "
                  f"${row['cost']:.4f}")
            if client.over_hard_cap():
                print("Budget exhausted; stopping the comparison.")
                break
    sink.emit(None, "run_finished", stopped_early=client.over_hard_cap())
    sink.close()

    print()
    print(f"{'harness':10s} {'score':>6s} {'pixel':>6s} {'style':>6s} {'time':>6s} {'calls':>5s} {'cost':>8s} "
          f"{'in':>7s} {'cached':>7s} {'out':>6s} {'think':>6s}")
    for r in rows:
        share = f"{r['cached'] / r['tokens_in']:.0%}" if r["tokens_in"] else "n/a"
        print(f"{r['harness']:10s} {r['score']:6.3f} {r['pixel']:6.3f} {r['style']:6.3f} {r['wall']:5.0f}s "
              f"{r['calls']:5d} ${r['cost']:7.4f} {r['tokens_in']:7d} {share:>7s} {r['tokens_out']:6d} {r['thinking']:6d}")
    _print_usage(client.usage)
    conn = connect(args.db, readonly=True)
    print(f"\nTranscripts (run `uv run conveyor serve --db {args.db}` first):")
    for r in rows:
        first = conn.execute("SELECT id FROM llm_calls WHERE trace_id=? ORDER BY started LIMIT 1", (r["trace_id"],)).fetchone()
        if first:
            print(f"  {r['harness']} #{r['repeat']}: http://127.0.0.1:8765/?run={sink.run_id}&call={first['id']}")


def main() -> None:
    # Progress lines should show up promptly even when stdout is redirected to a file.
    sys.stdout.reconfigure(line_buffering=True)
    from conveyor.llm import DEFAULT_PROVIDER
    from conveyor.llm import PROVIDERS

    parser = argparse.ArgumentParser(prog="conveyor")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="Run the painting graph and serve the dashboard while it runs")
    run = sub.add_parser("run", help="Run the painting graph without the dashboard")
    serve = sub.add_parser("serve", help="Serve the dashboard for an existing database")
    compare = sub.add_parser("compare", help="Paint one target with each harness and compare score, time and cost")

    def model_args(p: argparse.ArgumentParser, default_budget: float = 1.0) -> None:
        p.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER,
                       help="Where models are called. Each has its own API key; see the README.")
        p.add_argument("--model", default=None,
                       help="Model id. Defaults to the provider's: "
                            + ", ".join(f"{k} {v.default_model}" for k, v in sorted(PROVIDERS.items())))
        p.add_argument("--reasoning", choices=["none", "minimal", "low", "medium", "high"], default="low",
                       help="Thinking effort. On OpenRouter 'none' turns thinking off; elsewhere an effort the "
                            "model doesn't publish is dropped and the model decides for itself.")
        p.add_argument("--budget", type=float, default=default_budget,
                       help="USD. Past it the run stops after the current iteration; calls fail past 1.5x.")

    for p in (demo, run):
        p.add_argument("--cycles", type=int, default=None, help="Default 25 scripted, 20 with --painter llm")
        p.add_argument("--parents", type=int, default=None,
                       help="Parents sampled per iteration. Default 4 with --painter llm, 3 scripted.")
        p.add_argument("--concurrency", type=int, default=None,
                       help="Evaluations running at once per node, with 4 more model lanes. Default 8 for llm.")
        p.add_argument("--width", type=int, default=None,
                       help="Canvas width in pixels. Default 80 scripted, 128 llm. Brush sizes scale with it.")
        p.add_argument("--name", default=None)
        p.add_argument("--painter", choices=["scripted", "llm"], default="scripted")
        p.add_argument("--strokes", type=int, default=500,
                       help="Stroke cap per painting for --painter llm. Fixed during a run; evolution doesn't change it.")
        p.add_argument("--harness", choices=["pi", "stateless"], default="pi",
                       help="How the LLM painter runs: one Pi conversation (default) or a fresh request per turn")
        p.add_argument("--paint-model", default=None,
                       help="Paint with this model instead of --model, which then only runs the mutators. Pi "
                            "dispatches on the model's own API, so a responses-API model can paint.")
        model_args(p, default_budget=5.0)
    compare.add_argument("--harnesses", default="stateless,pi")
    compare.add_argument("--strokes", type=int, default=500)
    compare.add_argument("--width", type=int, default=128)
    compare.add_argument("--repeats", type=int, default=1)
    compare.add_argument("--db", default="runs/compare.db")
    model_args(compare)
    for p in (demo, run, serve):
        p.add_argument("--db", default="runs/conveyor.db")
    for p in (demo, serve):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.cmd == "run":
        _run_painting(args)
        return
    if args.cmd == "compare":
        _compare(args)
        return

    from conveyor.server import make_server

    if args.cmd == "serve":
        server = make_server(args.db, args.host, args.port)
        print(f"Dashboard at http://{args.host}:{server.server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return

    # demo: the database must exist before the server opens it read-only, so start serving once the run is set up.
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    holder: dict = {}

    def start_server(run_id: str) -> None:
        server = make_server(args.db, args.host, args.port)
        holder["server"] = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Dashboard at http://{args.host}:{server.server_port}/?run={run_id}")

    _run_painting(args, on_ready=start_server)
    if "server" in holder:
        print("Run finished. The dashboard stays up until you press Ctrl+C.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            holder["server"].shutdown()


if __name__ == "__main__":
    main()
