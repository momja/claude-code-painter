from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _setup(args):
    from conveyor.painting.problem import Setup

    db = Path(args.db).resolve()
    weights = {"refine": getattr(args, "refine", 0.4), "invent": getattr(args, "invent", 0.4),
               "recombine": getattr(args, "recombine", 0.2)}
    return Setup(target=args.target, width=args.width, actions=args.actions, looks=args.looks,
                 mode="offline" if getattr(args, "offline", False) else "model",
                 seeds=[s.strip() for s in getattr(args, "seeds", "round").split(",") if s.strip()],
                 work_dir=db.parent / f"{db.stem}-sessions", parents=getattr(args, "parents", 2),
                 paint_budget_usd=getattr(args, "paint_cap", 8.0), mutate_budget_usd=getattr(args, "mutate_cap", 4.0),
                 operator_weights=weights, judge=getattr(args, "judge", False),
                 judge_weight=getattr(args, "judge_weight", 0.5), confirm=max(1, getattr(args, "confirm", 3)),
                 scope_views=getattr(args, "scope", False))


def _stall_timeout(args) -> float | None:
    from conveyor.harness import DEFAULT_STALL_TIMEOUT

    seconds = getattr(args, "stall_timeout", DEFAULT_STALL_TIMEOUT)
    return seconds if seconds and seconds > 0 else None


def _harness(kind: str, model: str | None, effort: str | None, args, store, meter, cache: dict):
    """One harness per distinct configuration, shared by every role that asks for it."""
    key = (kind, model, effort, args.provider if kind == "pi" else None)
    if key in cache:
        return cache[key]
    if kind == "claude":
        from conveyor.claude import DEFAULT_MODEL
        from conveyor.claude import ClaudeCode
        from conveyor.claude import Settings
        from conveyor.claude import available

        if available(args.claude) is None:
            sys.exit(f"Can't run `{args.claude} --version`. Install Claude Code, use --harness pi, or pass --offline.")
        harness = ClaudeCode(Settings(binary=args.claude, model=model or DEFAULT_MODEL, effort=effort or "high",
                                      autocompact=getattr(args, "autocompact", None),
                                      stall_timeout=_stall_timeout(args)),
                             store, meter, lanes=args.lanes)
    else:
        from conveyor.pi import PiAgent
        from conveyor.pi import available

        problem = available()
        if problem:
            sys.exit(f"The Pi harness can't run: {problem}.")
        harness = PiAgent(model, provider=args.provider, effort=effort or "high", store=store, meter=meter,
                          lanes=args.lanes, compact_every_looks=getattr(args, "compact_every_looks", None),
                          stall_timeout=_stall_timeout(args))
        if not harness.authenticated:
            if harness.provider.login_command:
                sys.exit(f"{harness.provider.label} is not signed in. Run `{harness.provider.login_command}`.")
            sys.exit(f"No {harness.provider.label} key. Set {harness.provider.env_var}, or put "
                     f"{harness.provider.env_var}=... in a .env file here or in a parent directory.")
    cache[key] = harness
    return harness


def _roles(args, store, meter, *, paint: bool = True, mutate: bool = True, judge: bool = True):
    """
    The harness for each role. `--harness`, `--model` and `--effort` set every role; `--paint-*`, `--mutate-*`
    and `--judge-*` override one. A role on a different harness from the global one doesn't inherit `--model`:
    a Claude model id means nothing to Pi and the other way round.
    """
    from conveyor.painting.problem import Roles

    cache: dict = {}

    def role(prefix: str):
        kind = getattr(args, f"{prefix}_harness", None) or args.harness
        model = getattr(args, f"{prefix}_model", None) or (args.model if kind == args.harness else None)
        effort = getattr(args, f"{prefix}_effort", None) or args.effort
        return _harness(kind, model, effort, args, store, meter, cache)

    roles = Roles(paint=role("paint") if paint else None, mutate=role("mutate") if mutate else None,
                  judge=role("judge") if judge and getattr(args, "judge", False) else None)
    for name, h in (("paint", roles.paint), ("mutate", roles.mutate), ("judge", roles.judge)):
        if h is not None:
            print(f"  {name:7s} {h.describe()}")
    print(f"  {args.lanes} sessions at a time per harness.")
    return roles


def _stop_all(roles) -> None:
    for h in roles.all() if roles else []:
        h.stop_all()


def _serve_in_background(db: Path, host: str, port: int, run_id: str):
    from conveyor.server import make_server

    server = make_server(db, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Dashboard at http://{host}:{server.server_port}/?run={run_id}")
    return server


def _money(x: float) -> str:
    """Dollars, with enough places that a cheap model's spend doesn't read as zero."""
    return f"${x:.2f}" if x >= 0.1 or x == 0 else f"${x:.4f}"


def _config(args) -> dict:
    return {k: v for k, v in vars(args).items() if k != "func"}


def _hold(server) -> None:
    if server is None:
        return
    print("The dashboard stays up until you press Ctrl+C.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


def cmd_run(args) -> None:
    from conveyor.claude import Meter
    from conveyor.evolve import Conductor
    from conveyor.painting.problem import build
    from conveyor.store import Store

    # A parent that ignores Ctrl+C (a background job, or the dashboard that started this run) passes that on to
    # its children; the run needs the signal to stop its sessions cleanly.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    setup = _setup(args)
    offline = setup.mode == "offline"
    store = Store(args.db, run_name=args.name or ("offline" if offline else f"{args.harness} {args.model or 'default model'}"),
                  config=_config(args))
    wait = args.wait if args.wait and args.wait > 0 else None

    def waiting(reason: str, until: float) -> None:
        store.emit("waiting", reason=reason, until=until)
        print(f"Waiting until {time.strftime('%a %H:%M', time.localtime(until))}: {reason}.")

    def resumed() -> None:
        store.emit("resumed")
        print("The usage limit has reset; carrying on.")

    meter = Meter(budget_usd=None if offline else args.budget, max_usage=args.max_usage, wait_hours=wait,
                  on_wait=waiting, on_resume=resumed)
    roles = None if offline else _roles(args, store, meter)
    print(f"Run {store.run_id} writing to {Path(args.db).resolve()}")
    server = None if args.no_serve else _serve_in_background(Path(args.db), args.host, args.port, store.run_id)
    nodes, _ = build(setup, store, roles)
    conductor = Conductor(nodes, store, lanes=args.lanes, schedule=[("painter", 1), ("instrument", 1)], log=print,
                          on_stop=lambda: _stop_all(roles), should_stop=meter.should_stop,
                          wait_out_limits=wait is not None)
    done = threading.Event()

    def report_spend() -> None:
        while not done.wait(15):
            store.emit("spend", spent=round(meter.spent, 4), sessions=meter.sessions, rate_limit=meter.rate_limit)

    threading.Thread(target=report_spend, daemon=True).start()
    try:
        conductor.run(args.cycles)
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the Claude sessions.")
        conductor.stop("interrupted")
        _stop_all(roles)
    finally:
        done.set()
        store.emit("spend", spent=round(meter.spent, 4), sessions=meter.sessions, rate_limit=meter.rate_limit)
        store.flush()
    for name, pop in conductor.pops.items():
        champ = pop.champion()
        if champ:
            print(f"{name}: champion {champ[0].id[:6]} scored {champ[1].score:.3f} "
                  f"({len(pop.organisms)} organisms; niches {', '.join(sorted(pop.niches())) or '-'})")
    if not offline:
        print(f"Model spend: {_money(meter.spent)} over {meter.sessions} sessions (list price for Claude Code).")
    store.close()
    _hold(server)


def _instrument_source(spec: str) -> str:
    from conveyor.painting.seeds import SEEDS

    return SEEDS[spec] if spec in SEEDS else Path(spec).read_text()


def cmd_paint(args) -> None:
    """One painting: an instrument (a seed name or a .py file) with the starting strategy, scored."""
    from conveyor.claude import Meter
    from conveyor.evolve import Organism
    from conveyor.harness import SessionFailed
    from conveyor.painting import prompts
    from conveyor.painting.problem import Painter
    from conveyor.painting.problem import make_instrument
    from conveyor.store import Store

    setup = _setup(args)
    store = Store(args.db, run_name=args.name or f"paint {args.instrument}", config=_config(args))
    meter = Meter(max_usage=args.max_usage)
    roles = None if setup.mode == "offline" else _roles(args, store, meter, mutate=False)
    server = None if args.no_serve else _serve_in_background(Path(args.db), args.host, args.port, store.run_id)
    painter = Painter(setup, store, roles)
    inst = make_instrument(_instrument_source(args.instrument), f"from {args.instrument}", setup, store,
                           painter.target.height)
    if not inst.viable:
        sys.exit(f"The instrument isn't valid: {inst.note}")
    text = Path(args.prompt).read_text() if args.prompt else prompts.INITIAL_STRATEGY
    prompt = Organism(node="painter", genome={"prompt": text})
    for org in (inst, prompt):
        store.organism({"id": org.id, "node": org.node, "created": org.created, "summary": org.summary,
                        "genome": org.genome, "text": org.genome.get("source") or org.genome.get("prompt"),
                        "niche": org.niche, "traits": org.traits, "sheet": org.sheet, "viable": 1})
    started = time.time()
    try:
        p = painter.paint(inst, prompt)
    except KeyboardInterrupt:
        _stop_all(roles)
        raise
    except SessionFailed as e:
        sys.exit(f"The painting session failed: {e}")
    for org, partner in ((inst, prompt), (prompt, inst)):
        ev = painter.evaluation(p, org, partner, "paint")
        store.evaluation({"id": ev.id, "node": org.node, "organism_id": org.id, "partner_id": partner.id,
                          "reason": "paint", "score": ev.score, "viable": int(ev.viable), "started": started,
                          "ended": time.time(), "details": ev.details, "artifacts": ev.artifacts,
                          "session_id": ev.session_id, "error": ev.error})
    st = p.details["stats"]
    print(f"Score {p.score:.3f} (pixel {p.critic['pixel']:.3f}, style {p.critic['style']:.3f}) in "
          f"{time.time() - started:.0f}s: {st['actions_used']} actions, {st['looks_used']} looks, {st['refused']} refused."
          + (f" Spend {_money(meter.spent)}." if roles else ""))
    if p.error:
        print(f"Error: {p.error}")
    print(f"Painter's note: {p.details['note'] or '(none)'}")
    store.close()
    _hold(server)


def cmd_mutate(args) -> None:
    """
    Mutations only, no paintings: start from one instrument, run an operator n times, and report which niches the
    results land in. The cheapest way to see how much variance the mutator produces.
    """
    from conveyor.claude import Meter
    from conveyor.evolve import Context
    from conveyor.painting.canvas import load_target
    from conveyor.painting.instrument import all_niches
    from conveyor.painting.instrument import niche_distance
    from conveyor.painting.problem import ClaudeInstrumentMutator
    from conveyor.painting.problem import make_instrument
    from conveyor.store import Store

    setup = _setup(args)
    store = Store(args.db, run_name=args.name or f"mutate {args.operator} x{args.n} from {args.instrument}",
                  config=_config(args))
    meter = Meter(budget_usd=args.budget, max_usage=args.max_usage)
    claude = _roles(args, store, meter, paint=False, judge=False).mutate
    height = load_target(setup.target, width=setup.width, patch=setup.n_patch).height
    parent = make_instrument(_instrument_source(args.instrument), f"start: {args.instrument}", setup, store, height)
    if not parent.viable:
        sys.exit(f"The instrument isn't valid: {parent.note}")
    store.organism({"id": parent.id, "node": "instrument", "created": parent.created, "summary": parent.summary,
                    "genome": parent.genome, "text": parent.genome["source"], "niche": parent.niche,
                    "traits": parent.traits, "sheet": parent.sheet, "viable": 1})
    mutator = ClaudeInstrumentMutator(args.operator, claude, setup, store, height, 1.0)
    # Ask for a different interface kind (state x argument structure) each time, farthest from the parent first,
    # so n asks cover as many kinds of interface as they can before repeating one.
    by_kind: dict[str, list[str]] = {}
    for n in sorted((n for n in all_niches() if n != parent.niche), key=lambda n: -niche_distance(parent.niche, n)):
        by_kind.setdefault(n.rsplit("/", 1)[0], []).append(n)
    empty = [k[i] for i in range(3) for k in by_kind.values() if i < len(k)]

    def one(i: int):
        wanted = empty[i % len(empty)]
        ctx = Context(node="instrument", parent=parent, parent_eval=None, partner=None, lineage=[], niches={},
                      empty_niches=empty, wanted_niche=wanted)
        try:
            child = mutator.propose(ctx)[0]
        except Exception as e:  # noqa: BLE001 - report it with the others
            return wanted, None, str(e)
        store.organism({"id": child.id, "node": "instrument", "parent_id": parent.id, "mutator": mutator.name,
                        "created": child.created, "summary": child.summary, "genome": child.genome,
                        "text": child.genome["source"], "niche": child.niche, "traits": child.traits,
                        "sheet": child.sheet, "session_id": child.session_id, "viable": int(child.viable),
                        "note": child.note})
        return wanted, child, None

    print(f"Parent {parent.id[:6]} is {parent.niche}. Running {args.operator} {args.n} times.")
    with ThreadPoolExecutor(max_workers=args.lanes) as pool:
        results = list(pool.map(one, range(args.n)))
    hits = 0
    for wanted, child, error in results:
        if child is None:
            print(f"  asked {wanted:26s} failed: {error[:200]}")
            continue
        hit = args.operator == "invent" and child.niche == wanted
        hits += hit
        print(f"  asked {wanted:26s} got {child.niche or 'not viable':26s} {child.summary[:120]}")
    landed = sorted({c.niche for _, c, _ in results if c is not None and c.viable and c.niche})
    if args.operator == "invent":
        print(f"{hits} of {len(results)} landed in the niche asked for.")
    print(f"{len(landed)} distinct niches: {', '.join(landed) or 'none'}. "
          f"Spend {_money(meter.spent)} over {meter.sessions} sessions. Run {store.run_id}.")
    store.close()


def cmd_probe(args) -> None:
    from conveyor.painting.canvas import load_target
    from conveyor.painting.instrument import probe
    from conveyor.painting.problem import Setup
    from conveyor.painting.workbench import describe

    setup = Setup(target=args.target, width=args.width)
    height = load_target(setup.target, width=setup.width, patch=setup.n_patch).height
    report = probe(_instrument_source(args.instrument), setup.width, height)
    sheet = report.pop("sheet")
    print(describe(report))
    if sheet and args.sheet:
        Path(args.sheet).write_bytes(sheet)
        print(f"Demo sheet written to {args.sheet}")


def cmd_auth(args) -> None:
    from conveyor.pi import AUTH_FILE_ENV
    from conveyor.pi import PI_DIR
    from conveyor.pi import PROVIDERS
    from conveyor.pi import credential_file
    from conveyor.pi import provider_authenticated

    provider = PROVIDERS["openai-codex"]
    if args.auth_action == "status":
        if provider_authenticated(provider):
            print("OpenAI Codex: signed in")
            return
        print(f"OpenAI Codex: not signed in. Run `{provider.login_command}`.")
        raise SystemExit(1)

    if args.auth_action == "logout":
        action = "logout"
    else:
        action = "login"
    script = PI_DIR / "auth.mjs"
    if not script.is_file():
        sys.exit(f"Pi auth helper is missing: {script}")
    env = dict(os.environ)
    env[AUTH_FILE_ENV] = str(credential_file())
    result = subprocess.run(["node", str(script), action], env=env, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def cmd_serve(args) -> None:
    from conveyor.launch import Launcher
    from conveyor.server import make_server
    from conveyor.store import init_db

    db = Path(args.db).resolve()
    if not db.exists():  # an existing file is left alone: it may be an archive
        init_db(db)
    launcher = None
    token = args.launch_token or os.environ.get("CONVEYOR_LAUNCH_TOKEN") or None
    if token and len(token) < 16:
        sys.exit("The launch token is too short to guard a model budget; use at least 16 characters.")
    if args.no_launch:
        pass
    elif args.host in LOOPBACK or token:
        launcher = Launcher(db, build_parser, token=token)
    else:
        print(f"Starting runs from the dashboard is off: it would let anyone who can reach {args.host} spend your "
              "model budget. Serve on 127.0.0.1, or set CONVEYOR_LAUNCH_TOKEN (or --launch-token) so that starting "
              "a run asks for it.")
    server = make_server(db, args.host, args.port, launcher)
    print(f"Dashboard at http://{args.host}:{server.server_port}"
          + ("  (start runs from the New run button)" if launcher else ""))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if launcher:
            launcher.shutdown()


def build_parser(parser_class=argparse.ArgumentParser) -> argparse.ArgumentParser:
    from conveyor.claude import DEFAULT_EFFORT
    from conveyor.claude import DEFAULT_MODEL

    parser = parser_class(prog="conveyor", description="Co-evolve a painter's prompt and its instrument.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the co-evolution and serve the dashboard while it runs")
    paint = sub.add_parser("paint", help="One painting with one instrument, scored")
    mutate = sub.add_parser("mutate", help="Mutations only, no paintings: which niches does an operator reach?")
    probe = sub.add_parser("probe", help="Check an instrument and draw its demo sheet")
    serve = sub.add_parser("serve", help="Serve the dashboard for an existing database")
    auth = sub.add_parser("auth", help="Manage model-provider sign-in")
    auth_sub = auth.add_subparsers(dest="auth_action", required=True)
    auth_login = auth_sub.add_parser("login", help="Sign in with a provider subscription")
    auth_login.add_argument("provider", choices=["openai"], help="Provider to sign in to")
    auth_logout = auth_sub.add_parser("logout", help="Remove saved provider credentials")
    auth_logout.add_argument("provider", choices=["openai"], help="Provider to sign out of")
    auth_sub.add_parser("status", help="Show OpenAI sign-in status")

    efforts = ["low", "medium", "high", "xhigh", "max"]
    for p in (run, paint, mutate):
        p.add_argument("--harness", default="claude", choices=["claude", "pi"],
                       help="What runs the models: Claude Code (claude -p) or Pi (a Node sidecar). Sets every role; "
                            "the --paint-/--mutate-/--judge- flags override one.")
        p.add_argument("--model", default=None,
                       help=f"Model for every role. Default: {DEFAULT_MODEL} on claude, the provider's default on pi "
                            "(glm-5.3-flash on OpenCode Go)")
        p.add_argument("--effort", default=DEFAULT_EFFORT, choices=efforts,
                       help="Thinking effort. Pi moves it to the nearest level the model publishes.")
        from conveyor.pi import PROVIDERS
        p.add_argument("--provider", default="opencode-go", choices=list(PROVIDERS),
                       help="Where Pi calls models. OpenAI Codex uses `conveyor auth login openai`; the other "
                            "providers use API keys.")
        for role in ("paint", "mutate", "judge"):
            p.add_argument(f"--{role}-harness", default=None, choices=["claude", "pi"])
            p.add_argument(f"--{role}-model", default=None)
            p.add_argument(f"--{role}-effort", default=None, choices=efforts)
        p.add_argument("--claude", default="claude", help="Path to the Claude Code CLI")
        p.add_argument("--stall-timeout", type=float, default=600.0, metavar="SECONDS",
                       help="Kill a session that prints nothing for this long and run its job again. A provider "
                            "can take a request and go silent until the connection drops. 0 turns it off.")
        p.add_argument("--compact-every-looks", type=int, default=None, metavar="N",
                       help="Pi only: summarize the conversation once N looks (canvas images) have piled up, "
                            "keeping the task and the newest look. Default: never.")
        p.add_argument("--autocompact", type=int, default=None, metavar="TOKENS",
                       help="Claude only: compact the context once it passes this many tokens (100000 to 1000000). "
                            "Default: the CLI's own threshold.")
        p.add_argument("--lanes", type=int, default=2, help="Sessions at once per harness. On a Claude "
                       "subscription they share one rate limit.")
        p.add_argument("--name", default=None)
        p.add_argument("--paint-cap", type=float, default=8.0, help="USD cap per painting session")
        p.add_argument("--mutate-cap", type=float, default=4.0, help="USD cap per mutation session")
        p.add_argument("--max-usage", type=float, default=0.85, help="Start no new session once a Claude usage "
                       "window (five-hour or weekly) is this full. The run shares those windows with everything else.")
    for p in (run, paint, mutate, probe):
        p.add_argument("--target", default="self_portrait")
        p.add_argument("--width", type=int, default=512)
        p.add_argument("--actions", type=int, default=200, help="Instrument calls per painting")
        p.add_argument("--looks", type=int, default=None, help="Looks per painting. Default: one per 25 actions, "
                       "at least 4. Negative (e.g. -1) means unlimited.")
    for p in (run, paint, mutate, serve):
        p.add_argument("--db", default="runs/conveyor.db")
    for p in (run, paint, serve):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8765)
    for p in (run, paint):
        p.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True,
                       help="A model judges each finished painting on likeness, colour and brushwork (default on)")
        p.add_argument("--judge-weight", type=float, default=0.5, help="Share of the score the judge carries")
        p.add_argument("--scope", action=argparse.BooleanOptionalAction, default=False,
                       help="The painter may set a scope and paint in it with local coordinates (default off)")
        p.add_argument("--no-serve", action="store_true", help="Don't serve the dashboard")
        p.add_argument("--offline", action="store_true", help="No model: the greedy painter and a scripted mutator")
    run.add_argument("--cycles", type=int, default=6)
    run.add_argument("--budget", type=float, default=30.0, help="USD at list price. Past it the run stops after "
                     "the current iteration, and no session starts past 1.25x.")
    run.add_argument("--wait", type=float, nargs="?", const=6.0, default=None, metavar="HOURS",
                     help="When a usage window passes --max-usage, wait for it to reset instead of stopping, and "
                     "repaint anything the limit cut off. Waits only for a reset within HOURS (default 6, enough "
                     "for the five-hour window); a later one, like the weekly window's, still stops the run.")
    run.add_argument("--parents", type=int, default=2, help="Instrument mutations per iteration")
    run.add_argument("--confirm", type=int, default=3, help="Paintings a challenger and the champion each need "
                     "before the champion changes; scores are the mean. 1 crowns on a single painting.")
    run.add_argument("--seeds", default="round", help="Seed instruments, comma separated: round, pen")
    run.add_argument("--refine", type=float, default=0.4, help="Weight of the refine operator")
    run.add_argument("--invent", type=float, default=0.4, help="Weight of the invent operator")
    run.add_argument("--recombine", type=float, default=0.2, help="Weight of the recombine operator")
    paint.add_argument("instrument", nargs="?", default="round", help="A seed name (round, pen) or an instrument .py")
    paint.add_argument("--prompt", default=None, help="A file holding the strategy prompt")
    mutate.add_argument("instrument", nargs="?", default="round", help="A seed name or an instrument .py")
    mutate.add_argument("--operator", default="invent", choices=["refine", "invent"])
    mutate.add_argument("-n", type=int, default=4)
    mutate.add_argument("--budget", type=float, default=15.0)
    probe.add_argument("instrument", help="A seed name or an instrument .py")
    probe.add_argument("--sheet", default=None, help="Write the demo sheet PNG here")
    serve.add_argument("--no-launch", action="store_true", help="Don't offer to start runs from the dashboard")
    serve.add_argument("--launch-token", default=None, help="Secret that starting or stopping a run asks for; also "
                       "read from CONVEYOR_LAUNCH_TOKEN. Setting one turns launching on for a non-loopback --host.")
    for p, fn in ((run, cmd_run), (paint, cmd_paint), (mutate, cmd_mutate), (probe, cmd_probe), (serve, cmd_serve)):
        p.set_defaults(func=fn)
    auth.set_defaults(func=cmd_auth)
    parser.commands = {"run": run, "paint": paint, "mutate": mutate, "probe": probe, "serve": serve,
                       "auth": auth}
    return parser


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
