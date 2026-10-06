"""One-off paintings from a run's saved prompt, instrument and model settings.

Requests belong to the original run, but never enter its evolutionary evaluations.
The launcher starts this module in a child process so stop/shutdown use the same controls as runs.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from conveyor.launch import LaunchError
from conveyor.store import Store, connect, dumps, init_db, new_id

MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_BRIEF = 12_000


def inherited_config(conn, run: str) -> dict:
    from conveyor.__main__ import build_parser

    row = conn.execute("SELECT config FROM runs WHERE id=?", (run,)).fetchone()
    if row is None:
        raise LaunchError("Run not found.")
    config = {k: v for k, v in vars(build_parser().parse_args(["run"])).items() if k != "func"}
    config.update(json.loads(row["config"]))
    saved = {}
    for row in conn.execute("SELECT node, data FROM events WHERE run_id=? AND kind='role_settings' ORDER BY seq", (run,)):
        saved[row["node"]] = json.loads(row["data"])
    for role in ("paint", "judge"):
        settings = saved.get(role)
        if not settings:  # Older runs recorded resolved model/effort in sessions, not role_settings.
            purpose = "paint%" if role == "paint" else "judge%"
            row = conn.execute("SELECT model, effort, request FROM sessions WHERE run_id=? AND purpose LIKE ? "
                               "AND model!='greedy' ORDER BY started LIMIT 1", (run, purpose)).fetchone()
            if row:
                request = json.loads(row["request"] or "{}")
                settings = {"model": row["model"], "effort": row["effort"],
                            "harness": request.get("harness", config.get(f"{role}_harness") or config["harness"])}
        if settings:
            for key in ("harness", "model", "effort"):
                config[f"{role}_{key}"] = settings[key]
            if settings.get("provider"):
                config["provider"] = settings["provider"]
    config["saved_roles"] = saved
    return config


def decode_image(value: str, width: int) -> bytes:
    if not isinstance(value, str) or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 256:
        raise LaunchError("Image must be at most 6 MB.")
    try:
        encoded = value.split(",", 1)[1] if value.startswith("data:image/") else value
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise LaunchError("Image must be at most 6 MB.")
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise LaunchError("Image must be at most 16 million pixels.")
            image = ImageOps.exif_transpose(image)
            min_height = 16 * max(1, round(width / 128))
            if not min_height <= round(image.height * width / image.width) <= 2048:
                raise LaunchError(f"Image aspect ratio must give a canvas height between {min_height} and 2048 pixels.")
            image = image.convert("RGB")
            image.thumbnail((2048, 2048))
            out = io.BytesIO()
            image.save(out, format="PNG")
            return out.getvalue()
    except (binascii.Error, ValueError, IndexError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise LaunchError("Upload a valid PNG, JPEG or WebP image.") from None


def create_request(db: Path, run: str, body: dict) -> dict:
    if not isinstance(body, dict) or set(body) - {"instrument_id", "prompt_id", "text", "image"}:
        raise LaunchError("Expected instrument_id, prompt_id, text and optional image. Model settings cannot be changed.")
    brief = body.get("text", "")
    if not isinstance(brief, str) or len(brief) > MAX_BRIEF:
        raise LaunchError("Text must be at most 12000 characters.")
    brief = brief.strip()
    if not brief and not body.get("image"):
        raise LaunchError("Enter some text or upload an image.")
    init_db(db)
    conn = connect(db)
    try:
        config = inherited_config(conn, run)
        for key, node in (("instrument_id", "instrument"), ("prompt_id", "painter")):
            oid = body.get(key)
            if not isinstance(oid, str):
                raise LaunchError(f"Choose a {node} from this run.")
            org = conn.execute("SELECT viable, genome FROM organisms WHERE id=? AND run_id=? AND node=?",
                               (oid, run, node)).fetchone()
            if org is None or org["viable"] == 0:
                raise LaunchError(f"Choose a viable {node} from this run.")
            genome = json.loads(org["genome"])
            if not genome.get("source" if node == "instrument" else "prompt"):
                raise LaunchError(f"The selected {node} has no saved source.")
        target = None
        if body.get("image"):
            data = decode_image(body["image"], config["width"])
            import hashlib
            target = hashlib.sha256(data).hexdigest()[:24] + ".png"
            conn.execute("INSERT OR IGNORE INTO artifacts (name, data) VALUES (?, ?)", (target, data))
        elif config.get("offline"):
            raise LaunchError("Text-only painting needs a model. This run uses the offline image-copying painter.")
        # Preserve the original canvas shape for text-only requests, even if the target file later disappears.
        row = conn.execute("SELECT details FROM evaluations WHERE run_id=? ORDER BY ended LIMIT 1", (run,)).fetchone()
        dimensions = json.loads(row["details"] or "{}").get("canvas", {}) if row else {}
        config["canvas_height"] = dimensions.get("height")
        if not target and config["canvas_height"] is None:
            from conveyor.__main__ import _setup
            from conveyor.painting.canvas import load_target
            setup = _setup(argparse.Namespace(**config))
            try:
                config["canvas_height"] = load_target(setup.target, setup.width, setup.n_patch).height
            except OSError:
                raise LaunchError("The original target is missing, so its canvas height cannot be recovered.") from None
        request_id = new_id()
        conn.execute("INSERT INTO painting_requests (id, run_id, instrument_id, prompt_id, created, status, brief, "
                     "target, config) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                     (request_id, run, body["instrument_id"], body["prompt_id"], time.time(), brief, target, dumps(config)))
        conn.commit()
        return {"id": request_id, "run_id": run, "config": config}
    finally:
        conn.close()


def run_request(db: Path, request_id: str) -> None:
    from conveyor.__main__ import _roles, _setup, _stop_all
    from conveyor.evolve import Organism
    from conveyor.harness import Meter
    from conveyor.painting.problem import Painter

    conn = connect(db, readonly=True)
    try:
        row = dict(conn.execute("SELECT * FROM painting_requests WHERE id=?", (request_id,)).fetchone())
        orgs = []
        for oid in (row["instrument_id"], row["prompt_id"]):
            org = dict(conn.execute("SELECT * FROM organisms WHERE id=? AND run_id=?", (oid, row["run_id"])).fetchone())
            orgs.append(Organism(node=org["node"], id=org["id"], genome=json.loads(org["genome"]), sheet=org["sheet"]))
        target_data = conn.execute("SELECT data FROM artifacts WHERE name=?", (row["target"],)).fetchone() if row["target"] else None
    finally:
        conn.close()
    store = Store(db, run_id=row["run_id"])
    roles = None
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        config = json.loads(row["config"])
        config["db"] = str(db.resolve())
        args = argparse.Namespace(**config)
        setup = _setup(args)
        setup.brief = row["brief"]
        setup.target = None
        setup.height = config.get("canvas_height")
        if target_data:
            folder = setup.work_dir / store.run_id / f"request-{request_id}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "target.png"
            path.write_bytes(target_data[0])
            setup.target = str(path.resolve())
        store.update_painting_request(request_id, status="running")
        store.flush()
        meter = Meter(max_usage=args.max_usage)
        roles = None if setup.mode == "offline" else _roles(args, store, meter, mutate=False,
                                                          judge=setup.target is not None, record=False)
        if roles:
            for name in ("paint", "judge"):
                harness = getattr(roles, name)
                saved = config.get("saved_roles", {}).get(name, {})
                if harness and harness.name == "pi":
                    for key in ("definition", "max_tokens", "max_turns", "heap_mb", "compact_every_looks", "paint_context_turns"):
                        if key in saved:
                            setattr(harness, key, saved[key])
        painter = Painter(setup, store, roles)
        # Publish the session id immediately so the graph can show snapshots while painting.
        original_start = store.start_session

        def start_session(sid, **fields):
            original_start(sid, **fields)
            if fields["purpose"].startswith("paint"):
                store.update_painting_request(request_id, session_id=sid)

        store.start_session = start_session
        if roles:
            # Keep signal handling outside the harness's blocking stream reader. Its cleanup waits for the
            # child, so the main thread must stop that child before waiting for the painting thread to exit.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(painter.paint, *orgs)
                try:
                    painting = future.result()
                except KeyboardInterrupt:
                    _stop_all(roles)
                    raise
        else:
            painting = painter.paint(*orgs)
        store.update_painting_request(request_id, status="finished" if painting.viable else "failed", ended=time.time(),
                                      session_id=painting.session_id, artifacts=painting.artifacts,
                                      details=painting.details, error=painting.error)
        print(f"Painting {request_id}: {painting.artifacts['painting']}, score {painting.score}")
        if not painting.viable:
            raise RuntimeError(painting.error)
    except BaseException as error:
        store.update_painting_request(request_id, status="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                                      ended=time.time(), error=str(error) or "Interrupted")
        raise
    finally:
        _stop_all(roles)
        store.close()


if __name__ == "__main__":
    run_request(Path(sys.argv[1]), sys.argv[2])
