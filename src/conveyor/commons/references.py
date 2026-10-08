"""
Reference images: pictures an agent on the Pi harness generates to study before it paints, a hand from three
angles before painting hands, say. They never land on the canvas. An agent sees its own, and may share one with
every other agent through `broadcast`.

The picture comes from `pi-agent/image.mjs`, which asks the ChatGPT subscription's Codex backend through the
`openai-codex` login (`conveyor auth login openai`), the way the pi-codex-image-gen extension does. Tests set
CONVEYOR_IMAGE_COMMAND to a stand-in that speaks the same JSON on stdin and stdout.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shlex
import shutil
import subprocess

from PIL import Image

from conveyor.pi import PI_DIR

COMMAND_ENV = "CONVEYOR_IMAGE_COMMAND"
MAX_PROMPT = 2000  # characters in one prompt
TIMEOUT = 330  # seconds; the helper gives the backend five minutes
SHOWN_SIDE = 512  # the longest side of a reference as agents see it
STORED_SIDE = 1024  # and as the database keeps it


class ReferenceFailed(RuntimeError):
    pass


def available() -> bool:
    """Whether references can be made here: the stand-in is set, or Pi is installed and signed in to Codex."""
    if os.environ.get(COMMAND_ENV):
        return True
    from conveyor.pi import PROVIDERS
    from conveyor.pi import available as pi_available
    from conveyor.pi import provider_authenticated

    return (pi_available() is None and (PI_DIR / "image.mjs").is_file()
            and provider_authenticated(PROVIDERS["openai-codex"]))


def generate(prompt: str, images: list[bytes] = (), session_id: str | None = None) -> tuple[bytes, str | None]:
    """A PNG for `prompt`, worked from `images` (PNGs) when given, and the prompt as the backend rewrote it."""
    from conveyor.harness import child_env
    from conveyor.pi import AUTH_FILE_ENV
    from conveyor.pi import credential_file

    command = os.environ.get(COMMAND_ENV)
    argv = shlex.split(command) if command else [shutil.which("node") or "node", str(PI_DIR / "image.mjs")]
    env = child_env()
    env.setdefault(AUTH_FILE_ENV, str(credential_file()))
    request = {"prompt": prompt, "sessionId": session_id,
               "images": [{"data": base64.b64encode(b).decode(), "mimeType": "image/png"} for b in images]}
    try:
        proc = subprocess.run(argv, input=json.dumps(request).encode(), capture_output=True, timeout=TIMEOUT, env=env,
                              cwd=PI_DIR if not command else None)
    except subprocess.TimeoutExpired:
        raise ReferenceFailed("The image took too long and was dropped.") from None
    except OSError as e:
        raise ReferenceFailed(f"Couldn't start the image helper: {e}") from None
    lines = proc.stdout.decode(errors="replace").strip().splitlines()
    try:
        out = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise ReferenceFailed(f"The image helper failed{': ' + why if (why := _why(proc.stderr)) else ''}.") from None
    if not out.get("ok"):
        raise ReferenceFailed(str(out.get("error") or "The image helper failed."))
    return base64.b64decode(out["image"]), out.get("revised_prompt")


def _why(stderr: bytes) -> str:
    """The line of a crash that says what went wrong, not the end of its stack trace."""
    lines = [line.strip() for line in stderr.decode(errors="replace").splitlines() if line.strip()]
    said = next((line for line in lines if "Error" in line and not line.startswith("at ")), lines[-1] if lines else "")
    return said[:300]


def shrink(png: bytes, side: int) -> bytes:
    """`png` with its longest side at most `side`, as a PNG."""
    im = Image.open(io.BytesIO(png))
    im.load()
    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    if max(im.size) > side:
        im.thumbnail((side, side), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    return out.getvalue()
