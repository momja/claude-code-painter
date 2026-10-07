"""A stand-in for pi-agent/image.mjs: the same JSON on stdin and stdout, and no network.

It answers with a 1200 x 800 picture, or with the backend's quota error when the prompt says FAIL. Each request
is appended to the file FAKE_IMAGE_LOG names, if set, with how many images came with it."""

import base64
import io
import json
import os
import sys

from PIL import Image

request = json.loads(sys.stdin.read())
if os.environ.get("FAKE_IMAGE_LOG"):
    with open(os.environ["FAKE_IMAGE_LOG"], "a") as log:
        log.write(json.dumps({"prompt": request["prompt"], "images": len(request.get("images") or [])}) + "\n")
if "FAIL" in request["prompt"]:
    print(json.dumps({"ok": False, "error": "The ChatGPT subscription's image quota is used up for now."}))
    sys.exit(1)
out = io.BytesIO()
Image.new("RGB", (1200, 800), (40, 90, 160)).save(out, "PNG")
print(json.dumps({"ok": True, "image": base64.b64encode(out.getvalue()).decode(), "revised_prompt": request["prompt"]}))
