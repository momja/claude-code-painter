"""
A minimal MCP server over stdio: newline-delimited JSON-RPC 2.0, tools only.

Claude Code launches the server named in `--mcp-config` as a child process and talks to it over stdin/stdout.
The protocol a tools-only server needs is small (initialize, tools/list, tools/call, ping), so this module
implements it directly rather than pulling in an SDK.

Claude Code sends the tool calls of one reply one at a time, each after the previous result, so a server that
handles requests in order sees the calls in the order the model wrote them. The paint server relies on that:
strokes land on one canvas in sequence.

stdout is the protocol channel. `serve()` points `sys.stdout` at stderr before handling anything, so a stray
print (or a library warning) can't corrupt the stream.
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from typing import Any
from typing import TextIO


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def image(png: bytes) -> dict:
    return {"type": "image", "data": base64.b64encode(png).decode(), "mimeType": "image/png"}


class ToolFailure(Exception):
    """Raise inside `call` to send an error result (isError) with this message."""


class StdioServer:
    name = "server"
    version = "1"

    def tools(self) -> list[dict]:
        """[{name, description, inputSchema}]"""
        raise NotImplementedError

    def call(self, name: str, args: dict, meta: dict) -> list[dict]:
        """Run a tool. Return content blocks; raise ToolFailure for an error result."""
        raise NotImplementedError

    def structured_content(self, meta: dict) -> dict | None:
        """Optional machine-readable state alongside tool text, including after a partial failure."""
        return None

    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        stdin = stdin or sys.stdin
        out = stdout or sys.stdout
        sys.stdout = sys.stderr
        for line in stdin:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[{self.name}] not JSON: {line[:200]!r}", file=sys.stderr)
                continue
            reply = self.handle(msg)
            if reply is not None:
                out.write(json.dumps(reply) + "\n")
                out.flush()
            if getattr(self, "closing", False):
                break

    def handle(self, msg: dict) -> dict | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # a notification; nothing to answer
            return None
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                result: Any = {
                    "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                }
            elif method == "tools/list":
                result = {"tools": self.tools()}
            elif method == "tools/call":
                meta = params.get("_meta") or {}
                try:
                    content = self.call(params.get("name", ""), params.get("arguments") or {}, meta)
                    result = {"content": content, "isError": False}
                except ToolFailure as e:
                    result = {"content": [text(str(e))], "isError": True}
                structured = self.structured_content(meta)
                if structured is not None:
                    result["structuredContent"] = structured
            elif method == "ping":
                result = {}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
        except Exception as e:  # noqa: BLE001 - report instead of dying mid-conversation
            traceback.print_exc(file=sys.stderr)
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}
