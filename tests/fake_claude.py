"""
A stand-in for the `claude` CLI, speaking the same stream-json protocol, for tests.

It reads the first user message from stdin, launches the MCP server named in --mcp-config, and drives it:
  - the canvas server: calls each instrument tool once with arguments built from its schema, then look and finish
  - the shared canvas: looks, paints, writes a message, moves, batches, takes an overview, then looks until its
    budget runs out;
    if FAKE_CLAUDE_HANDOFF names a file that exists, it deletes it and calls spawn_successor instead of looking
  - the workbench: submits the source in FAKE_CLAUDE_SUBMIT (or tries it, then submits)
  - no server and --json-schema: returns FAKE_CLAUDE_STRUCTURED as the structured output
  - FAKE_CLAUDE_CUT_OFF names a file: if it exists, the painter deletes it and hits the usage limit after its
    first call, the way the CLI reports it (a rejected rate_limit_event, then an error result), with the limit
    resetting FAKE_CLAUDE_RESETS_IN seconds later (default 1)
  - FAKE_CLAUDE_STALL names a file: if it exists, the session deletes it, prints its init and then goes silent,
    the way a provider that takes a request and never answers does
  - FAKE_CLAUDE_DROP names a file: if it exists, the painter deletes it and dies after its first call with no
    result, the way a dropped stream does
Everything it does comes back on stdout as the CLI would print it, so the host's recording path is the real one.
"""

import json
import os
import subprocess
import sys
import time

args = sys.argv[1:]


def opt(name):
    return args[args.index(name) + 1] if name in args else None


def emit(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


if "--version" in args:
    print("9.9.9 (fake Claude Code)")
    sys.exit(0)

first = json.loads(sys.stdin.readline())
assert first["type"] == "user"
with open(os.environ.get("FAKE_CLAUDE_LOG", os.devnull), "a") as log:
    log.write(json.dumps({"argv": args, "first": [b.get("type") for b in first["message"]["content"]]}) + "\n")


class Server:
    def __init__(self, command, argv):
        self.proc = subprocess.Popen([command, *argv], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.n = 0

    def rpc(self, method, params=None):
        self.n += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def sample(schema):
    """A plausible value for one parameter."""
    t = schema.get("type")
    if t in ("number", "integer"):
        lo, hi = schema.get("minimum", 0), schema.get("maximum", 40)
        v = lo + (hi - lo) * 0.4
        return int(v) if t == "integer" else v
    if t == "boolean":
        return True
    if t == "string":
        return schema["enum"][0] if "enum" in schema else "#556677"
    if t == "array":
        if schema.get("items", {}).get("type") == "array":
            return [[20, 30], [50, 40], [80, 60]][: max(schema.get("minItems", 1), 2)]
        return [1.0, 2.0][: max(schema.get("minItems", 1), 1)]
    return None


def call(server, i, name, arguments):
    tid = f"toolu_fake_{i}"
    emit({"type": "assistant", "message": {"id": f"msg_{i}", "content": [
        {"type": "tool_use", "id": tid, "name": f"mcp__{server_name}__{name}", "input": arguments}]}})
    resp = server.rpc("tools/call", {"name": name, "arguments": arguments, "_meta": {"claudecode/toolUseId": tid}})
    result = resp.get("result") or {"content": [{"type": "text", "text": json.dumps(resp.get("error"))}], "isError": True}
    content = []
    for block in result["content"]:
        if block["type"] == "image":
            content.append({"type": "image", "source": {"type": "base64", "media_type": block["mimeType"], "data": block["data"]}})
        else:
            content.append(block)
    emit({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": result.get("isError", False)}]}})
    return result


server_name = None
turns = 1
emit({"type": "system", "subtype": "init", "model": opt("--model"), "tools": [], "mcp_servers": []})
stall = os.environ.get("FAKE_CLAUDE_STALL")
if stall and os.path.exists(stall):
    os.remove(stall)
    time.sleep(60)
cfg = opt("--mcp-config")
if cfg:
    server_name, spec = next(iter(json.loads(cfg)["mcpServers"].items()))
    server = Server(spec["command"], spec.get("args", []))
    server.rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "fake"}})
    tools = server.rpc("tools/list")["result"]["tools"]
    emit({"type": "system", "subtype": "init", "model": opt("--model"), "tools": [f"mcp__{server_name}__{t['name']}" for t in tools],
          "mcp_servers": [{"name": server_name, "status": "connected"}]})
    emit({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.1}}}})
    emit({"type": "assistant", "message": {"id": "msg_think", "content": [{"type": "thinking", "thinking": "Planning the work."}]}})
    i = 0
    if server_name == "canvas":
        batch = os.environ.get("FAKE_CLAUDE_BATCH")
        if batch:
            i += 1
            call(server, i, "paint_batch", json.loads(batch))
        for t in ([] if batch else tools):
            if t["name"] in ("look", "finish", "paint_batch"):
                continue
            props = t["inputSchema"].get("properties", {})
            arguments = {k: sample(v) for k, v in props.items() if k in t["inputSchema"].get("required", [])}
            for _ in range(int(os.environ.get("FAKE_CLAUDE_REPEAT", "2"))):
                i += 1
                call(server, i, t["name"], arguments)
                drop = os.environ.get("FAKE_CLAUDE_DROP")
                if drop and os.path.exists(drop):
                    os.remove(drop)
                    sys.stderr.write("terminated\n")
                    server.close()
                    sys.exit(1)
                flag = os.environ.get("FAKE_CLAUDE_CUT_OFF")
                if flag and os.path.exists(flag):
                    os.remove(flag)
                    resets = int(time.time() + float(os.environ.get("FAKE_CLAUDE_RESETS_IN", "1")))
                    emit({"type": "rate_limit_event", "rate_limit_info": {
                        "status": "rejected", "resetsAt": resets, "rateLimitType": "five_hour",
                        "unifiedWindows": {"five_hour": {"utilization": 1.0, "resetsAt": resets}}}})
                    emit({"type": "result", "subtype": "success", "is_error": True, "num_turns": i + 1,
                          "total_cost_usd": 0.004, "usage": {}, "result": "You've hit your limit"})
                    server.close()
                    sys.exit(1)
        i += 1
        call(server, i, "look", {})
        i += 1
        call(server, i, "finish", {"note": "The fake painter wanted thinner lines."})
    elif server_name == "commons":
        # A shared-canvas agent: one of each kind of call, then looks until the budget refuses one.
        names = {t["name"] for t in tools}
        paint = next(t for t in tools if t["name"] not in ("look", "move_viewport", "write_message", "paint_batch")
                     and "x" in t["inputSchema"].get("properties", {}))
        props = paint["inputSchema"].get("properties", {})
        arguments = {k: sample(v) for k, v in props.items() if k in paint["inputSchema"].get("required", [])}
        script = [("look", {}), (paint["name"], arguments),
                  ("write_message", {"text": os.environ.get("FAKE_CLAUDE_MESSAGE", "hello from a fake"), "x": 8, "y": 8}),
                  ("move_viewport", {"angle": 0, "distance": 10_000})]
        if "paint_batch" in names:
            script.append(("paint_batch", {"tool": paint["name"], "calls": [arguments, arguments]}))
        script.append(("overview", {}))
        for name, arguments in script:
            i += 1
            call(server, i, name, arguments)
        handoff = os.environ.get("FAKE_CLAUDE_HANDOFF")
        if handoff and os.path.exists(handoff) and "spawn_successor" in names:
            os.remove(handoff)
            i += 1
            call(server, i, "spawn_successor", {})
            i = 300  # and stop, the way a model told its session is over would
        while i < 300:
            i += 1
            if call(server, i, "look", {}).get("isError"):
                break
    elif server_name == "bench":
        source = os.environ.get("FAKE_CLAUDE_SUBMIT", "")
        i += 1
        call(server, i, "try_instrument", {"source": source})
        i += 1
        call(server, i, "submit_instrument", {"source": source, "summary": "fake submission"})
    turns = i + 1
    server.close()

emit({"type": "assistant", "message": {"id": "msg_end", "content": [{"type": "text", "text": "All done."}]}})
result = {"type": "result", "subtype": "success", "is_error": False, "num_turns": turns, "total_cost_usd": 0.01,
          "usage": {"input_tokens": 100, "output_tokens": 20}, "result": "All done.", "terminal_reason": "completed"}
if opt("--json-schema"):
    if (opt("--system-prompt") or "").startswith("You judge copies"):
        verdict = {"likeness": 5, "colour": 6, "brushwork": 4, "overall": 5, "critique": "The face sits too far left."}
        result["structured_output"] = json.loads(os.environ.get("FAKE_CLAUDE_JUDGE", json.dumps(verdict)))
    else:
        result["structured_output"] = json.loads(os.environ.get("FAKE_CLAUDE_STRUCTURED", "{}"))
emit(result)
