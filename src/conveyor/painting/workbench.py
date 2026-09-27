"""
The instrument designer's workbench, in its own process.

    python -m conveyor.painting.workbench <session_dir>

An MCP server with two tools. `try_instrument` compiles a draft, runs its examples, and returns the demo sheet
and the niche the draft lands in. `submit_instrument` does the same checks and, if they pass, writes
`submitted.json`, which is the mutation's result. A designer that can see what its code draws before it commits
can afford a bolder design: a draft that doesn't compile or doesn't paint costs a try, not a whole evaluation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from conveyor.mcp import StdioServer
from conveyor.mcp import ToolFailure
from conveyor.mcp import image
from conveyor.mcp import text
from conveyor.painting.instrument import probe


def describe(report: dict, wanted_niche: str | None = None) -> str:
    if not report["ok"]:
        return "Not valid:\n" + "\n".join(f"- {e}" for e in report["errors"])
    t = report["traits"]
    reach = f"median reach per call {t['reach'] * 100:.0f}% of the canvas diagonal" if t.get("reach") is not None else "no reach measured"
    tools = ", ".join(f"{x['name']}({', '.join(x['params'])})" for x in report.get("tools", []))
    lines = [
        f"Valid. Niche: {report['niche']} (keeps state between calls: {'yes' if t['stateful'] else 'no'}; "
        f"list parameters: {'yes' if t['list_params'] else 'no'}; {reach}; reads the canvas: "
        f"{'yes' if t['reads_canvas'] else 'no'}).",
        f"Tools: {tools}.",
    ]
    if wanted_niche:
        lines.append(f"You were asked for {wanted_niche}: " + ("this matches." if report["niche"] == wanted_niche
                                                                 else "this lands somewhere else."))
    if report["warnings"]:
        lines.append("Warnings:\n" + "\n".join(f"- {w}" for w in report["warnings"]))
    lines.append("The picture shows each EXAMPLES sequence run on its own blank canvas.")
    return "\n".join(lines)


class Workbench(StdioServer):
    name = "bench"

    def __init__(self, session_dir: Path) -> None:
        self.dir = Path(session_dir)
        self.job = json.loads((self.dir / "job.json").read_text())
        self.width, self.height = int(self.job["width"]), int(self.job["height"])
        self.tries_left = int(self.job.get("max_tries", 8))
        self.wanted = self.job.get("wanted_niche")
        self.n = 0
        self.submitted = False
        (self.dir / "tries").mkdir(exist_ok=True)

    def tools(self) -> list[dict]:
        src = {"type": "string", "description": "The whole instrument module, as Python source."}
        return [
            {"name": "try_instrument",
             "description": "Compile a draft instrument, run its EXAMPLES, and see the demo sheet and the niche it "
                            f"lands in. You have {self.tries_left} tries.",
             "inputSchema": {"type": "object", "properties": {"source": src}, "required": ["source"]}},
            {"name": "submit_instrument",
             "description": "Submit the finished instrument. It is checked the same way as a try; if it passes, "
                            "that's your answer and you are done.",
             "inputSchema": {"type": "object", "properties": {
                 "source": src,
                 "summary": {"type": "string", "description": "One or two sentences: what you changed and why."}},
                 "required": ["source", "summary"]}},
        ]

    def _probe(self, source: str, kind: str) -> dict:
        self.n += 1
        stem = self.dir / "tries" / f"{self.n:02d}_{kind}"
        stem.with_suffix(".py").write_text(source)
        report = probe(source, self.width, self.height)
        sheet = report.pop("sheet")
        if sheet:
            stem.with_suffix(".png").write_bytes(sheet)
        stem.with_suffix(".json").write_text(json.dumps({**report, "at": time.time()}))
        report["sheet"] = sheet
        return report

    def call(self, name: str, args: dict, meta: dict) -> list[dict]:
        source = args.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ToolFailure("Send the whole module as `source`.")
        if name == "try_instrument":
            if self.tries_left <= 0:
                raise ToolFailure("No tries left. Submit your best version with submit_instrument.")
            self.tries_left -= 1
            report = self._probe(source, "try")
            out = [text(describe(report, self.wanted) + f"\n{self.tries_left} tries left.")]
            if report["sheet"]:
                out.append(image(report["sheet"]))
            return out
        if name == "submit_instrument":
            if self.submitted:
                raise ToolFailure("Already submitted. You're done.")
            report = self._probe(source, "submit")
            if not report["ok"]:
                raise ToolFailure(describe(report) + "\nFix these and submit again.")
            self.submitted = True
            record = {"source": source, "summary": str(args.get("summary", ""))[:600], "niche": report["niche"],
                      "traits": report["traits"], "warnings": report["warnings"], "at": time.time()}
            (self.dir / "submitted.json").write_text(json.dumps(record))
            return [text(f"Accepted ({report['niche']}). You're done; end your turn without further tool calls.")]
        raise ToolFailure(f"No tool named {name}.")


if __name__ == "__main__":
    Workbench(Path(sys.argv[1])).serve()
