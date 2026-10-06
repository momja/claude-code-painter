"""The dashboard must not lose underlying calls when a batch shares one transcript tool id."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).parents[1] / "src/conveyor/dashboard.html"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_dashboard_script_syntax():
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", DASHBOARD.read_text(), re.DOTALL)
    assert scripts
    for script in scripts:
        subprocess.run(["node", "--check", "--input-type=module"], input=script, text=True, check=True,
                       capture_output=True, timeout=10)


@needs_node
def test_batch_details_show_all_calls_and_aggregate_score():
    session = {"id": "s", "status": "ok", "model": "test", "started": 1, "ended": 2, "cost": 0,
               "events": [{"kind": "tool_use", "data": {"id": "b", "name": "paint_batch", "input": {}}},
                          {"kind": "tool_result", "data": {"tool_use_id": "b", "text": "Applied 2/2", "images": []}}],
               "strokes": [
                   {"data": {"tool_use_id": "b", "batch_call": 1, "tool": "start", "status": "applied",
                             "args": {"x": 10}, "score_before": 0.4, "score_after": 0.5, "i": 1}, "snapshot": "one.png"},
                   {"data": {"tool_use_id": "b", "batch_call": 2, "tool": "move", "status": "applied",
                             "args": {"dx": 10}, "score_before": 0.5, "score_after": 0.6, "i": 2}, "snapshot": "two.png"},
               ]}
    script = '''
import { readFileSync } from "node:fs";
const source = readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function sessionDetail(s)");
const end = source.indexOf("function bindReplay(s)", start);
const render = new Function("s", "esc", "fmt", "dur", "money", "short", "img", "compactArgs",
    source.slice(start, end) + "return sessionDetail(s);");
const session = JSON.parse(readFileSync(0, "utf8"));
console.log(render(session, (x) => String(x ?? ""), (x, n) => x.toFixed(n ?? 3), () => "duration",
    () => "$0", (x) => x, (x) => `<img src="${x}">`, (x) => JSON.stringify(x)));
'''
    result = subprocess.run(["node", "--input-type=module", "-e", script, str(DASHBOARD)], input=json.dumps(session),
                            text=True, capture_output=True, check=True, timeout=10)
    html = result.stdout
    assert "2 underlying calls" in html and "1. start" in html and "2. move" in html
    assert "score 0.6000 (+0.2000)" in html
    assert 'src="/artifacts/two.png"' in html
