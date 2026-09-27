"""
The Claude Code harness: each job is one headless `claude -p` process.

    claude -p --input-format stream-json --output-format stream-json --verbose
           --model claude-opus-5-5 --effort high --thinking-display summarized
           --system-prompt <...> --tools "" --setting-sources "" --strict-mcp-config
           --mcp-config <one stdio server of ours> --allowedTools <its tools>
           --no-session-persistence --disable-slash-commands

The flags that matter, and why:
  --tools ""               no built-in tools: a painter with Read or Bash could open the target file and copy it
  --setting-sources ""     none of this machine's hooks, CLAUDE.md files or settings leak into a painting
  --strict-mcp-config      only our server; otherwise the account's claude.ai connectors load too, and a probe
                           run cost ten times as much in tool definitions alone
  --thinking-display       thinking comes back empty in print mode unless asked for; with `summarized` the
                           transcript shows the model's reasoning (a hidden flag in the CLI, so it's dropped
                           and retried without if a CLI version refuses it)
  --task-budget            a token budget the model paces itself against (hidden too, handled the same way)

The first user message goes in over stdin as stream-json, which is how it carries images (the target, the
demo sheet). stdin is then closed, and the CLI runs the agent loop to the end of the turn and exits.

Auth is whatever the local `claude` uses. On a subscription the reported cost is list price, not a bill, but it
is still the best single measure of how much of the plan's rate limit a run eats; `Meter` tracks both.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from conveyor.harness import BudgetExhausted  # noqa: F401 - re-exported for callers of this module
from conveyor.harness import Job
from conveyor.harness import Meter
from conveyor.harness import Outcome  # noqa: F401
from conveyor.harness import ProcessHarness
from conveyor.harness import RateLimited  # noqa: F401
from conveyor.harness import child_env
from conveyor.harness import content_blocks
from conveyor.store import Store

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"


@dataclass
class Settings:
    binary: str = "claude"
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    thinking_display: str | None = "summarized"
    timeout: float = 45 * 60  # seconds per session


class ClaudeCode(ProcessHarness):
    name = "claude"

    def __init__(self, settings: Settings | None = None, store: Store | None = None, meter: Meter | None = None,
                 lanes: int = 2) -> None:
        self.settings = settings or Settings()
        super().__init__(model=self.settings.model, effort=self.settings.effort, store=store, meter=meter,
                         lanes=lanes, timeout=self.settings.timeout)
        self._thinking_flag_ok = True
        self._task_budget_ok = True

    def _attempt(self, job: Job) -> Outcome:
        outcome = self._run_once(job)
        if outcome.error and "--thinking-display" in outcome.error and self._thinking_flag_ok:
            self._thinking_flag_ok = False  # this CLI doesn't know the hidden flag; do without summaries
            outcome = self._run_once(job)
        if outcome.error and job.task_budget and "task" in outcome.error.lower() and "budget" in outcome.error.lower() \
                and self._task_budget_ok and not outcome.tool_calls:
            self._task_budget_ok = False  # the CLI or the model refused task budgets; run without them
            outcome = self._run_once(job)
        return outcome

    def argv(self, job: Job) -> list[str]:
        s = self.settings
        argv = [s.binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--model", s.model, "--system-prompt", job.system_prompt, "--tools", "",
                "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
                "--disable-slash-commands"]
        if s.effort:
            argv += ["--effort", s.effort]
        if s.thinking_display and self._thinking_flag_ok:
            argv += ["--thinking-display", s.thinking_display]
        if job.mcp:
            server = {"command": job.mcp["command"], "args": job.mcp.get("args", [])}
            if job.mcp.get("env"):
                server["env"] = job.mcp["env"]
            argv += ["--mcp-config", json.dumps({"mcpServers": {job.mcp["name"]: server}})]
            if job.tools:
                argv += ["--allowedTools", ",".join(f"mcp__{job.mcp['name']}__{t}" for t in job.tools)]
        if job.json_schema:
            argv += ["--json-schema", json.dumps(job.json_schema)]
        if job.max_budget_usd:
            argv += ["--max-budget-usd", f"{job.max_budget_usd:.2f}"]
        if job.task_budget and self._task_budget_ok:
            argv += ["--task-budget", str(int(job.task_budget))]
        return argv

    def command(self, job: Job) -> tuple[list[str], bytes, dict[str, str]]:
        message = {"type": "user", "message": {"role": "user", "content": content_blocks(job)}}
        return self.argv(job), (json.dumps(message) + "\n").encode(), child_env()


def available(binary: str = "claude") -> str | None:
    """The CLI's version string, or None if it isn't installed."""
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None
