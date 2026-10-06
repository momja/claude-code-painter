"""
The Pi harness: each job runs Pi's agent loop in a Node sidecar (`pi-agent/agent.mjs`), on any model OpenCode Go
or OpenRouter serves.

The sidecar is a drop-in for `claude -p`. It launches the job's MCP server itself and hands its tools to the
model, and it prints Claude Code's stream-json events, so the rest of conveyor records and shows a Pi session
exactly like a Claude one. Structured output, which Claude Code gets from `--json-schema`, is a `respond` tool
here: the model calls it once with the answer, and the sidecar returns its arguments as `structured_output`.

Two things differ from Claude Code, both handled in the sidecar. Pi keeps calling tools until the model stops, so
a job names its `stop_tools` (a painting's `finish`, the workbench's `submit_instrument`) and the loop ends after
one of them succeeds. And some providers cap images per request (GLM on OpenCode Go takes 8), so the sidecar
drops the oldest canvas views from the context, never the target, when a conversation would go over.

Keys come from the provider's environment variable, or a `.env` file in the working directory or a parent.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from conveyor.catalog import _catalog
from conveyor.catalog import model_info
from conveyor.harness import DEFAULT_STALL_TIMEOUT
from conveyor.harness import Job
from conveyor.harness import Meter
from conveyor.harness import ProcessHarness
from conveyor.harness import child_env
from conveyor.store import Store
from conveyor.store import new_id

PI_DIR = Path(__file__).resolve().parents[2] / "pi-agent"
PI_SCRIPT = PI_DIR / "agent.mjs"
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    env_var: str
    default_model: str
    session_header: str | None = None  # OpenCode Go rejects requests without one (400 MissingSessionID)
    max_images: int | None = None  # images per request; GLM on OpenCode Go answers 400 too_many_images above 8


PROVIDERS = {p.key: p for p in (
    Provider("opencode-go", "OpenCode Go", "https://opencode.ai/zen/go/v1", "OPENCODE_API_KEY", "glm-5.3-flash",
             session_header="x-opencode-session", max_images=8),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "z-ai/glm-5.3-flash"),
)}
DEFAULT_PROVIDER = "opencode-go"


def load_api_key(provider: Provider, env_file: str | Path | None = None) -> str | None:
    """The provider's key from the environment, else from `env_file` or the nearest `.env` up from the cwd."""
    key = os.environ.get(provider.env_var, "").strip()
    if key:
        return key
    here = Path.cwd()
    candidates = [Path(env_file)] if env_file else [d / ".env" for d in (here, *here.parents)]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                if name.strip().removeprefix("export ").strip() == provider.env_var:
                    return value.strip().strip('"').strip("'") or None
    return None


# Which Pi API a model speaks when Pi's own table doesn't list it, from the SDK package models.dev names for it.
API_BY_PACKAGE = {"@ai-sdk/openai": "openai-responses", "@ai-sdk/anthropic": "anthropic-messages"}


def model_api(provider: Provider, model_id: str) -> str:
    entry = ((_catalog().get(provider.key) or {}).get("models") or {}).get(model_id) or {}
    return API_BY_PACKAGE.get((entry.get("provider") or {}).get("npm"), "openai-completions")


def model_def(provider: Provider, model_id: str) -> dict:
    """Pi's model definition from the models.dev catalog: modalities, limits, $/million prices, and the API."""
    info = model_info(provider.key, model_id)
    return {
        "api": model_api(provider, model_id),
        "id": model_id,
        "name": info.name,
        "reasoning": info.reasoning,
        "input": [x for x in info.inputs if x in ("text", "image")],
        "cost": {"input": info.cost.input, "output": info.cost.output,
                 "cacheRead": info.cost.cache_read, "cacheWrite": info.cost.cache_write},
        "contextWindow": info.context,
        "maxTokens": info.max_tokens,
        "known": info.known,
        "efforts": list(info.efforts) if info.efforts else None,
    }


def thinking_level(effort: str | None, allowed: list[str] | None) -> str:
    """Pi's thinking level for an effort, moved to the nearest one the model publishes when it lists them."""
    if not effort or effort == "none":
        return "off"
    effort = effort if effort in EFFORTS else "high"
    if not allowed:
        return effort
    ranked = [e for e in EFFORTS if e in allowed]
    if not ranked:
        return effort
    want = EFFORTS.index(effort)
    return min(ranked, key=lambda e: (abs(EFFORTS.index(e) - want), -EFFORTS.index(e)))


def available(node: str | None = None) -> str | None:
    """Why the Pi harness can't run here, or None when it can."""
    if not (node or shutil.which("node")):
        return "node is not on PATH; install Node 20 or later"
    if not (PI_DIR / "node_modules" / "@earendil-works" / "pi-agent-core").is_dir():
        return f"Pi isn't installed; run `npm install` in {PI_DIR}"
    return None


class PiAgent(ProcessHarness):
    name = "pi"

    def __init__(self, model: str | None = None, *, provider: str = DEFAULT_PROVIDER, effort: str | None = "high",
                 store: Store | None = None, meter: Meter | None = None, lanes: int = 2, api_key: str | None = None,
                 max_tokens: int = 16_384, max_turns: int = 400, heap_mb: int = 384,
                 compact_every_looks: int | None = None, stall_timeout: float | None = DEFAULT_STALL_TIMEOUT, timeout: float = 45 * 60,
                 node: str | None = None, faux: list[dict] | None = None) -> None:
        self.provider = PROVIDERS[provider]
        model = model or self.provider.default_model
        super().__init__(model=model, effort=effort, store=store, meter=meter, lanes=lanes, timeout=timeout,
                         stall_timeout=stall_timeout)
        self.node = node or shutil.which("node") or "node"
        self.api_key = api_key if api_key is not None else load_api_key(self.provider)
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.compact_every_looks = compact_every_looks  # summarize the history once this many looks pile up
        self.heap_mb = heap_mb
        self.faux = faux  # scripted replies for tests: the real agent loop and MCP plumbing, no network
        self.definition = model_def(self.provider, model) if faux is None else {"id": model, "input": ["text", "image"]}

    def describe(self) -> str:
        known = "" if self.faux is not None or self.definition.get("known") else " (not in the catalog: cost reads $0)"
        return f"pi: {self.model} on {self.provider.label}, thinking {self.level}{known}"

    @property
    def level(self) -> str:
        return thinking_level(self.effort, self.definition.get("efforts"))

    def command(self, job: Job) -> tuple[list[str], bytes, dict[str, str]]:
        content = []
        for b in job.content:
            if b["type"] == "png":
                content.append({"type": "image", "data": base64.b64encode(b["data"]).decode(), "mimeType": "image/png"})
            else:
                content.append({"type": "text", "text": b["text"]})
        config = {
            "provider": self.provider.key, "baseUrl": self.provider.base_url, "model": self.definition,
            "sessionHeader": self.provider.session_header, "sessionId": f"conveyor-{new_id()}",
            "maxImages": self.provider.max_images, "thinkingLevel": self.level, "maxTokens": self.max_tokens,
            "systemPrompt": job.system_prompt, "content": content, "mcp": job.mcp, "tools": job.tools,
            "stopTools": job.stop_tools, "jsonSchema": job.json_schema, "maxTurns": self.max_turns,
            "maxBudgetUsd": job.max_budget_usd, "compactEveryLooks": self.compact_every_looks, "faux": self.faux,
        }
        env = child_env()
        if self.api_key:
            env[self.provider.env_var] = self.api_key
        argv = [self.node, f"--max-old-space-size={self.heap_mb}", str(PI_SCRIPT)]
        return argv, (json.dumps(config) + "\n").encode(), env
