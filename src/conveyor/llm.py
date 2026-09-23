"""
Chat client that records every call in full.

Each call gets a row in `llm_calls`: the exact request (images kept as artifacts), the model's thinking and
reply as they stream in, every tool call, what the caller did with each one, and usage. It also emits an
`llm_call` event for totals and, inside an evaluation, a span on the evaluation's trace. Uses the
OpenAI-compatible /chat/completions endpoint over plain httpx, streaming so thinking is visible while the
model is still working.

Two providers are wired up, both OpenAI-compatible on the wire but not in the details: OpenCode Go (the
default) and OpenRouter. They differ in how you ask for usage, how you ask for thinking, and whether the
reply prices itself, so `Provider` holds those three differences and everything else is shared.
"""

from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable

import httpx

from conveyor.catalog import ModelInfo
from conveyor.catalog import model_info
from conveyor.events import EventSink
from conveyor.events import current_context
from conveyor.events import current_trace


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    env_var: str
    default_model: str
    # OpenRouter takes `reasoning: {"effort": ...}` and prices each reply in `usage.cost` when asked with
    # `usage: {"include": true}`. OpenCode Go is plain OpenAI: `reasoning_effort`, `stream_options`, no cost,
    # so spend is worked out here from the catalog's per-million prices.
    openrouter_extensions: bool
    # Header carrying a stable id for one conversation. OpenCode Go rejects requests without it
    # (400 MissingSessionID); it uses the id to route and to hit its prompt cache.
    session_header: str | None = None
    # Images allowed in one request. GLM through Console Go answers 400 too_many_images above 8, and a Pi
    # painting keeps every image it has sent. None means no limit we know of.
    max_images: int | None = None


OPENCODE_GO = Provider(
    key="opencode-go",
    label="OpenCode Go",
    base_url="https://opencode.ai/zen/go/v1",
    env_var="OPENCODE_API_KEY",
    default_model="glm-5.3-flash",
    openrouter_extensions=False,
    session_header="x-opencode-session",
    max_images=8,
)
OPENROUTER = Provider(
    key="openrouter",
    label="OpenRouter",
    base_url="https://openrouter.ai/api/v1",
    env_var="OPENROUTER_API_KEY",
    default_model="z-ai/glm-5.3-flash",
    openrouter_extensions=True,
)
PROVIDERS = {p.key: p for p in (OPENCODE_GO, OPENROUTER)}
DEFAULT_PROVIDER = OPENCODE_GO.key
DEFAULT_MODEL = OPENCODE_GO.default_model
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# Past the budget the Conductor stops after the current iteration. Past this multiple of it, calls fail outright.
HARD_CAP_MULTIPLE = 1.5
EXCERPT = 600
LIVE_PUSH_INTERVAL = 0.4
_UNSET = object()


def load_api_key(env_file: str | Path | None = None, provider: str = DEFAULT_PROVIDER) -> str | None:
    """The provider's key from the environment, else from a .env file: `env_file`, or the nearest .env found
    in the current directory or one of its parents."""
    var = PROVIDERS[provider].env_var
    key = os.environ.get(var, "").strip()
    if key:
        return key
    if env_file:
        path = Path(env_file)
    else:
        here = Path.cwd()
        path = next((d / ".env" for d in (here, *here.parents) if (d / ".env").is_file()), here / ".env")
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip().removeprefix("export ").strip() == var:
            return value.strip().strip('"').strip("'") or None
    return None


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def function_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None when the model sent arguments that aren't a JSON object
    raw: str


@dataclass
class LLMReply:
    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    cost: float | None
    latency: float
    finish_reason: str | None
    model: str
    call_id: str = ""
    reasoning_tokens: int | None = None
    usage: dict = field(default_factory=dict)


@dataclass
class Usage:
    calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


class _Retry(Exception):
    pass


Transport = Callable[[dict], dict]


def _cached_tokens(reply: LLMReply | None) -> int:
    """Input tokens the provider served from its prompt cache (OpenRouter's prompt_tokens_details)."""
    if reply is None:
        return 0
    return int((reply.usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def _reasoning_from(obj: dict) -> str:
    """Readable reasoning from a message or a stream delta. Both fields can carry the same text; use one."""
    plain = obj.get("reasoning")
    if isinstance(plain, str) and plain:
        return plain
    parts = []
    for d in obj.get("reasoning_details") or []:
        if isinstance(d, dict):
            text = d.get("text") or d.get("summary")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _has_encrypted(obj: dict) -> bool:
    return any(isinstance(d, dict) and d.get("type") == "reasoning.encrypted" for d in obj.get("reasoning_details") or [])


class _Stream:
    """Accumulates SSE chunks into the same shape as a non-streaming response."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.tools: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.model: str | None = None
        self.encrypted = False
        self.got_data = False

    def add(self, chunk: dict) -> None:
        self.got_data = True
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str) and delta["content"]:
                self.content.append(delta["content"])
            thought = _reasoning_from(delta)
            if thought:
                self.reasoning.append(thought)
            self.encrypted = self.encrypted or _has_encrypted(delta)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index")
                slot = self.tools.setdefault(len(self.tools) if idx is None else idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not slot["name"]:
                    slot["name"] = fn["name"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    slot["arguments"] += args
                elif isinstance(args, dict):
                    slot["arguments"] = json.dumps(args)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    @property
    def text(self) -> str:
        return "".join(self.content)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning)

    def partial_tool_calls(self) -> list[dict]:
        return [dict(self.tools[i]) for i in sorted(self.tools)]

    def response(self) -> dict:
        return {
            "model": self.model,
            "choices": [{
                "finish_reason": self.finish_reason,
                "message": {
                    "content": self.text,
                    "reasoning": self.reasoning_text,
                    "reasoning_hidden": self.encrypted,
                    "tool_calls": [
                        {"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}}
                        for t in self.partial_tool_calls()
                    ],
                },
            }],
            "usage": self.usage or {},
        }


class _LivePush:
    """Writes partial thinking, reply, and tool calls to the call's row while it streams, at most every 0.4s."""

    def __init__(self, sink: EventSink | None, call_id: str) -> None:
        self.sink = sink
        self.call_id = call_id
        self.first_token: float | None = None
        self._last = 0.0

    def __call__(self, stream: _Stream) -> None:
        now = time.time()
        first = self.first_token is None
        if first:
            self.first_token = now
        if self.sink is None or not (first or now - self._last >= LIVE_PUSH_INTERVAL):
            return
        self._last = now
        self.sink.update_call(
            self.call_id,
            first_token=self.first_token,
            reasoning=stream.reasoning_text,
            content=stream.text,
            tool_calls=stream.partial_tool_calls(),
        )


class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        provider: str = DEFAULT_PROVIDER,
        base_url: str | None = None,
        reasoning_effort: str | None = "low",
        temperature: float = 0.7,
        max_concurrency: int = 6,
        timeout: float = 180.0,
        max_retries: int = 3,
        budget_usd: float | None = None,
        sink: EventSink | None = None,
        on_budget: Callable[[float], None] | None = None,
        transport: Transport | None = None,
        http_transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        `transport` replaces the HTTP layer with a function from request payload to response dict (no streaming).
        `http_transport` keeps the real streaming code path but swaps httpx's transport, for tests.
        """
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        self.provider = PROVIDERS[provider]
        if transport is None and not api_key:
            raise ValueError(
                f"An {self.provider.label} API key is required. "
                f"Set {self.provider.env_var} or put it in .env."
            )
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
        self.model = model
        self.api_key = api_key  # the Pi harness hands it to its sidecar through the environment
        # One id for every call this client makes, unless a caller passes its own for a single conversation.
        self.session_id = f"conveyor-{uuid.uuid4().hex[:16]}"
        self.reasoning_effort = reasoning_effort
        self._info: ModelInfo | None = None
        self.temperature = temperature
        self.max_retries = max_retries
        self.budget_usd = budget_usd
        self.sink = sink
        self.on_budget = on_budget
        self.usage = Usage()
        self._transport = transport
        self._http = (
            None
            if transport
            else httpx.Client(
                base_url=base_url or self.provider.base_url,
                timeout=httpx.Timeout(timeout, connect=15.0),
                headers={"Authorization": f"Bearer {api_key}", "X-Title": "conveyor"},
                transport=http_transport,
            )
        )
        self._sem = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self._budget_announced = False

    def _session_headers(self, session_id: str | None) -> dict[str, str]:
        name = self.provider.session_header
        return {name: session_id or self.session_id} if name else {}

    @property
    def info(self) -> ModelInfo:
        """Catalog facts for this model, looked up on first use rather than in `__init__`, so building a
        client for a test or an offline run never waits on the network."""
        if self._info is None:
            self._info = model_info(self.provider.key, self.model)
        return self._info

    def chat(
        self,
        messages: list[dict],
        *,
        purpose: str,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        max_tokens: int = 4096,
        record_span: bool = True,
        reasoning_effort: Any = _UNSET,
        session_id: str | None = None,
    ) -> LLMReply:
        """
        `reasoning_effort` overrides the client's default for this call only. `session_id` groups calls that
        belong to one conversation, for providers that route on it.
        """
        if self.budget_usd is not None and self.usage.cost >= self.budget_usd * HARD_CAP_MULTIPLE:
            raise BudgetExceeded(f"spent ${self.usage.cost:.2f}, hard cap is ${self.budget_usd * HARD_CAP_MULTIPLE:.2f}")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.provider.openrouter_extensions:
            payload["usage"] = {"include": True}
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        payload.update(self._reasoning(self.reasoning_effort if reasoning_effort is _UNSET else reasoning_effort))

        headers = self._session_headers(session_id)
        context = current_context()
        trace = current_trace()
        call_id = uuid.uuid4().hex[:16]
        collector = context.get("llm_calls")
        if isinstance(collector, list):
            collector.append(call_id)

        reply: LLMReply | None = None
        error: str | None = None
        with self._sem:
            started = time.perf_counter()
            if self.sink is not None:
                self.sink.start_call(
                    call_id,
                    node=context.get("node"),
                    mutator=context.get("mutator"),
                    organism_id=context.get("organism_id"),
                    trace_id=trace.id if trace else None,
                    purpose=purpose,
                    model=self.model,
                    request=self._request_record(payload),
                )
            live = _LivePush(self.sink, call_id)
            try:
                raw = self._send(payload, live, headers)
                reply = self._parse(raw, time.perf_counter() - started, call_id)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - started
        self._finish(call_id, reply, error, live.first_token)
        self._record(purpose, context, trace, call_id, reply, error, latency, len(tools or []), record_span)
        if error is not None:
            raise LLMError(error)
        return reply

    def annotate(self, call_id: str, **fields: Any) -> None:
        """Attach what the caller did with a reply, such as `tool_results`, to the call's row."""
        if self.sink is not None and call_id:
            self.sink.update_call(call_id, **fields)

    def over_hard_cap(self) -> bool:
        return self.budget_usd is not None and self.usage.cost >= self.budget_usd * HARD_CAP_MULTIPLE

    # Calls made outside this client, such as by the Pi sidecar, are recorded through these two methods so
    # they share the budget, the totals, the call rows, and the trace spans with calls made by `chat`.

    def external_start(self, call_id: str, *, purpose: str, request: dict, model: str | None = None) -> None:
        """`model` names the model actually being called, which is not this client's when the painter runs a
        different one. Without it an in-flight row would name the mutator's model until the call ends."""
        context = current_context()
        trace = current_trace()
        collector = context.get("llm_calls")
        if isinstance(collector, list):
            collector.append(call_id)
        if self.sink is not None:
            self.sink.start_call(
                call_id,
                node=context.get("node"),
                mutator=context.get("mutator"),
                organism_id=context.get("organism_id"),
                trace_id=trace.id if trace else None,
                purpose=purpose,
                model=model or self.model,
                request=request,
            )

    def external_finish(
        self,
        call_id: str,
        reply: LLMReply | None,
        error: str | None,
        first_token: float | None,
        *,
        purpose: str,
        latency: float,
        n_tools: int,
        record_span: bool,
    ) -> None:
        context = current_context()
        self._finish(call_id, reply, error, first_token)
        self._record(purpose, context, current_trace(), call_id, reply, error, latency, n_tools, record_span)

    # ---- internals --------------------------------------------------------------------------------------

    def _reasoning(self, effort: str | None) -> dict:
        """How this provider is asked to think. OpenRouter takes any effort and reads `none` as thinking off.
        OpenCode Go is plain OpenAI, where each model publishes the efforts it takes and there is no `none`:
        an effort it doesn't list is dropped rather than sent and rejected."""
        if not effort:
            return {}
        if self.provider.openrouter_extensions:
            return {"reasoning": {"effort": effort}}
        allowed = self.info.efforts
        if effort == "none" or (allowed is not None and effort not in allowed):
            return {}
        return {"reasoning_effort": effort}

    def _cost(self, usage: dict) -> float | None:
        """What the reply cost. OpenRouter prices it; for everyone else, tokens times the catalog's rates."""
        if usage.get("cost") is not None:
            return float(usage["cost"])
        if not self.info.known:
            return None
        details = usage.get("prompt_tokens_details") or {}
        cache_read = int(details.get("cached_tokens") or 0)
        cache_write = int(details.get("cache_write_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        return self.info.cost.of(
            input_tokens=max(prompt - cache_read - cache_write, 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cache_read=cache_read,
            cache_write=cache_write,
        )

    def _request_record(self, payload: dict) -> dict:
        """The request as sent, with inline images swapped for artifact references so the row stays small."""
        messages = []
        for m in payload["messages"]:
            content = m.get("content")
            if not isinstance(content, list):
                messages.append(m)
                continue
            parts = []
            for p in content:
                url = (p.get("image_url") or {}).get("url", "") if p.get("type") == "image_url" else ""
                if url.startswith("data:") and self.sink is not None:
                    header, b64 = url.split(",", 1)
                    ext = "jpg" if "jpeg" in header else "webp" if "webp" in header else "png"
                    parts.append({"type": "image", "artifact": self.sink.store_artifact(base64.b64decode(b64), ext)})
                elif p.get("type") == "image_url":
                    parts.append({"type": "image", "url": url[:200]})
                else:
                    parts.append(p)
            messages.append({**m, "content": parts})
        record = {"messages": messages, "tools": payload.get("tools")}
        for key in ("temperature", "max_tokens", "tool_choice", "reasoning", "reasoning_effort"):
            if key in payload:
                record[key] = payload[key]
        return record

    def _send(self, payload: dict, on_delta: Callable[[_Stream], None], headers: dict[str, str] | None = None) -> dict:
        if self._transport is not None:
            return self._transport(payload)
        body = {**payload, "stream": True}
        if not self.provider.openrouter_extensions:
            body["stream_options"] = {"include_usage": True}  # OpenRouter sends usage unasked; OpenAI needs this
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            stream = _Stream()
            try:
                with self._http.stream("POST", "/chat/completions", json=body, headers=headers) as r:
                    if r.status_code != 200:
                        text = r.read().decode(errors="replace")[:300]
                        if r.status_code in RETRY_STATUS and not last:
                            raise _Retry()
                        raise LLMError(f"HTTP {r.status_code}: {text}")
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue  # blank separators and comment keep-alives such as ": OPENROUTER PROCESSING"
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        err = chunk.get("error")
                        if err:
                            if not stream.got_data and err.get("code") in RETRY_STATUS and not last:
                                raise _Retry()
                            raise LLMError(f"{self.provider.label} error {err.get('code')}: {err.get('message')}")
                        stream.add(chunk)
                        on_delta(stream)
                if not stream.got_data:
                    if last:
                        raise LLMError("the stream ended without data")
                    raise _Retry()
                return stream.response()
            except _Retry:
                pass
            except (httpx.TimeoutException, httpx.TransportError):
                # Once the model has started answering, a retry would bill a second answer. Give up instead.
                if last or stream.got_data:
                    raise
            time.sleep(delay + random.random() * 0.5)
            delay *= 2
        raise LLMError("retries exhausted")

    def _parse(self, raw: dict, latency: float, call_id: str) -> LLMReply:
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            if isinstance(raw_args, dict):
                args, raw_text = raw_args, json.dumps(raw_args)
            else:
                raw_text = raw_args or ""
                try:
                    args = json.loads(raw_text) if raw_text else {}
                except json.JSONDecodeError:
                    args = None
                if not isinstance(args, dict):
                    args = None
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args, raw=raw_text))
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = dict(raw.get("usage") or {})
        if msg.get("reasoning_hidden") or _has_encrypted(msg):
            usage["reasoning_hidden"] = True
        return LLMReply(
            text=content,
            reasoning=_reasoning_from(msg),
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost=self._cost(usage),
            latency=latency,
            finish_reason=choice.get("finish_reason"),
            model=raw.get("model") or self.model,
            call_id=call_id,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            usage=usage,
        )

    def _finish(self, call_id: str, reply: LLMReply | None, error: str | None, first_token: float | None) -> None:
        if self.sink is None:
            return
        fields: dict[str, Any] = {"status": "error" if error else "ok", "ended": time.time(), "error": error}
        if first_token is not None:
            fields["first_token"] = first_token
        if reply is not None:
            fields.update(
                model=reply.model,
                reasoning=reply.reasoning,
                content=reply.text,
                finish_reason=reply.finish_reason,
                usage=reply.usage,
                tool_calls=[
                    {"id": c.id, "name": c.name, "arguments": c.arguments if c.arguments is not None else c.raw}
                    for c in reply.tool_calls
                ],
            )
        self.sink.update_call(call_id, **fields)

    def _record(
        self,
        purpose: str,
        context: dict,
        trace,
        call_id: str,
        reply: LLMReply | None,
        error: str | None,
        latency: float,
        n_tools: int,
        record_span: bool,
    ) -> None:
        crossed = False
        with self._lock:
            self.usage.calls += 1
            if error is not None:
                self.usage.errors += 1
            cached = _cached_tokens(reply)
            if reply is not None:
                self.usage.prompt_tokens += reply.prompt_tokens
                self.usage.completion_tokens += reply.completion_tokens
                self.usage.cached_tokens += cached
                self.usage.reasoning_tokens += reply.reasoning_tokens or 0
                self.usage.cost += reply.cost or 0.0
            spent = self.usage.cost
            if self.budget_usd is not None and spent >= self.budget_usd and not self._budget_announced:
                self._budget_announced = crossed = True

        data = dict(
            call_id=call_id,
            purpose=purpose,
            model=reply.model if reply else self.model,
            mutator=context.get("mutator"),
            prompt_tokens=reply.prompt_tokens if reply else 0,
            completion_tokens=reply.completion_tokens if reply else 0,
            reasoning_tokens=reply.reasoning_tokens if reply else None,
            cached_tokens=cached,
            cost=reply.cost if reply else None,
            latency=latency,
            finish_reason=reply.finish_reason if reply else None,
            n_tools=n_tools,
            n_tool_calls=len(reply.tool_calls) if reply else 0,
            trace_id=trace.id if trace else None,
        )
        if error is not None:
            data["error"] = error
        if self.sink is not None:
            self.sink.emit(context.get("node"), "llm_call", context.get("organism_id"), **data)
        if record_span and trace is not None:
            result = {k: data[k] for k in ("prompt_tokens", "completion_tokens", "cost", "finish_reason", "n_tool_calls")}
            if reply is not None:
                result["text"] = reply.text[:EXCERPT]
                result["reasoning"] = reply.reasoning[:EXCERPT]
            if error is not None:
                result["error"] = error
            trace.span(
                f"model call ({purpose})",
                args={"model": data["model"], "tools": n_tools, "call_id": call_id},
                result=result,
                duration=latency,
            )
        if crossed and self.on_budget is not None:
            self.on_budget(spent)
