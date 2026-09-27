"""
Model facts (prices, modalities, reasoning efforts, limits) from the models.dev catalog.

Both providers the Pi harness talks to are listed there under one schema, so the sidecar gets a model
definition, prices included, even for a model the installed pi-ai predates. Prices are dollars per million tokens. The catalog
is fetched once per process; if it can't be reached, `model_info` returns a conservative guess rather than
failing a run, and says so through `known`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache

import json
import urllib.request

CATALOG_URL = "https://models.dev/api.json"
TIMEOUT = 20.0


@dataclass(frozen=True)
class Cost:
    """Dollars per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0

    def of(self, *, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
        """`input_tokens` is the uncached part of the prompt: cache reads and writes bill at their own rates."""
        return (
            self.input * input_tokens
            + self.output * output_tokens
            + self.cache_read * cache_read
            + self.cache_write * cache_write
        ) / 1_000_000


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    cost: Cost
    inputs: tuple[str, ...] = ("text",)
    reasoning: bool = True
    efforts: tuple[str, ...] | None = None  # None: the catalog says nothing, so pass any effort through
    context: int = 131_072
    max_tokens: int = 32_768
    known: bool = True  # False when the model isn't in the catalog and these are defaults

    @property
    def takes_images(self) -> bool:
        return "image" in self.inputs


UNKNOWN = ModelInfo(id="", name="", cost=Cost(), inputs=("text", "image"), known=False)


@lru_cache(maxsize=1)
def _catalog() -> dict:
    try:
        # models.dev answers 403 to urllib's default User-Agent.
        request = urllib.request.Request(CATALOG_URL, headers={"User-Agent": "conveyor/0.2"})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - an unreachable catalog must not take a run down
        return {}


def _efforts(entry: dict) -> tuple[str, ...] | None:
    for option in entry.get("reasoning_options") or []:
        if isinstance(option, dict) and option.get("type") == "effort" and option.get("values"):
            return tuple(option["values"])
    return None


@lru_cache(maxsize=64)
def model_info(provider: str, model_id: str) -> ModelInfo:
    entry = ((_catalog().get(provider) or {}).get("models") or {}).get(model_id)
    if not entry:
        return ModelInfo(**{**vars(UNKNOWN), "id": model_id, "name": model_id})
    price = entry.get("cost") or {}
    limit = entry.get("limit") or {}
    return ModelInfo(
        id=model_id,
        name=entry.get("name") or model_id,
        cost=Cost(
            input=float(price.get("input") or 0),
            output=float(price.get("output") or 0),
            cache_read=float(price.get("cache_read") or 0),
            cache_write=float(price.get("cache_write") or 0),
        ),
        inputs=tuple((entry.get("modalities") or {}).get("input") or ("text",)),
        reasoning=bool(entry.get("reasoning")),
        efforts=_efforts(entry),
        context=int(limit.get("context") or 131_072),
        max_tokens=int(limit.get("output") or 32_768),
    )


def refresh() -> None:
    """Drop the cached catalog, for tests and long-lived processes."""
    _catalog.cache_clear()
    model_info.cache_clear()
