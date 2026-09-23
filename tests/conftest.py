"""Stub the models.dev catalog so tests never reach the network and prices stay fixed."""

import pytest

from conveyor import catalog

CATALOG = {
    "opencode-go": {"models": {"glm-5.3-flash": {
        "name": "GLM-5.3-Flash", "reasoning": True,
        "cost": {"input": 0.15, "output": 0.5, "cache_read": 0.03},
        "modalities": {"input": ["text", "image"]},
        "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}],
        "limit": {"context": 1_000_000, "output": 131_072},
    }}},
    "openrouter": {"models": {"z-ai/glm-5.3-flash": {
        "name": "GLM-5.3-Flash", "reasoning": True,
        "cost": {"input": 0.15, "output": 0.5, "cache_read": 0.03},
        "modalities": {"input": ["text", "image"]},
        "limit": {"context": 1_310_720, "output": 131_072},
    }}},
}


@pytest.fixture(autouse=True)
def _stub_catalog(monkeypatch):
    monkeypatch.setattr(catalog, "_catalog", lambda: CATALOG)
    catalog.model_info.cache_clear()
    yield
    catalog.model_info.cache_clear()
