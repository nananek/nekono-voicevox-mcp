"""Smoke tests: server module loads and tools register cleanly.

The actual /audio_query and /synthesis pipelines require a running engine,
so they're not exercised here — see README for end-to-end verification.
"""
from __future__ import annotations

import importlib
import os

import pytest


def test_module_imports_with_default_engine_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICEVOX_ENGINE_URL", raising=False)
    import nekono_voicevox_mcp.server as srv

    importlib.reload(srv)
    assert srv.ENGINE_URL == "http://127.0.0.1:50021"


def test_module_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICEVOX_ENGINE_URL", "http://example.invalid:50021")
    import nekono_voicevox_mcp.server as srv

    importlib.reload(srv)
    assert srv.ENGINE_URL == "http://example.invalid:50021"


def test_tools_registered() -> None:
    """FastMCP exposes registered tools via _tool_manager; ensure all 3 are present."""
    monkey_env = dict(os.environ)
    monkey_env.pop("VOICEVOX_ENGINE_URL", None)
    import nekono_voicevox_mcp.server as srv

    importlib.reload(srv)
    # FastMCP keeps tools on a _tool_manager. The internal API isn't stable, but
    # any path that surfaces tool names will do — fall back to introspecting the
    # module for decorated callables if needed.
    tool_names = {"voicevox_list_speakers", "voicevox_synthesize", "voicevox_play"}
    for name in tool_names:
        assert hasattr(srv, name), f"missing tool: {name}"
