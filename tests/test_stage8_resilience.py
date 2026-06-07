"""Tests for Stage 8 resilience patches.

Covers:
- comfyui.probe_comfyui_alive: True on HTTP 200, False on ConnectError / non-200.
- comfy_supervisor._resolve_comfy_exe: env-var precedence + candidate fallback.
"""

import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock

import httpx
import pytest

import comfyui
import comfy_supervisor


# ---------------------------------------------------------------------------
# probe_comfyui_alive
# ---------------------------------------------------------------------------

def _run(coro):
    # asyncio.run() builds a fresh loop, avoids interfering with other
    # async tests in the suite that may have left a loop closed.
    return asyncio.run(coro)


def test_probe_returns_true_on_200():
    fake_response = MagicMock()
    fake_response.status_code = 200

    class _FakeClient:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, _url):
            return fake_response

    with patch.object(comfyui.httpx, "AsyncClient", _FakeClient):
        assert _run(comfyui.probe_comfyui_alive()) is True


def test_probe_returns_false_on_connect_error():
    class _FakeClient:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, _url):
            raise httpx.ConnectError("simulated down")

    with patch.object(comfyui.httpx, "AsyncClient", _FakeClient):
        assert _run(comfyui.probe_comfyui_alive()) is False


def test_probe_returns_false_on_non_200():
    fake_response = MagicMock()
    fake_response.status_code = 503

    class _FakeClient:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, _url):
            return fake_response

    with patch.object(comfyui.httpx, "AsyncClient", _FakeClient):
        assert _run(comfyui.probe_comfyui_alive()) is False


# ---------------------------------------------------------------------------
# _resolve_comfy_exe (supervisor)
# ---------------------------------------------------------------------------

def test_supervisor_env_var_takes_precedence(tmp_path, monkeypatch):
    custom = tmp_path / "custom" / "ComfyUI.exe"
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_text("stub")
    monkeypatch.setenv("COMFY_EXE_PATH", str(custom))
    assert comfy_supervisor._resolve_comfy_exe() == str(custom)


def test_supervisor_env_var_empty_falls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_EXE_PATH", "")
    # No candidate exists in tmp -> first candidate as last-resort default
    monkeypatch.setattr(
        comfy_supervisor, "_COMFY_EXE_CANDIDATES",
        [str(tmp_path / "missing-A.exe"), str(tmp_path / "missing-B.exe")],
    )
    out = comfy_supervisor._resolve_comfy_exe()
    assert out == str(tmp_path / "missing-A.exe")


def test_supervisor_returns_first_existing_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_EXE_PATH", raising=False)
    a = tmp_path / "a.exe"
    b = tmp_path / "b.exe"
    b.write_text("stub")  # only B exists
    monkeypatch.setattr(
        comfy_supervisor, "_COMFY_EXE_CANDIDATES",
        [str(a), str(b)],
    )
    assert comfy_supervisor._resolve_comfy_exe() == str(b)


def test_supervisor_returns_first_candidate_when_none_exist(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_EXE_PATH", raising=False)
    a = tmp_path / "a.exe"
    b = tmp_path / "b.exe"
    monkeypatch.setattr(
        comfy_supervisor, "_COMFY_EXE_CANDIDATES",
        [str(a), str(b)],
    )
    # neither exists -> first candidate returned as last-resort
    assert comfy_supervisor._resolve_comfy_exe() == str(a)


# ---------------------------------------------------------------------------
# Stage 9: _supervisor_url derives /start from configured /restart
# ---------------------------------------------------------------------------

def test_supervisor_url_strips_restart_and_appends_start(monkeypatch):
    monkeypatch.setattr(
        comfyui, "COMFY_RESTART_SERVICE_URL",
        "http://host.docker.internal:8787/restart",
    )
    assert comfyui._supervisor_url("/start") == "http://host.docker.internal:8787/start"
    assert comfyui._supervisor_url("/restart") == "http://host.docker.internal:8787/restart"


def test_supervisor_url_handles_base_without_known_suffix(monkeypatch):
    # operator misconfigured URL without /restart suffix
    monkeypatch.setattr(comfyui, "COMFY_RESTART_SERVICE_URL", "http://supervisor:9000")
    assert comfyui._supervisor_url("/start") == "http://supervisor:9000/start"


def test_supervisor_url_strips_existing_start_suffix(monkeypatch):
    monkeypatch.setattr(
        comfyui, "COMFY_RESTART_SERVICE_URL",
        "http://host.docker.internal:8787/start",
    )
    assert comfyui._supervisor_url("/restart") == "http://host.docker.internal:8787/restart"
