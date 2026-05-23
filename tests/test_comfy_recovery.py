"""Unit tests for RC11 ComfyUI auto-recovery wrapper (queue_and_wait_with_recovery).

NOTE: Do NOT patch comfyui.asyncio.sleep with a bare AsyncMock — that returns
without yielding to the event loop, which turns the wrapper's polling loop
(``while not wait_task.done(): await asyncio.sleep(2)``) into a tight CPU/RAM
spin until the system OOMs. Use ``_yield_sleep`` (real ``asyncio.sleep(0)``)
when you need to skip the 5s tier-1 wait while keeping the loop cooperative.
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfyui  # noqa: E402
from comfyui import (  # noqa: E402
    ComfyUnavailableError,
    SlowdownAbortError,
    _is_comfy_unavailable_error,
    queue_and_wait_with_recovery,
)


# Cache the real asyncio.sleep BEFORE any patch replaces asyncio.sleep globally.
# `patch("comfyui.asyncio.sleep")` patches the asyncio module itself (since
# comfyui.asyncio IS asyncio), so calling `asyncio.sleep` inside _yield_sleep
# after patching would recurse into _yield_sleep itself.
_REAL_ASYNCIO_SLEEP = asyncio.sleep


async def _yield_sleep(_seconds):
    """Replacement for asyncio.sleep that still yields to the event loop."""
    await _REAL_ASYNCIO_SLEEP(0)


# Hard per-test kill switch in case a future change re-introduces a tight loop.
# Without this, a runaway test can OOM the host (we already did this once).
pytestmark = pytest.mark.timeout(10)


@pytest.fixture(autouse=True)
def _recovery_config():
    """Force deterministic recovery config across all tests."""
    comfyui.COMFY_AUTO_RECOVER_ENABLED = True
    comfyui.COMFY_RESTART_COMMAND = "echo restart"  # never actually executed (mocked)
    comfyui.COMFY_RESTART_MAX_PER_JOB = 2
    comfyui.COMFY_SLOWDOWN_ENABLED = False  # disable WS monitor — not under test here
    yield


def _conn_error() -> httpx.ConnectError:
    return httpx.ConnectError("ECONNREFUSED")


def test_is_comfy_unavailable_matches_httpx_connect_error():
    assert _is_comfy_unavailable_error(_conn_error())


def test_is_comfy_unavailable_matches_string_markers():
    assert _is_comfy_unavailable_error(Exception("connection refused (foo)"))
    assert _is_comfy_unavailable_error(Exception("Read timed out reading body"))
    assert not _is_comfy_unavailable_error(ValueError("workflow node missing"))


@pytest.mark.asyncio
async def test_success_first_try_skips_recovery():
    with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-1")), \
         patch("comfyui.wait_for_completion_async", new=AsyncMock(return_value={"filename": "x.png"})), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)) as free_mock, \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()) as restart_mock:
        pid, info = await queue_and_wait_with_recovery({})
    assert pid == "pid-1"
    assert info == {"filename": "x.png"}
    free_mock.assert_not_awaited()
    restart_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_tier1_free_then_retry_success():
    """First wait fails with conn error → /free → second wait succeeds. No restart."""
    wait = AsyncMock(side_effect=[_conn_error(), {"filename": "x.png"}])
    with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-2")), \
         patch("comfyui.wait_for_completion_async", new=wait), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)) as free_mock, \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()) as restart_mock:
        pid, info = await queue_and_wait_with_recovery({})
    assert pid == "pid-2"
    assert info == {"filename": "x.png"}
    free_mock.assert_awaited_once()
    restart_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_tier2_restart_after_repeat_failure():
    """Two failures → /free first, full restart second, third attempt succeeds."""
    wait = AsyncMock(side_effect=[_conn_error(), _conn_error(), {"filename": "x.png"}])
    with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-3")), \
         patch("comfyui.wait_for_completion_async", new=wait), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)) as free_mock, \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()) as restart_mock, \
         patch("comfyui.asyncio.sleep", new=_yield_sleep):  # skip the 5s tier-1 wait
        pid, info = await queue_and_wait_with_recovery({})
    assert pid == "pid-3"
    free_mock.assert_awaited_once()
    restart_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_exhausted_recovery_raises_last_error():
    """All 3 attempts fail → original error type bubbles up."""
    wait = AsyncMock(side_effect=[_conn_error(), _conn_error(), _conn_error()])
    with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-4")), \
         patch("comfyui.wait_for_completion_async", new=wait), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)), \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()), \
         patch("comfyui.asyncio.sleep", new=_yield_sleep):
        with pytest.raises(httpx.ConnectError):
            await queue_and_wait_with_recovery({})


@pytest.mark.asyncio
async def test_non_recoverable_error_raises_immediately():
    """ValueError (e.g. malformed workflow) should NOT trigger recovery."""
    wait = AsyncMock(side_effect=ValueError("bad workflow node"))
    with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-5")), \
         patch("comfyui.wait_for_completion_async", new=wait), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)) as free_mock, \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()) as restart_mock:
        with pytest.raises(ValueError):
            await queue_and_wait_with_recovery({})
    free_mock.assert_not_awaited()
    restart_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_triggers_recovery():
    """asyncio.TimeoutError counts as recoverable (mirrors slow ComfyUI hang)."""
    wait = AsyncMock(side_effect=[asyncio.TimeoutError(), {"filename": "x.png"}])
    with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-6")), \
         patch("comfyui.wait_for_completion_async", new=wait), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)) as free_mock, \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()) as restart_mock, \
         patch("comfyui.asyncio.sleep", new=_yield_sleep):
        pid, info = await queue_and_wait_with_recovery({})
    assert pid == "pid-6"
    free_mock.assert_awaited_once()
    restart_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_recovery_passes_through():
    """COMFY_AUTO_RECOVER_ENABLED=False → no wrapping; first failure propagates."""
    comfyui.COMFY_AUTO_RECOVER_ENABLED = False
    try:
        with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-7")), \
             patch("comfyui.wait_for_completion_async", new=AsyncMock(side_effect=_conn_error())), \
             patch("comfyui.free_comfy", new=AsyncMock()) as free_mock:
            with pytest.raises(httpx.ConnectError):
                await queue_and_wait_with_recovery({})
        free_mock.assert_not_awaited()
    finally:
        comfyui.COMFY_AUTO_RECOVER_ENABLED = True


def test_slowdown_abort_error_is_exception_subclass():
    assert issubclass(SlowdownAbortError, Exception)
    assert issubclass(ComfyUnavailableError, Exception)


# --------------------------------------------------------------------------
# Restart routing: HTTP supervisor (container) vs local subprocess (native)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_routes_to_supervisor_when_url_set():
    """When restart_service_url is configured, the HTTP path runs and
    the local subprocess path is NOT touched."""
    comfyui.COMFY_RESTART_SERVICE_URL = "http://host.docker.internal:8787/restart"
    try:
        with patch("comfyui._restart_via_host_supervisor", new=AsyncMock()) as http_mock, \
             patch("comfyui._restart_via_local_subprocess", new=AsyncMock()) as local_mock:
            await comfyui._restart_comfy_process_and_wait()
        http_mock.assert_awaited_once()
        local_mock.assert_not_awaited()
    finally:
        comfyui.COMFY_RESTART_SERVICE_URL = ""


@pytest.mark.asyncio
async def test_restart_routes_to_subprocess_when_url_empty():
    """Without restart_service_url, fall back to local subprocess path."""
    comfyui.COMFY_RESTART_SERVICE_URL = ""
    with patch("comfyui._restart_via_host_supervisor", new=AsyncMock()) as http_mock, \
         patch("comfyui._restart_via_local_subprocess", new=AsyncMock()) as local_mock:
        await comfyui._restart_comfy_process_and_wait()
    http_mock.assert_not_awaited()
    local_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_unreachable_raises_runtime_error():
    """Container-side: if the supervisor can't be reached, _restart_via_host_supervisor
    must raise — that bubbles up into the wrapper's last-attempt failure path."""
    comfyui.COMFY_RESTART_SERVICE_URL = "http://host.docker.internal:8787/restart"
    try:
        # AsyncClient.post raises → we expect RuntimeError out of supervisor path
        async def _raise(*a, **kw):
            raise httpx.ConnectError("supervisor offline")

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, *a, **kw): await _raise()

        with patch("comfyui.httpx.AsyncClient", new=_FakeClient):
            with pytest.raises(RuntimeError, match="host_supervisor unreachable"):
                await comfyui._restart_via_host_supervisor()
    finally:
        comfyui.COMFY_RESTART_SERVICE_URL = ""


@pytest.mark.asyncio
async def test_supervisor_reports_failure_raises():
    """Supervisor returns ok=false → restart treated as failure."""
    comfyui.COMFY_RESTART_SERVICE_URL = "http://host.docker.internal:8787/restart"
    try:
        class _Resp:
            status_code = 503
            text = ""
            def json(self): return {"ok": False, "error": "exe_not_found"}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, *a, **kw): return _Resp()

        with patch("comfyui.httpx.AsyncClient", new=_FakeClient):
            with pytest.raises(RuntimeError, match="host_supervisor restart failed"):
                await comfyui._restart_via_host_supervisor()
    finally:
        comfyui.COMFY_RESTART_SERVICE_URL = ""
