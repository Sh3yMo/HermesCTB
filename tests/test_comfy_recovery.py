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
    class _FakeResp:
        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _FakeResp()

    wait = AsyncMock(side_effect=[_conn_error(), {"filename": "x.png"}])
    with patch("comfyui.queue_prompt_async", new=AsyncMock(return_value="pid-2")), \
         patch("comfyui.wait_for_completion_async", new=wait), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)) as free_mock, \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()) as restart_mock, \
         patch("comfyui.httpx.AsyncClient", new=_FakeClient), \
         patch("comfyui.asyncio.sleep", new=_yield_sleep):
        pid, info = await queue_and_wait_with_recovery({})
    assert pid == "pid-2"
    assert info == {"filename": "x.png"}
    free_mock.assert_awaited_once()
    restart_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_orphan_prompt_interrupted_before_retry():
    """Stage O8: an aborted attempt must /interrupt + dequeue its orphan
    prompt BEFORE re-queuing — otherwise ComfyUI renders it to completion in
    parallel with the retry (job 67ed4b23: duplicate segment ltx2_00797)."""
    posts: list[tuple[str, object]] = []

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kwargs):
            posts.append((url, kwargs.get("json")))
            return _FakeResp()

    wait = AsyncMock(side_effect=[_conn_error(), {"filename": "x.png"}])
    with patch("comfyui.queue_prompt_async", new=AsyncMock(side_effect=["pid-a", "pid-b"])), \
         patch("comfyui.wait_for_completion_async", new=wait), \
         patch("comfyui.free_comfy", new=AsyncMock(return_value=True)), \
         patch("comfyui._restart_comfy_process_and_wait", new=AsyncMock()), \
         patch("comfyui.httpx.AsyncClient", new=_FakeClient), \
         patch("comfyui.asyncio.sleep", new=_yield_sleep):
        pid, info = await queue_and_wait_with_recovery({})

    assert pid == "pid-b"
    assert info == {"filename": "x.png"}
    urls = [u for u, _ in posts]
    assert any(u.endswith("/interrupt") for u in urls), f"no /interrupt sent: {urls}"
    queue_deletes = [j for u, j in posts if u.endswith("/queue")]
    assert {"delete": ["pid-a"]} in queue_deletes, (
        f"orphan pid-a not removed from queue: {posts}"
    )


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
    must raise — that bubbles up into the wrapper's last-attempt failure path.

    Stage B2: health probe is now first; mock it to return True so we still
    exercise the POST path (the case where /health is reachable but /restart
    times out at the supervisor)."""
    comfyui.COMFY_RESTART_SERVICE_URL = "http://host.docker.internal:8787/restart"
    try:
        async def _raise(*a, **kw):
            raise httpx.ConnectError("supervisor offline")

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, *a, **kw): await _raise()

        with patch("comfyui._probe_supervisor_health", new=AsyncMock(return_value=True)), \
             patch("comfyui.httpx.AsyncClient", new=_FakeClient):
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

        with patch("comfyui._probe_supervisor_health", new=AsyncMock(return_value=True)), \
             patch("comfyui.httpx.AsyncClient", new=_FakeClient):
            with pytest.raises(RuntimeError, match="host_supervisor.*failed"):
                await comfyui._restart_via_host_supervisor()
    finally:
        comfyui.COMFY_RESTART_SERVICE_URL = ""


@pytest.mark.asyncio
async def test_supervisor_down_fail_fast(monkeypatch):
    """Stage B2 (2026-06-07): when /health probe fails, raise
    SupervisorDownError immediately — no 150s POST timeout."""
    comfyui.COMFY_RESTART_SERVICE_URL = "http://host.docker.internal:8787/restart"
    try:
        with patch("comfyui._probe_supervisor_health", new=AsyncMock(return_value=False)):
            with pytest.raises(comfyui.SupervisorDownError, match="health probe failed"):
                await comfyui._restart_via_host_supervisor()
    finally:
        comfyui.COMFY_RESTART_SERVICE_URL = ""


@pytest.mark.asyncio
async def test_supervisor_down_does_not_send_interrupt(monkeypatch):
    """Stage B2 fail-fast path must NOT bother with /interrupt — ComfyUI is
    almost certainly offline too if the supervisor crashed. Verify the
    SupervisorDownError propagates straight up out of the recovery wrapper."""
    comfyui.COMFY_RESTART_SERVICE_URL = "http://host.docker.internal:8787/restart"
    comfyui.COMFY_AUTO_RECOVER_ENABLED = True
    comfyui.COMFY_RESTART_MAX_PER_JOB = 2
    try:
        sup_down = comfyui.SupervisorDownError("crashed")
        interrupt_calls = []

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, *a, **kw):
                interrupt_calls.append(url)
                class R: status_code = 200
                return R()

        # Fail twice so we reach Tier-2 then SupervisorDownError raised.
        call_count = {"n": 0}
        async def fake_queue(_wf):
            call_count["n"] += 1
            raise ComfyUnavailableError("simulated")

        async def fake_wait(_pid, _to):
            return {}

        with patch("comfyui.queue_prompt_async", new=fake_queue), \
             patch("comfyui.wait_for_completion_async", new=fake_wait), \
             patch("comfyui._restart_comfy_process_and_wait", side_effect=sup_down), \
             patch("comfyui.asyncio.sleep", new=_yield_sleep), \
             patch("comfyui.httpx.AsyncClient", new=_FakeClient):
            with pytest.raises(comfyui.SupervisorDownError):
                await queue_and_wait_with_recovery({"node": {"class_type": "X"}})
        # /free is fine (Tier-1 path runs before Tier-2). /interrupt is the
        # one we must skip when SupervisorDownError fires.
        interrupt_only = [u for u in interrupt_calls if u.endswith("/interrupt")]
        assert interrupt_only == [], (
            f"/interrupt must NOT be called on supervisor_down, got: {interrupt_only}"
        )
    finally:
        comfyui.COMFY_RESTART_SERVICE_URL = ""


@pytest.mark.asyncio
async def test_slowdown_no_events_timeout_property():
    """Stage B3 (revised 2026-06-07 evening): WS-silence watchdog.

    Fires ONLY when at least one WS event has been seen AND the WS has
    been silent for `cap` seconds afterwards. The original wall-clock-
    from-monitor-start variant produced false positives during legitimately
    event-sparse setup phases (ACE-Step text encoding, T2I, model load)
    and aborted job 97ab5cce-... three times in a row."""
    import time as _time
    comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = 10.0  # cap = max(300, 20) = 300
    m = comfyui.SlowdownMonitor("test-pid")

    # Not armed yet — zero events received → must NOT fire even if the
    # monitor has been running for a long time (this is the ACE-Step /
    # cold-load case the revised B3 must tolerate).
    m._monitor_started_ts = _time.time() - 600.0
    m._last_any_event_ts = None
    assert m.no_events_timeout_expired is False

    # Armed by one WS event, but the silence since then is still under cap.
    m._last_any_event_ts = _time.time() - 100.0
    assert m.no_events_timeout_expired is False

    # Armed and now silent past the cap → real WS-wedge, must fire.
    m._last_any_event_ts = _time.time() - 301.0
    assert m.no_events_timeout_expired is True
