"""ComfyUI utility functions for HermesCTB — extracted from CTB/ctb.py."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

# ---------------------------------------------------------------------------
# Globals — overridden by init_config()
# ---------------------------------------------------------------------------
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOWS_DIR = "./Workflows"
COMFY_VIEW_TIMEOUT = 60
COMFY_VIEW_RETRIES = 3
COMFY_JOB_TIMEOUT = 2400
COMFY_HISTORY_RETRIES = 60
COMFY_HISTORY_RETRY_DELAY = 3

# Recovery (RC11): hybrid auto-recovery — /free first, then full process restart.
COMFY_AUTO_RECOVER_ENABLED = True
COMFY_RESTART_COMMAND = ""
COMFY_RESTART_SERVICE_URL = ""  # host_supervisor URL (used when API runs in a container)
COMFY_RESTART_MAX_PER_JOB = 2
COMFY_HEALTHCHECK_TIMEOUT_SECONDS = 120
COMFY_HEALTHCHECK_INTERVAL_SECONDS = 2
COMFY_SLOWDOWN_ENABLED = True
COMFY_SLOWDOWN_THRESHOLD = 3.0
COMFY_SLOWDOWN_GRACE_SECONDS = 120
COMFY_SLOWDOWN_MIN_STEPS = 3


def init_config(config: dict) -> None:
    global COMFYUI_URL, WORKFLOWS_DIR, COMFY_VIEW_TIMEOUT, COMFY_VIEW_RETRIES
    global COMFY_JOB_TIMEOUT, COMFY_HISTORY_RETRIES, COMFY_HISTORY_RETRY_DELAY
    global COMFY_AUTO_RECOVER_ENABLED, COMFY_RESTART_COMMAND, COMFY_RESTART_SERVICE_URL
    global COMFY_RESTART_MAX_PER_JOB
    global COMFY_HEALTHCHECK_TIMEOUT_SECONDS, COMFY_HEALTHCHECK_INTERVAL_SECONDS
    global COMFY_SLOWDOWN_ENABLED, COMFY_SLOWDOWN_THRESHOLD
    global COMFY_SLOWDOWN_GRACE_SECONDS, COMFY_SLOWDOWN_MIN_STEPS
    COMFYUI_URL = config.get("comfyui_url", COMFYUI_URL)
    WORKFLOWS_DIR = config.get("workflows_dir", WORKFLOWS_DIR)
    rv = config.get("comfy_recovery", {})
    COMFY_JOB_TIMEOUT = rv.get("job_timeout_seconds", COMFY_JOB_TIMEOUT)
    COMFY_HISTORY_RETRIES = rv.get("history_retry_attempts", COMFY_HISTORY_RETRIES)
    COMFY_HISTORY_RETRY_DELAY = rv.get("history_retry_delay_seconds", COMFY_HISTORY_RETRY_DELAY)
    COMFY_VIEW_TIMEOUT = config.get("comfy_view_timeout_seconds", COMFY_VIEW_TIMEOUT)
    COMFY_VIEW_RETRIES = config.get("comfy_view_retries", COMFY_VIEW_RETRIES)
    COMFY_AUTO_RECOVER_ENABLED = bool(rv.get("enabled", COMFY_AUTO_RECOVER_ENABLED))
    COMFY_RESTART_COMMAND = rv.get("restart_command", COMFY_RESTART_COMMAND) or ""
    COMFY_RESTART_SERVICE_URL = rv.get("restart_service_url", COMFY_RESTART_SERVICE_URL) or ""
    COMFY_RESTART_MAX_PER_JOB = int(rv.get("max_restarts_per_job", COMFY_RESTART_MAX_PER_JOB))
    COMFY_HEALTHCHECK_TIMEOUT_SECONDS = int(rv.get("healthcheck_timeout_seconds", COMFY_HEALTHCHECK_TIMEOUT_SECONDS))
    COMFY_HEALTHCHECK_INTERVAL_SECONDS = float(rv.get("healthcheck_interval_seconds", COMFY_HEALTHCHECK_INTERVAL_SECONDS))
    sd = rv.get("slowdown_detection", {})
    COMFY_SLOWDOWN_ENABLED = bool(sd.get("enabled", COMFY_SLOWDOWN_ENABLED))
    COMFY_SLOWDOWN_THRESHOLD = float(sd.get("threshold_multiplier", COMFY_SLOWDOWN_THRESHOLD))
    COMFY_SLOWDOWN_GRACE_SECONDS = float(sd.get("grace_period_seconds", COMFY_SLOWDOWN_GRACE_SECONDS))
    COMFY_SLOWDOWN_MIN_STEPS = int(sd.get("min_steps_before_detection", COMFY_SLOWDOWN_MIN_STEPS))


# ---------------------------------------------------------------------------
# Workflow lookup
# ---------------------------------------------------------------------------

def get_workflow_path_by_name(name: str) -> Optional[str]:
    wf_dir = Path(WORKFLOWS_DIR)
    candidate = wf_dir / f"{name}.json"
    if candidate.exists():
        return str(candidate)
    # fuzzy: case-insensitive
    for p in wf_dir.glob("*.json"):
        if p.stem.lower() == name.lower():
            return str(p)
    return None


def load_workflow(name: str) -> dict:
    path = get_workflow_path_by_name(name)
    if path is None:
        raise ValueError(f"Workflow not found: {name!r}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inject helpers (ported from CTB/ctb.py)
# ---------------------------------------------------------------------------

def inject_prompt(workflow: dict, prompt_text: str, global_prompt: str | None = None) -> dict:
    text_encode_nodes = []
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type == "CLIPTextEncode":
            if "inputs" in node and "text" in node["inputs"]:
                text_encode_nodes.append({
                    "id": node_id,
                    "title": node.get("_meta", {}).get("title", ""),
                    "node": node,
                    "input_field": "text",
                    "kind": "clip",
                })
        elif "TextEncodeQwen" in class_type:
            if "inputs" in node and "prompt" in node["inputs"]:
                text_encode_nodes.append({
                    "id": node_id,
                    "title": node.get("_meta", {}).get("title", ""),
                    "node": node,
                    "input_field": "prompt",
                    "kind": "qwen",
                })
        elif class_type == "PromptRelaySmartEncode":
            if "inputs" in node and "smart_prompt" in node["inputs"]:
                text_encode_nodes.append({
                    "id": node_id,
                    "title": node.get("_meta", {}).get("title", ""),
                    "node": node,
                    "input_field": "smart_prompt",
                    "kind": "relay",
                })
    if not text_encode_nodes:
        return workflow
    positive_nodes = [n for n in text_encode_nodes if "negative" not in n["title"].lower()] or [text_encode_nodes[0]]
    for n in positive_nodes:
        n["node"]["inputs"][n["input_field"]] = prompt_text
        if n["kind"] == "relay" and global_prompt is not None and "global_prompt" in n["node"]["inputs"]:
            n["node"]["inputs"]["global_prompt"] = global_prompt
    return workflow


def build_smart_prompt(beats: list[str]) -> str:
    """Build PromptRelay block-syntax string from list of beat strings.

    Each beat becomes `Scene N:` header on its own line followed by beat text.
    Empty / whitespace-only beats skipped. Returns empty string for empty input.
    """
    cleaned = [b.strip() for b in beats if b and b.strip()]
    if not cleaned:
        return ""
    out_lines: list[str] = []
    for i, beat in enumerate(cleaned, start=1):
        out_lines.append(f"Scene {i}:")
        out_lines.append(beat)
        if i < len(cleaned):
            out_lines.append("")
    return "\n".join(out_lines)


def has_relay_smart_node(workflow: dict) -> bool:
    """True when the workflow contains a PromptRelaySmartEncode node.

    Used by the music-video render loop to decide between the relay
    multi-beat path and the legacy single-prompt path. Cheap class-type
    scan, no edge inspection.
    """
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "PromptRelaySmartEncode":
            return True
    return False


def inject_input_image(workflow: dict, image_filename: str) -> dict:
    """Patch the workflow's source LoadImage node with the given filename.

    Stage C4 note (2026-06-07): The MCA workflow (F2K9B MCA.json) used by the
    multi-frame variant batcher has exactly ONE LoadImage node feeding the
    SimplePromptBatcher. All batched prompt variants share that source frame.
    The "Guitar Solo got Duet frame" symptom on f3a5adf6 was therefore NOT
    a missed-second-LoadImage bug — it was upstream role routing (api.py
    `effective_role_groups`): story segments fell back to the nearest
    annotated neighbour's portrait. Stage C3 fixes that by scanning the
    segment prompt for explicit performer keywords (male/female/duet) and
    routing the segment to that role's portrait before MCA dispatch.
    No node-targeting change is needed here for current topology; we keep
    the single-target behaviour and document the contract."""
    nodes = []
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoadImage" and "inputs" in node and "image" in node["inputs"]:
            nodes.append({"id": node_id, "title": node.get("_meta", {}).get("title", ""), "node": node})
    if not nodes:
        return workflow
    target = next((n for n in nodes if "input" in n["title"].lower()), nodes[0])
    target["node"]["inputs"]["image"] = image_filename
    return workflow


def inject_input_audio(workflow: dict, audio_filename: str) -> dict:
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoadAudio":
            node["inputs"]["audio"] = audio_filename
            break
    return workflow


def inject_segment_duration(workflow: dict, duration_seconds: float, max_duration: float = 30.0) -> dict:
    clamped = min(duration_seconds, max_duration)
    LTX_FPS = 24  # LTX2.3 (4.2) renders at 24 fps (verified via ffprobe on real output)
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if ct in ("ImpactInt", "PrimitiveNode"):
            title = node.get("_meta", {}).get("title", "").lower()
            if "duration" in title or "length" in title or "frames" in title:
                if "inputs" in node and "value" in node["inputs"]:
                    if "frames" in title:
                        node["inputs"]["value"] = max(1, int(clamped * 24))
                    else:
                        node["inputs"]["value"] = clamped
        elif ct == "EmptyLTXVLatentVideo":
            node.setdefault("inputs", {})
            node["inputs"]["length"] = max(1, int(clamped * LTX_FPS))
        elif ct == "LTXVEmptyLatentAudio":
            node.setdefault("inputs", {})
            node["inputs"]["frames_number"] = max(1, int(clamped * LTX_FPS))
    return workflow


_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1),
    "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9),
    "2:3": (2, 3), "3:2": (3, 2), "9:7": (9, 7),
}

_RESOLUTION_LATENT_TYPES = (
    "EmptyFlux2LatentImage",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyLTXVLatentVideo",
)


def inject_resolution(workflow: dict, aspect_ratio: str, megapixels: float = 0.59) -> dict:
    import math
    ratio = _ASPECT_RATIOS.get(aspect_ratio)
    if ratio is None:
        parts = aspect_ratio.split(":")
        if len(parts) == 2:
            try:
                ratio = (int(parts[0]), int(parts[1]))
            except ValueError:
                return workflow
        else:
            return workflow
    w_r, h_r = ratio
    total = int(megapixels * 1_000_000)
    height = int(math.sqrt(total * h_r / w_r) / 64) * 64
    width = int(math.sqrt(total * w_r / h_r) / 64) * 64
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") in _RESOLUTION_LATENT_TYPES:
            node.setdefault("inputs", {})
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
    return workflow


def randomize_seeds(workflow: dict) -> dict:
    SEED_FIELDS = ("seed", "noise_seed")
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for field in SEED_FIELDS:
            if field not in inputs:
                continue
            val = inputs[field]
            if isinstance(val, (int, float)):
                inputs[field] = random.randint(0, 4294967295)
            elif isinstance(val, list) and len(val) == 2:
                src_id = str(val[0])
                if src_id in workflow:
                    src_inputs = workflow[src_id].get("inputs", {})
                    if "value" in src_inputs and isinstance(src_inputs["value"], (int, float)):
                        src_inputs["value"] = random.randint(0, 4294967295)
    return workflow


def preprocess_workflow(workflow: dict) -> dict:
    """Deepcopy only. Full preprocessing (ComfyMathExpression etc.) can be ported from CTB/ctb.py."""
    return copy.deepcopy(workflow)


def _normalize_image_ref(path: str) -> str:
    """Strip absolute/pasted path noise — return just the filename ComfyUI expects."""
    if path.startswith("pasted/"):
        return path
    return os.path.basename(path)


def inject_custom_loras(workflow: dict, custom_loras: list) -> dict:
    """Stub — inject LoRA overrides into workflow. Extend as needed."""
    return workflow


# ---------------------------------------------------------------------------
# Async ComfyUI API calls
# ---------------------------------------------------------------------------

async def upload_file_to_comfy(file_bytes: bytes, filename: str, file_type: str = "image") -> str:
    # ComfyUI exposes only /upload/image; it stores any uploaded file in the
    # input/ dir by filename (LoadAudio/LoadImage then read it by name).
    # There is no /upload/audio endpoint (POSTing to it returns HTTP 405).
    endpoint = "upload/image"
    content_type = "audio/wav" if file_type == "audio" else "image/png"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COMFYUI_URL}/{endpoint}",
            files={"image": (filename, file_bytes, content_type)},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", filename)


async def probe_comfyui_alive(timeout: float = 3.0) -> bool:
    """Cheap pre-submit health probe — True when ComfyUI answers /queue.

    Used by the FastAPI submit endpoints to fail fast (HTTP 412) instead
    of letting a long-running job land in the queue, fire the Tier-2
    recovery path, hit the supervisor, and bubble up `ConnectError` 30
    minutes into the render.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{COMFYUI_URL}/queue")
            return r.status_code == 200
    except Exception:
        return False


async def queue_prompt_async(workflow: dict) -> str:
    workflow = preprocess_workflow(workflow)
    workflow = randomize_seeds(workflow)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": str(uuid.uuid4())},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]


async def free_comfy() -> bool:
    """RC10: ask ComfyUI to unload models + free VRAM/RAM. Best-effort —
    counters cumulative slowdown across many generations in one session.
    Never raises (perf recovery must not break a job)."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{COMFYUI_URL}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=30,
            )
            r.raise_for_status()
        print("ComfyUI /free: models unloaded, memory freed")
        return True
    except Exception as e:
        print(f"ComfyUI /free failed (non-fatal): {e!r}")
        return False


# ---------------------------------------------------------------------------
# RC11: Auto-recovery — Hybrid /free + full process restart, slowdown detection
# Ported from CTB/ctb.py (queue_and_wait_with_recovery / SlowdownMonitor).
# ---------------------------------------------------------------------------


class ComfyUnavailableError(Exception):
    """ComfyUI unreachable / offline during generation."""


class SlowdownAbortError(Exception):
    """Generation aborted because step time exceeded slowdown threshold."""


def _is_comfy_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
        ConnectionRefusedError,
        ConnectionResetError,
        ConnectionAbortedError,
    )):
        return True
    text = str(exc).lower()
    markers = (
        "comfyui unreachable",
        "failed to establish",
        "max retries exceeded",
        "connection refused",
        "read timed out",
        "could not fetch history",
        "queue endpoint unreachable",
        "remote protocol error",
        "server disconnected",
    )
    return any(m in text for m in markers)


async def _is_comfy_online(timeout_seconds: float = 3.0) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{COMFYUI_URL}/system_stats", timeout=timeout_seconds)
            return r.status_code == 200
    except Exception:
        return False


def _kill_comfy_processes() -> int:
    """Kill any running ComfyUI processes. Returns count killed.

    Best-effort: tries psutil first, falls back to taskkill on Windows."""
    killed = 0
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None

    if psutil is not None:
        targets = ("comfyui", "comfyui.exe", "main.py")
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if any(t in name for t in targets) or "comfyui" in cmdline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return killed

    # Fallback: Windows taskkill
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["taskkill", "/F", "/IM", "ComfyUI.exe"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                killed += 1
        except Exception as e:
            print(f"[recovery] taskkill fallback failed: {e!r}")
    return killed


def _supervisor_url(endpoint: str) -> str:
    """Derive a sibling supervisor endpoint URL from the configured /restart URL.

    Configured value is e.g. http://host.docker.internal:8787/restart;
    Stage 9 needs /start (cold-start, idempotent) as well. Returns the
    configured base + endpoint so future endpoints can be added without
    a second config knob.
    """
    base = COMFY_RESTART_SERVICE_URL or ""
    for suffix in ("/restart", "/start"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base.rstrip('/')}{endpoint}"


class SupervisorDownError(RuntimeError):
    """Raised when host_supervisor /health probe fails before Tier-2 restart.

    Stage B2 (2026-06-07): differentiates "supervisor process gone" from
    "supervisor up but restart attempt failed". Callers can pattern-match
    this to fail-fast a job rather than retrying for 40 min on a dead
    supervisor (the f3a5adf6 hang scenario)."""
    pass


async def _probe_supervisor_health(timeout_seconds: float = 2.0) -> bool:
    """Quick health probe to confirm supervisor is alive before Tier-2 POST.

    Returns True iff supervisor responds 200 to GET /health within timeout.
    Used by _restart_via_host_supervisor to fail-fast when CMD window of
    start_supervisor.bat has crashed (no auto-restart pre-B1)."""
    base_url = _supervisor_url("/health")
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            r = await client.get(base_url)
        return r.status_code == 200
    except Exception:
        return False


async def _restart_via_host_supervisor() -> None:
    """Container path: POST to the Windows-host comfy_supervisor service.

    Stage 9: probe ComfyUI first. If it is reachable but the wrapper still
    needs Tier-2 (e.g. hung after slowdown), use /restart (kill + relaunch).
    If ComfyUI is fully offline, use /start (no kill, idempotent) so the
    supervisor does not waste a kill cycle hunting a process that never
    existed.

    Stage B2 (2026-06-07): before the main POST, probe /health for 2s.
    If the supervisor itself is down (CMD window crashed without auto-restart
    pre-B1), raise SupervisorDownError immediately so the wrapper can
    fail-fast the job instead of hanging 40 min on the main POST timeout.

    The supervisor handles the actual spawn on the host and only returns
    once ComfyUI is responding to /system_stats (or its own deadline hits)."""
    if not await _probe_supervisor_health(timeout_seconds=2.0):
        raise SupervisorDownError(
            f"host_supervisor /health probe failed at "
            f"{_supervisor_url('/health')} — supervisor process not running. "
            f"Start it via I:\\HermesCTB\\start_supervisor.bat (Stage B1 "
            f"watchdog will auto-restart future crashes)."
        )
    # 2s probe is enough — a hung-but-listening ComfyUI still returns the
    # TCP RST/200 within that window; a stopped one will not.
    endpoint = "/restart" if await _is_comfy_online(timeout_seconds=2.0) else "/start"
    url = _supervisor_url(endpoint)
    # Generous client-side timeout: must exceed supervisor's HEALTHCHECK_TIMEOUT
    # (default 120s on supervisor) + spawn/kill overhead.
    client_timeout = max(60, COMFY_HEALTHCHECK_TIMEOUT_SECONDS + 30)
    print(f"[recovery] POST {url} (endpoint={endpoint}, timeout {client_timeout}s)")
    try:
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            r = await client.post(url)
    except Exception as e:
        raise RuntimeError(f"host_supervisor unreachable at {url}: {e!r}") from e

    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    # Stage 9 supervisor returns {"status": "<restarted|started|already_online>",
    # "online": True} on success and non-200 with {"ok": False, "error": ...}
    # on failure (e.g. exe_not_found, timeout).
    success = r.status_code == 200 and (
        data.get("online") is True or data.get("ok") is True
    )
    if not success:
        raise RuntimeError(
            f"host_supervisor {endpoint} failed (HTTP {r.status_code}): {data}"
        )
    print(
        f"[recovery] supervisor reported success on {endpoint}: "
        f"status={data.get('status')!r} online={data.get('online')}"
    )
    # Defensive: confirm ourselves before returning to the wrapper.
    if not await _is_comfy_online(timeout_seconds=5.0):
        raise RuntimeError(
            "host_supervisor reported ok but /system_stats still unreachable"
        )


async def _restart_via_local_subprocess() -> None:
    """Native (non-container) path: kill+spawn ComfyUI locally."""
    if not COMFY_RESTART_COMMAND:
        raise RuntimeError(
            "Auto-recovery enabled but neither 'restart_service_url' nor "
            "'restart_command' is configured."
        )

    killed = _kill_comfy_processes()
    print(f"[recovery] killed {killed} ComfyUI process(es); waiting 3s before relaunch")
    await asyncio.sleep(3)

    print(f"[recovery] launching: {COMFY_RESTART_COMMAND}")
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP detaches from this Python; let ComfyUI live past job.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(
            shlex.split(COMFY_RESTART_COMMAND, posix=False),
            creationflags=flags,
            close_fds=True,
        )
    else:
        subprocess.Popen(shlex.split(COMFY_RESTART_COMMAND), close_fds=True)

    deadline = time.time() + max(15, COMFY_HEALTHCHECK_TIMEOUT_SECONDS)
    interval = max(0.5, COMFY_HEALTHCHECK_INTERVAL_SECONDS)
    while time.time() < deadline:
        if await _is_comfy_online(timeout_seconds=3.0):
            print("[recovery] ComfyUI back online")
            await asyncio.sleep(2)
            return
        await asyncio.sleep(interval)

    raise RuntimeError(
        f"ComfyUI restart timed out after {COMFY_HEALTHCHECK_TIMEOUT_SECONDS}s"
    )


async def _restart_comfy_process_and_wait() -> None:
    """Trigger a full ComfyUI restart.

    Routes to the host_supervisor HTTP service when configured (the container
    setup), otherwise falls back to direct subprocess launch (native setup)."""
    if COMFY_RESTART_SERVICE_URL:
        await _restart_via_host_supervisor()
    else:
        await _restart_via_local_subprocess()


class SlowdownMonitor:
    """Watches ComfyUI WS progress events; flags VRAM-overflow slowdowns.

    Sync thread (websocket-client). Async wrapper polls grace_period_expired.
    """

    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        self.step_times: list[float] = []
        self._last_step_ts: float | None = None
        self._slowdown_detected = False
        self._slowdown_ts: float | None = None
        self._current_step = 0
        self._max_steps = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Stage B3 (2026-06-07, revised 2026-06-07 evening): wall-clock
        # silence fallback. Original B3 measured time since monitor.start();
        # that produced false positives during legitimately event-sparse
        # ComfyUI phases (model load, ACE-Step text encoding, T2I setup —
        # several minutes of no `progress` events is normal there). Revised
        # design tracks the timestamp of the LAST WS message of ANY type
        # (progress, executing, executed, status, etc). The watchdog only
        # arms once at least one WS event has arrived, and fires when the
        # WS has been completely silent for `cap` seconds afterwards.
        # That distinguishes "ComfyUI alive but slow" from "WS wedged".
        self._monitor_started_ts: float | None = None
        self._last_any_event_ts: float | None = None

    @property
    def slowdown_detected(self) -> bool:
        with self._lock:
            return self._slowdown_detected

    @property
    def grace_period_expired(self) -> bool:
        with self._lock:
            if not self._slowdown_detected or self._slowdown_ts is None:
                return False
            return (time.time() - self._slowdown_ts) >= COMFY_SLOWDOWN_GRACE_SECONDS

    @property
    def no_events_timeout_expired(self) -> bool:
        """Stage B3 (revised 2026-06-07 evening): WS-silence watchdog.

        Fires only when:
        1. At least one WS event of ANY type has arrived (monitor armed), AND
        2. The WS has been silent for `cap` seconds since the last event.

        That keeps the f3a5adf6 frozen-sampling defence (WS goes dead during
        a real ComfyUI hang) without producing false positives during
        legitimately event-sparse setup phases (model load, ACE-Step text
        encoding, VAE warmup) — those phases still emit periodic `executing`
        events even when no `progress` event has fired yet.

        Cap = max(300s, 2 * absolute_step_cap) — generous enough to tolerate
        a single slow sampling step without aborting."""
        with self._lock:
            if self._last_any_event_ts is None:
                return False  # not yet armed: zero events received at all
            abs_cap = float(getattr(sys.modules[__name__], "COMFY_SLOWDOWN_ABSOLUTE_SEC", 0.0) or 0.0)
            cap = max(300.0, abs_cap * 2.0)
            return (time.time() - self._last_any_event_ts) >= cap

    def start(self) -> None:
        if not COMFY_SLOWDOWN_ENABLED:
            return
        self._stop.clear()
        with self._lock:
            self._monitor_started_ts = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            import websocket as ws_lib  # websocket-client
        except ImportError:
            print("[slowdown] websocket-client not installed; monitor disabled")
            return

        ws_url = COMFYUI_URL.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId=slowdown-{uuid.uuid4().hex[:8]}"

        try:
            sock = ws_lib.create_connection(ws_url, timeout=10)
        except Exception as e:
            print(f"[slowdown] WS connect failed: {e!r}")
            return

        sock.settimeout(2.0)
        try:
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except ws_lib.WebSocketTimeoutException:
                    continue
                except Exception:
                    break
                # Stage B3 revision: any successful sock.recv() that returns a
                # message means the WS pipe is alive. Arm/reset the silence
                # watchdog on EVERY frame (incl. binary preview frames and
                # non-progress event types) so we only fire when the pipe
                # actually goes dead — not when we happen to be in a phase
                # that emits only `executing`/`status` events.
                with self._lock:
                    self._last_any_event_ts = time.time()
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("type") == "progress":
                    data = msg.get("data", {})
                    if data.get("prompt_id") == self.prompt_id:
                        self._on_progress(int(data.get("value", 0)), int(data.get("max", 0)))
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _on_progress(self, value: int, max_steps: int) -> None:
        now = time.time()
        with self._lock:
            if max_steps != self._max_steps and self._max_steps > 0:
                # Phase change (sampling → VAE etc.) — reset baseline
                self.step_times.clear()
                self._last_step_ts = None
            self._current_step = value
            self._max_steps = max_steps
            if self._last_step_ts is not None:
                dur = now - self._last_step_ts
                self.step_times.append(dur)
                if (
                    not self._slowdown_detected
                    and len(self.step_times) >= COMFY_SLOWDOWN_MIN_STEPS
                ):
                    prior = self.step_times[:-1]
                    avg = sum(prior) / len(prior) if prior else 0.0
                    if avg > 0 and dur > avg * COMFY_SLOWDOWN_THRESHOLD:
                        self._slowdown_detected = True
                        self._slowdown_ts = now
                        print(
                            f"[slowdown] step {value}/{max_steps}: "
                            f"{dur:.1f}s vs avg {avg:.1f}s ({dur / avg:.1f}x)"
                        )
            self._last_step_ts = now


async def queue_and_wait_with_recovery(
    workflow: dict,
    job_timeout: Optional[int] = None,
) -> tuple[str, dict]:
    """Queue + wait wrapper with hybrid auto-recovery.

    Tier 1 (attempt 2): /free + retry
    Tier 2 (attempt 3): full ComfyUI process restart + retry

    Recoverable errors: ComfyUnavailableError, SlowdownAbortError, TimeoutError,
    or anything _is_comfy_unavailable_error matches. Other errors raise immediately.
    """
    if not COMFY_AUTO_RECOVER_ENABLED:
        prompt_id = await queue_prompt_async(workflow)
        info = await wait_for_completion_async(prompt_id, job_timeout)
        return prompt_id, info

    max_restarts = max(0, COMFY_RESTART_MAX_PER_JOB)
    attempt = 0
    last_error: Exception | None = None

    while attempt <= max_restarts:
        monitor: SlowdownMonitor | None = None
        wait_task: asyncio.Task | None = None
        try:
            prompt_id = await queue_prompt_async(workflow)
            print(f"[queue] prompt {prompt_id} (attempt {attempt + 1}/{max_restarts + 1})")

            if COMFY_SLOWDOWN_ENABLED:
                monitor = SlowdownMonitor(prompt_id)
                monitor.start()

            wait_task = asyncio.create_task(wait_for_completion_async(prompt_id, job_timeout))
            while not wait_task.done():
                if monitor is not None and monitor.grace_period_expired:
                    wait_task.cancel()
                    try:
                        await wait_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise SlowdownAbortError(
                        f"Slowdown grace expired for job {prompt_id}"
                    )
                # Stage B3 (2026-06-07): no-events wall-clock fallback.
                # Frozen sampling thread never emits progress → relative
                # and absolute step-time checks both stay dormant. Fire
                # SlowdownAbortError when monitor has been alive past the
                # no-events cap with zero events received.
                if monitor is not None and monitor.no_events_timeout_expired:
                    wait_task.cancel()
                    try:
                        await wait_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise SlowdownAbortError(
                        f"No progress events received for job {prompt_id} "
                        f"after no-events cap — sampling thread likely frozen"
                    )
                await asyncio.sleep(2)
            info = await wait_task
            return prompt_id, info

        except Exception as e:
            last_error = e
            is_slowdown = isinstance(e, SlowdownAbortError)
            is_unavail = (
                isinstance(e, (ComfyUnavailableError, TimeoutError, asyncio.TimeoutError))
                or _is_comfy_unavailable_error(e)
            )
            if not (is_slowdown or is_unavail):
                raise
            if attempt >= max_restarts:
                raise

            attempt += 1
            if attempt == 1:
                print(f"[recovery] Tier-1 (attempt {attempt}/{max_restarts}): /free + retry — reason: {e!r}")
                await free_comfy()
                await asyncio.sleep(5)
            else:
                print(f"[recovery] Tier-2 (attempt {attempt}/{max_restarts}): full restart — reason: {e!r}")
                try:
                    await _restart_comfy_process_and_wait()
                except SupervisorDownError as sup_err:
                    # Stage B2 (2026-06-07): supervisor process crashed before B1
                    # watchdog could restart it (or B1 not yet active). No point
                    # sending /interrupt to ComfyUI — it is almost certainly
                    # offline too if supervisor was the spawner. Fail-fast so
                    # the job surfaces this as "supervisor_down" instead of
                    # hanging up to 40 min on the next attempt's slowdown grace.
                    print(f"[recovery] fail-fast: {sup_err}")
                    raise
                except Exception as restart_err:
                    # Supervisor/restart unreachable — orphan render keeps GPU. Send
                    # /interrupt so ComfyUI at least stops the hanging prompt.
                    try:
                        async with httpx.AsyncClient(timeout=5) as c:
                            await c.post(f"{COMFYUI_URL}/interrupt")
                        print("[recovery] sent /interrupt to ComfyUI after restart failure")
                    except Exception as int_err:
                        print(f"[recovery] /interrupt also failed: {int_err!r}")
                    raise restart_err
        finally:
            if monitor is not None:
                monitor.stop()

    raise last_error or RuntimeError("Auto-recovery exhausted with no error captured")


async def wait_for_completion_async(prompt_id: str, timeout: Optional[int] = None) -> dict:
    max_wait = timeout or COMFY_JOB_TIMEOUT
    check_interval = 2
    import time
    start = time.time()
    async with httpx.AsyncClient() as client:
        while (time.time() - start) < max_wait:
            # Transient poll failures (ComfyUI GPU-saturated during a long
            # render → /queue slow/unreachable) must NOT kill the job. Skip
            # the tick; max_wait still bounds total wall time.
            try:
                resp = await client.get(f"{COMFYUI_URL}/queue", timeout=5)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                elapsed = int(time.time() - start)
                print(f"Job {prompt_id}: queue poll transient error ({elapsed}s): {e!r} — retrying")
                await asyncio.sleep(check_interval)
                continue
            running = data.get("queue_running", [])
            pending = data.get("queue_pending", [])
            if not any(item[1] == prompt_id for item in running + pending):
                break
            elapsed = int(time.time() - start)
            print(f"Job {prompt_id}: waiting... ({elapsed}s)")
            await asyncio.sleep(check_interval)
        else:
            raise TimeoutError(f"Job {prompt_id} timed out after {max_wait}s")

        for attempt in range(COMFY_HISTORY_RETRIES):
            try:
                resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
                resp.raise_for_status()
                history = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                print(f"Job {prompt_id}: history poll transient error (attempt {attempt}): {e!r} — retrying")
                await asyncio.sleep(COMFY_HISTORY_RETRY_DELAY)
                continue
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    return _extract_output(outputs)
            await asyncio.sleep(COMFY_HISTORY_RETRY_DELAY)

    raise Exception(f"Job {prompt_id}: no outputs after {COMFY_HISTORY_RETRIES} retries")


def _extract_output(outputs: dict) -> dict:
    for node_output in outputs.values():
        for key in ("audio", "gifs", "videos"):
            if node_output.get(key):
                return node_output[key][0]
        if node_output.get("filenames"):
            return {"filename": node_output["filenames"][0], "subfolder": "", "type": "output"}
    for node_output in outputs.values():
        images = node_output.get("images", [])
        saved = [i for i in images if i.get("type") != "temp"]
        if saved:
            return saved[0]
        if images:
            return images[0]
    raise Exception("No output found in ComfyUI job history")


async def get_file_bytes(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    for attempt in range(max(1, COMFY_VIEW_RETRIES)):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{COMFYUI_URL}/view", params=params, timeout=COMFY_VIEW_TIMEOUT)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            if attempt < COMFY_VIEW_RETRIES - 1:
                await asyncio.sleep(1.5)
            else:
                raise Exception(f"Failed to fetch '{filename}': {e}") from e
    raise Exception("get_file_bytes: unreachable")


async def download_output_to_local(output_info: dict, output_dir: str) -> str:
    filename = output_info.get("filename", "output")
    subfolder = output_info.get("subfolder", "")
    folder_type = output_info.get("type", "output")
    data = await get_file_bytes(filename, subfolder, folder_type)
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, filename)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


# ---------------------------------------------------------------------------
# comfy_caller factory (for FilmPipeline)
# ---------------------------------------------------------------------------

def make_comfy_caller(output_dir: str) -> Callable:
    """Returns an async callable(workflow_name, params) -> dict for FilmPipeline."""

    async def comfy_caller(workflow_name: str, params: dict) -> dict:
        workflow = load_workflow(workflow_name)

        # Clear stale pasted image refs
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                node.get("inputs", {})["image"] = ""

        # Inject first/last/middle frame by title keyword
        FRAME_KEYS = {
            "first_frame": ["first frame", "first_frame", " ff"],
            "last_frame": ["last frame", "last_frame", " lf"],
            "middle_frame": ["middle frame", "middle_frame", " mf"],
        }
        if params:
            for node in workflow.values():
                if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                    title_lower = node.get("_meta", {}).get("title", "").lower()
                    for param_key, keywords in FRAME_KEYS.items():
                        val = params.get(param_key)
                        if val and isinstance(val, str) and any(kw in title_lower for kw in keywords):
                            node["inputs"]["image"] = _normalize_image_ref(val)
                            break

        # Node-title overrides
        if params:
            for node in workflow.values():
                if isinstance(node, dict):
                    title = node.get("_meta", {}).get("title", "")
                    if title in params:
                        override = dict(params[title])
                        for key, val in override.items():
                            if key == "image" and isinstance(val, str) and val:
                                override[key] = _normalize_image_ref(val)
                        node.get("inputs", {}).update(override)

        # LoRAs
        loras = (params or {}).get("loras", [])
        if loras:
            workflow = inject_custom_loras(workflow, loras)

        # Prompt
        if params and params.get("prompt"):
            workflow = inject_prompt(workflow, params["prompt"])

        # Audio
        if params and params.get("audio_clip"):
            audio_path = params["audio_clip"]
            if os.path.exists(audio_path):
                with open(audio_path, "rb") as af:
                    audio_bytes = af.read()
                uploaded = await upload_file_to_comfy(audio_bytes, os.path.basename(audio_path), "audio")
                workflow = inject_input_audio(workflow, uploaded)

        # Duration
        if params and params.get("duration"):
            workflow = inject_segment_duration(workflow, float(params["duration"]))

        # Queue and wait
        prompt_id = await queue_prompt_async(workflow)
        output_info = await wait_for_completion_async(prompt_id)

        # Download primary output
        os.makedirs(output_dir, exist_ok=True)
        out_path = await download_output_to_local(output_info, output_dir)
        result = {"prompt_id": prompt_id, "output_path": out_path}

        # Collect all images for first/last frame detection
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
                history = resp.json().get(prompt_id, {})
            all_images = []
            for node_out in history.get("outputs", {}).values():
                for img in node_out.get("images", []):
                    img_path = await download_output_to_local(img, output_dir)
                    all_images.append(img_path)
            if len(all_images) >= 2:
                result["first_frame"] = all_images[0]
                result["last_frame"] = all_images[1]
            elif len(all_images) == 1:
                result["first_frame"] = all_images[0]
        except Exception:
            pass

        return result

    return comfy_caller
