"""ComfyUI Host Supervisor — runs natively on Windows host (NOT in Docker).

Why this exists
---------------
HermesCTB's FastAPI runs in a Linux Docker container, but ComfyUI runs as a
Windows process on the host. The container can call `/free` over HTTP, but it
cannot kill or relaunch `ComfyUI.exe` — Linux processes can't launch Windows
binaries on the host directly.

This supervisor closes that gap. It is a tiny HTTP service running on the
Windows host. The container POSTs `/restart` and this process does the
kill+spawn locally where it actually works.

Endpoints
---------
GET  /health          — supervisor alive
GET  /status          — is ComfyUI process running + responding to /system_stats
POST /restart         — kill all ComfyUI.exe + relaunch + wait for /system_stats

Configuration
-------------
Env vars (with defaults):
    COMFY_EXE_PATH      C:\\Users\\SheyMo\\AppData\\Local\\Programs\\ComfyUI\\ComfyUI.exe
    COMFY_URL           http://127.0.0.1:8188
    SUPERVISOR_HOST     0.0.0.0   (must be 0.0.0.0 so the container can reach it)
    SUPERVISOR_PORT     8787
    HEALTHCHECK_TIMEOUT 120       (seconds — max wait for ComfyUI to come back online)
    KILL_TIMEOUT        10        (seconds — terminate grace before kill)

Run
---
    py host_supervisor\\comfy_supervisor.py

For autostart see start_supervisor.bat in this directory.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import psutil  # type: ignore
except ImportError:
    print("ERROR: psutil missing. Run: py -m pip install psutil", file=sys.stderr)
    sys.exit(2)


COMFY_EXE_PATH = os.environ.get(
    "COMFY_EXE_PATH",
    r"C:\Users\SheyMo\AppData\Local\Programs\ComfyUI\ComfyUI.exe",
)
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
SUPERVISOR_HOST = os.environ.get("SUPERVISOR_HOST", "0.0.0.0")
SUPERVISOR_PORT = int(os.environ.get("SUPERVISOR_PORT", "8787"))
HEALTHCHECK_TIMEOUT = int(os.environ.get("HEALTHCHECK_TIMEOUT", "120"))
KILL_TIMEOUT = int(os.environ.get("KILL_TIMEOUT", "10"))

# Single in-flight restart guard. The container retries on transient errors —
# without this lock, two concurrent restarts can race and double-spawn ComfyUI.
_RESTART_LOCK = threading.Lock()


def _comfy_online(timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(f"{COMFY_URL}/system_stats")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, socket.timeout, ConnectionError):
        return False
    except Exception:
        return False


def _list_comfy_procs() -> list[psutil.Process]:
    """Return all ComfyUI-related processes (best-effort match)."""
    out: list[psutil.Process] = []
    exe_lower = os.path.basename(COMFY_EXE_PATH).lower()
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower()
            exe = (proc.info.get("exe") or "").lower()
            if name == exe_lower or exe == COMFY_EXE_PATH.lower():
                out.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def _kill_all_comfy() -> int:
    killed = 0
    procs = _list_comfy_procs()
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    gone, alive = psutil.wait_procs(procs, timeout=KILL_TIMEOUT)
    killed += len(gone)
    for p in alive:
        try:
            p.kill()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


def _launch_comfy() -> int:
    if not os.path.exists(COMFY_EXE_PATH):
        raise FileNotFoundError(f"ComfyUI.exe not found at: {COMFY_EXE_PATH}")
    # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP so the spawned ComfyUI is not
    # a child of this supervisor — survives supervisor restart, no IO inheritance.
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    )
    proc = subprocess.Popen(
        [COMFY_EXE_PATH],
        creationflags=flags,
        close_fds=True,
        cwd=os.path.dirname(COMFY_EXE_PATH) or None,
    )
    return proc.pid


def _wait_until_online() -> tuple[bool, int]:
    deadline = time.time() + HEALTHCHECK_TIMEOUT
    elapsed_start = time.time()
    while time.time() < deadline:
        if _comfy_online():
            return True, int((time.time() - elapsed_start) * 1000)
        time.sleep(2)
    return False, int((time.time() - elapsed_start) * 1000)


def do_restart() -> dict:
    """Full restart: kill + spawn + wait for /system_stats."""
    if not _RESTART_LOCK.acquire(blocking=False):
        return {
            "ok": False,
            "error": "restart_in_progress",
            "detail": "Another restart is already running; ignored.",
        }
    try:
        t0 = time.time()
        killed = _kill_all_comfy()
        time.sleep(3)  # GPU/driver settle window
        try:
            spawn_pid = _launch_comfy()
        except FileNotFoundError as e:
            return {"ok": False, "error": "exe_not_found", "detail": str(e)}
        online, wait_ms = _wait_until_online()
        return {
            "ok": online,
            "killed": killed,
            "spawn_pid": spawn_pid,
            "comfy_online": online,
            "wait_for_online_ms": wait_ms,
            "total_ms": int((time.time() - t0) * 1000),
            "comfy_url": COMFY_URL,
            "exe_path": COMFY_EXE_PATH,
            "error": None if online else "healthcheck_timeout",
        }
    finally:
        _RESTART_LOCK.release()


class _Handler(BaseHTTPRequestHandler):
    server_version = "ComfySupervisor/1.0"

    def _json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}\n")

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True, "service": "comfy-supervisor"})
            return
        if self.path == "/status":
            procs = _list_comfy_procs()
            self._json(200, {
                "ok": True,
                "comfy_online": _comfy_online(),
                "comfy_procs": [{"pid": p.pid} for p in procs],
                "comfy_url": COMFY_URL,
                "exe_path": COMFY_EXE_PATH,
            })
            return
        self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/restart":
            try:
                result = do_restart()
                code = 200 if result.get("ok") else 503
                self._json(code, result)
            except Exception as e:
                self._json(500, {
                    "ok": False,
                    "error": "exception",
                    "detail": repr(e),
                    "traceback": traceback.format_exc(),
                })
            return
        self._json(404, {"ok": False, "error": "not_found"})


def main() -> None:
    print(f"ComfyUI Supervisor starting on {SUPERVISOR_HOST}:{SUPERVISOR_PORT}")
    print(f"  COMFY_EXE_PATH: {COMFY_EXE_PATH}")
    print(f"  COMFY_URL:      {COMFY_URL}")
    print(f"  Healthcheck:    {HEALTHCHECK_TIMEOUT}s timeout")
    if not os.path.exists(COMFY_EXE_PATH):
        print(f"WARNING: {COMFY_EXE_PATH} does not exist — /restart will fail until path is fixed.")
    srv = ThreadingHTTPServer((SUPERVISOR_HOST, SUPERVISOR_PORT), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()
