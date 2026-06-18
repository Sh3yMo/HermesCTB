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
    r"C:\Users\SheyMo\AppData\Local\Programs\ComfyUI\Comfy Desktop\Comfy Desktop.exe",
)
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
SUPERVISOR_HOST = os.environ.get("SUPERVISOR_HOST", "0.0.0.0")
SUPERVISOR_PORT = int(os.environ.get("SUPERVISOR_PORT", "8787"))
HEALTHCHECK_TIMEOUT = int(os.environ.get("HEALTHCHECK_TIMEOUT", "120"))
KILL_TIMEOUT = int(os.environ.get("KILL_TIMEOUT", "10"))


def _comfy_port() -> int:
    try:
        return int(COMFY_URL.rsplit(":", 1)[1].split("/")[0])
    except (IndexError, ValueError):
        return 8188


COMFY_PORT = _comfy_port()

# --- Direct headless server launch -----------------------------------------
# Comfy Desktop 0.4.x shows a cloud/local picker on GUI start, so relaunching
# the Electron exe no longer brings the server online unattended (it sits on
# the picker; the port never binds). And the real server runs as a plain
# 'python.exe' (uv-managed) the name scan never matches. So: kill by the
# socket on COMFY_PORT (+ its process tree) and relaunch the ComfyUI *server*
# process directly — the exact command the desktop app spawns. All paths
# env-overridable; defaults match the live install.
import glob as _glob


def _resolve_server_python() -> str:
    explicit = os.environ.get("COMFY_SERVER_PYTHON", "").strip()
    if explicit:
        return explicit
    # The Comfy Desktop launch entrypoint is the project venv's python — a uv
    # shim that sets up the environment (sys.path / managed runtime) and then
    # execs CPython. Launching the managed cpython under uv\python directly
    # (with -s) skips that setup -> ImportError on ComfyUI's deps. So prefer the
    # venv python; only fall back to the managed cpython glob if no venv exists.
    base_dir = os.environ.get("COMFY_BASE_DIR", r"I:\ComfyUI")
    venv = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv):
        return venv
    uv = os.path.expanduser(r"~\AppData\Roaming\uv\python")
    hits = sorted(_glob.glob(os.path.join(uv, "cpython-*", "python.exe")))
    if not hits:
        return ""
    pref = os.environ.get("COMFY_SERVER_PY_VERSION", "cpython-3.12")
    matching = [h for h in hits if os.path.basename(os.path.dirname(h)).startswith(pref)]
    return (matching or hits)[-1]


COMFY_SERVER_PYTHON = _resolve_server_python()
COMFY_SERVER_CWD = os.environ.get(
    "COMFY_SERVER_CWD", os.path.expanduser(r"~\ComfyUI-Installs\ComfyUI"),
)
COMFY_BASE_DIR = os.environ.get("COMFY_BASE_DIR", r"I:\ComfyUI")
_SHARED_MODEL_PATHS = os.environ.get(
    "COMFY_EXTRA_MODEL_PATHS",
    os.path.join(
        os.environ.get("APPDATA", os.path.expanduser(r"~\AppData\Roaming")),
        r"Comfy Desktop\shared_model_paths.yaml",
    ),
)
COMFY_SERVER_ARGS = [
    "-s", r"ComfyUI\main.py",
    "--feature-flag", "show_signin_button=true",
    "--base-directory", COMFY_BASE_DIR,
    "--user-directory", os.path.join(COMFY_BASE_DIR, "user"),
    "--database-url", f"sqlite:///{os.path.join(COMFY_BASE_DIR, 'user', 'comfyui.db')}",
    "--enable-manager", "--fast-disk",
    "--extra-model-paths-config", _SHARED_MODEL_PATHS,
    "--input-directory", os.path.join(COMFY_BASE_DIR, "input"),
    "--output-directory", os.path.join(COMFY_BASE_DIR, "output"),
]

_COMFY_PROC_NEEDLES = ("comfyui", "comfy desktop")


def _launch_available() -> bool:
    if (COMFY_SERVER_PYTHON and os.path.exists(COMFY_SERVER_PYTHON)
            and os.path.isdir(COMFY_SERVER_CWD)):
        return True
    return os.path.exists(COMFY_EXE_PATH)


def _launch_detail() -> str:
    return (
        f"no launchable ComfyUI found (server_python={COMFY_SERVER_PYTHON!r}, "
        f"cwd={COMFY_SERVER_CWD!r}, exe={COMFY_EXE_PATH!r})"
    )


def _server_listener_pids(port: int) -> set[int]:
    pids: set[int] = set()
    try:
        for c in psutil.net_connections(kind="inet"):
            if (c.laddr and c.laddr.port == port
                    and c.status == psutil.CONN_LISTEN and c.pid):
                pids.add(c.pid)
    except (psutil.AccessDenied, RuntimeError):
        pass
    return pids

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
    """ComfyUI Electron processes by image name ('Comfy Desktop.exe' /
    'ComfyUI.exe'). Used for /status display and as part of the kill set. The
    actual server is a plain 'python.exe' — that one is found via its socket,
    not here (see _kill_targets)."""
    out: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if any(needle in name for needle in _COMFY_PROC_NEEDLES):
                out.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def _kill_targets() -> list[psutil.Process]:
    """Electron processes + whatever binds COMFY_PORT and its whole tree
    (the uv-managed python.exe server). Never the supervisor itself; never a
    bare 'python' name match, so unrelated pythons survive."""
    self_pid = os.getpid()
    targets: dict[int, psutil.Process] = {}
    for p in _list_comfy_procs():
        targets[p.pid] = p
    for pid in _server_listener_pids(COMFY_PORT):
        try:
            proc = psutil.Process(pid)
            for member in [proc] + proc.children(recursive=True):
                targets[member.pid] = member
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    targets.pop(self_pid, None)
    return list(targets.values())


def _kill_all_comfy() -> int:
    killed = 0
    procs = _kill_targets()
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
    if not _launch_available():
        raise FileNotFoundError(_launch_detail())
    # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP so the spawned ComfyUI is not
    # a child of this supervisor — survives supervisor restart, no IO inheritance.
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    )
    # Prefer the direct headless server launch (no Electron cloud/local picker).
    if (COMFY_SERVER_PYTHON and os.path.exists(COMFY_SERVER_PYTHON)
            and os.path.isdir(COMFY_SERVER_CWD)):
        print(f"  launching server: {COMFY_SERVER_PYTHON} (cwd={COMFY_SERVER_CWD})")
        proc = subprocess.Popen(
            [COMFY_SERVER_PYTHON, *COMFY_SERVER_ARGS],
            creationflags=flags,
            close_fds=True,
            cwd=COMFY_SERVER_CWD,
        )
        return proc.pid
    # Fallback: the Electron desktop exe (may require a manual picker click).
    print(f"  launching desktop exe (fallback): {COMFY_EXE_PATH}")
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
    print(f"  SERVER_PYTHON:  {COMFY_SERVER_PYTHON or '(none — will use exe)'}")
    print(f"  SERVER_CWD:     {COMFY_SERVER_CWD}")
    if not _launch_available():
        print(f"WARNING: {_launch_detail()} — /restart will fail until paths are fixed.")
    srv = ThreadingHTTPServer((SUPERVISOR_HOST, SUPERVISOR_PORT), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()
