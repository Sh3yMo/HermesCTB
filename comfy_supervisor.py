"""
ComfyUI Host Supervisor -- tiny HTTP bridge so the Docker container can
kill + relaunch ComfyUI.exe on the Windows host.

Endpoints:
  GET  /health  -> 200 {"status": "ok"}
  GET  /status  -> process + HTTP reachability info
  POST /restart -> kill all ComfyUI.exe, relaunch, wait until online
"""

import json
import os
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

# ComfyUI Desktop reshuffled its install path in the 0.24.x line — the
# Electron build now lives under @comfyorgcomfyui-electron, while older
# installs sat directly under \Programs\ComfyUI\. Probe known locations
# instead of hardcoding a single one so a future installer reshuffle
# does not silently break Tier-2 recovery.
_LOCALAPPDATA = os.environ.get(
    "LOCALAPPDATA",
    os.path.expanduser(r"~\AppData\Local"),
)
_COMFY_EXE_CANDIDATES = [
    os.path.join(_LOCALAPPDATA, r"Programs\@comfyorgcomfyui-electron\ComfyUI.exe"),
    os.path.join(_LOCALAPPDATA, r"Programs\ComfyUI\ComfyUI.exe"),
    os.path.join(_LOCALAPPDATA, r"Programs\comfyui-electron\ComfyUI.exe"),
    os.path.join(_LOCALAPPDATA, r"Programs\comfyui\Comfy Desktop\Comfy Desktop.exe"),
]


def _resolve_comfy_exe() -> str:
    """Return the first existing ComfyUI.exe path.

    Precedence: explicit COMFY_EXE_PATH env-var > known install-path
    candidates > first candidate as last-resort default (lets the
    /restart endpoint return a useful exe_not_found message instead of
    crashing the supervisor at import time).
    """
    explicit = os.environ.get("COMFY_EXE_PATH", "").strip()
    if explicit:
        return explicit
    for candidate in _COMFY_EXE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return _COMFY_EXE_CANDIDATES[0]


COMFY_EXE_PATH = _resolve_comfy_exe()
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
# the picker; the port never binds). We instead relaunch the ComfyUI *server*
# process directly — the exact command the desktop app spawns — which binds the
# port with no picker. All paths are env-overridable; defaults match the live
# install (uv-managed CPython under AppData\Roaming\uv\python).
import glob as _glob


def _resolve_server_python() -> str:
    explicit = os.environ.get("COMFY_SERVER_PYTHON", "").strip()
    if explicit:
        return explicit
    base = os.path.join(os.path.expanduser(r"~\AppData\Roaming\uv\python"))
    hits = sorted(_glob.glob(os.path.join(base, "cpython-*", "python.exe")))
    if not hits:
        return ""
    # Comfy Desktop's managed runtime is CPython 3.12 — that env (not a newer
    # uv python) is the one with ComfyUI's dependencies installed. Prefer the
    # newest 3.12.x; only fall back to the newest overall if no 3.12 exists.
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
# main.py is relative to COMFY_SERVER_CWD, matching the desktop launch.
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


def _launch_available() -> bool:
    """True when either the direct server launch or the GUI exe is runnable."""
    if (COMFY_SERVER_PYTHON and os.path.exists(COMFY_SERVER_PYTHON)
            and os.path.isdir(COMFY_SERVER_CWD)):
        return True
    return os.path.exists(COMFY_EXE_PATH)


def _launch_detail() -> str:
    return (
        f"no launchable ComfyUI found (server_python={COMFY_SERVER_PYTHON!r}, "
        f"cwd={COMFY_SERVER_CWD!r}, exe={COMFY_EXE_PATH!r})"
    )


# Process image names to match when killing. The Electron desktop build runs
# as "Comfy Desktop.exe" (no "comfyui" substring), the portable/electron build
# as "ComfyUI.exe" -- match both so /restart actually finds the running app.
_COMFY_PROC_NEEDLES = ("comfyui", "comfy desktop")


def _comfy_processes():
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            name = (p.info["name"] or "").lower()
            if any(needle in name for needle in _COMFY_PROC_NEEDLES):
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs


def _is_comfy_online():
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3):
            return True
    except Exception:
        return False


def _server_listener_pids(port: int):
    """PIDs LISTENing on the ComfyUI port. The real server is a plain
    'python.exe' (uv-managed) that the name-based scan above never matches —
    so the only reliable handle on it is the socket it binds."""
    pids = set()
    try:
        for c in psutil.net_connections(kind="inet"):
            if (c.laddr and c.laddr.port == port
                    and c.status == psutil.CONN_LISTEN and c.pid):
                pids.add(c.pid)
    except (psutil.AccessDenied, RuntimeError):
        pass
    return pids


def _kill_targets():
    """Electron 'Comfy Desktop' processes + the process bound to the ComfyUI
    port and its whole tree. Never the supervisor itself (kill by socket/tree,
    never by a bare 'python' name match, so unrelated pythons survive)."""
    self_pid = os.getpid()
    targets: dict[int, "psutil.Process"] = {}
    for p in _comfy_processes():
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


def _kill_comfy():
    procs = _kill_targets()
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(procs, timeout=KILL_TIMEOUT)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _launch_comfy():
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    # Prefer the direct headless server launch (no Electron cloud/local picker).
    if (COMFY_SERVER_PYTHON and os.path.exists(COMFY_SERVER_PYTHON)
            and os.path.isdir(COMFY_SERVER_CWD)):
        print(f"[supervisor] launching server: {COMFY_SERVER_PYTHON} "
              f"(cwd={COMFY_SERVER_CWD})", flush=True)
        subprocess.Popen(
            [COMFY_SERVER_PYTHON, *COMFY_SERVER_ARGS],
            cwd=COMFY_SERVER_CWD, creationflags=flags,
        )
        return
    # Fallback: the Electron desktop exe (may require a manual picker click).
    print(f"[supervisor] launching desktop exe (fallback): {COMFY_EXE_PATH}", flush=True)
    subprocess.Popen([COMFY_EXE_PATH], creationflags=flags)


def _wait_for_online():
    deadline = time.time() + HEALTHCHECK_TIMEOUT
    while time.time() < deadline:
        if _is_comfy_online():
            return True
        time.sleep(2)
    return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[supervisor] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/status":
            procs = _comfy_processes()
            online = _is_comfy_online()
            self._send_json(200, {
                "processes": [{"pid": p.pid, "name": p.name()} for p in procs],
                "http_online": online,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/restart":
            print("[supervisor] /restart received -- killing ComfyUI...", flush=True)
            _kill_comfy()
            time.sleep(1)
            if not _launch_available():
                print(f"[supervisor] {_launch_detail()}", flush=True)
                self._send_json(503, {
                    "ok": False,
                    "error": "exe_not_found",
                    "detail": _launch_detail(),
                })
                return
            print("[supervisor] Launching ComfyUI...", flush=True)
            _launch_comfy()
            online = _wait_for_online()
            if online:
                print("[supervisor] ComfyUI online.", flush=True)
                self._send_json(200, {"status": "restarted", "online": True})
            else:
                print("[supervisor] ComfyUI did not come online in time.", flush=True)
                self._send_json(504, {"status": "timeout", "online": False})
        elif self.path == "/start":
            # Stage 9: idempotent cold-start. If ComfyUI is already
            # answering /system_stats, do nothing. Otherwise launch and
            # wait. Used by Tier-2 recovery when ComfyUI was never
            # running (vs. /restart which always kills first).
            if _is_comfy_online():
                print("[supervisor] /start: ComfyUI already online, no-op.", flush=True)
                self._send_json(200, {"status": "already_online", "online": True})
                return
            if not _launch_available():
                print(f"[supervisor] /start: {_launch_detail()}", flush=True)
                self._send_json(503, {
                    "ok": False,
                    "error": "exe_not_found",
                    "detail": _launch_detail(),
                })
                return
            print("[supervisor] /start: launching ComfyUI...", flush=True)
            _launch_comfy()
            online = _wait_for_online()
            if online:
                print("[supervisor] /start: ComfyUI online.", flush=True)
                self._send_json(200, {"status": "started", "online": True})
            else:
                print("[supervisor] /start: ComfyUI did not come online in time.", flush=True)
                self._send_json(504, {"status": "timeout", "online": False})
        else:
            self._send_json(404, {"error": "not found"})


def main():
    server = ThreadingHTTPServer((SUPERVISOR_HOST, SUPERVISOR_PORT), Handler)
    print(
        f"[supervisor] Listening on {SUPERVISOR_HOST}:{SUPERVISOR_PORT} "
        f"(ComfyUI exe: {COMFY_EXE_PATH})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[supervisor] Shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
