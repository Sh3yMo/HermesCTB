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


def _comfy_processes():
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if "comfyui" in p.info["name"].lower():
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


def _kill_comfy():
    procs = _comfy_processes()
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
    subprocess.Popen(
        [COMFY_EXE_PATH],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )


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
            if not os.path.exists(COMFY_EXE_PATH):
                print(f"[supervisor] EXE not found: {COMFY_EXE_PATH}", flush=True)
                self._send_json(503, {
                    "ok": False,
                    "error": "exe_not_found",
                    "detail": f"ComfyUI.exe not found at: {COMFY_EXE_PATH}",
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
            if not os.path.exists(COMFY_EXE_PATH):
                print(f"[supervisor] /start: EXE not found: {COMFY_EXE_PATH}", flush=True)
                self._send_json(503, {
                    "ok": False,
                    "error": "exe_not_found",
                    "detail": f"ComfyUI.exe not found at: {COMFY_EXE_PATH}",
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
