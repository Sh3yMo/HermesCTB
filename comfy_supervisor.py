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

COMFY_EXE_PATH = os.environ.get(
    "COMFY_EXE_PATH",
    r"C:\Users\SheyMo\AppData\Local\Programs\ComfyUI\ComfyUI.exe",
)
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
            print("[supervisor] Launching ComfyUI...", flush=True)
            _launch_comfy()
            online = _wait_for_online()
            if online:
                print("[supervisor] ComfyUI online.", flush=True)
                self._send_json(200, {"status": "restarted", "online": True})
            else:
                print("[supervisor] ComfyUI did not come online in time.", flush=True)
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
