# ComfyUI Host Supervisor

Tiny HTTP service that lets the **Linux Docker container** (HermesCTB API)
restart **Windows ComfyUI.exe** on the host. Runs natively on Windows, exposes
port `8787` to localhost (and thus to `host.docker.internal:8787` from inside
the container).

## Why

`comfyui.py` inside the container can call `/free` on ComfyUI over HTTP, but it
cannot do a full kill + relaunch of `ComfyUI.exe` — Linux containers cannot
spawn Windows processes on the host. The supervisor closes that gap.

## Endpoints

| Method | Path      | Description                                          |
|--------|-----------|------------------------------------------------------|
| GET    | /health   | Supervisor alive                                     |
| GET    | /status   | Are ComfyUI processes running + responding?          |
| POST   | /restart  | Kill all `ComfyUI.exe` + relaunch + wait for online  |

## Run manually

```cmd
py comfy_supervisor.py
```

## Run on logon (recommended)

1. `Win+R` → `shell:startup`
2. Right-click → New → Shortcut → Browse to `start_supervisor.bat`
3. Done. Supervisor starts on every login.

Or use Task Scheduler with trigger "At log on" pointing at `start_supervisor.bat`.

## Configuration

Environment variables (defaults in parentheses):

- `COMFY_EXE_PATH` (`C:\Users\SheyMo\AppData\Local\Programs\ComfyUI\ComfyUI.exe`)
- `COMFY_URL` (`http://127.0.0.1:8188`)
- `SUPERVISOR_HOST` (`0.0.0.0`)
- `SUPERVISOR_PORT` (`8787`)
- `HEALTHCHECK_TIMEOUT` (`120` seconds)
- `KILL_TIMEOUT` (`10` seconds)

## Verifying

From your host:

```cmd
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/status
```

From inside the Docker container:

```bash
docker exec hermes-ctb-api-1 curl -s http://host.docker.internal:8787/status
```

Triggering a restart manually:

```bash
docker exec hermes-ctb-api-1 curl -s -X POST http://host.docker.internal:8787/restart
```
