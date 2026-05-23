@echo off
REM ComfyUI Host Supervisor launcher.
REM
REM Autostart: place a shortcut to this .bat in the Windows Startup folder:
REM   Win+R -> shell:startup -> right-click -> New shortcut -> point at this file.
REM
REM Or schedule via Task Scheduler with trigger "At log on".

setlocal
cd /d "%~dp0"

REM Optional overrides (uncomment + edit as needed):
REM set "COMFY_EXE_PATH=C:\Users\SheyMo\AppData\Local\Programs\ComfyUI\ComfyUI.exe"
REM set "SUPERVISOR_PORT=8787"
REM set "HEALTHCHECK_TIMEOUT=120"

py comfy_supervisor.py
endlocal
