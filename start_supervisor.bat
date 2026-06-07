@echo off
REM Supervisor auto-restart watchdog (Stage B1, 2026-06-07)
REM Loops the supervisor: if it exits for any reason, log timestamp and restart
REM after 5s. Keeps the CMD window open as visual status signal for the user.

cd /d I:\HermesCTB
py -m pip install --quiet psutil >> supervisor.log 2>&1

:loop
echo [%date% %time%] starting comfy_supervisor.py >> supervisor.log
py comfy_supervisor.py >> supervisor.log 2>&1
echo [%date% %time%] supervisor exited (errorlevel=%errorlevel%) - restarting in 5s >> supervisor.log
echo [%date% %time%] supervisor exited (errorlevel=%errorlevel%) - restarting in 5s
timeout /t 5 /nobreak >nul
goto loop
