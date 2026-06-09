@echo off
REM Stage L2: launch HermesCTB API with forced UTF-8 IO so Unicode in
REM print/log statements never crashes the process on Windows cp1252.
REM Default port 8001 to avoid the trading bot squatting on 8000.

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d I:\HermesCTB

set PORT=%1
if "%PORT%"=="" set PORT=8001

py -X utf8 -m uvicorn api:app --host 127.0.0.1 --port %PORT% --log-level info
