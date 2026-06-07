@echo off
cd /d I:\HermesCTB
py -m pip install --quiet psutil >> supervisor.log 2>&1
py comfy_supervisor.py >> supervisor.log 2>&1
