@echo off
setlocal
cd /d "%~dp0"
if not defined LR1_PYTHON set "LR1_PYTHON=C:\Users\FF\Documents\Codex\2026-10-26\ban\work\lr0-env\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "LR1_PYTHON=%CD%\.venv\Scripts\python.exe"
set "MLFLOW_DISABLE_AGENT_HINT=1"
echo Open http://127.0.0.1:5002 after startup.
"%LR1_PYTHON%" launch_mlflow.py
pause
