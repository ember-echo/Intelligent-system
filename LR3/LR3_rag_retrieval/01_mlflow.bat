@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "LAB_PY=C:\Users\FF\Documents\Codex\2026-10-26\ban\work\lr0-env\Scripts\python.exe"
if not exist "%LAB_PY%" set "LAB_PY=python"
set "PYTHONIOENCODING=utf-8"
set "MLFLOW_DISABLE_AGENT_HINT=1"
echo Open http://127.0.0.1:5004 after startup.
"%LAB_PY%" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5004 --workers 1
pause
