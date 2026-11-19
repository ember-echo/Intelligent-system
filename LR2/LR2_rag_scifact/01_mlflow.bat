@echo off
setlocal
cd /d "%~dp0"
if not defined LR2_PYTHON set "LR2_PYTHON=C:\Users\FF\Documents\Codex\2026-10-26\ban\work\lr0-env\Scripts\python.exe"
set "MLFLOW_DISABLE_AGENT_HINT=1"
echo Open http://127.0.0.1:5003 after startup.
"%LR2_PYTHON%" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5003 --workers 1
pause
