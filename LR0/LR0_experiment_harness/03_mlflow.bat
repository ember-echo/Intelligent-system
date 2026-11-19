@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "LAB_PY=C:\Users\FF\Documents\Codex\2026-10-26\ban\work\lr0-env\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "MLFLOW_DISABLE_AGENT_HINT=1"
echo Open http://127.0.0.1:5001 after startup.
"%LAB_PY%" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001 --workers 1
pause
