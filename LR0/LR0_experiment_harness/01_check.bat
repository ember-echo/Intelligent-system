@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "LAB_PY=C:\Users\FF\Documents\Codex\2026-10-26\ban\work\lr0-env\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "MLFLOW_DISABLE_AGENT_HINT=1"
"%LAB_PY%" -m unittest discover -s tests -v
if errorlevel 1 goto end
"%LAB_PY%" verify_results.py --final
:end
pause
