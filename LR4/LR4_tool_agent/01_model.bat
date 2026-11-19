@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "SERVER=%~dp0..\..\LR1\runtime\bin\llama-server.exe"
set "MODEL=%~dp0..\..\LR1\runtime\model.gguf"
"%SERVER%" -m "%MODEL%" --alias Qwen3-4B-Instruct-2507-Q4_K_M --host 127.0.0.1 --port 8024 -c 16384 -np 1 -ngl 99 -t 8 -tb 8 --jinja --metrics
pause
